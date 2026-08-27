#!/usr/bin/env python3
"""Run tclaude with bounded attempts and attempt-scoped diagnostics."""

from __future__ import annotations

import argparse
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from typing import BinaryIO

_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\Z")


class _SignalState:
    def __init__(self) -> None:
        self.signum: int | None = None
        self.interrupt_count = 0
        # Python signal handlers run on the main thread and may interrupt a
        # force_kill property read; RLock avoids self-deadlock on that path.
        self._lock = threading.RLock()

    def handle(self, signum: int, _frame: object) -> None:
        with self._lock:
            if signum == signal.SIGINT:
                self.interrupt_count += 1
            if self.signum is None:
                self.signum = signum

    @property
    def force_kill(self) -> bool:
        with self._lock:
            return self.interrupt_count >= 2


def _bounded_decimal(name: str, minimum: int, maximum: int):
    def parse(value: str) -> int:
        if not _DECIMAL.fullmatch(value):
            raise argparse.ArgumentTypeError(f"{name} must be a decimal integer")
        number = int(value, 10)
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between {minimum} and {maximum}"
            )
        return number

    return parse


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout-seconds", required=True, type=_bounded_decimal("timeout", 1, 86400)
    )
    parser.add_argument(
        "--grace-seconds", required=True, type=_bounded_decimal("grace", 1, 300)
    )
    parser.add_argument(
        "--max-retries", required=True, type=_bounded_decimal("retries", 0, 10)
    )
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--stdout-suffix", required=True, choices=("jsonl", "json"))
    parser.add_argument("--success-path-file", required=True, type=Path)
    parser.add_argument("--forward-stdout", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def _run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{os.getppid()}-{secrets.token_hex(3)}"


def _copy_stream(
    source: BinaryIO,
    raw_file: BinaryIO,
    forwarded: BinaryIO | None,
    errors: Queue[BaseException],
) -> None:
    try:
        while chunk := os.read(source.fileno(), 65536):
            raw_file.write(chunk)
            raw_file.flush()
            if forwarded is not None:
                forwarded.write(chunk)
                forwarded.flush()
    except BaseException as exc:  # surfaced by the supervising thread
        errors.put(exc)
    finally:
        source.close()


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _signal_exit_code(returncode: int) -> int:
    return 128 + (-returncode) if returncode < 0 else returncode


def _signal_process_group(process: subprocess.Popen[bytes], signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _wait_for_exit(process: subprocess.Popen[bytes], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    return process.poll() is not None


def _wait_for_signal_cleanup(
    process: subprocess.Popen[bytes], seconds: float, signals: _SignalState
) -> bool:
    deadline = time.monotonic() + seconds
    while process.poll() is None and time.monotonic() < deadline:
        if signals.force_kill:
            return False
        time.sleep(0.02)
    return process.poll() is not None


def _cleanup_for_external_signal(
    process: subprocess.Popen[bytes], signals: _SignalState
) -> int:
    signum = signals.signum
    assert signum is not None
    if signum == signal.SIGINT:
        _signal_process_group(process, signal.SIGINT)
        if not _wait_for_signal_cleanup(process, 1.0, signals):
            if signals.force_kill:
                _signal_process_group(process, signal.SIGKILL)
            else:
                _signal_process_group(process, signal.SIGTERM)
                if not _wait_for_signal_cleanup(process, 2.0, signals):
                    _signal_process_group(process, signal.SIGKILL)
        exit_code = 130
    else:
        _signal_process_group(process, signal.SIGTERM)
        if not _wait_for_signal_cleanup(process, 2.0, signals):
            _signal_process_group(process, signal.SIGKILL)
        exit_code = 128 + signum
    _wait_for_exit(process, 0.5)
    return exit_code


def _join_readers(
    process: subprocess.Popen[bytes], threads: tuple[threading.Thread, threading.Thread]
) -> None:
    for thread in threads:
        thread.join(timeout=0.4)
    if any(thread.is_alive() for thread in threads):
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        for thread in threads:
            thread.join(timeout=0.1)


def _run_once(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    *,
    forward_stdout: bool,
    timeout_seconds: int,
    grace_seconds: int,
    signals: _SignalState,
) -> int:
    try:
        stdout_file = stdout_path.open("wb")
        stderr_file = stderr_path.open("wb")
    except OSError as exc:
        print(f"❌ 无法创建 tclaude attempt 日志: {exc}", file=sys.stderr)
        return 1

    try:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            print(f"❌ 无法启动 tclaude: {exc}", file=sys.stderr)
            return 127
        except OSError as exc:
            print(f"❌ 无法启动 tclaude: {exc}", file=sys.stderr)
            return 1

        assert process.stdout is not None
        assert process.stderr is not None
        errors: Queue[BaseException] = Queue()
        stdout_thread = threading.Thread(
            target=_copy_stream,
            args=(
                process.stdout,
                stdout_file,
                sys.stdout.buffer if forward_stdout else None,
                errors,
            ),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_copy_stream,
            args=(process.stderr, stderr_file, None, errors),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        deadline = time.monotonic() + timeout_seconds
        timed_out = False
        output_failed = False
        external_exit_code: int | None = None
        while process.poll() is None:
            if signals.signum is not None:
                external_exit_code = _cleanup_for_external_signal(process, signals)
                break
            if not errors.empty():
                output_failed = True
                _signal_process_group(process, signal.SIGTERM)
                if not _wait_for_exit(process, min(grace_seconds, 1)):
                    _signal_process_group(process, signal.SIGKILL)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _signal_process_group(process, signal.SIGTERM)
                grace_deadline = time.monotonic() + grace_seconds
                while process.poll() is None and time.monotonic() < grace_deadline:
                    if signals.signum is not None:
                        timed_out = False
                        external_exit_code = _cleanup_for_external_signal(process, signals)
                        break
                    time.sleep(0.02)
                if external_exit_code is None and process.poll() is None:
                    _signal_process_group(process, signal.SIGKILL)
                break
            time.sleep(0.02)

        returncode = process.wait()
        _join_readers(process, (stdout_thread, stderr_thread))
        if external_exit_code is not None:
            return external_exit_code
        if output_failed or not errors.empty():
            print(f"❌ 记录 tclaude 输出失败: {errors.get()}", file=sys.stderr)
            return 1
        if timed_out:
            return 124
        return _signal_exit_code(returncode)
    finally:
        stdout_file.close()
        stderr_file.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        args.raw_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"❌ 无法创建 raw 目录: {exc}", file=sys.stderr)
        return 1

    run_id = _run_id()
    signal_state = _SignalState()
    handled_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {
        signum: signal.signal(signum, signal_state.handle) for signum in handled_signals
    }
    try:
        maximum_attempts = args.max_retries + 1
        for attempt in range(1, maximum_attempts + 1):
            stdout_path = (
                args.raw_dir
                / f"{args.job_id}.{run_id}.attempt-{attempt}.stdout.{args.stdout_suffix}"
            )
            stderr_path = (
                args.raw_dir / f"{args.job_id}.{run_id}.attempt-{attempt}.stderr.log"
            )
            print(
                f"ℹ️  tclaude 尝试 {attempt}/{maximum_attempts}: "
                f"stdout={stdout_path} stderr={stderr_path}",
                file=sys.stderr,
            )
            returncode = _run_once(
                args.command,
                stdout_path,
                stderr_path,
                forward_stdout=args.forward_stdout,
                timeout_seconds=args.timeout_seconds,
                grace_seconds=args.grace_seconds,
                signals=signal_state,
            )
            if returncode == 0:
                try:
                    _atomic_write_text(args.success_path_file, f"{stdout_path}\n")
                except OSError as exc:
                    print(f"❌ 无法写 success-path: {exc}", file=sys.stderr)
                    return 1
                return 0
            if returncode != 124:
                return returncode
            if attempt < maximum_attempts:
                print(
                    f"⚠️  tclaude 第 {attempt} 次尝试超过 {args.timeout_seconds} 秒，"
                    "使用相同模型和参数重试…",
                    file=sys.stderr,
                )
            else:
                print(
                    f"❌ tclaude 连续 {maximum_attempts} 次尝试均超过 "
                    f"{args.timeout_seconds} 秒",
                    file=sys.stderr,
                )
        return 124
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


if __name__ == "__main__":
    raise SystemExit(main())
