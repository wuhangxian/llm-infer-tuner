from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / "runners/tclaude_guard.py"


def _process_is_alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    fields = stat.read_text().split()
    return len(fields) >= 3 and fields[2] != "Z"


def _wait_until_processes_exit(pids: list[int], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_process_is_alive(pid) for pid in pids):
            return
        time.sleep(0.02)
    assert not [pid for pid in pids if _process_is_alive(pid)]


def _wait_for_file(path: Path, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def _guard_command(
    tmp_path: Path,
    child_command: list[str],
    *,
    timeout_seconds: str = "5",
    grace_seconds: str = "1",
    max_retries: str = "1",
    forward_stdout: bool = False,
) -> tuple[list[str], Path, Path]:
    raw_dir = tmp_path / "raw"
    success_path = tmp_path / "success-path"
    command = [
        sys.executable,
        str(GUARD),
        "--timeout-seconds",
        timeout_seconds,
        "--grace-seconds",
        grace_seconds,
        "--max-retries",
        max_retries,
        "--raw-dir",
        str(raw_dir),
        "--job-id",
        "test-job",
        "--stdout-suffix",
        "jsonl",
        "--success-path-file",
        str(success_path),
    ]
    if forward_stdout:
        command.append("--forward-stdout")
    command.extend(["--", *child_command])
    return command, raw_dir, success_path


def test_success_records_attempt_and_success_path(tmp_path: Path) -> None:
    child = [
        sys.executable,
        "-c",
        "import sys; print('stdout-line'); print('stderr-line', file=sys.stderr)",
    ]
    command, raw_dir, success_path = _guard_command(tmp_path, child)

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    stdout_files = list(raw_dir.glob("*.attempt-1.stdout.jsonl"))
    stderr_files = list(raw_dir.glob("*.attempt-1.stderr.log"))
    assert len(stdout_files) == 1
    assert len(stderr_files) == 1
    assert stdout_files[0].read_text() == "stdout-line\n"
    assert stderr_files[0].read_text() == "stderr-line\n"
    assert success_path.read_text().strip() == str(stdout_files[0])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout_seconds", "0"),
        ("timeout_seconds", "86401"),
        ("timeout_seconds", "08"),
        ("grace_seconds", "0"),
        ("grace_seconds", "301"),
        ("max_retries", "-1"),
        ("max_retries", "11"),
        ("max_retries", "01"),
        ("max_retries", "abc"),
    ],
)
def test_invalid_config_fails_before_starting_child(
    tmp_path: Path, field: str, value: str
) -> None:
    marker = tmp_path / "child-started"
    child = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]
    options = {field: value}
    command, _, success_path = _guard_command(tmp_path, child, **options)

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 2
    assert not marker.exists()
    assert not success_path.exists()


def test_timeout_retries_same_command_once_then_succeeds(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    argv_log = tmp_path / "argv.jsonl"
    fake = tmp_path / "fake_tclaude.py"
    fake.write_text(
        "import json, sys, time\n"
        "from pathlib import Path\n"
        f"counter = Path({str(counter)!r})\n"
        f"argv_log = Path({str(argv_log)!r})\n"
        "attempt = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(attempt))\n"
        "with argv_log.open('a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print(f'partial-stdout-{attempt}', flush=True)\n"
        "print(f'partial-stderr-{attempt}', file=sys.stderr, flush=True)\n"
        "if attempt == 1:\n"
        "    time.sleep(30)\n"
        "print('success', flush=True)\n"
    )
    child = [sys.executable, str(fake), "--model", "claude-hy3"]
    command, raw_dir, success_path = _guard_command(
        tmp_path,
        child,
        timeout_seconds="1",
        grace_seconds="1",
        max_retries="1",
    )

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    assert counter.read_text() == "2"
    recorded_argv = [json.loads(line) for line in argv_log.read_text().splitlines()]
    assert recorded_argv == [
        ["--model", "claude-hy3"],
        ["--model", "claude-hy3"],
    ]
    attempt_1 = list(raw_dir.glob("*.attempt-1.stdout.jsonl"))
    attempt_2 = list(raw_dir.glob("*.attempt-2.stdout.jsonl"))
    assert len(attempt_1) == 1
    assert len(attempt_2) == 1
    assert "partial-stdout-1" in attempt_1[0].read_text()
    assert "partial-stdout-2" in attempt_2[0].read_text()
    assert success_path.read_text().strip() == str(attempt_2[0])


def test_exhausted_timeouts_return_124_without_success_path(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    fake = tmp_path / "always_block.py"
    fake.write_text(
        "import time\n"
        "from pathlib import Path\n"
        f"counter = Path({str(counter)!r})\n"
        "attempt = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(attempt))\n"
        "print(f'partial-{attempt}', flush=True)\n"
        "time.sleep(30)\n"
    )
    command, raw_dir, success_path = _guard_command(
        tmp_path,
        [sys.executable, str(fake)],
        timeout_seconds="1",
        grace_seconds="1",
        max_retries="1",
    )

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 124
    assert counter.read_text() == "2"
    assert len(list(raw_dir.glob("*.stdout.jsonl"))) == 2
    assert not success_path.exists()


def test_ordinary_failure_is_not_retried(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    child = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(counter)!r}).write_text('1'); raise SystemExit(42)",
    ]
    command, _, success_path = _guard_command(tmp_path, child)

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 42
    assert counter.read_text() == "1"
    assert not success_path.exists()


def test_child_signal_exit_maps_to_shell_status_without_retry(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    child = [
        sys.executable,
        "-c",
        (
            "import os, signal; from pathlib import Path; "
            f"Path({str(counter)!r}).write_text('1'); "
            "os.kill(os.getpid(), signal.SIGUSR1)"
        ),
    ]
    command, _, success_path = _guard_command(tmp_path, child)

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 128 + 10
    assert counter.read_text() == "1"
    assert not success_path.exists()


def test_timeout_kills_term_ignoring_process_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids"
    fake = tmp_path / "ignores_term.py"
    fake.write_text(
        "import os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n"
        f"Path({str(pid_file)!r}).write_text(f'{{os.getpid()}} {{child.pid}}')\n"
        "print('started', flush=True)\n"
        "time.sleep(30)\n"
    )
    command, _, success_path = _guard_command(
        tmp_path,
        [sys.executable, str(fake)],
        timeout_seconds="1",
        grace_seconds="1",
        max_retries="0",
    )

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 124, completed.stderr
    pids = [int(value) for value in pid_file.read_text().split()]
    _wait_until_processes_exit(pids)
    assert not success_path.exists()


def test_sigint_exits_130_without_retry_or_orphans(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    pid_file = tmp_path / "pids"
    fake = tmp_path / "ignores_int.py"
    fake.write_text(
        "import os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n"
        f"counter = Path({str(counter)!r})\n"
        "attempt = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(attempt))\n"
        f"Path({str(pid_file)!r}).write_text(f'{{os.getpid()}} {{child.pid}}')\n"
        "time.sleep(30)\n"
    )
    command, _, success_path = _guard_command(
        tmp_path,
        [sys.executable, str(fake)],
        timeout_seconds="30",
        grace_seconds="1",
        max_retries="1",
    )
    guard = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pids: list[int] = []
    try:
        _wait_for_file(pid_file)
        pids = [int(value) for value in pid_file.read_text().split()]
        os.kill(guard.pid, signal.SIGINT)
        _, stderr = guard.communicate(timeout=5)

        assert guard.returncode == 130, stderr
        assert counter.read_text() == "1"
        _wait_until_processes_exit(pids)
        assert not success_path.exists()
    finally:
        if guard.poll() is None:
            os.kill(guard.pid, signal.SIGKILL)
            guard.wait()
        if pids and _process_is_alive(pids[0]):
            os.killpg(pids[0], signal.SIGKILL)


def test_second_sigint_forces_immediate_kill(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids"
    fake = tmp_path / "ignores_two_ints.py"
    fake.write_text(
        "import os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n"
        f"Path({str(pid_file)!r}).write_text(f'{{os.getpid()}} {{child.pid}}')\n"
        "time.sleep(30)\n"
    )
    command, _, _ = _guard_command(
        tmp_path,
        [sys.executable, str(fake)],
        timeout_seconds="30",
        max_retries="1",
    )
    guard = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pids: list[int] = []
    try:
        _wait_for_file(pid_file)
        pids = [int(value) for value in pid_file.read_text().split()]
        started = time.monotonic()
        os.kill(guard.pid, signal.SIGINT)
        time.sleep(0.1)
        os.kill(guard.pid, signal.SIGINT)
        guard.communicate(timeout=2)

        assert guard.returncode == 130
        assert time.monotonic() - started < 1.5
        _wait_until_processes_exit(pids)
    finally:
        if guard.poll() is None:
            guard.kill()
            guard.wait()
        if pids and _process_is_alive(pids[0]):
            os.killpg(pids[0], signal.SIGKILL)


@pytest.mark.parametrize(
    ("signum", "expected"),
    [(signal.SIGTERM, 143), (signal.SIGHUP, 129)],
)
def test_external_signal_cleans_child_and_maps_status(
    tmp_path: Path, signum: signal.Signals, expected: int
) -> None:
    marker = tmp_path / "started"
    child = [
        sys.executable,
        "-c",
        f"from pathlib import Path; import time; Path({str(marker)!r}).touch(); time.sleep(30)",
    ]
    command, _, success_path = _guard_command(tmp_path, child)
    guard = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        _wait_for_file(marker)
        os.kill(guard.pid, signum)
        guard.communicate(timeout=3)

        assert guard.returncode == expected
        assert not success_path.exists()
    finally:
        if guard.poll() is None:
            guard.kill()
            guard.wait()


def test_closed_forwarding_consumer_stops_child(tmp_path: Path) -> None:
    child = [
        sys.executable,
        "-c",
        "import time\nwhile True:\n print('stream-data', flush=True)\n time.sleep(0.01)",
    ]
    command, _, success_path = _guard_command(
        tmp_path, child, timeout_seconds="5", forward_stdout=True
    )
    guard = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert guard.stdout is not None
    guard.stdout.close()

    guard.wait(timeout=3)

    assert guard.returncode != 0
    assert not success_path.exists()


def test_raw_directory_failure_does_not_start_child(tmp_path: Path) -> None:
    marker = tmp_path / "started"
    child = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]
    command, raw_dir, success_path = _guard_command(tmp_path, child)
    raw_dir.write_text("not a directory")

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 1
    assert not marker.exists()
    assert not success_path.exists()


def test_success_path_failure_is_not_retried(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    child = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(counter)!r}).write_text('1')",
    ]
    command, _, _ = _guard_command(tmp_path, child)
    missing_success_path = tmp_path / "missing" / "success-path"
    index = command.index("--success-path-file") + 1
    command[index] = str(missing_success_path)

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 1
    assert counter.read_text() == "1"
    assert not missing_success_path.exists()


def test_sigint_during_timeout_grace_takes_priority_over_retry(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    fake = tmp_path / "ignore_term_during_grace.py"
    fake.write_text(
        "import signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"counter = Path({str(counter)!r})\n"
        "attempt = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(attempt))\n"
        "time.sleep(30)\n"
    )
    command, _, _ = _guard_command(
        tmp_path,
        [sys.executable, str(fake)],
        timeout_seconds="1",
        grace_seconds="5",
        max_retries="1",
    )
    guard = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        _wait_for_file(counter)
        time.sleep(1.2)
        started = time.monotonic()
        os.kill(guard.pid, signal.SIGINT)
        guard.communicate(timeout=7)

        assert guard.returncode == 130
        assert time.monotonic() - started < 4.0
        assert counter.read_text() == "1"
    finally:
        if guard.poll() is None:
            guard.kill()
            guard.wait()
