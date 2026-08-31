"""Build deterministic sglang.bench_serving commands and run them in-container."""

from __future__ import annotations

import json
import re
import shlex
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from runners.container import Container
from runners.remote import CommandResult
from schemas.job_spec import JobSpec


def benchmark_artifact_paths(
    base_dir: str,
    *,
    round_label: str,
    concurrency: int,
    repeat: int,
    recovery_attempt: int,
    replica_index: int,
) -> tuple[str, str, str, str]:
    """Return collision-proof result/log/status paths with full probe identity."""
    identity = (
        f"{round_label}_c{int(concurrency)}_repeat{int(repeat)}_"
        f"recovery{int(recovery_attempt)}_replica{int(replica_index)}_"
        f"{time.time_ns()}_{uuid.uuid4().hex}"
    )
    root = base_dir.rstrip("/")
    return (
        f"{root}/{identity}.jsonl",
        f"{root}/{identity}.log",
        f"{root}/{identity}.status",
        f"{root}/{identity}.pid",
    )


def _load_benchmark_method(project_root: Path, method_id: str) -> dict:
    directory = project_root / "references" / "benchmark_methods"
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("method_id") == method_id:
            return data
    raise ValueError(f"unknown benchmark_method: {method_id}")


def _token_value(workload: dict, field: str, *, workload_id: str) -> int:
    raw: object = workload.get(field)
    if isinstance(raw, dict):
        raw = raw.get("value")
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise ValueError(f"workload {workload_id} has invalid {field}: {raw!r}")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"workload {workload_id} has invalid {field}: {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"workload {workload_id} has invalid {field}: {raw!r}")
    return value


def _lookup_dotted(data: dict, path: str) -> object:
    current: object = data
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current


def build_benchmark_command_template(
    job: JobSpec, *, project_root: str | Path
) -> str:
    """Build the canonical bench_serving template from checked-in data only."""
    root = Path(project_root)
    workloads_path = root / "catalogs" / "workloads.yaml"
    try:
        workload_catalog = yaml.safe_load(
            workloads_path.read_text(encoding="utf-8")
        ) or {}
    except OSError as exc:
        raise ValueError(f"cannot read workload catalog: {workloads_path}") from exc
    workloads = workload_catalog.get("workloads", {}) or {}
    workload = workloads.get(job.workload)
    if not isinstance(workload, dict):
        raise ValueError(f"unknown workload: {job.workload}")

    input_tokens = _token_value(workload, "input_tokens", workload_id=job.workload)
    output_tokens = _token_value(workload, "output_tokens", workload_id=job.workload)
    method = _load_benchmark_method(root, job.benchmark_method)
    if method.get("engine") != job.engine:
        raise ValueError(
            f"benchmark_method {job.benchmark_method} engine mismatch: "
            f"expected {job.engine}, got {method.get('engine')}"
        )

    traffic = method.get("traffic", {}) or {}
    concurrency_values = traffic.get("concurrency_values") or [1]
    try:
        concurrency = int(concurrency_values[0])
        multiplier = int(traffic.get("num_prompts_multiplier", 4))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"benchmark_method {job.benchmark_method} has invalid traffic settings"
        ) from exc
    if concurrency <= 0 or multiplier <= 0:
        raise ValueError(
            f"benchmark_method {job.benchmark_method} has invalid traffic settings"
        )

    output_template = str(
        (method.get("result", {}) or {}).get(
            "output_file_template", "result_{job_id}_{timestamp}.jsonl"
        )
    )
    output_template = output_template.replace("{job_id}", "${JOB_ID}").replace(
        "{timestamp}", "${TIMESTAMP}"
    )
    values = {
        "fixed_args": method.get("fixed_args", {}) or {},
        "runtime_args": method.get("runtime_args", {}) or {},
        "traffic": {"concurrency": concurrency},
        "workload": {
            "input_tokens": {"value": input_tokens},
            "output_tokens": {"value": output_tokens},
        },
        "result": {"output_file": output_template},
        "derived": {"num_prompts": concurrency * multiplier},
    }

    entrypoint = method.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError(f"benchmark_method {job.benchmark_method} has no entrypoint")
    parts = shlex.split(entrypoint)
    argument_mapping = method.get("argument_mapping", {}) or {}
    if not isinstance(argument_mapping, dict):
        raise ValueError(
            f"benchmark_method {job.benchmark_method} has invalid argument_mapping"
        )
    for source, flag in argument_mapping.items():
        value = _lookup_dotted(values, str(source))
        if value is None or value == "":
            continue
        parts.extend([str(flag), str(value)])

    if "--context-length" in parts:
        raise ValueError("benchmark command must not contain --context-length")
    return shlex.join(parts)


def substitute_placeholders(
    command: str,
    *,
    host: str,
    port: int,
    model_path: str,
    job_id: str,
    timestamp: str,
    dataset_path: str = "",
) -> str:
    """Replace the runtime placeholders left in a generated bench command."""
    replacements = {
        "${BENCHMARK_HOST}": host,
        "${BENCHMARK_PORT}": str(port),
        "${MODEL_PATH}": model_path,
        "${JOB_ID}": job_id,
        "${TIMESTAMP}": timestamp,
        "${DATASET_PATH}": dataset_path,
    }
    for placeholder, value in replacements.items():
        command = command.replace(placeholder, value)
    return command


def _set_flag(parts: list[str], flag: str, value: str) -> list[str]:
    """Set ``flag value`` in a shlex-split arg list; replace in place or append.

    Handles both ``--flag value`` and ``--flag=value`` spellings and preserves the
    position of an existing flag so the rest of the command is untouched.
    """
    for index, part in enumerate(parts):
        if part == flag and index + 1 < len(parts):
            parts[index + 1] = value
            return parts
        if part.startswith(f"{flag}="):
            parts[index] = f"{flag}={value}"
            return parts
    parts.extend([flag, value])
    return parts


def rewrite_bench_command(
    template: str, *, concurrency: int, multiplier: int
) -> tuple[str, int]:
    """Rewrite ONE bench command template for a given concurrency.

    The fairness rule: every candidate must be pressed
    with the *same* workload, and per-concurrency num_prompts must scale as a
    constant ratio of C. So we take the job's locally built template and
    deterministically rewrite ONLY ``--max-concurrency``
    (-> concurrency) and ``--num-prompts`` (-> concurrency * multiplier), leaving
    every other flag — model, input/output len, dataset, range-ratio, seed —
    byte-for-byte identical across all candidates and all concurrency probes.

    Pure and idempotent. Returns (command, num_prompts).
    """
    num_prompts = concurrency * multiplier
    parts = shlex.split(template)
    _set_flag(parts, "--max-concurrency", str(concurrency))
    _set_flag(parts, "--num-prompts", str(num_prompts))
    return shlex.join(parts), num_prompts


def run_benchmark(
    container: Container, command: str, *, timeout: int | None = 0
) -> CommandResult:
    """Execute a fully-substituted bench command inside the container."""
    return container.exec(command, timeout=timeout)


def _parse_monitored_state(result: CommandResult) -> tuple[str, int | None] | None:
    if not result.ok:
        return None
    text = result.stdout.strip()
    if text == "RUNNING":
        return "running", None
    if text == "MISSING":
        return "missing", None
    match = re.fullmatch(r"EXIT (-?\d+)", text)
    if match is not None:
        return "exit", int(match.group(1))
    return None


def _parse_process_group_state(result: CommandResult) -> str | None:
    if not result.ok:
        return None
    text = result.stdout.strip()
    return text.lower() if text in {"RUNNING", "MISSING"} else None


def _monitor_failure(
    detail: str,
    *,
    result: CommandResult | None = None,
) -> CommandResult:
    return CommandResult(
        returncode=(result.returncode if result is not None and not result.ok else 125),
        stdout=(result.stdout if result is not None else ""),
        stderr=detail,
        failure_kind=(result.failure_kind if result is not None else None),
    )


def _terminate_monitored_process(
    container: Container,
    *,
    pid: int,
    status_path: str,
    reason: str,
    now: Callable[[], float],
    sleep: Callable[[float], None],
    poll_interval_s: float,
    terminate_grace_s: float,
    poll_timeout_s: int,
    primary_result: CommandResult | None = None,
) -> CommandResult:
    diagnostics: list[str] = [reason]
    if primary_result is not None and not primary_result.ok:
        primary_detail = primary_result.stderr.strip()
        if primary_detail and primary_detail not in diagnostics:
            diagnostics.append(primary_detail)
    terminated = container.signal_process_group(
        pid, "TERM", timeout=poll_timeout_s
    )
    if not terminated.ok:
        diagnostics.append(
            terminated.stderr.strip() or f"TERM failed rc={terminated.returncode}"
        )
    deadline = now() + terminate_grace_s
    while True:
        state_result = container.process_group_state(pid, timeout=poll_timeout_s)
        group_state = _parse_process_group_state(state_result)
        if group_state is None:
            diagnostics.append(
                state_result.stderr.strip()
                or f"invalid process state: {state_result.stdout!r}"
            )
        elif group_state == "missing":
            return _monitor_failure(
                "; ".join(diagnostics), result=primary_result or terminated
            )
        remaining = deadline - now()
        if remaining <= 0:
            break
        sleep(min(poll_interval_s, remaining))

    killed = container.signal_process_group(pid, "KILL", timeout=poll_timeout_s)
    if not killed.ok:
        diagnostics.append(killed.stderr.strip() or f"KILL failed rc={killed.returncode}")
    proved_absent = False
    for _ in range(5):
        proof = container.process_group_state(pid, timeout=poll_timeout_s)
        parsed_proof = _parse_process_group_state(proof)
        if parsed_proof == "missing":
            proved_absent = True
            break
        sleep(poll_interval_s)
    if not proved_absent:
        diagnostics.append("process still present after KILL; cleanup proof failed")
    return _monitor_failure(
        "; ".join(diagnostics), result=primary_result or killed
    )


def run_benchmark_monitored(
    container: Container,
    command: str,
    *,
    log_path: str,
    result_path: str,
    status_path: str,
    pid_path: str | None = None,
    server_ports: Sequence[int],
    starter: Callable[[], CommandResult] | None = None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_s: float = 1.0,
    stall_timeout_s: float = 300.0,
    terminate_grace_s: float = 10.0,
    poll_timeout_s: int = 5,
    cancelled: Callable[[], bool | str] | None = None,
) -> CommandResult:
    """Run a benchmark without a successful-runtime ceiling.

    Only lack of independent log/result progress for ``stall_timeout_s`` is a
    time limit.  Every remote observation is a short bounded command.
    """
    effective_pid_path = pid_path or f"{status_path}.pid"
    started = (
        starter()
        if starter is not None
        else container.start_monitored(
            command,
            log_path,
            status_path,
            effective_pid_path,
            timeout=poll_timeout_s,
        )
    )
    if not started.ok:
        return _monitor_failure("benchmark start failed", result=started)
    pid_text = started.stdout.strip()
    if re.fullmatch(r"[0-9]+", pid_text) is None or int(pid_text) <= 1:
        return _monitor_failure(f"invalid benchmark pid: {started.stdout!r}")
    pid = int(pid_text)

    progress_paths = [log_path, result_path]
    expected_paths = set(progress_paths)
    last_progress: dict[str, str] | None = None
    last_progress_at = now()
    while True:
        cancellation = cancelled() if cancelled is not None else False
        if cancellation:
            cancellation_reason = (
                cancellation if isinstance(cancellation, str) else "executor lifecycle"
            )
            return _terminate_monitored_process(
                container,
                pid=pid,
                status_path=status_path,
                reason=f"benchmark cancelled by {cancellation_reason}",
                now=now,
                sleep=sleep,
                poll_interval_s=poll_interval_s,
                terminate_grace_s=terminate_grace_s,
                poll_timeout_s=poll_timeout_s,
            )
        state_result = container.monitored_state(
            pid, status_path, timeout=poll_timeout_s
        )
        parsed_state = _parse_monitored_state(state_result)
        if parsed_state is None:
            return _terminate_monitored_process(
                container,
                pid=pid,
                status_path=status_path,
                reason=(
                    state_result.stderr.strip()
                    or f"invalid benchmark process state: {state_result.stdout!r}"
                ),
                now=now,
                sleep=sleep,
                poll_interval_s=poll_interval_s,
                terminate_grace_s=terminate_grace_s,
                poll_timeout_s=poll_timeout_s,
                primary_result=state_result,
            )
        state, exit_code = parsed_state
        if state == "exit":
            if exit_code is None:
                return _monitor_failure(
                    "benchmark exit sentinel did not contain an exit code"
                )
            return CommandResult(returncode=exit_code, stdout="", stderr="")
        if state == "missing":
            return _monitor_failure(
                "benchmark process disappeared without an exit-status sentinel"
            )

        for port in server_ports:
            health = container.health(int(port), timeout=poll_timeout_s)
            if not health.ok:
                return _terminate_monitored_process(
                    container,
                    pid=pid,
                    status_path=status_path,
                    reason=f"server port {port} became unhealthy",
                    now=now,
                    sleep=sleep,
                    poll_interval_s=poll_interval_s,
                    terminate_grace_s=terminate_grace_s,
                    poll_timeout_s=poll_timeout_s,
                    primary_result=health,
                )

        progress_result = container.file_progress(
            progress_paths, timeout=poll_timeout_s
        )
        if not progress_result.ok:
            return _terminate_monitored_process(
                container,
                pid=pid,
                status_path=status_path,
                reason=(
                    "benchmark progress transport failed: "
                    + (progress_result.stderr.strip() or "unknown failure")
                ),
                now=now,
                sleep=sleep,
                poll_interval_s=poll_interval_s,
                terminate_grace_s=terminate_grace_s,
                poll_timeout_s=poll_timeout_s,
                primary_result=progress_result,
            )
        try:
            progress = json.loads(progress_result.stdout)
        except (TypeError, json.JSONDecodeError):
            progress = None
        if (
            not isinstance(progress, dict)
            or set(progress) != expected_paths
            or not all(isinstance(value, str) for value in progress.values())
        ):
            return _terminate_monitored_process(
                container,
                pid=pid,
                status_path=status_path,
                reason=f"invalid benchmark progress state: {progress_result.stdout!r}",
                now=now,
                sleep=sleep,
                poll_interval_s=poll_interval_s,
                terminate_grace_s=terminate_grace_s,
                poll_timeout_s=poll_timeout_s,
            )
        current = now()
        if last_progress is None or progress != last_progress:
            last_progress = progress
            last_progress_at = current
        elif current - last_progress_at >= stall_timeout_s:
            return _terminate_monitored_process(
                container,
                pid=pid,
                status_path=status_path,
                reason=f"benchmark stalled for {stall_timeout_s:g}s without progress",
                now=now,
                sleep=sleep,
                poll_interval_s=poll_interval_s,
                terminate_grace_s=terminate_grace_s,
                poll_timeout_s=poll_timeout_s,
            )
        sleep(poll_interval_s)


def cleanup_monitored_benchmark(
    container: Container,
    *,
    pid_path: str,
    status_path: str,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_s: float = 1.0,
    terminate_grace_s: float = 10.0,
    poll_timeout_s: int = 5,
) -> list[str]:
    """Checked cleanup callback for a possibly-started benchmark resource."""
    pid_result = container.monitored_pid(pid_path, timeout=poll_timeout_s)
    pid_text = pid_result.stdout.strip()
    if (
        not pid_result.ok
        or re.fullmatch(r"[0-9]+", pid_text) is None
        or int(pid_text) <= 1
    ):
        detail = pid_result.stderr.strip() or repr(pid_result.stdout)
        return [f"cannot read strict benchmark pid from {pid_path}: {detail}"]
    pid = int(pid_text)
    state_result = container.process_group_state(pid, timeout=poll_timeout_s)
    group_state = _parse_process_group_state(state_result)
    if group_state == "missing":
        return []
    terminated = _terminate_monitored_process(
        container,
        pid=pid,
        status_path=status_path,
        reason="lifecycle benchmark cleanup",
        now=now,
        sleep=sleep,
        poll_interval_s=poll_interval_s,
        terminate_grace_s=terminate_grace_s,
        poll_timeout_s=poll_timeout_s,
        primary_result=state_result,
    )
    if "cleanup proof failed" in terminated.stderr:
        return [terminated.stderr]
    return []
