"""Build deterministic sglang.bench_serving commands and run them in-container."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import yaml

from runners.container import Container
from runners.remote import CommandResult
from schemas.job_spec import JobSpec


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
