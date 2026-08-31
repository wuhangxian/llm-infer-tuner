"""Phase-2 executor: run generated configs on a remote host and collect metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from runners.bench_runner import (
    benchmark_artifact_paths,
    build_benchmark_command_template,
    cleanup_monitored_benchmark,
    rewrite_bench_command,
    run_benchmark_monitored,
    substitute_placeholders,
)
from runners.concurrency_search import (
    SearchOutcome,
    required_sample_budget,
    search_saturation,
)
from runners.container import Container, ContainerConfig
from runners.lifecycle import (
    ExecutorLifecycle,
    LifecycleAborted,
    LifecycleInterrupted,
)
from runners.metrics import (
    SEARCH_VERDICT_STATUSES,
    ProbeStatus,
    RunResult,
    classify_known_runtime_issue,
    failed_probe_result,
    parse_bench_text,
)
from runners.preflight import (
    CandidatePlacement,
    FillHostPlacement,
    PreflightRequest,
    prepare_remote_host,
    validate_local_preflight,
)
from runners.ranker import data_health_check, passes_sla, rank_candidates
from runners.readiness import (
    ReadinessTransportError,
    make_health_probe,
    wait_until_ready,
)
from runners.remote import CommandFailureKind, CommandResult, RemoteRunner
from runners.reporting import (
    annotate_baseline_threshold,
    build_candidate_rows,
    render_candidate_preview,
    write_reports,
)
from schemas.document_io import load_candidates, load_job, load_target
from schemas.job_spec import JobSpec, SearchBudget
from schemas.parameter_contract import MAMBA_STRATEGY_PARAMETERS, normalise_parameter_name
from schemas.target_spec import TargetSpec

# Round-1 = coarse adaptive expansion over ALL candidates (no bisection); round-2
# = precise bisection over ALL candidates. Round-1 coordinates only guide probe
# order; every authoritative Round-2 concurrency is freshly sampled three times.
ROUND1_MAX_PROBES = 7   # reaches ~C=64 via 1,2,4,8,16,32,64 in one candidate
ROUND1_CONFIRM = 1      # coarse: precise round 2 will confirm the boundary
ROUND2_CONFIRM = 3      # precise: fixed three-sample majority at every concurrency
DEFAULT_TOP_K = 5
DEFAULT_MAX_CAP = 256
WARMUP_CONCURRENCY = 2  # server ready 后、正式搜索前的预热并发档(结果丢弃,只热 kernel)
# The bracketed spelling matches the real module name while keeping the literal
# ``sglang.launch_server`` out of pgrep/pkill's own shell command line.
SERVER_PROCESS_PATTERN = "[s]glang[.]launch_server"
ENGINE_VERSION_COMMAND = (
    "python -c \"import importlib.metadata as m; print(m.version('sglang'))\""
)
SERVER_CLEANUP_TERM_POLL_ATTEMPTS = 21
SERVER_CLEANUP_TERM_POLL_INTERVAL_S = 0.5
SERVER_CLEANUP_KILL_POLL_ATTEMPTS = 5
SERVER_CLEANUP_KILL_POLL_INTERVAL_S = 0.1
REMOTE_START_COMMAND_TIMEOUT_S = 30
LIFECYCLE_CLEANUP_COMMAND_TIMEOUT_S = 10
BENCHMARK_MAX_ATTEMPTS = 3
PROCESS_TERMINATE_GRACE_S = 10

DEFAULT_WORKLOADS_PATH = Path(__file__).resolve().parents[1] / "catalogs" / "workloads.yaml"


def _log(message: str) -> None:
    """Progress line to stderr (line-buffered) so a live Monitor can follow the run.

    Kept off stdout, which carries the final JSON summary; a Monitor greps these.
    """
    print(f"[executor] {message}", file=sys.stderr, flush=True)


@dataclass
class ExecutorConfig:
    """Everything the executor needs to run one job's candidates end to end."""

    job_path: Path
    configs_path: Path
    results_dir: Path
    ssh_target: str
    image_ref: str
    model_host_dir: str
    model_container_path: str
    project_root: Path
    max_candidates: int = 1
    concurrencies: list[int] | None = None  # deprecated: adaptive search now chooses probes
    port: int = 30000
    container_name: str = "llm-infer-tuner-exec"
    # Absolute path on the remote host; default:
    # $HOME/llm-infer-tuner-outputs/<job>.
    remote_outputs_dir: str = ""
    target_gpu_model: str = ""
    target_gpu_count: int = 0
    target_gpu_memory_gb: float = 0.0
    top_k: int = DEFAULT_TOP_K    # deprecated compatibility input; all candidates enter round 2
    max_cap: int = DEFAULT_MAX_CAP  # upper bound on concurrency the search will probe
    ssh_password: str = field(default="", repr=False)  # empty = key-based SSH
    max_parallel: int = 8  # 批内并发跑多少个候选(每个独占容器+GPU+端口);1 = 退回串行
    fill_host: bool = False  # round2 是否按 topology 复制实际可放置实例做整机满载实测
    allow_cross_numa: bool = False  # 显式 opt-in；默认每个 TP/副本必须保持 NUMA-local
    exclusive_host: bool = False  # 只有显式授权时才允许整机清理
    startup_stall_timeout_s: int = 300
    startup_hard_timeout_s: int = 900
    startup_max_attempts: int = 3


class CleanupError(RuntimeError):
    """A checked lifecycle cleanup failed, so resources cannot be safely reused."""


def validate_port_span(base_port: int, span: int) -> tuple[int, int]:
    """Validate one contiguous assigned/cleanup port span without side effects."""
    if type(base_port) is not int or not 1 <= base_port <= 65535:
        raise ValueError("base port must be an integer in 1..65535")
    if type(span) is not int or span < 1:
        raise ValueError("port span must be a positive integer")
    end_port = base_port + span - 1
    if end_port > 65535:
        raise ValueError(
            f"port span {base_port}-{end_port} exceeds 65535 (span={span})"
        )
    return base_port, end_port


def _prepare_local_results_dir(path: Path) -> None:
    """Create and write-probe the local result directory before any SSH work."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path,
            prefix=".llm-infer-tuner-write-check-",
        ):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"local results directory is not writable: {path}"
        ) from exc


def _load_job(job_path: Path) -> JobSpec:
    return load_job(job_path)


def _load_output_len(workload: str, *, workloads_path: Path = DEFAULT_WORKLOADS_PATH) -> int:
    """Read ``output_tokens.value`` for the job's workload; the health check's target length."""
    data = yaml.safe_load(workloads_path.read_text(encoding="utf-8")) or {}
    workloads = data.get("workloads", {}) or {}
    entry = workloads.get(workload, {}) or {}
    output_tokens = entry.get("output_tokens", 0)
    if isinstance(output_tokens, dict):
        value = output_tokens.get("value", 0)
    else:
        value = output_tokens
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_num_prompts_multiplier(method_id: str, *, project_root: Path) -> int:
    """Read ``traffic.num_prompts_multiplier`` from the benchmark-method json (default 4).

    Matched by ``method_id`` inside the file, not by filename, so it stays aligned
    with how cli/main.py resolves methods.
    """
    directory = project_root / "references" / "benchmark_methods"
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("method_id") == method_id:
            traffic = data.get("traffic", {}) or {}
            try:
                return int(traffic.get("num_prompts_multiplier", 4))
            except (TypeError, ValueError):
                return 4
    return 4


def _load_candidates(configs_path: Path, *, search: SearchBudget) -> list[dict[str, Any]]:
    return load_candidates(configs_path, search=search)


def _run_result_dict(result: RunResult) -> dict[str, Any]:
    return asdict(result)


def _search_sample_evidence(outcome: SearchOutcome) -> dict[str, Any]:
    """Serialize complete and interrupted statistical groups without flattening them."""

    return {
        "complete_groups": [
            {
                "concurrency": concurrency,
                "qualifies": group.qualifies,
                "representative": _run_result_dict(group.representative),
                "samples": [_run_result_dict(sample) for sample in group.samples],
            }
            for concurrency, group in outcome.sample_groups.items()
        ],
        "incomplete_groups": [
            {
                "concurrency": concurrency,
                "samples": [_run_result_dict(sample) for sample in samples],
            }
            for concurrency, samples in outcome.incomplete_samples.items()
        ],
    }


def _force_output_file(command: str, output_path: str) -> str:
    """Point the bench command's --output-file at a known in-container path.

    The benchmark method emits ``--output-file result_...jsonl`` (a bare
    relative name); rewrite that argument so the file lands where the executor reads it.
    Appends the flag if the command lacks one.
    """
    parts = shlex.split(command)
    for index, part in enumerate(parts):
        if part == "--output-file" and index + 1 < len(parts):
            parts[index + 1] = output_path
            return shlex.join(parts)
        if part.startswith("--output-file="):
            parts[index] = f"--output-file={output_path}"
            return shlex.join(parts)
    parts.extend(["--output-file", output_path])
    return shlex.join(parts)


@dataclass
class _CandidateContext:
    """Shared handles threaded into every candidate's server lifecycle + evaluate()."""

    container: Container
    config: ExecutorConfig
    job: JobSpec
    bench_template: str
    multiplier: int
    outputs_container_path: str
    port: int = 30000  # per-candidate port (overrides config.port in parallel mode)
    # 整机满载(fill_host)时,本候选复制成 N 个实例的端口与 GPU 切片(一一对应)。
    # None = 单实例,只用 port 起一个 server —— round1 粗筛与非满载路径的原行为。
    replica_ports: list[int] | None = None
    replica_gpus: list[str] | None = None  # 每个副本的 CUDA_VISIBLE_DEVICES 值(如 "0,1")
    container_ready: bool = False
    container_present: bool = False
    container_start_failures: list[dict[str, Any]] = field(default_factory=list)
    engine_version: str = ""
    lifecycle: ExecutorLifecycle | None = None
    container_resource: str | None = None
    replica_server_logs: dict[int, str] = field(default_factory=dict)


def _resolve_bench_template(
    job: JobSpec, *, config: ExecutorConfig
) -> str:
    """Build one locally validated command template for every candidate/probe."""
    return build_benchmark_command_template(
        job, project_root=config.project_root
    )


def _aggregate_replicas(replicas: list[RunResult], *, expected: int) -> RunResult:
    """把整机满载的 N 个相同实例、同一并发档的实测结果聚成一条 per-host 结果。

    这是「整机满载实测」口径落地的核心,也是防双重计数的关键:
    · total_throughput / request_throughput / output_throughput / completed /
      total_output_tokens = N 个副本求和(整机真实吞吐,不再靠 floor(gpu/tp) 外推);
    · instances = 实际参与求和的副本数,full_host_measured=True 明确告诉 ranker
      直接使用该实测总和,不得按理论 floor(gpu/tp) 补齐 NUMA 碎片;
    · 延迟(ttft/tpot)取各副本里最差的一个 —— 满载下 SLA 必须每个副本都过;
    · success_rate / avg_output_tokens 取最差,duration 取最长;
    · 只要有副本缺失(len<expected)或本身不健康,整体标记为失败,绝不拿"部分副本"
      的求和冒充满载 goodput(那会高估)。
    """
    healthy = [r for r in replicas if r is not None and r.status == ProbeStatus.OK]
    concurrency = replicas[0].concurrency if replicas else 0
    candidate_id = replicas[0].candidate_id if replicas else "unknown"
    tp_size = replicas[0].tp_size if replicas else 1
    if not healthy or len(healthy) < expected:
        failure_priority = {
            ProbeStatus.RUNTIME_FAILED: 0,
            ProbeStatus.TRANSPORT_FAILED: 1,
            ProbeStatus.INVALID_RESULT: 2,
            ProbeStatus.STARTUP_FAILED: 3,
            ProbeStatus.BENCHMARK_FAILED: 4,
        }
        failed = min(
            (
                result
                for result in replicas
                if result is not None and result.status != ProbeStatus.OK
            ),
            key=lambda result: (
                failure_priority.get(ProbeStatus(result.status), 99),
                # A peer cancellation is consequential evidence, not the
                # originating failure.  Do not let replica ordering replace a
                # same-class bench/transport diagnostic from the failed peer.
                1
                if "peer replica" in (result.failure_reason or "").lower()
                else 0,
            ),
            default=None,
        )
        status = (
            ProbeStatus(failed.status)
            if failed is not None
            else ProbeStatus.INVALID_RESULT
        )
        num_prompts = sum(result.num_prompts for result in replicas if result is not None)
        if replicas and len(replicas) < expected:
            num_prompts += replicas[0].num_prompts * (expected - len(replicas))
        agg = _probe_failure_result(
            candidate_id,
            (
                f"满载副本不齐或不健康:健康 {len(healthy)}/{expected}"
                f"(C={concurrency}); {failed.failure_reason if failed else 'missing replica'}"
            ),
            status=status,
            tp_size=tp_size,
            concurrency=concurrency,
            num_prompts=num_prompts,
            known_issue=failed.known_issue if failed is not None else None,
        )
        agg.concurrency = concurrency
        agg.tp_size = tp_size
        agg.instances = expected
        agg.full_host_measured = True
        if failed is not None:
            agg.raw.update(failed.raw)
        return agg
    aggregate_values = {
        "success_rate": min(r.success_rate for r in healthy),
        "request_throughput": sum(r.request_throughput for r in healthy),
        "output_throughput": sum(r.output_throughput for r in healthy),
        "total_throughput": sum(r.total_throughput for r in healthy),
        "mean_ttft_ms": max(r.mean_ttft_ms for r in healthy),
        "p99_ttft_ms": max(r.p99_ttft_ms for r in healthy),
        "mean_tpot_ms": max(r.mean_tpot_ms for r in healthy),
        "p99_tpot_ms": max(r.p99_tpot_ms for r in healthy),
        "avg_output_tokens": min(r.avg_output_tokens for r in healthy),
        "duration": max(r.duration for r in healthy),
    }
    invalid_derived = next(
        (
            name
            for name, value in aggregate_values.items()
            if not _is_finite_nonnegative_metric(value)
        ),
        None,
    )
    if invalid_derived is not None:
        agg = _probe_failure_result(
            candidate_id,
            f"满载聚合产生非法指标: {invalid_derived}",
            status=ProbeStatus.INVALID_RESULT,
            tp_size=tp_size,
            concurrency=concurrency,
            num_prompts=sum(r.num_prompts for r in healthy),
        )
        agg.instances = expected
        agg.full_host_measured = True
        return agg

    return RunResult(
        candidate_id=candidate_id,
        concurrency=concurrency,
        num_prompts=sum(r.num_prompts for r in healthy),
        completed=sum(r.completed for r in healthy),
        success_rate=aggregate_values["success_rate"],
        request_throughput=aggregate_values["request_throughput"],
        output_throughput=aggregate_values["output_throughput"],
        total_throughput=aggregate_values["total_throughput"],
        mean_ttft_ms=aggregate_values["mean_ttft_ms"],
        p99_ttft_ms=aggregate_values["p99_ttft_ms"],
        mean_tpot_ms=aggregate_values["mean_tpot_ms"],
        p99_tpot_ms=aggregate_values["p99_tpot_ms"],
        total_output_tokens=sum(r.total_output_tokens for r in healthy),
        avg_output_tokens=aggregate_values["avg_output_tokens"],
        duration=aggregate_values["duration"],
        tp_size=tp_size,
        instances=len(healthy),
        full_host_measured=True,
        status="ok",
    )


def _is_finite_nonnegative_metric(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric >= 0


def _probe_failure_result(
    candidate_id: str,
    reason: str,
    *,
    status: ProbeStatus,
    tp_size: int,
    concurrency: int,
    num_prompts: int,
    known_issue: str | None = None,
) -> RunResult:
    return failed_probe_result(
        candidate_id,
        status=status,
        reason=reason,
        tp_size=tp_size,
        concurrency=concurrency,
        num_prompts=num_prompts,
        known_issue=known_issue,
    )


def _is_transport_failure(result: CommandResult | Any) -> bool:
    kind = getattr(result, "failure_kind", None)
    return kind in {CommandFailureKind.TRANSPORT, CommandFailureKind.TIMEOUT} or (
        kind is None and getattr(result, "returncode", None) == 255
    )


def _command_failure_detail(result: CommandResult | Any) -> str:
    detail = (getattr(result, "stderr", "") or getattr(result, "stdout", "")).strip()
    return detail or f"exit {getattr(result, 'returncode', 'unknown')}"


def _candidate_param(candidate: dict[str, Any] | None, name: str) -> str | None:
    if not candidate:
        return None
    params = candidate.get("params", {})
    if not isinstance(params, dict):
        return None
    for spelling in (name, name.replace("_", "-")):
        value = params.get(spelling)
        if value is not None:
            return str(value)
    return None


def _classify_probe_for_search(result: RunResult, *, output_len: int, sla: Any) -> bool:
    """Assign the sole two verdict statuses after strict data-health validation."""

    if result.status != ProbeStatus.OK:
        return False
    healthy, reason = data_health_check(result, output_len=output_len)
    if not healthy:
        result.status = ProbeStatus.INVALID_RESULT
        result.failure_reason = reason or "probe data health failed"
        return False
    if passes_sla(result, sla):
        return True
    result.status = ProbeStatus.SLA_FAILED
    result.failure_reason = (
        "SLA failed: "
        f"ttft={result.mean_ttft_ms}, tpot={result.mean_tpot_ms}, "
        f"success_rate={result.success_rate}"
    )
    return False


def _make_evaluate(
    ctx: _CandidateContext,
    candidate_id: str,
    candidate_dir: Path,
    tp_size: int = 1,
    ports: list[int] | None = None,
    *,
    output_len: int,
    round_label: str = "r1",
    candidate: dict[str, Any] | None = None,
    recover_servers=None,
):
    """Build the (evaluate, warmup) closures the search uses.

    Returns a tuple: ``evaluate(concurrency) -> RunResult`` (the closure the
    adaptive search calls per probe) and ``warmup(concurrency=WARMUP_CONCURRENCY)
    -> None`` (a one-shot pre-search benchmark whose result is discarded, run once
    per server after /health passes to absorb the first-run kernel-compile spike).

    ``ports`` 是本候选要同时压的实例端口列表:
    · 缺省(None)= 单实例,只压 ctx.port,instances=1 —— round1 粗筛的原行为,逐字不变;
    · 整机满载 = topology 实际可放置的副本端口,每档并发压全部端口、由 _aggregate_replicas
      求和成一条 instances=N 的 per-host 结果(round2 fill_host)。

    每次 evaluate 把单一模板按并发档改写,在容器里跑 bench,把 console + 结果拉回
    candidate_dir(每个真实 bench 恰好落一份证据)并解析指标。探测异常收敛成 typed
    RunResult；搜索会把该候选标为 incomplete/unknown，绝不把基础设施故障当作 SLA 边界。
    """
    container = ctx.container
    candidate_out = f"{ctx.outputs_container_path}/{candidate_id}"
    full_host_measurement = ports is not None
    bench_ports = ports if ports is not None else [ctx.port]
    lifecycle = ctx.lifecycle
    if lifecycle is None:  # pragma: no cover - production always supplies it
        raise RuntimeError("executor lifecycle is required")

    def _bench_one_port(
        concurrency: int,
        port: int,
        tag: str,
        *,
        recovery_attempt: int = 0,
        repeat: int = 0,
        peer_cancelled=None,
        monitor_server_ports: list[int] | None = None,
    ) -> RunResult:
        command_tpl, num_prompts = rewrite_bench_command(
            ctx.bench_template, concurrency=concurrency, multiplier=ctx.multiplier
        )
        timestamp = str(time.time_ns())
        replica_match = re.search(r"_i(\d+)$", tag)
        replica_index = int(replica_match.group(1)) if replica_match else 0
        (
            result_container_path,
            log_container_path,
            status_container_path,
            pid_container_path,
        ) = benchmark_artifact_paths(
            candidate_out,
            round_label=round_label,
            concurrency=concurrency,
            repeat=repeat,
            recovery_attempt=recovery_attempt,
            replica_index=replica_index,
        )
        result_name = Path(result_container_path).name
        log_name = Path(log_container_path).name
        base_command = substitute_placeholders(
            command_tpl,
            host="127.0.0.1",
            port=port,
            model_path=ctx.config.model_container_path,
            job_id=ctx.job.job_id,
            timestamp=timestamp,
        )
        command = _force_output_file(base_command, result_container_path)
        cleared = container.exec(
            "rm -f -- "
            + " ".join(
                shlex.quote(path)
                for path in (
                    result_container_path,
                    log_container_path,
                    status_container_path,
                    pid_container_path,
                )
            ),
            timeout=5,
        )
        if not cleared.ok:
            status = (
                ProbeStatus.TRANSPORT_FAILED
                if _is_transport_failure(cleared)
                else ProbeStatus.BENCHMARK_FAILED
            )
            return _probe_failure_result(
                candidate_id,
                f"cannot clear prior benchmark result: {_command_failure_detail(cleared)}",
                status=status,
                tp_size=tp_size,
                concurrency=concurrency,
                num_prompts=num_prompts,
            )
        _log(
            f"    [{candidate_id}] bench C={concurrency}{tag} @port{port} "
            f"(num_prompts={num_prompts}) ..."
        )
        resource_name = f"benchmark:{Path(pid_container_path).name}"

        def _start_benchmark() -> CommandResult:
            return lifecycle.start_resource(
                resource_name,
                starter=lambda: container.start_monitored(
                    command,
                    log_container_path,
                    status_container_path,
                    pid_container_path,
                    timeout=5,
                ),
                cleanup=lambda: cleanup_monitored_benchmark(
                    container,
                    pid_path=pid_container_path,
                    status_path=status_container_path,
                    terminate_grace_s=PROCESS_TERMINATE_GRACE_S,
                    poll_timeout_s=5,
                ),
            )

        cleanup_failures: list[str] = []
        try:
            bench_run = run_benchmark_monitored(
                container,
                command,
                log_path=log_container_path,
                result_path=result_container_path,
                status_path=status_container_path,
                pid_path=pid_container_path,
                server_ports=(
                    [port]
                    if monitor_server_ports is None
                    else monitor_server_ports
                ),
                starter=_start_benchmark,
                cancelled=lambda: (
                    "executor lifecycle"
                    if lifecycle.cancelled
                    else (
                        "peer replica infrastructure failure"
                        if peer_cancelled is not None and peer_cancelled()
                        else False
                    )
                ),
                terminate_grace_s=PROCESS_TERMINATE_GRACE_S,
                poll_timeout_s=5,
            )
        finally:
            active_exception = sys.exc_info()[1]
            if lifecycle.is_registered(resource_name):
                try:
                    cleanup_failures = cleanup_monitored_benchmark(
                        container,
                        pid_path=pid_container_path,
                        status_path=status_container_path,
                        terminate_grace_s=PROCESS_TERMINATE_GRACE_S,
                        poll_timeout_s=5,
                    )
                except LifecycleInterrupted:
                    raise
                except Exception as exc:
                    cleanup_failures = [f"checked cleanup raised {exc!r}"]
            if cleanup_failures:
                detail = (
                    f"benchmark cleanup failed for {candidate_id} C={concurrency}: "
                    + "; ".join(cleanup_failures)
                )
                lifecycle.abort(detail)
                if not isinstance(active_exception, LifecycleInterrupted):
                    raise CleanupError(detail)
            elif lifecycle.is_registered(resource_name):
                lifecycle.release(resource_name)

        console_result = container.exec(
            f"cat {shlex.quote(log_container_path)}", timeout=5
        )
        bench_console = (
            f"$ {command}\n"
            f"# exit code: {bench_run.returncode}\n\n"
            f"----- stdout -----\n"
            f"{console_result.stdout if console_result.ok else bench_run.stdout}\n"
            f"----- stderr -----\n{bench_run.stderr}\n"
        )
        (candidate_dir / log_name).write_text(bench_console, encoding="utf-8")

        if _is_transport_failure(bench_run):
            return _probe_failure_result(
                candidate_id,
                f"benchmark transport failed: {_command_failure_detail(bench_run)}",
                status=ProbeStatus.TRANSPORT_FAILED,
                tp_size=tp_size,
                concurrency=concurrency,
                num_prompts=num_prompts,
            )

        failed_server_match = re.search(
            r"server port (\d+) became unhealthy", bench_run.stderr
        )
        if failed_server_match is not None:
            failed_port = int(failed_server_match.group(1))
            failed_replica_index = bench_ports.index(failed_port)
            failed_log_path = (
                ctx.replica_server_logs.get(failed_replica_index)
                or f"{candidate_out}/server_replica{failed_replica_index}.log"
            )
            failed_log = container.exec(
                f"cat {shlex.quote(failed_log_path)}", timeout=5
            )
            result = _probe_failure_result(
                candidate_id,
                f"server port {failed_port} became unhealthy during benchmark",
                status=ProbeStatus.RUNTIME_FAILED,
                tp_size=tp_size,
                concurrency=concurrency,
                num_prompts=num_prompts,
                known_issue=classify_known_runtime_issue(
                    engine_version=ctx.engine_version,
                    attention_backend=_candidate_param(candidate, "attention_backend"),
                    speculative_algorithm=_candidate_param(
                        candidate, "speculative_algorithm"
                    ),
                    traceback_text=failed_log.stdout if failed_log.ok else "",
                ),
            )
            result.raw.update(
                {"replica_index": failed_replica_index, "port": failed_port}
            )
            return result

        post_health = container.health(port, timeout=5)
        if _is_transport_failure(post_health):
            return _probe_failure_result(
                candidate_id,
                f"post-benchmark health transport failed: "
                f"{_command_failure_detail(post_health)}",
                status=ProbeStatus.TRANSPORT_FAILED,
                tp_size=tp_size,
                concurrency=concurrency,
                num_prompts=num_prompts,
            )
        if not post_health.ok:
            server_log_path = (
                ctx.replica_server_logs.get(replica_index)
                or f"{candidate_out}/server_replica{replica_index}.log"
            )
            server_log_result = container.exec(
                f"cat {shlex.quote(server_log_path)}", timeout=5
            )
            traceback_text = server_log_result.stdout if server_log_result.ok else ""
            known_issue = classify_known_runtime_issue(
                engine_version=ctx.engine_version,
                attention_backend=_candidate_param(candidate, "attention_backend"),
                speculative_algorithm=_candidate_param(
                    candidate, "speculative_algorithm"
                ),
                traceback_text=traceback_text,
            )
            return _probe_failure_result(
                candidate_id,
                f"server unhealthy after benchmark: {_command_failure_detail(post_health)}",
                status=ProbeStatus.RUNTIME_FAILED,
                tp_size=tp_size,
                concurrency=concurrency,
                num_prompts=num_prompts,
                known_issue=known_issue,
            )
        if not bench_run.ok:
            return _probe_failure_result(
                candidate_id,
                f"bench exit {bench_run.returncode}; see {log_name}: "
                f"{_command_failure_detail(bench_run)}",
                status=ProbeStatus.BENCHMARK_FAILED,
                tp_size=tp_size,
                concurrency=concurrency,
                num_prompts=num_prompts,
            )

        read_result = container.exec(
            f"cat {shlex.quote(result_container_path)}", timeout=5
        )
        if _is_transport_failure(read_result):
            return _probe_failure_result(
                candidate_id,
                f"benchmark result transport failed: "
                f"{_command_failure_detail(read_result)}",
                status=ProbeStatus.TRANSPORT_FAILED,
                tp_size=tp_size,
                concurrency=concurrency,
                num_prompts=num_prompts,
            )
        if not read_result.ok:
            return _probe_failure_result(
                candidate_id,
                f"benchmark result read failed: {_command_failure_detail(read_result)}",
                status=ProbeStatus.BENCHMARK_FAILED,
                tp_size=tp_size,
                concurrency=concurrency,
                num_prompts=num_prompts,
            )
        text = read_result.stdout
        run_result = parse_bench_text(
            text,
            candidate_id=candidate_id,
            concurrency=concurrency,
            num_prompts=num_prompts,
            tp_size=tp_size,
            output_len=output_len,
        )
        (candidate_dir / result_name).write_text(text, encoding="utf-8")
        return run_result

    attempt_failures: list[dict[str, Any]] = []

    repeat_counters: dict[int, int] = {}

    def _run_group(
        concurrency: int, recovery_attempt: int, repeat: int
    ) -> RunResult:
        if not full_host_measurement:
            result = _bench_one_port(
                concurrency,
                bench_ports[0],
                "",
                recovery_attempt=recovery_attempt,
                repeat=repeat,
            )
            result.raw.update({"replica_index": 0, "port": bench_ports[0]})
            return result
        peer_failed = threading.Event()
        group_done = threading.Event()
        health_root: list[RunResult] = []
        health_exception: list[BaseException] = []

        def _watch_group_health() -> None:
            try:
                while not group_done.is_set() and not lifecycle.cancelled:
                    observation = container.health_many(bench_ports, timeout=5)
                    if observation.ok:
                        group_done.wait(1.0)
                        continue
                    failed_match = re.fullmatch(
                        r"FAILED (\d+) (-?\d+)", observation.stdout.strip()
                    )
                    failed_port = (
                        int(failed_match.group(1))
                        if failed_match is not None
                        else None
                    )
                    failed_replica = (
                        bench_ports.index(failed_port)
                        if failed_port in bench_ports
                        else None
                    )
                    peer_failed.set()
                    status = (
                        ProbeStatus.TRANSPORT_FAILED
                        if _is_transport_failure(observation)
                        else ProbeStatus.RUNTIME_FAILED
                    )
                    traceback_text = ""
                    if failed_replica is not None:
                        log_path = (
                            ctx.replica_server_logs.get(failed_replica)
                            or f"{candidate_out}/server_replica{failed_replica}.log"
                        )
                        pulled = container.exec(
                            f"cat {shlex.quote(log_path)}", timeout=5
                        )
                        if pulled.ok:
                            traceback_text = pulled.stdout
                    result = _probe_failure_result(
                        candidate_id,
                        (
                            f"server port {failed_port} became unhealthy"
                            if failed_port is not None
                            else "replica group health observation failed: "
                            + _command_failure_detail(observation)
                        ),
                        status=status,
                        tp_size=tp_size,
                        concurrency=concurrency,
                        num_prompts=concurrency * ctx.multiplier,
                        known_issue=classify_known_runtime_issue(
                            engine_version=ctx.engine_version,
                            attention_backend=_candidate_param(
                                candidate, "attention_backend"
                            ),
                            speculative_algorithm=_candidate_param(
                                candidate, "speculative_algorithm"
                            ),
                            traceback_text=traceback_text,
                        ),
                    )
                    if failed_replica is not None and failed_port is not None:
                        result.raw.update(
                            {"replica_index": failed_replica, "port": failed_port}
                        )
                    health_root.append(result)
                    return
            except BaseException as exc:
                health_exception.append(exc)
                peer_failed.set()

        health_thread = threading.Thread(target=_watch_group_health, daemon=True)
        health_thread.start()

        def _one(i_port):
            i, port = i_port
            if peer_failed.is_set():
                skipped = _probe_failure_result(
                    candidate_id,
                    "benchmark cancelled by peer replica infrastructure failure",
                    status=ProbeStatus.BENCHMARK_FAILED,
                    tp_size=tp_size,
                    concurrency=concurrency,
                    num_prompts=concurrency * ctx.multiplier,
                )
                skipped.raw.update({"replica_index": i, "port": port})
                return skipped
            try:
                result = _bench_one_port(
                    concurrency,
                    port,
                    f"_i{i}",
                    recovery_attempt=recovery_attempt,
                    repeat=repeat,
                    peer_cancelled=peer_failed.is_set,
                    monitor_server_ports=[],
                )
                result.raw.setdefault("replica_index", i)
                result.raw.setdefault("port", port)
            except (
                AssertionError,
                LifecycleInterrupted,
                LifecycleAborted,
                CleanupError,
            ):
                # A peer that never returns a RunResult is still the origin of
                # a group-wide infrastructure cancellation.  Wake monitored
                # siblings before ThreadPoolExecutor waits for them.
                peer_failed.set()
                raise
            except Exception as exc:
                peer_failed.set()
                result = _probe_failure_result(
                    candidate_id,
                    f"replica {i} port {port} raised: {exc!r}",
                    status=ProbeStatus.BENCHMARK_FAILED,
                    tp_size=tp_size,
                    concurrency=concurrency,
                    num_prompts=concurrency * ctx.multiplier,
                )
                result.raw.update({"replica_index": i, "port": port})
            if result.status not in SEARCH_VERDICT_STATUSES:
                peer_failed.set()
            return result

        try:
            with ThreadPoolExecutor(max_workers=len(bench_ports)) as pool:
                replicas = list(pool.map(_one, enumerate(bench_ports)))
        finally:
            group_done.set()
            health_thread.join(timeout=6)
        if health_thread.is_alive():
            raise CleanupError("replica group health watchdog did not stop")
        if health_exception:
            raise health_exception[0]
        if health_root:
            root = health_root[0]
            root_index = root.raw.get("replica_index")
            if isinstance(root_index, int) and 0 <= root_index < len(replicas):
                replicas[root_index] = root
            else:
                replicas[0] = root
        return _aggregate_replicas(replicas, expected=len(bench_ports))

    def _record_attempt_failure(result: RunResult, attempt: int) -> None:
        attempt_failures.append(
            {
                "failed_at": datetime.now(UTC).astimezone().isoformat(),
                "round": int(round_label.removeprefix("r") or 0),
                "concurrency": result.concurrency,
                "num_prompts": result.num_prompts,
                "tp_size": result.tp_size,
                "attempt": attempt,
                "stage": "probe_recovery",
                "status": str(result.status),
                "reason": result.failure_reason,
                "known_issue": result.known_issue,
                "replica_index": result.raw.get("replica_index"),
                "port": result.raw.get("port"),
            }
        )

    def _retry_allowed(counts: dict[str, int], result: RunResult) -> bool:
        category = str(result.status)
        counts[category] = counts.get(category, 0) + 1
        return counts[category] < BENCHMARK_MAX_ATTEMPTS

    def _attempt_exception(concurrency: int, exc: Exception, *, stage: str) -> RunResult:
        if isinstance(exc, AssertionError):
            raise exc
        if isinstance(exc, (CleanupError, LifecycleAborted)):
            raise exc
        return _probe_failure_result(
            candidate_id,
            f"{stage} raised at C={concurrency}: {exc!r}",
            status=ProbeStatus.BENCHMARK_FAILED,
            tp_size=tp_size,
            concurrency=concurrency,
            num_prompts=concurrency * ctx.multiplier,
        )

    def evaluate(concurrency: int) -> RunResult:
        try:
            run_result: RunResult | None = None
            repeat = repeat_counters.get(concurrency, 0)
            repeat_counters[concurrency] = repeat + 1
            category_counts: dict[str, int] = {}
            for attempt in range(1, BENCHMARK_MAX_ATTEMPTS * 5 + 1):
                if attempt > 1:
                    if recover_servers is None:
                        break
                    try:
                        restart_failure = recover_servers(attempt, concurrency, repeat)
                    except Exception as exc:
                        restart_failure = _attempt_exception(
                            concurrency, exc, stage="server recovery"
                        )
                    if restart_failure is not None:
                        run_result = restart_failure
                        _record_attempt_failure(run_result, attempt)
                        if not _retry_allowed(category_counts, run_result):
                            break
                        continue
                    # Every fresh server group is warmed before retrying the
                    # exact same concurrency; warmup evidence is discarded.
                    warmup_failure = None
                    for i, port in enumerate(bench_ports):
                        tag = "_warmup" if len(bench_ports) == 1 else f"_warmup_i{i}"
                        try:
                            warmed = _bench_one_port(
                                WARMUP_CONCURRENCY,
                                port,
                                tag,
                                recovery_attempt=attempt - 1,
                                repeat=-1,
                            )
                        except Exception as exc:
                            warmed = _attempt_exception(
                                WARMUP_CONCURRENCY, exc, stage="recovery warmup"
                            )
                        if warmed.status not in SEARCH_VERDICT_STATUSES:
                            warmup_failure = warmed
                            break
                    if warmup_failure is not None:
                        run_result = warmup_failure
                        _record_attempt_failure(run_result, attempt)
                        if not _retry_allowed(category_counts, run_result):
                            break
                        continue
                try:
                    run_result = _run_group(concurrency, attempt - 1, repeat)
                except Exception as exc:
                    run_result = _attempt_exception(concurrency, exc, stage="probe")
                if run_result.status in SEARCH_VERDICT_STATUSES:
                    break
                _record_attempt_failure(run_result, attempt)
                if not _retry_allowed(category_counts, run_result):
                    break
            if run_result is None:  # pragma: no cover - max attempts is positive
                raise RuntimeError("benchmark recovery made no attempt")
            _log(
                f"      [{candidate_id}] C={concurrency}: tput={run_result.total_throughput:.0f} "
                f"(x{run_result.instances}) "
                f"ttft={run_result.mean_ttft_ms:.0f}ms tpot={run_result.mean_tpot_ms:.1f}ms "
                f"succ={run_result.success_rate:.2f} status={run_result.status}"
            )
            return run_result
        except AssertionError:
            raise
        except (CleanupError, LifecycleAborted) as exc:
            lifecycle.abort(str(exc))
            raise
        except Exception as exc:  # typed incomplete evidence, never an SLA verdict
            return _probe_failure_result(
                candidate_id,
                f"evaluate raised at C={concurrency}: {exc!r}",
                status=ProbeStatus.BENCHMARK_FAILED,
                tp_size=tp_size,
                concurrency=concurrency,
                num_prompts=concurrency * ctx.multiplier,
            )

    def warmup(concurrency: int = WARMUP_CONCURRENCY) -> RunResult | None:
        """server ready 后、正式并发搜索前跑一次预热压测,结果**丢弃不计入搜索**。

        目的:吸收首次运行的 kernel 编译/JIT 尖峰(见 client knowledge §「丢弃第一条」),
        让正式 probe 的每一档都从热 kernel 起步,不再出现「首测 ttft 断崖高、复测断崖低」。
        用固定小并发(WARMUP_CONCURRENCY),整机满载时对所有副本端口各预热一次。
        因候选默认已 pin `--disable-radix-cache`,预热不会给正式 probe 留下 prefix 缓存,
        故只热 kernel、不喂缓存,保持各候选/各档横向可比。预热若出现基础设施
        故障则返回 typed failure，调用方把候选标成 incomplete，而不是继续发布结果。
        """
        try:
            last_failure: RunResult | None = None
            category_counts: dict[str, int] = {}
            for attempt in range(1, BENCHMARK_MAX_ATTEMPTS * 5 + 1):
                if attempt > 1:
                    if recover_servers is None:
                        break
                    try:
                        restart_failure = recover_servers(attempt, concurrency, -1)
                    except Exception as exc:
                        restart_failure = _attempt_exception(
                            concurrency, exc, stage="server recovery"
                        )
                    if restart_failure is not None:
                        last_failure = restart_failure
                        _record_attempt_failure(last_failure, attempt)
                        if not _retry_allowed(category_counts, last_failure):
                            break
                        continue
                failed = None
                for i, port in enumerate(bench_ports):
                    tag = "_warmup" if len(bench_ports) == 1 else f"_warmup_i{i}"
                    _log(
                        f"      [{candidate_id}] warmup C={concurrency} @port{port} "
                        f"(预热,结果丢弃) ..."
                    )
                    try:
                        r = _bench_one_port(
                            concurrency,
                            port,
                            tag,
                            recovery_attempt=attempt - 1,
                            repeat=-1,
                        )
                    except Exception as exc:
                        r = _attempt_exception(concurrency, exc, stage="warmup")
                    _log(
                        f"      [{candidate_id}] warmup done "
                        f"(ttft={r.mean_ttft_ms:.0f}ms status={r.status},不计入搜索)"
                    )
                    if r.status not in SEARCH_VERDICT_STATUSES:
                        failed = r
                        break
                if failed is None:
                    return None
                last_failure = failed
                _record_attempt_failure(last_failure, attempt)
                if not _retry_allowed(category_counts, last_failure):
                    break
            return last_failure
        except AssertionError:
            raise
        except (CleanupError, LifecycleAborted) as exc:
            lifecycle.abort(str(exc))
            raise
        except Exception as exc:
            return _probe_failure_result(
                candidate_id,
                f"warmup raised at C={concurrency}: {exc!r}",
                status=ProbeStatus.BENCHMARK_FAILED,
                tp_size=tp_size,
                concurrency=concurrency,
                num_prompts=concurrency * ctx.multiplier,
            )
        return None

    evaluate.attempt_failures = attempt_failures  # type: ignore[attr-defined]
    return evaluate, warmup


def _detect_numa_groups(remote: RemoteRunner, gpu_count: int) -> list[list[int]]:
    """Detect NUMA groups from nvidia-smi topo -m.

    Returns a list of NUMA groups, each containing GPU IDs.
    Example: [[0,1,2,3], [4,5,6,7]] for a dual-NUMA 8-GPU server.
    Detection is fail-closed: inventing one flat NUMA group would silently allow
    cross-socket TP placement and make benchmark comparisons topology-dependent.
    """
    try:
        result = remote.run("nvidia-smi topo -m", timeout=30)
    except Exception as exc:
        raise RuntimeError("NUMA topology detection command failed") from exc
    if not result.ok:
        raise RuntimeError("NUMA topology detection command failed")

    # Parse NUMA affinity column.
    groups: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        # 表头行以空白缩进开头,天然被下面这条跳过;
        # 不能用 not startswith("GPU0") 过滤,否则会误删 GPU1~GPU7(以及 16 卡机的 GPU10+)。
        if not line.startswith("GPU"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            gpu_id = int(parts[0].removeprefix("GPU"))
        except ValueError:
            continue
        if gpu_id >= gpu_count:
            continue
        # Find NUMA Affinity column (second to last or last numeric).
        numa_id = None
        for part in reversed(parts):
            try:
                numa_id = int(part)
                break
            except ValueError:
                continue
        if numa_id is not None:
            groups.setdefault(numa_id, []).append(gpu_id)
    if not groups:
        raise ValueError("NUMA topology output did not contain any GPU affinity rows")
    return _normalise_numa_groups(gpu_count, list(groups.values()))


def _normalise_numa_groups(
    gpu_count: int,
    numa_groups: list[list[int]] | None,
) -> list[list[int]]:
    """Validate and copy a complete physical GPU-to-NUMA topology."""
    if type(gpu_count) is not int or gpu_count < 1:
        raise ValueError("GPU count must be a positive integer")
    if numa_groups is None or not isinstance(numa_groups, list) or not numa_groups:
        raise ValueError("NUMA topology is missing")

    normalised: list[list[int]] = []
    seen: set[int] = set()
    for group in numa_groups:
        if not isinstance(group, list) or not group:
            raise ValueError("NUMA topology contains an empty or invalid group")
        copied_group: list[int] = []
        for gpu_id in group:
            if type(gpu_id) is not int:
                raise ValueError("NUMA topology GPU IDs must be integers")
            if not 0 <= gpu_id < gpu_count:
                raise ValueError(
                    f"NUMA topology GPU ID {gpu_id} is outside 0..{gpu_count - 1}"
                )
            if gpu_id in seen:
                raise ValueError(f"NUMA topology contains duplicate GPU ID {gpu_id}")
            seen.add(gpu_id)
            copied_group.append(gpu_id)
        normalised.append(copied_group)

    expected = set(range(gpu_count))
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"NUMA topology is incomplete; missing GPU IDs: {missing}")
    return normalised


def _candidate_tp_size(candidate: dict[str, Any]) -> int:
    """Return one candidate's strict positive TP, defaulting a missing TP to one."""
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    params = candidate.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"candidate {candidate.get('id', '?')}: params must be an object")
    tp_size = params.get("tp_size", 1)
    if type(tp_size) is not int or tp_size < 1:
        raise ValueError(
            f"candidate {candidate.get('id', '?')}: tp_size must be a positive integer"
        )
    return tp_size


def _allocate_gpus_and_ports(
    candidates: list[dict[str, Any]],
    gpu_count: int,
    base_port: int = 30000,
    numa_groups: list[list[int]] | None = None,
    *,
    allow_cross_numa: bool = False,
) -> list[tuple[dict[str, Any], str, int]]:
    """Assign GPU IDs and ports to each candidate with NUMA awareness.

    Same-TP candidate GPUs must stay within the same NUMA node.
    Fills one NUMA node before moving to the next.

    Example (8 GPUs, NUMA=[[0,1,2,3],[4,5,6,7]], 3 candidates TP1+TP2+TP4):
      TP1: GPU 0 (NUMA 0)
      TP2: GPU 1,2 (NUMA 0)
      TP4: GPU 4,5,6,7 (NUMA 1)
    """
    topology = _normalise_numa_groups(gpu_count, numa_groups)
    if type(allow_cross_numa) is not bool:
        raise ValueError("allow_cross_numa must be a boolean")
    validate_port_span(base_port, max(len(candidates), 1))

    result: list[tuple[dict[str, Any], str, int]] = []
    port_cursor = base_port
    free_by_group = [list(group) for group in topology]

    for candidate in candidates:
        tp_size = _candidate_tp_size(candidate)

        # Find a NUMA group that has enough remaining GPUs for this tp_size
        assigned = False
        for free_ids in free_by_group:
            if len(free_ids) >= tp_size:
                gpu_ids = free_ids[:tp_size]
                del free_ids[:tp_size]
                gpu_ids_str = "device=" + ",".join(str(g) for g in gpu_ids)
                result.append((candidate, gpu_ids_str, port_cursor))
                port_cursor += 1
                assigned = True
                break

        if not assigned:
            # Cross-NUMA placement is opt-in: the default executor policy keeps
            # every TP replica inside one NUMA group for stable measurements.
            all_gpus = [gpu_id for free_ids in free_by_group for gpu_id in free_ids]
            if len(all_gpus) >= tp_size and allow_cross_numa:
                gpu_ids = all_gpus[:tp_size]
                selected = set(gpu_ids)
                for free_ids in free_by_group:
                    free_ids[:] = [gpu_id for gpu_id in free_ids if gpu_id not in selected]
                gpu_ids_str = "device=" + ",".join(str(g) for g in gpu_ids)
                result.append((candidate, gpu_ids_str, port_cursor))
                port_cursor += 1
            elif len(all_gpus) >= tp_size:
                raise ValueError(
                    f"candidate {candidate.get('id', '?')}: tp_size={tp_size} "
                    "只能通过 cross-NUMA 放置；默认策略拒绝跨 NUMA。"
                )
            else:
                # 卡数 < tp_size:绝不静默降级成少卡启动,否则容器会以
                # --tp {tp_size} 却只暴露 len(all_gpus) 张卡的方式崩在
                # "CUDA error: invalid device ordinal"。显式报错挡在远程启动之前。
                raise ValueError(
                    f"candidate {candidate.get('id', '?')}: tp_size={tp_size} "
                    f"但只能分到 {len(all_gpus)} 张 GPU"
                    f"(gpu_count={gpu_count},NUMA={topology}),拒绝以少卡静默启动。"
                    f"请减小 tp_size 或提供足够的 GPU。"
                )

    return result


def _plan_candidate_batches(
    candidates: list[dict[str, Any]],
    gpu_count: int,
    base_port: int = 30000,
    numa_groups: list[list[int]] | None = None,
    *,
    allow_cross_numa: bool = False,
) -> list[list[dict[str, Any]]]:
    """Greedily form deterministic batches under the selected NUMA policy.

    By default, a candidate that would only fit by consuming fragments from
    multiple NUMA groups starts a fresh sequential batch. Explicit cross-NUMA
    policy permits such a placement without changing the default.
    """
    topology = _normalise_numa_groups(gpu_count, numa_groups)
    validate_port_span(base_port, 1)

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for candidate in candidates:
        # Validate the candidate independently so malformed/unplaceable input is
        # never misclassified as ordinary batch fragmentation.
        _allocate_gpus_and_ports(
            [candidate],
            gpu_count,
            base_port=base_port,
            numa_groups=topology,
            allow_cross_numa=allow_cross_numa,
        )
        trial = [*current, candidate]
        try:
            _allocate_gpus_and_ports(
                trial,
                gpu_count,
                base_port=base_port,
                numa_groups=topology,
                allow_cross_numa=allow_cross_numa,
            )
        except ValueError:
            if current:
                batches.append(current)
                current = [candidate]
            else:  # defensive: the independent placement above should catch this
                raise
        else:
            current = trial

    if current:
        batches.append(current)
    return batches


def _plan_fill_host_replica_slices(
    device_ids: list[int],
    tp_size: int,
    numa_groups: list[list[int]] | None,
    *,
    allow_cross_numa: bool = False,
) -> list[list[int]]:
    """Return container-visible GPU slices for the replicas that really fit."""
    topology = _normalise_numa_groups(len(device_ids), numa_groups)
    if type(tp_size) is not int or tp_size < 1:
        raise ValueError("tp_size must be a positive integer")
    if type(allow_cross_numa) is not bool:
        raise ValueError("allow_cross_numa must be a boolean")
    if len(device_ids) != len(set(device_ids)) or set(device_ids) != set(
        range(len(device_ids))
    ):
        raise ValueError("fill-host devices must contain each host GPU exactly once")

    host_slices: list[list[int]] = []
    leftovers: list[int] = []
    for group in topology:
        local_replica_count = len(group) // tp_size
        local_width = local_replica_count * tp_size
        host_slices.extend(
            group[offset:offset + tp_size]
            for offset in range(0, local_width, tp_size)
        )
        leftovers.extend(group[local_width:])
    if allow_cross_numa:
        cross_width = (len(leftovers) // tp_size) * tp_size
        host_slices.extend(
            leftovers[offset:offset + tp_size]
            for offset in range(0, cross_width, tp_size)
        )

    container_ordinal = {host_id: index for index, host_id in enumerate(device_ids)}
    return [
        [container_ordinal[host_id] for host_id in host_slice]
        for host_slice in host_slices
    ]


def _format_alloc(alloc: list[tuple[dict[str, Any], str, int]]) -> str:
    """把一批候选的 GPU/端口分配拼成可读字符串,用于命令行展示每个候选跑在哪几张卡上。

    形如: c001→GPU[0,1]@30000  c002→GPU[2,3]@30001
    """
    parts = []
    for candidate, gpu_ids_str, port in alloc:
        cid = candidate.get("id", "?")
        gpus = gpu_ids_str.replace("device=", "")
        parts.append(f"{cid}→GPU[{gpus}]@{port}")
    return "  ".join(parts)


def _alloc_from_preflight_batch(
    batch: tuple[CandidatePlacement, ...],
    candidates_by_id: dict[str, dict[str, Any]],
) -> list[tuple[dict[str, Any], str, int]]:
    """Translate one validated plan batch into the legacy execution tuple shape."""
    allocations: list[tuple[dict[str, Any], str, int]] = []
    for placement in batch:
        try:
            candidate = candidates_by_id[placement.candidate_id]
        except KeyError as exc:
            raise ValueError(
                f"preflight plan references unknown candidate {placement.candidate_id!r}"
            ) from exc
        gpu_spec = "device=" + ",".join(str(gpu_id) for gpu_id in placement.gpu_ids)
        allocations.append((candidate, gpu_spec, placement.port))
    return allocations


def _result_failure(label: str, result) -> str | None:
    if result.ok:
        return None
    detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
    return f"{label} rc={result.returncode}: {detail}"


def _container_inspection_state(container: Container, *, timeout: int) -> tuple[str, str]:
    """Return present/absent/unknown; only explicit Docker no-such proves absence."""
    try:
        inspection = container.inspect(timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - unknown state is unsafe
        return "unknown", f"inspect raised {exc!r}"
    if inspection.ok:
        return "present", ""
    detail = inspection.stderr.strip() or inspection.stdout.strip() or "no diagnostic"
    if inspection.returncode == 1 and re.search(
        r"\bno such (?:object|container)\b",
        detail,
        flags=re.IGNORECASE,
    ):
        return "absent", ""
    return "unknown", f"inspect rc={inspection.returncode}: {detail}"


def _cleanup_container_checked(
    container: Container,
    *,
    candidate_id: str,
    stop_first: bool,
    timeout: int,
) -> list[str]:
    """Try every container cleanup step and return all failures, including residue."""
    failures: list[str] = []
    if stop_first:
        try:
            failure = _result_failure(
                "stop",
                container.stop(timeout=timeout),
            )
            if failure:
                failures.append(failure)
        except Exception as exc:  # noqa: BLE001 - still attempt remove/postcheck
            failures.append(f"stop raised {exc!r}")
    try:
        failure = _result_failure(
            "remove",
            container.remove(force=True, timeout=timeout),
        )
        if failure:
            failures.append(failure)
    except Exception as exc:  # noqa: BLE001 - still perform postcheck
        failures.append(f"remove raised {exc!r}")
    state, detail = _container_inspection_state(container, timeout=timeout)
    if state == "present":
        failures.append("postcheck found container still present")
    elif state == "unknown":
        failures.append(f"postcheck {detail}")
    return [f"candidate {candidate_id}: {failure}" for failure in failures]


def _cleanup_servers_checked(
    container: Container,
    *,
    candidate_id: str,
    command_timeout_s: int = LIFECYCLE_CLEANUP_COMMAND_TIMEOUT_S,
) -> None:
    """Bound TERM/KILL cleanup and prove no launch_server process remains."""
    failures: list[str] = []
    pattern = shlex.quote(SERVER_PROCESS_PATTERN)

    def _poll_absent(
        *,
        label: str,
        attempts: int,
        interval_s: float,
    ) -> bool | None:
        """Return True/False for absent/still-present, None for unknown state."""
        for attempt in range(attempts):
            try:
                residual = container.exec(
                    f"pgrep -f {pattern}", timeout=command_timeout_s
                )
            except Exception as exc:  # noqa: BLE001 - unsafe unknown state
                failures.append(f"{label} pgrep raised {exc!r}")
                return None
            if residual.returncode == 1:
                return True
            if not residual.ok:
                failures.append(
                    _result_failure(f"{label} pgrep", residual)
                    or f"{label} pgrep failed"
                )
                return None
            if attempt + 1 < attempts:
                time.sleep(interval_s)
        return False

    try:
        terminated = container.exec(
            f"pkill -TERM -f {pattern}", timeout=command_timeout_s
        )
        if terminated.returncode not in (0, 1):
            failures.append(
                _result_failure("TERM pkill", terminated) or "TERM pkill failed"
            )
    except Exception as exc:  # noqa: BLE001 - still attempt proof/escalation
        failures.append(f"TERM pkill raised {exc!r}")

    absent = _poll_absent(
        label="post-TERM",
        attempts=SERVER_CLEANUP_TERM_POLL_ATTEMPTS,
        interval_s=SERVER_CLEANUP_TERM_POLL_INTERVAL_S,
    )
    if absent is False:
        try:
            killed = container.exec(
                f"pkill -KILL -f {pattern}", timeout=command_timeout_s
            )
            if killed.returncode not in (0, 1):
                failures.append(
                    _result_failure("KILL pkill", killed) or "KILL pkill failed"
                )
        except Exception as exc:  # noqa: BLE001 - still perform final proof
            failures.append(f"KILL pkill raised {exc!r}")
        absent = _poll_absent(
            label="post-KILL",
            attempts=SERVER_CLEANUP_KILL_POLL_ATTEMPTS,
            interval_s=SERVER_CLEANUP_KILL_POLL_INTERVAL_S,
        )

    if absent is False:
        failures.append("postcheck found launch_server process still running")
    elif absent is None and not failures:
        # Defensive fallback: _poll_absent records the diagnostic itself.
        failures.append("postcheck could not determine launch_server process state")
    if failures:
        raise CleanupError(
            f"server cleanup failed for candidate {candidate_id}: " + "; ".join(failures)
        )


def _run_batch_parallel(
    ctx_template: _CandidateContext,
    batch: list[tuple[dict[str, Any], str, int]],
    remote: RemoteRunner,
    outputs_host_dir: str,
    outputs_container_path: str,
    *,
    qualifies,
    output_len: int,
    round_label: str,
    max_probes: int,
    confirm: int,
    refine: bool,
    seeds_by_id: dict[str, list[RunResult]] | None = None,
    fill_host: bool = False,
    numa_groups: list[list[int]] | None = None,
    allow_cross_numa: bool = False,
    fill_host_placements: dict[str, FillHostPlacement] | None = None,
) -> dict[str, SearchOutcome]:
    """Run a batch of candidates in parallel, each in its own container.

    Each candidate gets its own docker container with specific GPUs and port.
    容器先整批一次性 start,随后每个候选的「等就绪 → 起搜索 → 压测 → 拆除」在各自
    线程里并发跑(线程安全:底层 RemoteRunner.run / container.exec 都是 subprocess.run,
    且每个候选独占容器+GPU+端口,互不串数据)。批内并发上限由 config.max_parallel 控制。

    fill_host=True(round2 整机满载):批内每个候选独占整机,容器拿到全部 gpu_count 张卡,
    在容器内按 topology 复制实际可放置的实例(各自 CUDA_VISIBLE_DEVICES 切片 +
    各自端口),压测求和 —— 实测整机满载 goodput,不再靠外推。此模式下候选应各自成批
    (调用方保证 batch 只含一个候选),批间串行。
    """
    config = ctx_template.config
    job = ctx_template.job
    bench_template = ctx_template.bench_template
    multiplier = ctx_template.multiplier
    lifecycle = ctx_template.lifecycle
    if lifecycle is None:  # pragma: no cover - all production entry points provide it
        raise RuntimeError("executor lifecycle is required")
    lifecycle_cleanup_failures: list[str] = []

    containers: list[tuple[str, Container, _CandidateContext, dict[str, Any], str, int]] = []

    # Create one container per candidate in this batch
    for candidate, gpu_ids_str, port in batch:
        candidate_id = str(candidate.get("id", "unknown"))
        _tp_size = _candidate_tp_size(candidate)
        _devices = [
            int(device_id)
            for device_id in gpu_ids_str.replace("device=", "").split(",")
            if device_id
        ]
        _n_devices = len(_devices)
        replica_ports: list[int] | None = None
        replica_gpus: list[str] | None = None
        if fill_host:
            planned = (fill_host_placements or {}).get(candidate_id)
            if planned is None:
                raise ValueError(
                    f"candidate {candidate_id}: fill-host preflight placement is missing"
                )
            replica_slices = [list(replica) for replica in planned.gpu_slices]
            n_replicas = len(replica_slices)
            if n_replicas < 1:
                raise ValueError(
                    f"candidate {candidate_id}: 整机 {_n_devices} 卡放不下一个 "
                    f"tp_size={_tp_size} 实例。"
                )
            replica_ports = list(planned.ports)
            if len(replica_ports) != n_replicas or replica_ports[0] != port:
                raise ValueError(
                    f"candidate {candidate_id}: fill-host preflight ports disagree with batch"
                )
            replica_gpus = [
                ",".join(str(gpu_id) for gpu_id in replica_slice)
                for replica_slice in replica_slices
            ]
            _log(f"[{round_label}] {candidate_id}: 整机满载 {n_replicas} 实例 "
                 f"(tp={_tp_size}×{n_replicas}={_tp_size * n_replicas}/"
                 f"{_n_devices}卡), 端口 {replica_ports}")
        else:
            # 启动前一致性断言:分到的 device 数必须等于 tp_size,否则容器会以
            # --tp {tp_size} 却少卡的方式崩在 CUDA "invalid device ordinal"。
            if _n_devices != _tp_size:
                raise ValueError(
                    f"candidate {candidate_id}: tp_size={_tp_size} 需要 {_tp_size} 张 GPU,"
                    f"实际分到 {_n_devices} 张({gpu_ids_str})。"
                    "拒绝启动以免 CUDA invalid device ordinal。"
                )
        container_name = f"{config.container_name}-{candidate_id}"
        container_config = ContainerConfig(
            image_ref=config.image_ref,
            name=container_name,
            model_host_dir=config.model_host_dir,
            model_container_path=config.model_container_path,
            outputs_host_dir=outputs_host_dir,
            outputs_container_path=outputs_container_path,
            port=port,
            gpus=gpu_ids_str,
            extra_run_args=(
                "--label",
                f"llm-infer-tuner.owner={config.container_name}",
            ),
        )
        container = Container(remote, container_config)
        ctx = _CandidateContext(
            container=container,
            config=config,
            job=job,
            bench_template=bench_template,
            multiplier=multiplier,
            outputs_container_path=outputs_container_path,
            port=port,
            replica_ports=replica_ports,
            replica_gpus=replica_gpus,
            lifecycle=lifecycle,
        )
        containers.append((candidate_id, container, ctx, candidate, gpu_ids_str, port))

    def _cleanup_created_containers() -> list[str]:
        failures: list[str] = []
        for candidate_id, container, ctx, _candidate, _gpu_ids, _port in containers:
            if not ctx.container_present:
                continue
            cleanup_failures = _cleanup_container_checked(
                container,
                candidate_id=candidate_id,
                stop_first=True,
                timeout=LIFECYCLE_CLEANUP_COMMAND_TIMEOUT_S,
            )
            failures.extend(cleanup_failures)
            if not cleanup_failures:
                ctx.container_present = False
                if ctx.container_resource is not None:
                    lifecycle.release(ctx.container_resource)
                    ctx.container_resource = None
        return failures

    # Start all containers in this batch. Container creation (including a
    # transient SSH 255) is part of bounded service startup recovery.
    abort_startup = False
    for candidate_id, container, ctx, candidate, gpu_ids_str, port in containers:
        if abort_startup:
            break
        try:
            failed_round = int(round_label.removeprefix("r"))
        except ValueError:
            failed_round = 0
        container_category_counts: dict[str, int] = {}
        per_category_container_budget = max(1, config.startup_max_attempts)
        for attempt in range(1, per_category_container_budget * 5 + 1):
            _log(
                f"[{round_label}] {candidate_id}: starting container "
                f"(GPUs={gpu_ids_str}, port={port}, "
                f"attempt {attempt}/{config.startup_max_attempts}) ..."
            )
            try:
                resource_name = (
                    f"container:{round_label}:{candidate_id}:attempt{attempt}:"
                    f"{id(ctx)}"
                )
                ctx.container_resource = resource_name
                started = lifecycle.start_resource(
                    resource_name,
                    starter=lambda container=container: container.start(
                        timeout=min(
                            config.startup_hard_timeout_s,
                            REMOTE_START_COMMAND_TIMEOUT_S,
                        )
                    ),
                    cleanup=lambda container=container, candidate_id=candidate_id: (
                        _cleanup_container_checked(
                            container,
                            candidate_id=candidate_id,
                            stop_first=True,
                            timeout=LIFECYCLE_CLEANUP_COMMAND_TIMEOUT_S,
                        )
                    ),
                )
                attempt_status = (
                    ProbeStatus.TRANSPORT_FAILED
                    if _is_transport_failure(started)
                    else ProbeStatus.STARTUP_FAILED
                )
                running_state = None
                if started.ok:
                    running_state = container.running_state(
                        timeout=min(
                            config.startup_hard_timeout_s,
                            REMOTE_START_COMMAND_TIMEOUT_S,
                        )
                    )
                    if _is_transport_failure(running_state):
                        attempt_status = ProbeStatus.TRANSPORT_FAILED
                    running = (
                        running_state.ok
                        and running_state.stdout.strip() == "true"
                    )
                else:
                    running = False
                reason = (
                    (
                        _command_failure_detail(running_state)
                        if running_state is not None and not running_state.ok
                        else ""
                    )
                    or started.stderr.strip()
                    or (started.stdout.strip() if not started.ok else "")
                    or "container not running"
                )
            except (AssertionError, CleanupError, LifecycleAborted):
                raise
            except Exception as exc:  # noqa: BLE001 - inspect decides safe retry
                running = False
                result = getattr(exc, "result", None)
                attempt_status = (
                    ProbeStatus.TRANSPORT_FAILED
                    if result is not None and _is_transport_failure(result)
                    else ProbeStatus.STARTUP_FAILED
                )
                reason = f"container start raised: {exc!r}"
            if running:
                ctx.container_ready = True
                ctx.container_present = True
                break
            ctx.container_start_failures.append(
                {
                    "failed_at": datetime.now(UTC).astimezone().isoformat(),
                    "round": failed_round,
                    "concurrency": 0,
                    "num_prompts": 0,
                    "tp_size": _candidate_tp_size(candidate),
                    "status": str(attempt_status),
                    "known_issue": None,
                    "attempt": attempt,
                    "stage": "container_start",
                    "reason": f"container did not start: {reason}",
                }
            )
            _log(f"[{round_label}] {candidate_id}: container start failed: {reason}")
            status_key = str(attempt_status)
            container_category_counts[status_key] = (
                container_category_counts.get(status_key, 0) + 1
            )
            category_exhausted = (
                container_category_counts[status_key]
                >= per_category_container_budget
            )
            state, inspect_detail = _container_inspection_state(
                container,
                timeout=REMOTE_START_COMMAND_TIMEOUT_S,
            )
            if state == "absent":
                ctx.container_present = False
                if ctx.container_resource is not None:
                    lifecycle.release(ctx.container_resource)
                    ctx.container_resource = None
                if category_exhausted:
                    break
                continue
            if state == "unknown":
                ctx.container_present = True
                lifecycle_cleanup_failures.append(
                    f"candidate {candidate_id}: startup {inspect_detail}"
                )
                abort_startup = True
                break
            ctx.container_present = True
            cleanup_failures = _cleanup_container_checked(
                container,
                candidate_id=candidate_id,
                stop_first=False,
                timeout=LIFECYCLE_CLEANUP_COMMAND_TIMEOUT_S,
            )
            if cleanup_failures:
                lifecycle_cleanup_failures.extend(cleanup_failures)
                abort_startup = True
                break
            ctx.container_present = False
            if ctx.container_resource is not None:
                lifecycle.release(ctx.container_resource)
                ctx.container_resource = None
            if category_exhausted:
                break

    if abort_startup:
        lifecycle_cleanup_failures.extend(_cleanup_created_containers())
        lifecycle.abort("; ".join(lifecycle_cleanup_failures))
        raise CleanupError(
            "executor cleanup failed during startup; refusing new resource starts: "
            + "; ".join(lifecycle_cleanup_failures)
        )

    # Run each candidate (server lifecycle + search) CONCURRENTLY within the batch.
    # 每个候选独占容器+GPU+端口,底层 subprocess.run 线程安全,故用线程池并发跑。
    # 单个候选内部抛异常(RemoteRunner OSError 等)被收敛成 typed failure
    # 结果而非中断整批;try/finally 仍保证本 batch 已启动的容器一定被 stop+remove,
    # 不会留下孤儿容器占住 GPU/端口导致下个 batch 撞车。
    def _run_one(entry) -> tuple[str, SearchOutcome]:
        candidate_id, container, ctx, candidate, gpu_ids_str, port = entry
        candidate_tp = _candidate_tp_size(candidate)
        if not ctx.container_ready:
            reason = ctx.container_start_failures[-1]["reason"]
            terminal_status = ProbeStatus(
                ctx.container_start_failures[-1]["status"]
            )
            return candidate_id, SearchOutcome(
                results=[
                    _probe_failure_result(
                        candidate_id,
                        reason,
                        status=terminal_status,
                        tp_size=candidate_tp,
                        concurrency=0,
                        num_prompts=0,
                    )
                ],
                c_star=None,
                stop_reason=terminal_status,
                last_pass=None,
                first_fail=None,
                num_evals=0,
                newly_probed=[],
                log=[reason],
                startup_attempts=len(ctx.container_start_failures),
                failures=list(ctx.container_start_failures),
                complete=False,
                certainty="unknown",
            )
        cand_seeds = None
        if seeds_by_id and candidate_id in seeds_by_id:
            cand_seeds = seeds_by_id[candidate_id]
        try:
            outcome = _run_candidate(
                ctx,
                candidate,
                qualifies=qualifies,
                output_len=output_len,
                round_label=round_label,
                max_probes=max_probes,
                confirm=confirm,
                refine=refine,
                seeds=cand_seeds,
            )
        except (AssertionError, CleanupError, LifecycleAborted):
            raise
        except Exception as exc:  # noqa: BLE001 - 单候选崩溃不拖垮同批其他候选
            _log(f"[{round_label}] {candidate_id}: 运行异常,记为失败: {exc!r}")
            if isinstance(exc, CleanupError):
                lifecycle_cleanup_failures.append(str(exc))
            try:
                failed_round = int(round_label.removeprefix("r"))
            except ValueError:
                failed_round = 0
            # RemoteRunner already turns transport failures into an explicit
            # CommandResult.failure_kind at the command boundary.  An exception
            # escaping this orchestration layer is therefore not safe to label
            # as transport merely because it inherits OSError/ConnectionError.
            status = ProbeStatus.RUNTIME_FAILED
            failure = {
                "failed_at": datetime.now(UTC).astimezone().isoformat(),
                "round": failed_round,
                "concurrency": 0,
                "num_prompts": 0,
                "tp_size": candidate_tp,
                "status": str(status),
                "known_issue": None,
                "attempt": 1,
                "reason": f"_run_candidate raised: {exc!r}",
            }
            outcome = SearchOutcome(
                results=[
                    _probe_failure_result(
                        candidate_id,
                        f"_run_candidate raised: {exc!r}",
                        status=status,
                        tp_size=candidate_tp,
                        concurrency=0,
                        num_prompts=0,
                    )
                ],
                c_star=None,
                stop_reason=status,
                last_pass=None,
                first_fail=None,
                num_evals=0,
                newly_probed=[],
                log=[f"_run_candidate raised: {exc!r}"],
                failures=[failure],
                complete=False,
                certainty="unknown",
            )
        return candidate_id, outcome

    outcomes: dict[str, SearchOutcome] = {}
    try:
        max_workers = max(1, min(len(containers), config.max_parallel))
        if max_workers == 1 or len(containers) == 1:
            for entry in containers:
                cid, outcome = _run_one(entry)
                outcomes[cid] = outcome
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for cid, outcome in pool.map(_run_one, containers):
                    outcomes[cid] = outcome
    finally:
        # Stop/remove/postcheck every still-present container. Never let one
        # failure skip cleanup of the remaining owned resources.
        # On a signal/abort, leave registered resources to the outer lifecycle's
        # reverse-order unwind: server/benchmark groups must be proved absent
        # while their owning container still exists, before container removal.
        if not lifecycle.cancelled:
            lifecycle_cleanup_failures.extend(_cleanup_created_containers())

        if lifecycle_cleanup_failures:
            lifecycle.abort("; ".join(lifecycle_cleanup_failures))
            raise CleanupError(
                "executor cleanup failed; refusing further resource reuse: "
                + "; ".join(lifecycle_cleanup_failures)
            )

    return outcomes


def _extract_failure_reason(log_path: Path) -> str:
    """Extract the last error/exception from a server log file.

    Scans the log backwards for lines containing Error/ValueError/
    AssertionError/Exception, returns the matching line plus 2 lines
    of context after it.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(server log not available)"
    lines = text.splitlines()
    # Search from the end for an error line
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        for keyword in ("ValueError", "AssertionError", "RuntimeError",
                         "Error:", "Exception", "Traceback"):
            if keyword in line:
                # Take this line + up to 2 following lines for context
                context = lines[i:i + 3]
                return " ".join(context).strip()[:500]
    # No error keyword found, return last 3 lines as fallback
    if lines:
        return " ".join(lines[-3:]).strip()[:500]
    return "(empty server log)"


def _build_cmd_from_params(
    params: dict,
    model_path_placeholder: str = "${MODEL_PATH}",
    port: int = 30000,
) -> str:
    """Build a complete launch_server command from a params dict.

    Rules:
    - Field name tp_size -> flag --tp-size (underscore to hyphen)
    - Boolean true -> bare flag (--trust-remote-code)
    - Boolean false -> omit
    - None/null -> omit
    - --model-path, --host, --port are executor-owned and auto-appended
    - --disable-radix-cache auto-appended if disable_radix_cache is true
    - Mamba radix/scheduler strategy is audit-only and never emitted
    """
    # Auto-included flags that should not be duplicated
    auto_flags = {"model_path", "host", "port"}

    parts = ["python -m sglang.launch_server"]

    # Always add model-path first
    parts.append(f"--model-path {model_path_placeholder}")

    for key, val in params.items():
        normalised_key = normalise_parameter_name(str(key))
        if (
            normalised_key in auto_flags
            or normalised_key == "is_baseline"
            or normalised_key in MAMBA_STRATEGY_PARAMETERS
        ):
            continue
        flag = "--" + normalised_key.replace("_", "-")
        if val is True:
            parts.append(flag)
        elif val is False or val is None:
            continue
        else:
            parts.append(f"{flag} {shlex.quote(str(val))}")

    # Runtime ownership means candidates cannot override these values.
    parts.append("--host 0.0.0.0")
    parts.append(f"--port {port}")

    return " ".join(parts)


def _force_disable_radix_cache(cmd: str) -> str:
    """无条件钉死关 radix/prefix cache —— 不论候选来自 cmd / params,
    也不论 config 有没有手写,最终启动命令都保证带且仅带一个 `--disable-radix-cache`。

    为什么在这里强制、而不是靠 config 写:SGLang 默认 `disable_radix_cache=False`
    (radix 开着),`_build_cmd_from_params` 又只翻译 params 里【列出】的字段 ——
    config 漏写 = flag 不拼 = 回落默认 = radix 开着。团队硬约束是「任何测试一律关
    radix」(固定 seed 复测会命中前缀,prefill 白嫖,ttft 断崖式下降污染对比),
    故在唯一的命令汇合点统一注入,任何来源都跑不掉。

    用户是否写该开关都不影响最终结果:先移除已有写法,再统一追加一个裸 flag。
    Mamba radix 策略不在这里改写;Radix Cache 关闭时该策略不生效,报告显示 inactive。
    """
    flag = "--disable-radix-cache"
    legacy_values = {"true", "false", "0", "1"}
    parts = shlex.split(cmd)
    effective_parts: list[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        normalized = part.lower()
        if normalized == flag:
            index += 1
            if index < len(parts) and parts[index].lower() in legacy_values:
                index += 1
            continue
        if normalized.startswith(f"{flag}="):
            index += 1
            continue
        effective_parts.append(part)
        index += 1
    effective_parts.append(flag)
    return shlex.join(effective_parts)


_RUNTIME_FLAG_ALIASES = {
    "--model-path": "model_path",
    "--host": "host",
    "--port": "port",
    "-p": "port",
}


def _runtime_flag_token(token: str) -> tuple[str, str | None] | None:
    for spelling, name in _RUNTIME_FLAG_ALIASES.items():
        if token == spelling:
            return name, None
        prefix = f"{spelling}="
        if token.startswith(prefix):
            return name, token[len(prefix):]
    return None


def _validate_supplied_runtime_value(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"launch --{name.replace('_', '-')} value must not be empty")
    if name == "port":
        try:
            supplied_port = int(value)
        except ValueError as exc:
            raise ValueError(f"launch port must be an integer, received {value!r}") from exc
        validate_port_span(supplied_port, 1)


def _remove_launch_runtime_args(
    argv: list[str], *, names: set[str]
) -> list[str]:
    """Remove every selected executor-owned runtime argument from tokenized argv."""
    effective: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        runtime = _runtime_flag_token(token)
        if runtime is None or runtime[0] not in names:
            effective.append(token)
            index += 1
            continue

        name, equals_value = runtime
        if equals_value is not None:
            _validate_supplied_runtime_value(name, equals_value)
            index += 1
            continue

        if index + 1 >= len(argv):
            raise ValueError(f"launch flag {token} is missing its value")
        supplied_value = argv[index + 1]
        if _runtime_flag_token(supplied_value) is not None:
            raise ValueError(f"launch flag {token} is missing its value")
        _validate_supplied_runtime_value(name, supplied_value)
        index += 2
    return effective


def _canonicalize_launch_runtime_argv(
    argv: list[str], *, model_path: str, port: int
) -> list[str]:
    """Return launch argv with exactly one executor-owned model, host, and port."""
    validate_port_span(port, 1)
    effective = _remove_launch_runtime_args(
        list(argv), names={"model_path", "host", "port"}
    )
    effective.extend(
        ["--model-path", model_path, "--host", "0.0.0.0", "--port", str(port)]
    )
    return effective


def _set_launch_runtime(cmd: str, *, model_path: str, port: int) -> str:
    """Strip candidate runtime flags and append the executor-owned values."""
    return shlex.join(
        _canonicalize_launch_runtime_argv(
            shlex.split(cmd), model_path=model_path, port=port
        )
    )


def _override_launch_port(cmd: str, port: int) -> str:
    """Set one launch command's ``--port`` regardless of its original spelling/value."""
    validate_port_span(port, 1)
    parts = _remove_launch_runtime_args(shlex.split(cmd), names={"port"})
    parts.extend(["--port", str(port)])
    return shlex.join(parts)


def _run_candidate(
    ctx: _CandidateContext,
    candidate: dict[str, Any],
    *,
    qualifies,
    output_len: int,
    round_label: str,
    max_probes: int,
    confirm: int,
    refine: bool,
    seeds: list[RunResult] | None,
) -> SearchOutcome:
    """Start the candidate's server, run one adaptive search round, tear it down.

    Returns the SearchOutcome (results = fresh median representatives only). On
    a server that never becomes ready, returns a one-element outcome with a
    startup_failed result so reporting still sees the candidate.
    """
    container = ctx.container
    config = ctx.config
    lifecycle = ctx.lifecycle
    if lifecycle is None:  # pragma: no cover - production always supplies it
        raise RuntimeError("executor lifecycle is required")
    candidate_id = str(candidate.get("id", "unknown"))
    tp_size = _candidate_tp_size(candidate)
    raw_cmd = candidate.get("cmd")
    cmd = raw_cmd if isinstance(raw_cmd, str) and raw_cmd.strip() else ""
    # Structured params are the primary input when the optional legacy cmd is absent.
    if not cmd:
        cmd = _build_cmd_from_params(candidate.get("params", {}), port=ctx.port)
    # 无条件钉死关 radix/prefix cache（团队硬约束）——不论命令来自 cmd/params，
    # 也不论 config 有没有手写，都在此汇合点统一注入，任何来源都跑不掉。
    cmd = _force_disable_radix_cache(cmd)
    candidate_dir = config.results_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = _set_launch_runtime(
        cmd, model_path=config.model_container_path, port=ctx.port
    )
    candidate_out = f"{ctx.outputs_container_path}/{candidate_id}"
    container.exec(f"mkdir -p {shlex.quote(candidate_out)}", timeout=5)

    # 副本计划:满载时用 ctx.replica_ports/replica_gpus(N 个),否则单实例 = [ctx.port]。
    replica_ports = ctx.replica_ports or [ctx.port]
    replica_gpus = ctx.replica_gpus or [None] * len(replica_ports)
    n_replicas = len(replica_ports)
    server_log = f"{candidate_out}/server.log"  # compatibility fallback only

    def _server_alive(pid: int) -> bool:
        state = container.process_state(pid, timeout=5)
        if _is_transport_failure(state):
            raise ReadinessTransportError(state, observation=f"server pid {pid}")
        return state.ok and state.stdout.strip() == "RUNNING"

    def _startup_progress(log_path: str) -> str:
        result = container.exec(
            f"stat -c '%s:%Y' {shlex.quote(log_path)}", timeout=5
        )
        if _is_transport_failure(result):
            raise ReadinessTransportError(
                result, observation=f"server log progress {log_path}"
            )
        return result.stdout.strip() if result.ok else ""

    startup_failures: list[dict[str, Any]] = list(ctx.container_start_failures)
    server_resources: list[str] = []
    server_replicas: list[tuple[int, int, str, str]] = []  # pid, port, log, pidfile

    def _cleanup_server_group() -> None:
        failures: list[str] = []
        for _pid, _port, _log, pid_path in list(server_replicas):
            try:
                failures.extend(
                    cleanup_monitored_benchmark(
                        container,
                        pid_path=pid_path,
                        status_path=f"{pid_path}.unused-status",
                        terminate_grace_s=PROCESS_TERMINATE_GRACE_S,
                        poll_timeout_s=5,
                    )
                )
            except LifecycleInterrupted:
                raise
            except Exception as exc:
                failures.append(f"{pid_path}: checked cleanup raised {exc!r}")
        try:
            _cleanup_servers_checked(container, candidate_id=candidate_id)
        except LifecycleInterrupted:
            raise
        except Exception as exc:
            failures.append(f"global server cleanup raised {exc!r}")
        if failures:
            detail = (
                f"server cleanup failed for candidate {candidate_id}: "
                + "; ".join(failures)
            )
            lifecycle.abort(detail)
            raise CleanupError(detail)
        for resource_name in server_resources:
            lifecycle.release(resource_name)
        server_resources.clear()
        server_replicas.clear()
        ctx.replica_server_logs.clear()

    ready = False
    startup_attempt = 0
    try:
        failed_round = int(round_label.removeprefix("r"))
    except ValueError:
        failed_round = 0

    startup_category_counts: dict[str, int] = {}
    per_category_startup_budget = max(1, config.startup_max_attempts)
    for startup_attempt in range(1, per_category_startup_budget * 5 + 1):
        launch_ok = True
        attempt_status = ProbeStatus.STARTUP_FAILED
        attempt_detail = ""
        failed_replica_index: int | None = None
        failed_replica_port: int | None = None
        failed_replica_log: str | None = None
        attempt_logs: list[tuple[int, int, str]] = []
        for idx, (rport, rgpu) in enumerate(zip(replica_ports, replica_gpus)):
            # 每个副本换到自己的端口;满载时再用 CUDA_VISIBLE_DEVICES 把它钉在自己那段卡上。
            rcmd = _override_launch_port(base_cmd, rport)
            if rgpu is not None:
                rcmd = f"env CUDA_VISIBLE_DEVICES={rgpu} {rcmd}"
            identity = f"{time.time_ns()}_{uuid.uuid4().hex}"
            rlog = (
                f"{candidate_out}/server_{round_label}_startup{startup_attempt}_"
                f"replica{idx}_{identity}.log"
            )
            pid_path = rlog.removesuffix(".log") + ".pid"
            attempt_logs.append((idx, rport, rlog))
            ctx.replica_server_logs[idx] = rlog
            tag = "" if n_replicas == 1 else (
                f" 副本 {idx + 1}/{n_replicas} (GPU={rgpu}@{rport})"
            )
            _log(
                f"[{round_label}] {candidate_id}: starting server{tag} "
                f"(attempt {startup_attempt}, per-class budget "
                f"{per_category_startup_budget}), "
                "waiting for /health ..."
            )
            resource_name = (
                f"server:{round_label}:{candidate_id}:attempt{startup_attempt}:"
                f"replica{idx}:port{rport}:{id(ctx)}"
            )
            server_resources.append(resource_name)
            launched = lifecycle.start_resource(
                resource_name,
                starter=lambda rcmd=rcmd, rlog=rlog, pid_path=pid_path: (
                    container.start_server_monitored(
                        rcmd,
                        rlog,
                        pid_path,
                        timeout=REMOTE_START_COMMAND_TIMEOUT_S,
                    )
                ),
                cleanup=lambda pid_path=pid_path: cleanup_monitored_benchmark(
                    container,
                    pid_path=pid_path,
                    status_path=f"{pid_path}.unused-status",
                    terminate_grace_s=PROCESS_TERMINATE_GRACE_S,
                    poll_timeout_s=5,
                ),
            )
            pid_text = launched.stdout.strip() if launched.ok else ""
            if (
                not launched.ok
                or re.fullmatch(r"[0-9]+", pid_text) is None
                or int(pid_text) <= 1
            ):
                launch_ok = False
                failed_replica_index = idx
                failed_replica_port = rport
                failed_replica_log = rlog
                attempt_detail = _command_failure_detail(launched)
                if _is_transport_failure(launched):
                    attempt_status = ProbeStatus.TRANSPORT_FAILED
                break
            server_replicas.append((int(pid_text), rport, rlog, pid_path))

        # 满载:所有副本端口都要 /health 通过才算就绪。
        ready = launch_ok and len(server_replicas) == len(replica_ports)
        for idx, (pid, rport, rlog, _pid_path) in enumerate(server_replicas):
            probe = make_health_probe(container, port=rport, timeout_s=5)
            try:
                replica_ready = wait_until_ready(
                    probe,
                    is_alive=lambda pid=pid: _server_alive(pid),
                    timeout_s=config.startup_hard_timeout_s,
                    stall_timeout_s=config.startup_stall_timeout_s,
                    progress=lambda rlog=rlog: _startup_progress(rlog),
                    cancelled=lambda: lifecycle.cancelled,
                )
            except ReadinessTransportError as exc:
                replica_ready = False
                attempt_status = ProbeStatus.TRANSPORT_FAILED
                attempt_detail = str(exc)
            if not replica_ready:
                ready = False
                failed_replica_index = idx
                failed_replica_port = rport
                failed_replica_log = rlog
                break
        if lifecycle.cancelled:
            lifecycle.assert_start_allowed()
        if ready:
            break

        failed_local_log: Path | None = None
        for idx, rport, rlog in attempt_logs:
            remote_name = Path(rlog).name
            replica_marker = f"_replica{idx}_"
            local_name = remote_name.replace(
                replica_marker,
                f"{replica_marker}port{rport}_",
                1,
            )
            local_log = candidate_dir / local_name
            _pull_container_file(container, rlog, local_log)
            if rlog == failed_replica_log:
                failed_local_log = local_log
        reason = attempt_detail
        if not reason and failed_local_log is not None:
            reason = _extract_failure_reason(failed_local_log)
        if not reason:
            reason = "server did not become ready"
        startup_failures.append(
            {
                "failed_at": datetime.now(UTC).astimezone().isoformat(),
                "round": failed_round,
                "concurrency": 0,
                "num_prompts": 0,
                "tp_size": tp_size,
                "status": str(attempt_status),
                "known_issue": None,
                "attempt": startup_attempt,
                "stage": "server_start",
                "replica_index": failed_replica_index,
                "port": failed_replica_port,
                "reason": f"server did not become ready: {reason}",
            }
        )
        _log(
            f"[{round_label}] {candidate_id}: startup attempt {startup_attempt} failed: {reason}"
        )
        _cleanup_server_group()
        status_key = str(attempt_status)
        startup_category_counts[status_key] = (
            startup_category_counts.get(status_key, 0) + 1
        )
        if startup_category_counts[status_key] >= per_category_startup_budget:
            break

    try:
        if not ready:
            _log(
                f"[{round_label}] {candidate_id}: server NOT ready after "
                f"{startup_attempt} attempt(s)"
            )
            reason = startup_failures[-1]["reason"] if startup_failures else "unknown"
            terminal_status = (
                ProbeStatus(startup_failures[-1]["status"])
                if startup_failures
                else ProbeStatus.STARTUP_FAILED
            )
            failed = _probe_failure_result(
                candidate_id,
                reason,
                status=terminal_status,
                tp_size=tp_size,
                concurrency=0,
                num_prompts=0,
            )
            outcome = SearchOutcome(
                results=[failed],
                c_star=None,
                stop_reason=terminal_status,
                last_pass=None,
                first_fail=None,
                num_evals=0,
                newly_probed=[],
                log=["server did not become ready"],
                startup_attempts=startup_attempt + len(ctx.container_start_failures),
                failures=startup_failures,
                complete=False,
                certainty="unknown",
            )
        else:
            _log(f"[{round_label}] {candidate_id}: server ready, probing concurrency ...")
            version_result = container.exec(ENGINE_VERSION_COMMAND, timeout=5)
            ctx.engine_version = (
                version_result.stdout.strip().splitlines()[0]
                if version_result.ok and version_result.stdout.strip()
                else ""
            )

            def _recover_servers(
                recovery_attempt: int, concurrency: int, repeat: int
            ) -> RunResult | None:
                _cleanup_server_group()
                launch_ok = True
                recovery_status = ProbeStatus.STARTUP_FAILED
                recovery_detail = ""
                failed_replica_index: int | None = None
                failed_replica_port: int | None = None
                failed_replica_log: str | None = None
                recovery_logs: list[tuple[int, int, str]] = []
                for idx, (rport, rgpu) in enumerate(
                    zip(replica_ports, replica_gpus)
                ):
                    rcmd = _override_launch_port(base_cmd, rport)
                    if rgpu is not None:
                        rcmd = f"env CUDA_VISIBLE_DEVICES={rgpu} {rcmd}"
                    identity = f"{time.time_ns()}_{uuid.uuid4().hex}"
                    rlog = (
                        f"{candidate_out}/server_{round_label}_c{concurrency}_"
                        f"repeat{repeat}_"
                        f"recovery{recovery_attempt}_replica{idx}_{identity}.log"
                    )
                    pid_path = rlog.removesuffix(".log") + ".pid"
                    recovery_logs.append((idx, rport, rlog))
                    ctx.replica_server_logs[idx] = rlog
                    resource_name = (
                        f"server:{round_label}:{candidate_id}:"
                        f"recovery{recovery_attempt}:replica{idx}:"
                        f"port{rport}:{id(ctx)}"
                    )
                    server_resources.append(resource_name)
                    launched = lifecycle.start_resource(
                        resource_name,
                        starter=lambda rcmd=rcmd, rlog=rlog, pid_path=pid_path: (
                            container.start_server_monitored(
                                rcmd,
                                rlog,
                                pid_path,
                                timeout=REMOTE_START_COMMAND_TIMEOUT_S,
                            )
                        ),
                        cleanup=lambda pid_path=pid_path: cleanup_monitored_benchmark(
                            container,
                            pid_path=pid_path,
                            status_path=f"{pid_path}.unused-status",
                            terminate_grace_s=PROCESS_TERMINATE_GRACE_S,
                            poll_timeout_s=5,
                        ),
                    )
                    pid_text = launched.stdout.strip() if launched.ok else ""
                    if (
                        not launched.ok
                        or re.fullmatch(r"[0-9]+", pid_text) is None
                        or int(pid_text) <= 1
                    ):
                        launch_ok = False
                        failed_replica_index = idx
                        failed_replica_port = rport
                        failed_replica_log = rlog
                        recovery_detail = _command_failure_detail(launched)
                        if _is_transport_failure(launched):
                            recovery_status = ProbeStatus.TRANSPORT_FAILED
                        break
                    server_replicas.append((int(pid_text), rport, rlog, pid_path))

                recovered = launch_ok and len(server_replicas) == len(replica_ports)
                for idx, (pid, rport, rlog, _pid_path) in enumerate(server_replicas):
                    try:
                        replica_ready = wait_until_ready(
                            make_health_probe(container, port=rport, timeout_s=5),
                            is_alive=lambda pid=pid: _server_alive(pid),
                            timeout_s=config.startup_hard_timeout_s,
                            stall_timeout_s=config.startup_stall_timeout_s,
                            progress=lambda rlog=rlog: _startup_progress(rlog),
                            cancelled=lambda: lifecycle.cancelled,
                        )
                    except ReadinessTransportError as exc:
                        replica_ready = False
                        recovery_status = ProbeStatus.TRANSPORT_FAILED
                        recovery_detail = str(exc)
                    if not replica_ready:
                        recovered = False
                        failed_replica_index = idx
                        failed_replica_port = rport
                        failed_replica_log = rlog
                        break
                if lifecycle.cancelled:
                    lifecycle.assert_start_allowed()
                if recovered:
                    return None
                failed_local_log: Path | None = None
                for idx, rport, rlog in recovery_logs:
                    local_log = candidate_dir / Path(rlog).name
                    _pull_container_file(container, rlog, local_log)
                    if rlog == failed_replica_log:
                        failed_local_log = local_log
                if not recovery_detail and failed_local_log is not None:
                    recovery_detail = _extract_failure_reason(failed_local_log)
                if not recovery_detail:
                    recovery_detail = "server did not become ready"
                _cleanup_server_group()
                failure = _probe_failure_result(
                    candidate_id,
                    f"server recovery {recovery_attempt} failed"
                    + (
                        f" on replica {failed_replica_index}"
                        f" port {failed_replica_port}: {recovery_detail}"
                    ),
                    status=recovery_status,
                    tp_size=tp_size,
                    concurrency=concurrency,
                    num_prompts=concurrency * ctx.multiplier,
                )
                failure.raw.update(
                    {
                        "replica_index": failed_replica_index,
                        "port": failed_replica_port,
                    }
                )
                return failure

            evaluate, warmup = _make_evaluate(
                ctx, candidate_id, candidate_dir, tp_size=tp_size,
                ports=ctx.replica_ports,
                output_len=output_len,
                round_label=round_label,
                candidate=candidate,
                recover_servers=_recover_servers,
            )
            recovery_failures = evaluate.attempt_failures  # type: ignore[attr-defined]

            def on_evaluate_exception(concurrency: int, exc: Exception) -> RunResult:
                if isinstance(exc, AssertionError):
                    raise exc
                if isinstance(exc, (CleanupError, LifecycleAborted)):
                    lifecycle.abort(str(exc))
                    raise exc
                return _probe_failure_result(
                    candidate_id,
                    f"evaluate raised at C={concurrency}: {exc!r}",
                    status=ProbeStatus.BENCHMARK_FAILED,
                    tp_size=tp_size,
                    concurrency=concurrency,
                    num_prompts=concurrency * ctx.multiplier,
                )

            warmup_failure = warmup()
            if warmup_failure is not None:
                outcome = SearchOutcome(
                    results=[warmup_failure],
                    c_star=None,
                    stop_reason=warmup_failure.status,
                    last_pass=None,
                    first_fail=None,
                    num_evals=0,
                    newly_probed=[],
                    log=[warmup_failure.failure_reason or "warmup failed"],
                    complete=False,
                    certainty="unknown",
                )
            else:
                outcome = search_saturation(
                    evaluate,
                    qualifies,
                    start=1,
                    factor=2,
                    max_cap=config.max_cap,
                    max_probes=max_probes,
                    refine=refine,
                    confirm=confirm,
                    seeds=seeds,
                    on_evaluate_exception=on_evaluate_exception,
                )
            terminal_infra = [
                result
                for result in outcome.results
                if result.status not in SEARCH_VERDICT_STATUSES
            ]
            if terminal_infra:
                failed_concurrencies = {result.concurrency for result in terminal_infra}
                outcome.results = [
                    result
                    for result in outcome.results
                    if result.status in SEARCH_VERDICT_STATUSES
                ]
                outcome.newly_probed = [
                    concurrency
                    for concurrency in outcome.newly_probed
                    if concurrency not in failed_concurrencies
                ]

            _pull_container_file(
                container,
                ctx.replica_server_logs.get(0, server_log),
                candidate_dir / f"server.{round_label}.log",
            )
            _write_json(
                candidate_dir / f"run_result.{round_label}.json",
                [_run_result_dict(r) for r in outcome.results],
            )
            _write_json(
                candidate_dir / f"search_samples.{round_label}.json",
                _search_sample_evidence(outcome),
            )
            _log(
                f"[{round_label}] {candidate_id}: done c_star={outcome.c_star} "
                f"stop={outcome.stop_reason} evals={outcome.num_evals}"
            )
            outcome.startup_attempts = startup_attempt + len(ctx.container_start_failures)
            probe_failures = [
                {
                    "failed_at": datetime.now(UTC).astimezone().isoformat(),
                    "round": failed_round,
                    "concurrency": result.concurrency,
                    "num_prompts": result.num_prompts,
                    "tp_size": result.tp_size,
                    "attempt": 1,
                    "stage": "probe",
                    "status": str(result.status),
                    "reason": result.failure_reason,
                    "known_issue": result.known_issue,
                }
                for result in terminal_infra
                if result.status not in SEARCH_VERDICT_STATUSES
            ]
            outcome.failures = startup_failures + recovery_failures
            seen_failures = {
                (
                    failure.get("round"),
                    failure.get("status"),
                    failure.get("concurrency"),
                    failure.get("reason"),
                )
                for failure in outcome.failures
            }
            for failure in probe_failures:
                identity = (
                    failure.get("round"),
                    failure.get("status"),
                    failure.get("concurrency"),
                    failure.get("reason"),
                )
                if identity not in seen_failures:
                    outcome.failures.append(failure)
                    seen_failures.add(identity)
    finally:
        _cleanup_server_group()

    return outcome


def _outcome_diag(outcome: SearchOutcome) -> dict[str, Any]:
    return {
        "c_star": outcome.c_star,
        "stop_reason": outcome.stop_reason,
        "last_pass": outcome.last_pass,
        "first_fail": outcome.first_fail,
        "num_evals": outcome.num_evals,
        "newly_probed": list(outcome.newly_probed),
        "num_results": len(outcome.results),
        "startup_attempts": outcome.startup_attempts,
        "complete": outcome.complete,
        "certainty": outcome.certainty,
    }
def _norm_gpu(name: str) -> str:
    """归一化 GPU 型号名以便比较。

    job.json 用 catalog key(带前缀,如 ``G24_pro5000``),target.json 用裸型号名
    (如 ``pro5000``),两者指同一张卡。剥掉 ``G<NN>_`` 前缀(gpu.yaml 里每个 key 的
    后缀全局唯一),再归一大小写/空格/连字符,兼容 ``PRO 5000`` 之类写法。
    注意:仅剥前缀,不做别名全解析,故 ``pro5000`` vs ``pro6000`` 仍判为不同,
    真正的硬件不匹配依旧会被拦住。
    """
    stripped = re.sub(r"(?i)^G\d+_", "", name.strip())
    return stripped.lower().replace(" ", "").replace("-", "")


def _check_hardware_match(job: JobSpec, config: ExecutorConfig) -> None:
    """Verify target.json GPU fields match job.json requirements before any remote work.

    Catches mismatched hardware early (e.g. job wants 8x72G but target only has 4x72G)
    instead of failing during container start or benchmark.
    """
    if not config.target_gpu_model:
        _log("⚠️  target.json 缺少 gpu_model 字段,跳过硬件校验")
        return
    if _norm_gpu(config.target_gpu_model) != _norm_gpu(job.gpu_model):
        raise SystemExit(
            f"❌ GPU 型号不匹配: job 要求 {job.gpu_model}, "
            f"target 提供 {config.target_gpu_model}"
        )
    if config.target_gpu_count < job.gpu_count:
        raise SystemExit(
            f"❌ GPU 卡数不足: job 要求 {job.gpu_count} 张, "
            f"target 只有 {config.target_gpu_count} 张"
        )
    if config.target_gpu_memory_gb < job.gpu_memory_gb:
        raise SystemExit(
            f"❌ 单卡显存不足: job 要求 {job.gpu_memory_gb}GB/卡, "
            f"target 只有 {config.target_gpu_memory_gb}GB/卡"
        )
    _log(
        f"✅ 硬件校验通过: {job.gpu_model} "
        f"{job.gpu_count}x{job.gpu_memory_gb}GB (target: "
        f"{config.target_gpu_count}x{config.target_gpu_memory_gb}GB)"
    )


def run_executor(
    config: ExecutorConfig,
    *,
    remote: RemoteRunner | None = None,
    lifecycle: ExecutorLifecycle | None = None,
) -> dict:
    """Run one invocation under the process-wide lifecycle and signal guard."""
    if lifecycle is not None:
        return _run_executor_impl(config, remote=remote, lifecycle=lifecycle)
    _prepare_local_results_dir(config.results_dir)
    owned_lifecycle = ExecutorLifecycle(
        config.results_dir, job_id=config.job_path.stem
    )
    result: dict | None = None
    with owned_lifecycle:
        job = _load_job(config.job_path)
        owned_lifecycle.job_id = job.job_id
        result = _run_executor_impl(
            config, remote=remote, lifecycle=owned_lifecycle
        )
    if result is None:  # defensive: a lifecycle context must never suppress errors
        raise RuntimeError("executor lifecycle suppressed execution without a result")
    return result


def _run_executor_impl(
    config: ExecutorConfig,
    *,
    remote: RemoteRunner | None = None,
    lifecycle: ExecutorLifecycle,
) -> dict:
    """Orchestrate the 8-step executor loop and return a summary dict.

    Steps: load job + workload output_len, load candidate configs, start the
    container, then per candidate start the server, wait for readiness, generate
    and run bench commands, parse results, and tear the server down; finally
    stop/remove the container and rank the candidates.
    """
    job = _load_job(config.job_path)
    output_len = _load_output_len(job.workload)
    multiplier = _load_num_prompts_multiplier(
        job.benchmark_method, project_root=config.project_root
    )
    candidates = _load_candidates(config.configs_path, search=job.search)
    _check_hardware_match(job, config)

    # Everything needed to construct the benchmark is local and deterministic.
    # Validate it before SSH preflight, which may clear containers/GPU processes.
    bench_template = _resolve_bench_template(job, config=config)

    target_gpu_count = config.target_gpu_count or job.gpu_count
    local_preflight = PreflightRequest(
        candidates=tuple(candidates),
        gpu_model=config.target_gpu_model or job.gpu_model,
        gpu_count=target_gpu_count,
        allocation_gpu_count=job.gpu_count,
        gpu_memory_gb=config.target_gpu_memory_gb or job.gpu_memory_gb,
        model_host_dir=config.model_host_dir,
        image_ref=config.image_ref,
        base_port=config.port,
        container_name=config.container_name,
        remote_outputs_dir=config.remote_outputs_dir,
        job_id=job.job_id,
        fill_host=config.fill_host,
        allow_cross_numa=config.allow_cross_numa,
        exclusive_host=config.exclusive_host,
    )
    validate_local_preflight(local_preflight)
    _prepare_local_results_dir(config.results_dir)

    remote = remote or RemoteRunner(config.ssh_target, ssh_password=config.ssh_password)
    preflight_plan = prepare_remote_host(remote, local_preflight)

    # The boundary predicate for the adaptive search: a run "qualifies" iff it is
    # both data-healthy (not truncated) and within SLA. Injected so the search
    # module stays ranker/SLA-agnostic.
    def qualifies(result: RunResult) -> bool:
        return _classify_probe_for_search(
            result, output_len=output_len, sla=job.sla
        )

    # docker -v resolves host paths on the REMOTE machine, so the outputs mount
    # must be an absolute path that exists on the remote host — a local relative
    # path like "outputs/<job>/results" is rejected by docker as a volume name.
    # Results are still read back to the local results_dir via `docker exec cat`.
    outputs_host_dir = preflight_plan.outputs_host_dir

    outputs_container_path = "/workspace/outputs"

    _log(
        f"job={job.job_id} candidates={len(candidates)} output_len={output_len} "
        f"multiplier={multiplier} precise_candidates=all gpu_count={job.gpu_count}"
    )

    results_by_candidate: dict[str, list[RunResult]] = {}
    round_results: dict[str, dict[int, list[RunResult]]] = {}
    candidate_summaries: dict[str, dict[str, Any]] = {}
    search_diagnostics: dict[str, dict[str, Any]] = {}

    # Template context (container is None here; each batch creates its own)
    ctx_template = _CandidateContext(
        container=None,  # type: ignore[arg-type]
        config=config,
        job=job,
        bench_template=bench_template,
        multiplier=multiplier,
        outputs_container_path=outputs_container_path,
        port=config.port,
        lifecycle=lifecycle,
    )

    candidate_by_id = {str(candidate["id"]): candidate for candidate in candidates}
    numa_groups = [list(group) for group in preflight_plan.numa_groups]
    _log(f"NUMA groups: {numa_groups}")
    batches = preflight_plan.round1_batches
    numa_policy = "cross-NUMA allowed" if config.allow_cross_numa else "NUMA-local"
    _log(f"{len(candidates)} candidates split into {len(batches)} batch(es) "
         f"(max {job.gpu_count} GPUs per batch, {numa_policy})")

    # ---- ROUND 1: coarse expansion over ALL candidates (batch-parallel) ----
    for batch_idx, batch in enumerate(batches):
        alloc = _alloc_from_preflight_batch(batch, candidate_by_id)
        _log(f"round-1 batch {batch_idx + 1}/{len(batches)}: "
             + _format_alloc(alloc))
        outcomes = _run_batch_parallel(
            ctx_template, alloc, remote, outputs_host_dir, outputs_container_path,
            qualifies=qualifies,
            output_len=output_len,
            round_label="r1",
            max_probes=ROUND1_MAX_PROBES,
            confirm=ROUND1_CONFIRM,
            refine=False,
            numa_groups=numa_groups,
        )
        for candidate_id, outcome in outcomes.items():
            results_by_candidate[candidate_id] = outcome.results
            round_results.setdefault(candidate_id, {})[1] = list(outcome.results)
            search_diagnostics[candidate_id] = _outcome_diag(outcome)
            candidate_summaries[candidate_id] = {
                "candidate_id": candidate_id,
                "round1": _outcome_diag(outcome),
                "round1_batch": f"{batch_idx + 1}/{len(batches)}",
                "round1_attempts": outcome.startup_attempts,
                "attempts": outcome.startup_attempts,
                "failures": [
                    {**failure, "batch": f"{batch_idx + 1}/{len(batches)}"}
                    for failure in outcome.failures
                ],
            }

    # Round-1 ranking is diagnostic only. Every candidate enters precise round 2.
    round1_ranking = rank_candidates(
        results_by_candidate, job.sla, output_len=output_len,
        gpu_count=job.gpu_count,
    )
    refine_ids = [str(candidate.get("id", "unknown")) for candidate in candidates]
    _log(
        "round-1 diagnostic/provisional ranking="
        f"{[(r['candidate_id'], r['goodput_per_host']) for r in round1_ranking]}"
    )
    _log(f"round-2 refining ALL {len(refine_ids)} candidates: {refine_ids}")

    # ---- ROUND 2: precise bisection on ALL candidates (batch-parallel) ----
    top_batches = preflight_plan.round2_batches
    planned_fill_host = {
        placement.candidate_id: placement
        for placement in preflight_plan.fill_host_placements
    }

    for batch_idx, batch in enumerate(top_batches):
        alloc = _alloc_from_preflight_batch(batch, candidate_by_id)
        _log(f"round-2 batch {batch_idx + 1}/{len(top_batches)}: "
             + _format_alloc(alloc))
        # Round-1 results only provide bracket coordinates. search_saturation
        # always remeasures hinted endpoints, so single-instance seed metrics can
        # safely guide full-host ordering without entering authoritative results.
        seeds = results_by_candidate
        outcomes = _run_batch_parallel(
            ctx_template, alloc, remote, outputs_host_dir, outputs_container_path,
            qualifies=qualifies,
            output_len=output_len,
            round_label="r2",
            max_probes=required_sample_budget(
                config.max_cap,
                samples_per_concurrency=ROUND2_CONFIRM,
                seed_hint_endpoints=2,
            ),
            confirm=ROUND2_CONFIRM,
            refine=True,
            seeds_by_id=seeds,
            fill_host=config.fill_host,
            numa_groups=numa_groups,
            allow_cross_numa=config.allow_cross_numa,
            fill_host_placements=planned_fill_host,
        )
        for candidate_id, outcome in outcomes.items():
            results_by_candidate[candidate_id] = outcome.results
            new_concurrencies = set(outcome.newly_probed)
            round_results.setdefault(candidate_id, {})[2] = [
                result for result in outcome.results if result.concurrency in new_concurrencies
            ]
            search_diagnostics[candidate_id] = _outcome_diag(outcome)
            candidate_summaries[candidate_id]["round2"] = _outcome_diag(outcome)
            candidate_summaries[candidate_id]["round2_batch"] = (
                f"{batch_idx + 1}/{len(top_batches)}"
            )
            candidate_summaries[candidate_id]["round2_attempts"] = outcome.startup_attempts
            candidate_summaries[candidate_id]["attempts"] += outcome.startup_attempts
            candidate_summaries[candidate_id]["failures"].extend(
                {**failure, "batch": f"{batch_idx + 1}/{len(top_batches)}"}
                for failure in outcome.failures
            )

    exact_results_by_candidate = {
        candidate_id: results
        for candidate_id, results in results_by_candidate.items()
        if candidate_summaries.get(candidate_id, {}).get("round2", {}).get("complete")
        is True
        and candidate_summaries.get(candidate_id, {})
        .get("round2", {})
        .get("certainty")
        == "exact"
    }
    ranking = rank_candidates(
        exact_results_by_candidate, job.sla, output_len=output_len,
        gpu_count=job.gpu_count,
    )

    threshold_pct = job.search.baseline_threshold_pct
    ranking = annotate_baseline_threshold(ranking, threshold_pct=threshold_pct)
    candidate_rows = build_candidate_rows(
        candidates,
        candidate_summaries,
        round_results,
        ranking,
        output_len=output_len,
    )
    completed_count = sum(row["status"] == "completed" for row in candidate_rows)
    all_completed = completed_count == len(candidate_rows)
    task_status = {
        "report_schema_version": 1,
        "job_id": job.job_id,
        "task_status": "COMPLETED" if all_completed else "INCOMPLETE",
        "ranking_status": "FINAL" if all_completed else "PROVISIONAL",
        "interrupted": False,
        "total_candidates": len(candidate_rows),
        "completed_candidates": completed_count,
        "failed_candidates": len(candidate_rows) - completed_count,
    }
    write_reports(
        config.results_dir,
        ranking=ranking,
        candidate_rows=candidate_rows,
        task_status=task_status,
    )
    _log("candidate result preview (one row per candidate):")
    for line in render_candidate_preview(candidate_rows):
        _log("  " + line)

    return {
        "job_id": job.job_id,
        "output_len": output_len,
        "num_prompts_multiplier": multiplier,
        **task_status,
        "candidates": list(candidate_summaries.values()),
        "candidate_results": candidate_rows,
        "search_diagnostics": search_diagnostics,
        "ranking": ranking,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pull_container_file(container: Container, container_path: str, local_path: Path) -> bool:
    """Copy a file from inside the container to a local path via `docker exec cat`.

    The outputs dir is mounted on the remote host, not locally, so evidence files
    (server.log, bench stdout) must be read back over the exec channel. Returns
    True when something was written.
    """
    result = container.exec(f"cat {shlex.quote(container_path)}", timeout=5)
    if not result.ok:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(result.stdout, encoding="utf-8")
    return True


def _parse_concurrencies(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def _resolve_target_password(target_path: Path) -> tuple[TargetSpec, str]:
    """Load one target and resolve its optional password without exposing it in argv."""
    target = load_target(target_path)
    if target.ssh_password_env is not None:
        password = os.environ.get(target.ssh_password_env, "")
        if not password:
            raise ValueError(
                f"{target_path}: environment variable {target.ssh_password_env!r} "
                "referenced by ssh_password_env is missing or empty"
            )
        return target, password
    if target.ssh_password is not None:
        return target, target.ssh_password.get_secret_value()
    return target, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run generated SGLang configs on a remote host and rank them.",
        allow_abbrev=False,
    )
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--configs", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--concurrencies", default=None,
                        help="deprecated: adaptive search now chooses probes")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="deprecated compatibility option; round-2 always refines all candidates",
    )
    parser.add_argument("--max-cap", type=int, default=DEFAULT_MAX_CAP,
                        help="upper bound on concurrency the search will probe")
    parser.add_argument("--container-name", default=None)
    parser.add_argument(
        "--fill-host", action="store_true",
        help="round2 按 NUMA topology 把每个候选复制成实际可放置的实例整机满载、"
             "多端口并发压测求和,得到实测整机 goodput(而非单实例 × 实例数 纸面外推)。"
             "round1 仍单实例粗筛。",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=8,
        help="批内候选并发上限(每候选独占容器+GPU+端口)。",
    )
    parser.add_argument("--startup-stall-timeout", type=int, default=300)
    parser.add_argument("--startup-hard-timeout", type=int, default=900)
    parser.add_argument("--startup-max-attempts", type=int, default=3)
    args = parser.parse_args(argv)

    _prepare_local_results_dir(args.results)
    lifecycle = ExecutorLifecycle(args.results, job_id=args.job.stem)
    try:
        with lifecycle:
            try:
                job = load_job(args.job)
                lifecycle.job_id = job.job_id
                target, ssh_password = _resolve_target_password(args.target)
            except ValueError as exc:
                parser.error(str(exc))

            config = ExecutorConfig(
                job_path=args.job,
                configs_path=args.configs,
                results_dir=args.results,
                ssh_target=target.ssh_target,
                ssh_password=ssh_password,
                image_ref=target.image_ref,
                model_host_dir=target.model_host_dir,
                model_container_path=target.model_container_path,
                project_root=args.project_root,
                max_candidates=job.search.max_candidates,
                concurrencies=_parse_concurrencies(args.concurrencies),
                port=target.port,
                container_name=args.container_name or f"llm-infer-tuner-{job.job_id}",
                remote_outputs_dir=target.remote_outputs_dir,
                top_k=args.top_k,
                max_cap=args.max_cap,
                target_gpu_model=target.gpu_model,
                target_gpu_count=target.gpu_count,
                target_gpu_memory_gb=target.gpu_memory_gb,
                fill_host=args.fill_host,
                allow_cross_numa=target.allow_cross_numa,
                exclusive_host=target.exclusive_host,
                max_parallel=args.max_parallel,
                startup_stall_timeout_s=args.startup_stall_timeout,
                startup_hard_timeout_s=args.startup_hard_timeout,
                startup_max_attempts=args.startup_max_attempts,
            )
            summary = run_executor(config, lifecycle=lifecycle)
    except LifecycleInterrupted as exc:
        return exc.exit_code
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
