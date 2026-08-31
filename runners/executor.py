"""Phase-2 executor: run generated configs on a remote host and collect metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from runners.bench_runner import (
    build_benchmark_command_template,
    rewrite_bench_command,
    run_benchmark,
    substitute_placeholders,
)
from runners.concurrency_search import SearchOutcome, search_saturation
from runners.container import Container, ContainerConfig
from runners.metrics import RunResult, parse_bench_text
from runners.preflight import (
    CandidatePlacement,
    FillHostPlacement,
    PreflightRequest,
    prepare_remote_host,
    validate_local_preflight,
)
from runners.ranker import data_health_check, passes_sla, rank_candidates
from runners.readiness import make_health_probe, wait_until_ready
from runners.remote import RemoteRunner
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
# = precise bisection over ALL candidates, reusing round-1 probes as seeds. The two
# rounds are cheap because expansion is log-scale and round-2 only re-probes new
# midpoints (seeds are served from cache, never re-benched).
ROUND1_MAX_PROBES = 7   # reaches ~C=64 via 1,2,4,8,16,32,64 in one candidate
ROUND1_CONFIRM = 1      # coarse: precise round 2 will confirm the boundary
ROUND2_MAX_PROBES = 14
ROUND2_CONFIRM = 2      # precise: re-probe boundary passes AND fails (see search module)
DEFAULT_TOP_K = 5
DEFAULT_MAX_CAP = 256
WARMUP_CONCURRENCY = 2  # server ready 后、正式搜索前的预热并发档(结果丢弃,只热 kernel)
# The bracketed spelling matches the real module name while keeping the literal
# ``sglang.launch_server`` out of pgrep/pkill's own shell command line.
SERVER_PROCESS_PATTERN = "[s]glang[.]launch_server"
SERVER_CLEANUP_TERM_POLL_ATTEMPTS = 21
SERVER_CLEANUP_TERM_POLL_INTERVAL_S = 0.5
SERVER_CLEANUP_KILL_POLL_ATTEMPTS = 5
SERVER_CLEANUP_KILL_POLL_INTERVAL_S = 0.1

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
    remote_outputs_dir: str = ""  # abs path on the remote host; default $HOME/llm-infer-tuner-outputs/<job>
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
    healthy = [r for r in replicas if r is not None and r.status == "ok"]
    concurrency = replicas[0].concurrency if replicas else 0
    candidate_id = replicas[0].candidate_id if replicas else "unknown"
    tp_size = replicas[0].tp_size if replicas else 1
    if not healthy or len(healthy) < expected:
        agg = _health_check_failed_result(
            candidate_id,
            f"满载副本不齐或不健康:健康 {len(healthy)}/{expected}(C={concurrency})",
        )
        agg.concurrency = concurrency
        agg.tp_size = tp_size
        agg.instances = expected
        agg.full_host_measured = True
        return agg
    return RunResult(
        candidate_id=candidate_id,
        concurrency=concurrency,
        num_prompts=sum(r.num_prompts for r in healthy),
        completed=sum(r.completed for r in healthy),
        success_rate=min(r.success_rate for r in healthy),
        request_throughput=sum(r.request_throughput for r in healthy),
        output_throughput=sum(r.output_throughput for r in healthy),
        total_throughput=sum(r.total_throughput for r in healthy),
        mean_ttft_ms=max(r.mean_ttft_ms for r in healthy),
        p99_ttft_ms=max(r.p99_ttft_ms for r in healthy),
        mean_tpot_ms=max(r.mean_tpot_ms for r in healthy),
        p99_tpot_ms=max(r.p99_tpot_ms for r in healthy),
        total_output_tokens=sum(r.total_output_tokens for r in healthy),
        avg_output_tokens=min(r.avg_output_tokens for r in healthy),
        duration=max(r.duration for r in healthy),
        tp_size=tp_size,
        instances=len(healthy),
        full_host_measured=True,
        status="ok",
    )


def _health_check_failed_result(candidate_id: str, reason: str) -> RunResult:
    return RunResult(
        candidate_id=candidate_id,
        tp_size=1,
        concurrency=0,
        num_prompts=0,
        completed=0,
        success_rate=0.0,
        request_throughput=0.0,
        output_throughput=0.0,
        total_throughput=0.0,
        mean_ttft_ms=0.0,
        p99_ttft_ms=0.0,
        mean_tpot_ms=0.0,
        p99_tpot_ms=0.0,
        total_output_tokens=0,
        avg_output_tokens=0.0,
        duration=0.0,
        status="health_check_failed",
        failure_reason=reason,
    )


def _make_evaluate(
    ctx: _CandidateContext,
    candidate_id: str,
    candidate_dir: Path,
    tp_size: int = 1,
    ports: list[int] | None = None,
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
    candidate_dir(每个真实 bench 恰好落一份证据)并解析指标。任何异常收敛成不合格
    RunResult,使过载塌缩记为边界失败而非中断整轮搜索。
    """
    container = ctx.container
    candidate_out = f"{ctx.outputs_container_path}/{candidate_id}"
    full_host_measurement = ports is not None
    bench_ports = ports if ports is not None else [ctx.port]

    def _bench_one_port(concurrency: int, port: int, tag: str) -> RunResult:
        command_tpl, num_prompts = rewrite_bench_command(
            ctx.bench_template, concurrency=concurrency, multiplier=ctx.multiplier
        )
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        result_name = f"bench_c{concurrency}{tag}_{timestamp}.jsonl"
        result_container_path = f"{candidate_out}/{result_name}"
        base_command = substitute_placeholders(
            command_tpl,
            host="127.0.0.1",
            port=port,
            model_path=ctx.config.model_container_path,
            job_id=ctx.job.job_id,
            timestamp=timestamp,
        )
        command = _force_output_file(base_command, result_container_path)
        _log(f"    [{candidate_id}] bench C={concurrency}{tag} @port{port} (num_prompts={num_prompts}) ...")
        bench_run = run_benchmark(container, command)

        log_name = f"bench_c{concurrency}{tag}_{timestamp}.log"
        bench_console = (
            f"$ {command}\n"
            f"# exit code: {bench_run.returncode}\n\n"
            f"----- stdout -----\n{bench_run.stdout}\n"
            f"----- stderr -----\n{bench_run.stderr}\n"
        )
        (candidate_dir / log_name).write_text(bench_console, encoding="utf-8")

        text = container.exec(f"cat {shlex.quote(result_container_path)}").stdout
        run_result = parse_bench_text(
            text,
            candidate_id=candidate_id,
            concurrency=concurrency,
            num_prompts=num_prompts,
            tp_size=tp_size,
        )
        if run_result.status == "bad_args" and bench_run.returncode != 0:
            tail = (bench_run.stderr or bench_run.stdout).strip().splitlines()[-3:]
            run_result.failure_reason = (
                f"bench exit {bench_run.returncode}; see {log_name}: "
                + " | ".join(tail)
            )
        (candidate_dir / result_name).write_text(text, encoding="utf-8")
        return run_result

    def evaluate(concurrency: int) -> RunResult:
        try:
            if not full_host_measurement:
                run_result = _bench_one_port(concurrency, bench_ports[0], "")
            else:
                # 整机满载:N 个副本端口并发各压一份(同一并发档),再求和。
                # 底层 subprocess.run 线程安全,每个副本写各自的结果文件,互不串。
                def _one(i_port):
                    i, port = i_port
                    return _bench_one_port(concurrency, port, f"_i{i}")

                with ThreadPoolExecutor(max_workers=len(bench_ports)) as pool:
                    replicas = list(pool.map(_one, enumerate(bench_ports)))
                run_result = _aggregate_replicas(replicas, expected=len(bench_ports))
            _log(
                f"      [{candidate_id}] C={concurrency}: tput={run_result.total_throughput:.0f} "
                f"(x{run_result.instances}) "
                f"ttft={run_result.mean_ttft_ms:.0f}ms tpot={run_result.mean_tpot_ms:.1f}ms "
                f"succ={run_result.success_rate:.2f} status={run_result.status}"
            )
            return run_result
        except Exception as exc:  # collapse != abort; the search treats this as a fail
            return _health_check_failed_result(
                candidate_id, f"evaluate raised at C={concurrency}: {exc!r}"
            )

    def warmup(concurrency: int = WARMUP_CONCURRENCY) -> None:
        """server ready 后、正式并发搜索前跑一次预热压测,结果**丢弃不计入搜索**。

        目的:吸收首次运行的 kernel 编译/JIT 尖峰(见 client knowledge §「丢弃第一条」),
        让正式 probe 的每一档都从热 kernel 起步,不再出现「首测 ttft 断崖高、复测断崖低」。
        用固定小并发(WARMUP_CONCURRENCY),整机满载时对所有副本端口各预热一次。
        因候选默认已 pin `--disable-radix-cache`,预热不会给正式 probe 留下 prefix 缓存,
        故只热 kernel、不喂缓存,保持各候选/各档横向可比。异常吞掉,预热失败不阻断搜索。
        """
        try:
            for i, port in enumerate(bench_ports):
                tag = "_warmup" if len(bench_ports) == 1 else f"_warmup_i{i}"
                _log(
                    f"      [{candidate_id}] warmup C={concurrency} @port{port} "
                    f"(预热,结果丢弃) ..."
                )
                r = _bench_one_port(concurrency, port, tag)
                _log(
                    f"      [{candidate_id}] warmup done "
                    f"(ttft={r.mean_ttft_ms:.0f}ms status={r.status},不计入搜索)"
                )
        except Exception as exc:  # 预热失败不应阻断搜索
            _log(f"      [{candidate_id}] warmup skipped ({exc!r})")

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


def _cleanup_servers_checked(container: Container, *, candidate_id: str) -> None:
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
                residual = container.exec(f"pgrep -f {pattern}")
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
        terminated = container.exec(f"pkill -TERM -f {pattern}")
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
            killed = container.exec(f"pkill -KILL -f {pattern}")
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
                    f"candidate {candidate_id}: 整机 {_n_devices} 卡放不下一个 tp_size={_tp_size} 实例。"
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
                    f"实际分到 {_n_devices} 张({gpu_ids_str})。拒绝启动以免 CUDA invalid device ordinal。"
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
                timeout=config.startup_hard_timeout_s,
            )
            failures.extend(cleanup_failures)
            if not cleanup_failures:
                ctx.container_present = False
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
        for attempt in range(1, max(1, config.startup_max_attempts) + 1):
            _log(
                f"[{round_label}] {candidate_id}: starting container "
                f"(GPUs={gpu_ids_str}, port={port}, "
                f"attempt {attempt}/{config.startup_max_attempts}) ..."
            )
            try:
                started = container.start(timeout=config.startup_hard_timeout_s)
                running = started.ok and container.is_running(
                    timeout=config.startup_hard_timeout_s
                )
                reason = (
                    started.stderr.strip()
                    or started.stdout.strip()
                    or "container not running"
                )
            except Exception as exc:  # noqa: BLE001 - inspect decides safe retry
                running = False
                reason = f"container start raised: {exc!r}"
            if running:
                ctx.container_ready = True
                ctx.container_present = True
                break
            ctx.container_start_failures.append(
                {
                    "failed_at": datetime.now(UTC).astimezone().isoformat(),
                    "round": failed_round,
                    "concurrency": None,
                    "attempt": attempt,
                    "stage": "container_start",
                    "reason": f"container did not start: {reason}",
                }
            )
            _log(f"[{round_label}] {candidate_id}: container start failed: {reason}")
            state, inspect_detail = _container_inspection_state(
                container,
                timeout=config.startup_hard_timeout_s,
            )
            if state == "absent":
                ctx.container_present = False
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
                timeout=config.startup_hard_timeout_s,
            )
            if cleanup_failures:
                lifecycle_cleanup_failures.extend(cleanup_failures)
                abort_startup = True
                break
            ctx.container_present = False

    if abort_startup:
        lifecycle_cleanup_failures.extend(_cleanup_created_containers())
        raise CleanupError(
            "executor cleanup failed during startup; refusing new resource starts: "
            + "; ".join(lifecycle_cleanup_failures)
        )

    # Run each candidate (server lifecycle + search) CONCURRENTLY within the batch.
    # 每个候选独占容器+GPU+端口,底层 subprocess.run 线程安全,故用线程池并发跑。
    # 单个候选内部抛异常(RemoteRunner OSError 等)被收敛成 health_check_failed
    # 结果而非中断整批;try/finally 仍保证本 batch 已启动的容器一定被 stop+remove,
    # 不会留下孤儿容器占住 GPU/端口导致下个 batch 撞车。
    def _run_one(entry) -> tuple[str, SearchOutcome]:
        candidate_id, container, ctx, candidate, gpu_ids_str, port = entry
        if not ctx.container_ready:
            reason = ctx.container_start_failures[-1]["reason"]
            return candidate_id, SearchOutcome(
                results=[_health_check_failed_result(candidate_id, reason)],
                c_star=None,
                stop_reason="health_check_failed",
                last_pass=None,
                first_fail=None,
                num_evals=0,
                newly_probed=[],
                log=[reason],
                startup_attempts=len(ctx.container_start_failures),
                failures=list(ctx.container_start_failures),
            )
        cand_seeds = None
        if seeds_by_id and candidate_id in seeds_by_id:
            cand_seeds = seeds_by_id[candidate_id]
        try:
            outcome = _run_candidate(
                ctx,
                candidate,
                qualifies=qualifies,
                round_label=round_label,
                max_probes=max_probes,
                confirm=confirm,
                refine=refine,
                seeds=cand_seeds,
            )
        except Exception as exc:  # noqa: BLE001 - 单候选崩溃不拖垮同批其他候选
            _log(f"[{round_label}] {candidate_id}: 运行异常,记为失败: {exc!r}")
            if isinstance(exc, CleanupError):
                lifecycle_cleanup_failures.append(str(exc))
            try:
                failed_round = int(round_label.removeprefix("r"))
            except ValueError:
                failed_round = 0
            failure = {
                "failed_at": datetime.now(UTC).astimezone().isoformat(),
                "round": failed_round,
                "concurrency": None,
                "attempt": 1,
                "reason": f"_run_candidate raised: {exc!r}",
            }
            outcome = SearchOutcome(
                results=[_health_check_failed_result(candidate_id, f"_run_candidate raised: {exc!r}")],
                c_star=None,
                stop_reason="health_check_failed",
                last_pass=None,
                first_fail=None,
                num_evals=0,
                newly_probed=[],
                log=[f"_run_candidate raised: {exc!r}"],
                failures=[failure],
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
        lifecycle_cleanup_failures.extend(_cleanup_created_containers())

        if lifecycle_cleanup_failures:
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


def _build_cmd_from_params(params: dict, model_path_placeholder: str = "${MODEL_PATH}", port: int = 30000) -> str:
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
    round_label: str,
    max_probes: int,
    confirm: int,
    refine: bool,
    seeds: list[RunResult] | None,
) -> SearchOutcome:
    """Start the candidate's server, run one adaptive search round, tear it down.

    Returns the SearchOutcome (results = deduped seeds + new probes). On a server
    that never becomes ready, returns a one-element outcome with a
    health_check_failed result so ranking still sees the candidate.
    """
    container = ctx.container
    config = ctx.config
    candidate_id = str(candidate.get("id", "unknown"))
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
    container.exec(f"mkdir -p {shlex.quote(candidate_out)}")

    # 副本计划:满载时用 ctx.replica_ports/replica_gpus(N 个),否则单实例 = [ctx.port]。
    replica_ports = ctx.replica_ports or [ctx.port]
    replica_gpus = ctx.replica_gpus or [None] * len(replica_ports)
    n_replicas = len(replica_ports)
    server_log = f"{candidate_out}/server.log"  # 首副本日志,崩溃诊断沿用它

    def _server_alive() -> bool:
        return container.exec(
            f"pgrep -f {shlex.quote(SERVER_PROCESS_PATTERN)}"
        ).ok

    def _startup_progress() -> str:
        result = container.exec(f"stat -c '%s:%Y' {shlex.quote(server_log)}")
        return result.stdout.strip() if result.ok else ""

    startup_failures: list[dict[str, Any]] = list(ctx.container_start_failures)
    ready = False
    startup_attempt = 0
    try:
        failed_round = int(round_label.removeprefix("r"))
    except ValueError:
        failed_round = 0

    for startup_attempt in range(1, max(1, config.startup_max_attempts) + 1):
        for idx, (rport, rgpu) in enumerate(zip(replica_ports, replica_gpus)):
            # 每个副本换到自己的端口;满载时再用 CUDA_VISIBLE_DEVICES 把它钉在自己那段卡上。
            rcmd = _override_launch_port(base_cmd, rport)
            if rgpu is not None:
                rcmd = f"env CUDA_VISIBLE_DEVICES={rgpu} {rcmd}"
            rlog = server_log if idx == 0 else f"{candidate_out}/server_i{idx}.log"
            tag = "" if n_replicas == 1 else (
                f" 副本 {idx + 1}/{n_replicas} (GPU={rgpu}@{rport})"
            )
            _log(
                f"[{round_label}] {candidate_id}: starting server{tag} "
                f"(attempt {startup_attempt}/{config.startup_max_attempts}), "
                "waiting for /health ..."
            )
            container.exec_detached(rcmd, rlog)

        # 满载:所有副本端口都要 /health 通过才算就绪。
        ready = True
        for rport in replica_ports:
            probe = make_health_probe(container, port=rport)
            if not wait_until_ready(
                probe,
                is_alive=_server_alive,
                timeout_s=config.startup_hard_timeout_s,
                stall_timeout_s=config.startup_stall_timeout_s,
                progress=_startup_progress,
            ):
                ready = False
                break
        if ready:
            break

        local_log = candidate_dir / f"server.{round_label}.attempt{startup_attempt}.log"
        _pull_container_file(container, server_log, local_log)
        reason = _extract_failure_reason(local_log)
        startup_failures.append(
            {
                "failed_at": datetime.now(UTC).astimezone().isoformat(),
                "round": failed_round,
                "concurrency": None,
                "attempt": startup_attempt,
                "stage": "server_start",
                "reason": f"server did not become ready: {reason}",
            }
        )
        _log(
            f"[{round_label}] {candidate_id}: startup attempt {startup_attempt} failed: {reason}"
        )
        _cleanup_servers_checked(container, candidate_id=candidate_id)

    try:
        if not ready:
            _log(
                f"[{round_label}] {candidate_id}: server NOT ready after "
                f"{startup_attempt} attempt(s)"
            )
            reason = startup_failures[-1]["reason"] if startup_failures else "unknown"
            failed = _health_check_failed_result(
                candidate_id, reason
            )
            outcome = SearchOutcome(
                results=[failed],
                c_star=None,
                stop_reason="health_check_failed",
                last_pass=None,
                first_fail=None,
                num_evals=0,
                newly_probed=[],
                log=["server did not become ready"],
                startup_attempts=startup_attempt + len(ctx.container_start_failures),
                failures=startup_failures,
            )
        else:
            _log(f"[{round_label}] {candidate_id}: server ready, probing concurrency ...")
            tp_size = int(candidate.get("params", {}).get("tp_size", 1))
            evaluate, warmup = _make_evaluate(
                ctx, candidate_id, candidate_dir, tp_size=tp_size,
                ports=ctx.replica_ports,
            )
            warmup()  # 正式搜索前预热一次(结果丢弃,吸收 kernel 编译尖峰)
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
            )
            _pull_container_file(
                container, server_log, candidate_dir / f"server.{round_label}.log"
            )
            _write_json(
                candidate_dir / f"run_result.{round_label}.json",
                [_run_result_dict(r) for r in outcome.results],
            )
            _log(
                f"[{round_label}] {candidate_id}: done c_star={outcome.c_star} "
                f"stop={outcome.stop_reason} evals={outcome.num_evals}"
            )
            outcome.startup_attempts = startup_attempt + len(ctx.container_start_failures)
            outcome.failures = startup_failures
    finally:
        _cleanup_servers_checked(container, candidate_id=candidate_id)

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
        healthy, _ = data_health_check(result, output_len=output_len)
        return healthy and passes_sla(result, job.sla)

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
    _log(f"round-1 done. ranking={[(r['candidate_id'], r['goodput_per_host']) for r in round1_ranking]}")
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
        # 满载 round2 的结果是 instances=N 的实测求和;round1 种子是 instances=1 的
        # 单实例外推。二者口径不同,绝不能混进同一候选的 results 里让 ranker 选到被
        # 高估的单实例种子 —— 满载模式下不喂种子,round2 从头实测。
        seeds = None if config.fill_host else results_by_candidate
        outcomes = _run_batch_parallel(
            ctx_template, alloc, remote, outputs_host_dir, outputs_container_path,
            qualifies=qualifies,
            round_label="r2",
            max_probes=ROUND2_MAX_PROBES,
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

    ranking = rank_candidates(
        results_by_candidate, job.sla, output_len=output_len,
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
    result = container.exec(f"cat {shlex.quote(container_path)}")
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
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="deprecated compatibility option; round-2 always refines all candidates")
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

    try:
        job = load_job(args.job)
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
    summary = run_executor(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
