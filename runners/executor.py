"""Phase-2 executor: run generated configs on a remote host and collect metrics."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from planner.claude_code_client import ClaudeCodeClient
from runners.bench_runner import (
    generate_benchmark_commands,
    rewrite_bench_command,
    run_benchmark,
    substitute_placeholders,
)
from runners.concurrency_search import SearchOutcome, search_saturation
from runners.container import Container, ContainerConfig
from runners.metrics import RunResult, parse_bench_text
from runners.ranker import data_health_check, passes_sla, rank_candidates
from runners.readiness import make_health_probe, wait_until_ready
from runners.remote import RemoteRunner
from schemas.job_spec import JobSpec

# Round-1 = coarse adaptive expansion over ALL candidates (no bisection); round-2
# = precise bisection over the top-K, reusing round-1 probes as seeds. The two
# rounds are cheap because expansion is log-scale and round-2 only re-probes new
# midpoints (seeds are served from cache, never re-benched).
ROUND1_MAX_PROBES = 7   # reaches ~C=64 via 1,2,4,8,16,32,64 in one candidate
ROUND1_CONFIRM = 1      # coarse: naive verdicts are fine for picking top-K
ROUND2_MAX_PROBES = 14
ROUND2_CONFIRM = 2      # precise: re-probe boundary passes AND fails (see search module)
DEFAULT_TOP_K = 5
DEFAULT_MAX_CAP = 256

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
    top_k: int = DEFAULT_TOP_K    # round-2 refines only the top-K candidates by round-1 goodput
    max_cap: int = DEFAULT_MAX_CAP  # upper bound on concurrency the search will probe
    ssh_password: str = ""  # optional; empty = key-based SSH


def _load_job(job_path: Path) -> JobSpec:
    data = json.loads(job_path.read_text(encoding="utf-8"))
    return JobSpec.model_validate(data)


def _load_output_len(workload: str, *, workloads_path: Path = DEFAULT_WORKLOADS_PATH) -> int:
    """Read ``output_tokens.value`` for the job's workload; the health check's target length."""
    data = yaml.safe_load(workloads_path.read_text(encoding="utf-8")) or {}
    workloads = data.get("workloads", {}) or {}
    entry = workloads.get(workload, {}) or {}
    output_tokens = entry.get("output_tokens", {}) or {}
    value = output_tokens.get("value", 0)
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


def _load_candidates(configs_path: Path, max_candidates: int) -> list[dict[str, Any]]:
    """Read the first ``max_candidates`` config rows (each row: id/params/cmd/reasons)."""
    candidates: list[dict[str, Any]] = []
    for line in configs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            candidates.append(row)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _run_result_dict(result: RunResult) -> dict[str, Any]:
    return asdict(result)


def _force_output_file(command: str, output_path: str) -> str:
    """Point the bench command's --output-file at a known in-container path.

    The client skill emits ``--output-file result_...jsonl`` (a bare relative
    name); rewrite that argument so the file lands where the executor reads it.
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


def _resolve_bench_template(
    job: JobSpec, *, config: ExecutorConfig, client: ClaudeCodeClient
) -> str:
    """Call the client skill ONCE and return a single bench command template.

    The skill emits one command per concurrency level; they differ only in
    --max-concurrency / --num-prompts, so any one of them serves as the template
    that rewrite_bench_command() re-parameterizes per probe. Prefer the lowest
    concurrency (fewest prompts) as the canonical template.
    """
    commands = generate_benchmark_commands(
        job, project_root=config.project_root, client=client
    )
    if not commands:
        raise RuntimeError("client skill returned no benchmark commands")
    commands.sort(key=lambda bc: bc.concurrency)
    return commands[0].command


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


def _make_evaluate(ctx: _CandidateContext, candidate_id: str, candidate_dir: Path, tp_size: int = 1):
    """Build the evaluate(concurrency) -> RunResult closure the search calls.

    Each call rewrites the single template for this concurrency, runs one bench in
    the container, pulls the console + result back to candidate_dir (evidence fires
    exactly once per real bench), and parses metrics. Any exception becomes a
    non-qualifying RunResult so an overload collapse counts as a boundary fail
    rather than aborting the whole search.
    """
    container = ctx.container
    candidate_out = f"{ctx.outputs_container_path}/{candidate_id}"

    def evaluate(concurrency: int) -> RunResult:
        try:
            command_tpl, num_prompts = rewrite_bench_command(
                ctx.bench_template, concurrency=concurrency, multiplier=ctx.multiplier
            )
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            result_name = f"bench_c{concurrency}_{timestamp}.jsonl"
            result_container_path = f"{candidate_out}/{result_name}"
            base_command = substitute_placeholders(
                command_tpl,
                host="127.0.0.1",
                port=ctx.port,
                model_path=ctx.config.model_container_path,
                job_id=ctx.job.job_id,
                timestamp=timestamp,
            )
            command = _force_output_file(base_command, result_container_path)
            _log(f"    {candidate_id}: bench C={concurrency} (num_prompts={num_prompts}) ...")
            bench_run = run_benchmark(container, command)

            log_name = f"bench_c{concurrency}_{timestamp}.log"
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
            _log(
                f"      C={concurrency}: tput={run_result.total_throughput:.0f} "
                f"ttft={run_result.mean_ttft_ms:.0f}ms tpot={run_result.mean_tpot_ms:.1f}ms "
                f"succ={run_result.success_rate:.2f} status={run_result.status}"
            )
            return run_result
        except Exception as exc:  # collapse != abort; the search treats this as a fail
            return _health_check_failed_result(
                candidate_id, f"evaluate raised at C={concurrency}: {exc!r}"
            )

    return evaluate


def _allocate_gpus_and_ports(
    candidates: list[dict[str, Any]],
    gpu_count: int,
    base_port: int = 30000,
) -> list[tuple[dict[str, Any], str, int]]:
    """Assign GPU IDs and ports to each candidate.

    Returns a list of (candidate, gpu_ids_str, port) tuples. Candidates are
    grouped into batches that fit within gpu_count: the sum of tp_size in each
    batch does not exceed gpu_count.

    Example (8 GPUs, 4 candidates all TP1):
      batch 1: cand1(GPU0, port 30000), cand2(GPU1, 30001),
               cand3(GPU2, 30002), cand4(GPU3, 30003)
      → 4 GPUs used, 4 idle (could fit 4 more TP1 candidates in same batch)

    Example (8 GPUs, 4 candidates all TP2):
      batch 1: cand1(GPU0,1, 30000), cand2(GPU2,3, 30001),
               cand3(GPU4,5, 30002), cand4(GPU6,7, 30003)
    """
    result: list[tuple[dict[str, Any], str, int]] = []
    gpu_cursor = 0
    port_cursor = base_port
    for candidate in candidates:
        tp_size = int(candidate.get("params", {}).get("tp_size", 1))
        tp_size = max(1, tp_size)
        gpu_ids = list(range(gpu_cursor, gpu_cursor + tp_size))
        gpu_cursor += tp_size
        gpu_ids_str = "device=" + ",".join(str(g) for g in gpu_ids)
        result.append((candidate, gpu_ids_str, port_cursor))
        port_cursor += 1
    return result


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
) -> dict[str, SearchOutcome]:
    """Run a batch of candidates in parallel, each in its own container.

    Each candidate gets its own docker container with specific GPUs and port.
    All containers are started, health-checked, and benchmarked concurrently.
    """
    config = ctx_template.config
    job = ctx_template.job
    bench_template = ctx_template.bench_template
    multiplier = ctx_template.multiplier

    containers: list[tuple[str, Container, _CandidateContext, dict[str, Any], str, int]] = []

    # Create one container per candidate in this batch
    for candidate, gpu_ids_str, port in batch:
        candidate_id = str(candidate.get("id", "unknown"))
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
        )
        containers.append((candidate_id, container, ctx, candidate, gpu_ids_str, port))

    # Start all containers in this batch
    for candidate_id, container, ctx, candidate, gpu_ids_str, port in containers:
        _log(f"[{round_label}] {candidate_id}: starting container (GPUs={gpu_ids_str}, port={port}) ...")
        started = container.start()
        if not started.ok or not container.is_running():
            _log(f"[{round_label}] {candidate_id}: container FAILED to start: {started.stderr.strip()}")

    # Run each candidate (server lifecycle + search) — still sequential within batch
    # but each has its own container with isolated GPUs and port
    outcomes: dict[str, SearchOutcome] = {}
    for candidate_id, container, ctx, candidate, gpu_ids_str, port in containers:
        cand_seeds = None
        if seeds_by_id and candidate_id in seeds_by_id:
            cand_seeds = seeds_by_id[candidate_id]
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
        outcomes[candidate_id] = outcome

    # Stop and remove all containers in this batch
    for candidate_id, container, ctx, candidate, gpu_ids_str, port in containers:
        container.stop()
        container.remove()

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
    cmd = str(candidate.get("cmd", ""))
    candidate_dir = config.results_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # configs.jsonl may carry either the ${MODEL_PATH} placeholder or the host
    # model dir baked straight into --model-path; rewrite both to the in-container
    # mount point so the server finds the weights.
    server_cmd = cmd.replace("${MODEL_PATH}", config.model_container_path)
    server_cmd = server_cmd.replace(config.model_host_dir, config.model_container_path)
    # Replace the port in the launch command with this candidate's assigned port
    candidate_port = ctx.port
    server_cmd = server_cmd.replace("--port 30000", f"--port {candidate_port}")
    server_cmd = server_cmd.replace("--host 0.0.0.0 --port 30000", f"--host 0.0.0.0 --port {candidate_port}")
    candidate_out = f"{ctx.outputs_container_path}/{candidate_id}"
    server_log = f"{candidate_out}/server.log"
    container.exec(f"mkdir -p {shlex.quote(candidate_out)}")
    _log(f"[{round_label}] {candidate_id}: starting server, waiting for /health ...")
    container.exec_detached(server_cmd, server_log)

    def _server_alive() -> bool:
        return container.exec("pgrep -f sglang.launch_server").ok

    probe = make_health_probe(container, port=candidate_port)
    ready = wait_until_ready(probe, is_alive=_server_alive)

    try:
        if not ready:
            _log(f"[{round_label}] {candidate_id}: server NOT ready (crash/timeout)")
            local_log = candidate_dir / f"server.{round_label}.log"
            _pull_container_file(container, server_log, local_log)
            reason = _extract_failure_reason(local_log)
            _log(f"[{round_label}] {candidate_id}: crash reason: {reason}")
            failed = _health_check_failed_result(
                candidate_id, f"server did not become ready: {reason}"
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
            )
        else:
            _log(f"[{round_label}] {candidate_id}: server ready, probing concurrency ...")
            tp_size = int(candidate.get("params", {}).get("tp_size", 1))
            evaluate = _make_evaluate(ctx, candidate_id, candidate_dir, tp_size=tp_size)
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
    finally:
        container.exec("pkill -f sglang.launch_server || true")

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
    }



def _preflight_checks(config: ExecutorConfig, remote: RemoteRunner) -> None:
    """Verify remote connectivity, model directory, and image availability before container start."""
    # 0. Clean up stale containers from previous runs (same job_id prefix)
    prefix = shlex.quote(config.container_name + "%")
    cleanup = remote.run(f"docker rm -f $(docker ps -a --filter name={prefix} -q) 2>/dev/null || true")
    if cleanup.ok and cleanup.stdout.strip():
        _log(f"Cleaned up stale containers: {cleanup.stdout.strip()}")

    # 1. SSH
    probe = remote.run("echo ok")
    if not probe.ok or probe.stdout.strip() != "ok":
        detail = probe.stderr.strip()
        raise SystemExit(
            f"SSH connection failed: {config.ssh_target}\n{detail}"
        )
    _log(f"SSH OK: {config.ssh_target}")

    # 2. Model dir
    q = shlex.quote(config.model_host_dir)
    check_dir = remote.run(f"test -d {q} && ls {q} | head -5")
    if not check_dir.ok:
        detail = check_dir.stderr.strip()
        raise SystemExit(
            f"Model dir not found: {config.model_host_dir}\n{detail}"
        )
    n = len(check_dir.stdout.strip().splitlines())
    _log(f"Model dir OK: {config.model_host_dir} ({n} files)")

    # 3. Image
    iq = shlex.quote(config.image_ref)
    image_check = remote.run(
        f"docker image inspect {iq} --format {{{{.Id}}}} 2>/dev/null || echo NOT_LOCAL"
    )
    if image_check.stdout.strip() == "NOT_LOCAL":
        _log(f"Image not cached, pulling: {config.image_ref}")
        pull = remote.run(f"docker pull {iq}", timeout=600)
        if not pull.ok:
            detail = pull.stderr.strip()
            raise SystemExit(
                f"Image pull failed: {config.image_ref}\n{detail}"
            )
        _log(f"Image pulled: {config.image_ref}")
    else:
        _log(f"Image local: {config.image_ref}")


def _check_hardware_match(job: JobSpec, config: ExecutorConfig) -> None:
    """Verify target.json GPU fields match job.json requirements before any remote work.

    Catches mismatched hardware early (e.g. job wants 8x72G but target only has 4x72G)
    instead of failing during container start or benchmark.
    """
    if not config.target_gpu_model:
        _log("⚠️  target.json 缺少 gpu_model 字段,跳过硬件校验")
        return
    if config.target_gpu_model != job.gpu_model:
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
    client: ClaudeCodeClient | None = None,
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
    candidates = _load_candidates(config.configs_path, config.max_candidates)
    _check_hardware_match(job, config)

    remote = remote or RemoteRunner(config.ssh_target, ssh_password=config.ssh_password)
    _preflight_checks(config, remote)
    client = client or ClaudeCodeClient()

    # FAIRNESS: the client skill (an LLM) is called EXACTLY ONCE per job to get a
    # bench command template. Every candidate and every probed concurrency reuses
    # that one template with only --max-concurrency / --num-prompts rewritten, so
    # input length / dataset / seed are byte-identical across the whole sweep. (The
    # old code called the skill once per candidate, letting the workload drift and
    # making candidates non-comparable.)
    bench_template = _resolve_bench_template(job, config=config, client=client)

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
    outputs_host_dir = config.remote_outputs_dir
    if not outputs_host_dir:
        home = remote.run("echo $HOME").stdout.strip() or "/root"
        outputs_host_dir = f"{home}/llm-infer-tuner-outputs/{job.job_id}"
    mkdir = remote.run(f"mkdir -p {shlex.quote(outputs_host_dir)}")
    if not mkdir.ok:
        raise RuntimeError(
            f"failed to create remote outputs dir {outputs_host_dir!r}: {mkdir.stderr.strip()}"
        )

    outputs_container_path = "/workspace/outputs"

    config.results_dir.mkdir(parents=True, exist_ok=True)
    _log(f"job={job.job_id} candidates={len(candidates)} "
         f"output_len={output_len} multiplier={multiplier} top_k={config.top_k} "
         f"gpu_count={job.gpu_count}")

    results_by_candidate: dict[str, list[RunResult]] = {}
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

    # ---- Split into batches that fit within gpu_count ----
    # Each batch reuses GPU IDs starting from 0 (batches run sequentially).
    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_gpus_used = 0
    for candidate in candidates:
        tp_size = int(candidate.get("params", {}).get("tp_size", 1))
        if current_gpus_used + tp_size > job.gpu_count and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_gpus_used = 0
        current_batch.append(candidate)
        current_gpus_used += tp_size
    if current_batch:
        batches.append(current_batch)

    _log(f"{len(candidates)} candidates split into {len(batches)} batch(es) "
         f"(max {job.gpu_count} GPUs per batch)")

    # ---- ROUND 1: coarse expansion over ALL candidates (batch-parallel) ----
    for batch_idx, batch in enumerate(batches):
        # Allocate GPUs within this batch (reset cursor to 0)
        alloc = _allocate_gpus_and_ports(batch, job.gpu_count, base_port=config.port)
        _log(f"round-1 batch {batch_idx + 1}/{len(batches)}: "
             f"{[c.get('id') for c in batch]}")
        outcomes = _run_batch_parallel(
            ctx_template, alloc, remote, outputs_host_dir, outputs_container_path,
            qualifies=qualifies,
            round_label="r1",
            max_probes=ROUND1_MAX_PROBES,
            confirm=ROUND1_CONFIRM,
            refine=False,
        )
        for candidate_id, outcome in outcomes.items():
            results_by_candidate[candidate_id] = outcome.results
            search_diagnostics[candidate_id] = _outcome_diag(outcome)
            candidate_summaries[candidate_id] = {
                "candidate_id": candidate_id,
                "round1": _outcome_diag(outcome),
            }

    # ---- pick top-K by round-1 goodput ------------------------------------
    round1_ranking = rank_candidates(
        results_by_candidate, job.sla, output_len=output_len,
        gpu_count=job.gpu_count,
    )
    top_ids = [row["candidate_id"] for row in round1_ranking[: config.top_k]]
    _log(f"round-1 done. ranking={[(r['candidate_id'], r['goodput_per_host']) for r in round1_ranking]}")
    _log(f"round-2 refining top-{config.top_k}: {top_ids}")

    # ---- ROUND 2: precise bisection on top-K (batch-parallel) ----
    candidate_by_id = {str(c.get("id", "unknown")): c for c in candidates}
    top_candidates = [candidate_by_id[cid] for cid in top_ids if cid in candidate_by_id]

    # Split top-K into batches too
    top_batches: list[list[dict[str, Any]]] = []
    current_batch = []
    current_gpus_used = 0
    for candidate in top_candidates:
        tp_size = int(candidate.get("params", {}).get("tp_size", 1))
        if current_gpus_used + tp_size > job.gpu_count and current_batch:
            top_batches.append(current_batch)
            current_batch = []
            current_gpus_used = 0
        current_batch.append(candidate)
        current_gpus_used += tp_size
    if current_batch:
        top_batches.append(current_batch)

    for batch_idx, batch in enumerate(top_batches):
        alloc = _allocate_gpus_and_ports(batch, job.gpu_count, base_port=config.port)
        _log(f"round-2 batch {batch_idx + 1}/{len(top_batches)}: "
             f"{[c.get('id') for c in batch]}")
        outcomes = _run_batch_parallel(
            ctx_template, alloc, remote, outputs_host_dir, outputs_container_path,
            qualifies=qualifies,
            round_label="r2",
            max_probes=ROUND2_MAX_PROBES,
            confirm=ROUND2_CONFIRM,
            refine=True,
            seeds_by_id=results_by_candidate,
        )
        for candidate_id, outcome in outcomes.items():
            results_by_candidate[candidate_id] = outcome.results
            search_diagnostics[candidate_id] = _outcome_diag(outcome)
            candidate_summaries[candidate_id]["round2"] = _outcome_diag(outcome)

    ranking = rank_candidates(
        results_by_candidate, job.sla, output_len=output_len,
        gpu_count=job.gpu_count,
    )

    # Baseline filtering: if baseline_threshold_pct > 0, find baseline goodput
    # and filter out candidates below baseline * (1 + pct/100)
    threshold_pct = job.search.baseline_threshold_pct
    baseline_goodput = None
    if threshold_pct > 0:
        for row in ranking:
            if row["candidate_id"] == "baseline":
                baseline_goodput = row["goodput_per_host"]
                break
        if baseline_goodput is not None and baseline_goodput > 0:
            threshold = baseline_goodput * (1 + threshold_pct / 100)
            filtered = [r for r in ranking if r["goodput_per_host"] >= threshold or r["candidate_id"] == "baseline"]
            _log(f"Baseline filter: baseline={baseline_goodput:.0f}, threshold=+{threshold_pct}%={threshold:.0f}, kept {len(filtered)}/{len(ranking)} candidates")
            ranking = filtered
        elif baseline_goodput is not None and baseline_goodput == 0:
            _log(f"Baseline failed (goodput=0), skipping threshold filter")
        else:
            _log(f"No baseline candidate found, skipping threshold filter")

    _write_json(config.results_dir / "ranking.json", ranking)

    return {
        "job_id": job.job_id,
        "output_len": output_len,
        "num_prompts_multiplier": multiplier,
        "top_k": config.top_k,
        "top_ids": top_ids,
        "candidates": list(candidate_summaries.values()),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run generated SGLang configs on a remote host and rank them."
    )
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--configs", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--ssh-password", default="", help="SSH password (empty=key-based)")
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--model-host-dir", required=True)
    parser.add_argument("--model-container-path", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-candidates", type=int, default=1)
    parser.add_argument("--concurrencies", default=None,
                        help="deprecated: adaptive search now chooses probes")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="round-2 refines only the top-K candidates by round-1 goodput")
    parser.add_argument("--max-cap", type=int, default=DEFAULT_MAX_CAP,
                        help="upper bound on concurrency the search will probe")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--container-name", default="llm-infer-tuner-exec")
    parser.add_argument("--remote-outputs-dir", default="")
    parser.add_argument("--target-gpu-model", default="")
    parser.add_argument("--target-gpu-count", type=int, default=0)
    parser.add_argument("--target-gpu-memory-gb", type=float, default=0.0)
    args = parser.parse_args(argv)

    config = ExecutorConfig(
        job_path=args.job,
        configs_path=args.configs,
        results_dir=args.results,
        ssh_target=args.ssh_target,
        ssh_password=args.ssh_password,
        image_ref=args.image_ref,
        model_host_dir=args.model_host_dir,
        model_container_path=args.model_container_path,
        project_root=args.project_root,
        max_candidates=args.max_candidates,
        concurrencies=_parse_concurrencies(args.concurrencies),
        port=args.port,
        container_name=args.container_name,
        remote_outputs_dir=args.remote_outputs_dir,
        top_k=args.top_k,
        max_cap=args.max_cap,
        target_gpu_model=args.target_gpu_model,
        target_gpu_count=args.target_gpu_count,
        target_gpu_memory_gb=args.target_gpu_memory_gb,
    )
    summary = run_executor(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
