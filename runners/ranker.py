"""Health checks, SLA gating, and goodput-based ranking of benchmark runs.

Ranking uses **per-GPU normalized goodput**: the single-instance throughput is
scaled by ``gpu_count / tp_size`` so that candidates using different numbers of
GPUs per instance are compared on an equal footing (e.g. TP2 ×4 instances vs
TP8 ×1 instance on the same 8-GPU machine).
"""

from __future__ import annotations

from runners.metrics import RunResult
from schemas.job_spec import SLA


def data_health_check(result: RunResult, *, output_len: int) -> tuple[bool, str | None]:
    """§5 data sanity: at least one completed request and outputs not truncated."""
    if result.completed <= 0:
        return False, "no_completed: 0 completed requests"
    threshold = output_len * 0.9
    if result.avg_output_tokens < threshold:
        return (
            False,
            f"truncated: avg_output_tokens={result.avg_output_tokens:.1f} "
            f"< {threshold:.1f} (target {output_len})",
        )
    return True, None


def passes_sla(result: RunResult, sla: SLA) -> bool:
    """Latency and reliability gate: mean TTFT, mean TPOT, and success rate."""
    return (
        result.mean_ttft_ms <= sla.max_avg_ttft_ms
        and result.mean_tpot_ms <= sla.max_avg_tpot_ms
        and result.success_rate >= sla.min_success_rate
    )


def _instances_per_host(tp_size: int, gpu_count: int) -> float:
    """How many server instances fit on one host (gpu_count / tp_size).

    A TP2 job on an 8-GPU host can run 4 instances; TP8 runs 1.  The per-host
    goodput is single-instance throughput × instances_per_host.
    """
    if tp_size <= 0:
        return 1.0
    return gpu_count / tp_size


def _best_qualifying(
    results: list[RunResult],
    sla: SLA,
    *,
    output_len: int,
    gpu_count: int = 1,
) -> tuple[float, float, int | None]:
    """Return (best_raw_goodput, best_per_gpu_goodput, best_concurrency) over
    healthy, SLA-passing runs.

    * raw_goodput       = total_throughput of the single instance
    * per_gpu_goodput   = raw_goodput × (gpu_count / tp_size)
    """
    best_raw = 0.0
    best_per_gpu = 0.0
    best_concurrency: int | None = None
    for result in results:
        healthy, _ = data_health_check(result, output_len=output_len)
        if not healthy or not passes_sla(result, sla):
            continue
        raw = result.total_throughput
        per_gpu = raw * _instances_per_host(result.tp_size, gpu_count)
        if best_concurrency is None or per_gpu > best_per_gpu:
            best_raw = raw
            best_per_gpu = per_gpu
            best_concurrency = result.concurrency
    if best_concurrency is None:
        return 0.0, 0.0, None
    return best_raw, best_per_gpu, best_concurrency


def candidate_goodput(
    results: list[RunResult],
    sla: SLA,
    *,
    output_len: int,
    gpu_count: int = 1,
) -> float:
    """Per-GPU normalized goodput (max over healthy, SLA-passing runs)."""
    _, per_gpu, _ = _best_qualifying(results, sla, output_len=output_len, gpu_count=gpu_count)
    return per_gpu


def rank_candidates(
    results_by_candidate: dict[str, list[RunResult]],
    sla: SLA,
    *,
    output_len: int,
    gpu_count: int = 1,
) -> list[dict]:
    """Rank candidates by per-GPU normalized goodput (descending).

    Each ranking row carries both the raw single-instance goodput and the
    per-GPU normalized value used for sorting.
    """
    ranking: list[dict] = []
    for candidate_id, results in results_by_candidate.items():
        raw, per_gpu, best_concurrency = _best_qualifying(
            results, sla, output_len=output_len, gpu_count=gpu_count
        )
        # tp_size from the first healthy result for display
        tp_size = 1
        for r in results:
            healthy, _ = data_health_check(r, output_len=output_len)
            if healthy and passes_sla(r, sla):
                tp_size = r.tp_size
                break
        ranking.append(
            {
                "candidate_id": candidate_id,
                "tp_size": tp_size,
                "instances_per_host": _instances_per_host(tp_size, gpu_count),
                "goodput_raw": raw,
                "goodput_per_gpu": per_gpu,
                "best_concurrency": best_concurrency,
            }
        )
    ranking.sort(key=lambda row: row["goodput_per_gpu"], reverse=True)
    return ranking
