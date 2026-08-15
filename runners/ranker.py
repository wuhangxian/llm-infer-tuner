"""Health checks, SLA gating, and goodput-based ranking of benchmark runs."""

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


def _best_qualifying(
    results: list[RunResult], sla: SLA, *, output_len: int
) -> tuple[float, int | None]:
    """Return (max total_throughput, its concurrency) over healthy, SLA-passing runs."""
    best_throughput = 0.0
    best_concurrency: int | None = None
    for result in results:
        healthy, _ = data_health_check(result, output_len=output_len)
        if not healthy or not passes_sla(result, sla):
            continue
        if best_concurrency is None or result.total_throughput > best_throughput:
            best_throughput = result.total_throughput
            best_concurrency = result.concurrency
    if best_concurrency is None:
        return 0.0, None
    return best_throughput, best_concurrency


def candidate_goodput(results: list[RunResult], sla: SLA, *, output_len: int) -> float:
    """Max total_throughput among runs that pass both the health check and the SLA."""
    goodput, _ = _best_qualifying(results, sla, output_len=output_len)
    return goodput


def rank_candidates(
    results_by_candidate: dict[str, list[RunResult]], sla: SLA, *, output_len: int
) -> list[dict]:
    """Rank candidates by goodput (descending), reporting the winning concurrency."""
    ranking: list[dict] = []
    for candidate_id, results in results_by_candidate.items():
        goodput, best_concurrency = _best_qualifying(results, sla, output_len=output_len)
        ranking.append(
            {
                "candidate_id": candidate_id,
                "goodput": goodput,
                "best_concurrency": best_concurrency,
            }
        )
    ranking.sort(key=lambda row: row["goodput"], reverse=True)
    return ranking
