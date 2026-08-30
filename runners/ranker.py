"""Health checks, SLA gating, and goodput-based ranking of benchmark runs.

Ranking uses **per-host goodput**: the single-instance throughput is
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
    """How many server instances fit on one host (floor(gpu_count / tp_size)).

    A TP2 job on an 8-GPU host can run 4 instances; TP8 runs 1.  The per-host
    goodput is single-instance throughput × instances_per_host.

    必须向下取整:半个实例在物理上放不下。例如 6 卡跑 TP4,只能放 1 个实例
    (剩 2 卡不够再起一个),而非 1.5 个——否则 per-host goodput 会被高估 50%,
    在非 2 的幂主机上把真正打包更优的候选(如 TP3×2)挤下去。
    tp_size > gpu_count 时返回 0(单主机连一个实例都放不下)。
    """
    if tp_size <= 0:
        return 1.0
    return float(gpu_count // tp_size)


def _best_qualifying(
    results: list[RunResult],
    sla: SLA,
    *,
    output_len: int,
    gpu_count: int = 1,
) -> tuple[float, float, int | None]:
    """Return (best_raw_goodput, best_per_host_goodput, best_concurrency) over
    healthy, SLA-passing runs.

    * raw_goodput       = total_throughput of the single instance
    * per_host_goodput   = raw_goodput × (gpu_count / tp_size)
    """
    best_raw = 0.0
    best_per_host = 0.0
    best_concurrency: int | None = None
    for result in results:
        healthy, _ = data_health_check(result, output_len=output_len)
        if not healthy or not passes_sla(result, sla):
            continue
        # total_throughput 是 result.instances 个并发实例的实测求和。
        # 先除回单实例吞吐,再乘"整机能放几个实例",得到 per-host goodput:
        #   per_host = (total / instances) × floor(gpu/tp)
        # · round1 单实例(instances=1):= single × floor —— 纸面外推,只用于粗筛选 top-K。
        # · round2 整机满载(instances=floor(gpu/tp)):= (S/N)×N = S —— 外推乘数被实测实例数
        #   除掉、塌成 ×1,得到的就是实测满载求和。这样 N 只被计一次,绝不会 S×N 双重计数。
        measured = max(1, int(getattr(result, "instances", 1)))
        raw = result.total_throughput / measured
        per_host = raw * _instances_per_host(result.tp_size, gpu_count)
        if best_concurrency is None or per_host > best_per_host:
            best_raw = raw
            best_per_host = per_host
            best_concurrency = result.concurrency
    if best_concurrency is None:
        return 0.0, 0.0, None
    return best_raw, best_per_host, best_concurrency


def candidate_goodput(
    results: list[RunResult],
    sla: SLA,
    *,
    output_len: int,
    gpu_count: int = 1,
) -> float:
    """Per-GPU normalized goodput (max over healthy, SLA-passing runs)."""
    _, per_host, _ = _best_qualifying(results, sla, output_len=output_len, gpu_count=gpu_count)
    return per_host


def rank_candidates(
    results_by_candidate: dict[str, list[RunResult]],
    sla: SLA,
    *,
    output_len: int,
    gpu_count: int = 1,
) -> list[dict]:
    """Rank candidates by per-host goodput (descending).

    Each ranking row carries both the raw single-instance goodput and the
    per-GPU normalized value used for sorting.
    """
    ranking: list[dict] = []
    for candidate_id, results in results_by_candidate.items():
        raw, per_host, best_concurrency = _best_qualifying(
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
                "goodput_per_host": per_host,
                "best_concurrency": best_concurrency,
                **_best_point_metrics(results, best_concurrency, sla, output_len),
            }
        )
    ranking.sort(key=lambda row: row["goodput_per_host"], reverse=True)
    return ranking


def _best_point_metrics(
    results: list[RunResult],
    best_concurrency: int | None,
    sla: SLA,
    output_len: int,
) -> dict:
    if best_concurrency is None:
        return {
            "request_throughput": 0.0,
            "output_throughput": 0.0,
            "total_throughput": 0.0,
            "mean_ttft_ms": 0.0,
            "p99_ttft_ms": 0.0,
            "mean_tpot_ms": 0.0,
            "p99_tpot_ms": 0.0,
            "success_rate": 0.0,
            "avg_output_tokens": 0.0,
        }
    for result in reversed(results):
        healthy, _ = data_health_check(result, output_len=output_len)
        if result.concurrency == best_concurrency and healthy and passes_sla(result, sla):
            return {
                "request_throughput": result.request_throughput,
                "output_throughput": result.output_throughput,
                "total_throughput": result.total_throughput,
                "mean_ttft_ms": result.mean_ttft_ms,
                "p99_ttft_ms": result.p99_ttft_ms,
                "mean_tpot_ms": result.mean_tpot_ms,
                "p99_tpot_ms": result.p99_tpot_ms,
                "success_rate": result.success_rate,
                "avg_output_tokens": result.avg_output_tokens,
            }
    return {}
