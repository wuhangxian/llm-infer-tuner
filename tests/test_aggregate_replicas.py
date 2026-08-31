"""Unit tests for _aggregate_replicas + ranker no-double-count (issue #2 core).

整机满载 N 实例:N 个相同实例的实测结果求和成一条 per-host 结果,instances=N。
ranker 再用 per_host = (total/instances) × floor(gpu/tp) 把外推乘数除回去,
使满载实测下乘数塌成 ×1、N 只被计一次。这些用例钉死聚合与防双重计数。
"""

from __future__ import annotations

from runners.executor import _aggregate_replicas
from runners.metrics import RunResult
from runners.ranker import _best_qualifying, _instances_per_host, rank_candidates
from schemas.job_spec import SLA


def _r(cid="c", *, conc=8, tput=1000.0, ttft=100.0, tpot=10.0, status="ok", tp=2):
    return RunResult(
        candidate_id=cid,
        concurrency=conc,
        num_prompts=conc * 4,
        completed=conc * 4,
        success_rate=1.0,
        request_throughput=float(conc),
        output_throughput=tput,
        total_throughput=tput,
        mean_ttft_ms=ttft,
        p99_ttft_ms=ttft * 2,
        mean_tpot_ms=tpot,
        p99_tpot_ms=tpot * 2,
        total_output_tokens=conc * 4 * 1000,
        avg_output_tokens=1000.0,
        duration=10.0,
        tp_size=tp,
        status=status,
    )


def test_aggregate_sums_throughput_sets_instances():
    """4 个相同实例求和:total = ΣN,instances = 4,延迟取最差。"""
    reps = [_r(tput=1000.0, ttft=100.0), _r(tput=1100.0, ttft=120.0),
            _r(tput=900.0, ttft=110.0), _r(tput=1050.0, ttft=105.0)]
    agg = _aggregate_replicas(reps, expected=4)
    assert agg.status == "ok"
    assert agg.total_throughput == 4050.0        # 求和
    assert agg.instances == 4
    assert agg.full_host_measured is True
    assert agg.mean_ttft_ms == 120.0             # 最差副本
    assert agg.duration == 10.0


def test_aggregate_missing_replica_fails():
    """副本不齐(只回来 3 个,期望 4)→ 整体失败,绝不拿部分求和冒充满载。"""
    reps = [_r(), _r(), _r()]
    agg = _aggregate_replicas(reps, expected=4)
    assert agg.status == "health_check_failed"
    assert agg.instances == 4  # 期望值,供 ranker 判定


def test_aggregate_one_unhealthy_fails():
    """有一个副本 status != ok → 整体失败。"""
    reps = [_r(), _r(), _r(), _r(status="bad_args")]
    agg = _aggregate_replicas(reps, expected=4)
    assert agg.status == "health_check_failed"


def test_no_double_count_fullload_equals_measured_sum():
    """核心防双重计数:满载聚合结果(instances=4)喂 ranker,
    per_host 恰等于实测求和,不再 ×4 外推。"""
    sla = SLA(max_avg_ttft_ms=1000.0, max_avg_tpot_ms=100.0)
    reps = [_r(tput=1000.0), _r(tput=1000.0), _r(tput=1000.0), _r(tput=1000.0)]
    agg = _aggregate_replicas(reps, expected=4)  # total=4000, instances=4, tp=2
    # gpu_count=8, tp=2 -> floor(8/2)=4。若双重计数会得 4000×4=16000。
    raw, per_host, _ = _best_qualifying([agg], sla, output_len=1000, gpu_count=8)
    assert _instances_per_host(2, 8) == 4.0
    assert per_host == 4000.0        # = 实测求和,NOT 16000
    assert raw == 1000.0             # 单实例等效吞吐 = total/instances


def test_fragmented_full_host_uses_actual_measured_replica_count() -> None:
    """NUMA fragmentation must not be filled in later by theoretical extrapolation."""
    sla = SLA(max_avg_ttft_ms=1000.0, max_avg_tpot_ms=100.0)
    aggregate = _aggregate_replicas(
        [_r(tput=1000.0, tp=2), _r(tput=1000.0, tp=2)],
        expected=2,
    )

    raw, per_host, _ = _best_qualifying(
        [aggregate], sla, output_len=1000, gpu_count=6
    )
    ranking = rank_candidates(
        {"c": [aggregate]}, sla, output_len=1000, gpu_count=6
    )

    assert raw == 1000.0
    assert per_host == 2000.0
    assert ranking[0]["goodput_per_host"] == 2000.0
    assert ranking[0]["instances_per_host"] == 2.0


def test_one_measured_full_host_replica_is_not_extrapolated() -> None:
    sla = SLA(max_avg_ttft_ms=1000.0, max_avg_tpot_ms=100.0)
    aggregate = _aggregate_replicas([_r(tput=1000.0, tp=2)], expected=1)

    raw, per_host, _ = _best_qualifying(
        [aggregate], sla, output_len=1000, gpu_count=4
    )
    ranking = rank_candidates(
        {"c": [aggregate]}, sla, output_len=1000, gpu_count=4
    )

    assert aggregate.full_host_measured is True
    assert raw == 1000.0
    assert per_host == 1000.0
    assert ranking[0]["instances_per_host"] == 1.0


def test_single_instance_path_unchanged():
    """单实例(instances=1)路径:per_host = single × floor,旧的粗筛外推行为不变。"""
    sla = SLA(max_avg_ttft_ms=1000.0, max_avg_tpot_ms=100.0)
    r = _r(tput=1000.0, tp=2)  # instances 默认 1
    assert r.instances == 1
    raw, per_host, _ = _best_qualifying([r], sla, output_len=1000, gpu_count=8)
    assert per_host == 4000.0   # 1000 × floor(8/2)=4 —— 外推,round1 粗筛用
