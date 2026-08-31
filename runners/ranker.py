"""Health checks, SLA gating, and goodput-based ranking of benchmark runs.

Ranking uses **per-host goodput**: the single-instance throughput is
scaled by ``gpu_count / tp_size`` so that candidates using different numbers of
GPUs per instance are compared on an equal footing (e.g. TP2 ×4 instances vs
TP8 ×1 instance on the same 8-GPU machine).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from statistics import median

from runners.concurrency_search import SampleGroup
from runners.metrics import SEARCH_VERDICT_STATUSES, ProbeStatus, RunResult
from schemas.job_spec import SLA


class MeasurementMode(StrEnum):
    FULL_HOST = "full_host"
    ESTIMATED = "estimated"


@dataclass(frozen=True)
class CandidateMeasurement:
    """The authoritative fresh Round-2 evidence for one candidate."""

    results: list[RunResult]
    sample_groups: Mapping[int, SampleGroup]
    round_number: int
    complete: bool
    certainty: str
    measurement_mode: MeasurementMode | str
    expected_instances: int


def _metric_sanity_error(
    result: RunResult, *, output_len: int | None = None
) -> str | None:
    if type(result.full_host_measured) is not bool:
        return (
            "full_host_measured must be a boolean, got "
            f"{result.full_host_measured!r}"
        )
    for field_name in ("concurrency", "num_prompts", "completed", "tp_size", "instances"):
        value = getattr(result, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return f"{field_name} must be a positive integer, got {value!r}"
    if not result.full_host_measured and result.instances != 1:
        return "instances must equal 1 when full_host_measured is false"
    if result.completed != result.num_prompts:
        return (
            f"completed={result.completed} does not match "
            f"num_prompts={result.num_prompts}"
        )
    if (
        isinstance(result.total_output_tokens, bool)
        or not isinstance(result.total_output_tokens, int)
        or result.total_output_tokens < 0
    ):
        return "total_output_tokens must be a non-negative integer"
    numeric_fields = (
        "success_rate",
        "request_throughput",
        "output_throughput",
        "total_throughput",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "avg_output_tokens",
        "duration",
    )
    for field_name in numeric_fields:
        value = getattr(result, field_name)
        if not _is_finite_nonnegative_number(value):
            return f"{field_name} must be finite and non-negative, got {value!r}"
    if not 0 <= result.success_rate <= 1:
        return f"success_rate must be in [0, 1], got {result.success_rate!r}"
    if output_len is not None:
        if isinstance(output_len, bool) or not isinstance(output_len, int) or output_len <= 0:
            return f"output_len must be a positive integer, got {output_len!r}"
        scaled_tokens = 10 * result.total_output_tokens
        lower_scaled = 9 * output_len * result.completed
        upper_scaled = 11 * output_len * result.completed
        if not lower_scaled <= scaled_tokens <= upper_scaled:
            return (
                "total_output_tokens is outside 90%..110% of the expected "
                "completed-token total"
            )
        try:
            average_numerator, average_denominator = (
                result.avg_output_tokens.as_integer_ratio()
            )
        except (AttributeError, OverflowError, ValueError):
            return "avg_output_tokens is not representable as a finite ratio"
        average_scaled = 10 * average_numerator
        average_lower_scaled = 9 * output_len * average_denominator
        average_upper_scaled = 11 * output_len * average_denominator
        if not average_lower_scaled <= average_scaled <= average_upper_scaled:
            return (
                f"avg_output_tokens={result.avg_output_tokens!r} outside "
                "90%..110% of output_len"
            )
    return None


def _is_finite_nonnegative_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric >= 0


def data_health_check(result: RunResult, *, output_len: int) -> tuple[bool, str | None]:
    """§5 data sanity: at least one completed request and outputs not truncated."""
    if result.status != ProbeStatus.OK:
        return False, f"probe_status: {result.status}"
    sanity_error = _metric_sanity_error(result, output_len=output_len)
    if sanity_error is not None:
        return False, f"invalid_metrics: {sanity_error}"
    return True, None


def passes_sla(result: RunResult, sla: SLA) -> bool:
    """Latency and reliability gate: mean TTFT, mean TPOT, and success rate."""
    if result.status != ProbeStatus.OK or _metric_sanity_error(result) is not None:
        return False
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
    if not _is_representable_positive_int(tp_size) or not _is_representable_positive_int(
        gpu_count
    ):
        return 0.0
    return float(gpu_count // tp_size)


def _is_representable_positive_int(value: object) -> bool:
    """Return whether a topology count is a strict, finite positive integer."""

    if type(value) is not int or value <= 0:
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


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
    if not _is_representable_positive_int(gpu_count) or not results or any(
        result.status not in SEARCH_VERDICT_STATUSES
        or _metric_sanity_error(result, output_len=output_len) is not None
        for result in results
    ):
        return 0.0, 0.0, None
    best_raw = 0.0
    best_per_host = 0.0
    best_concurrency: int | None = None
    for result in results:
        physical_capacity = gpu_count // result.tp_size
        if physical_capacity < 1 or (
            result.full_host_measured and result.instances > physical_capacity
        ):
            return 0.0, 0.0, None
        healthy, _ = data_health_check(result, output_len=output_len)
        if not healthy or not passes_sla(result, sla):
            continue
        # total_throughput 是 result.instances 个并发实例的实测求和。
        # round1 单实例仍按 floor(gpu/tp) 做兼容外推；round2 满载结果已经
        # 按真实 NUMA topology 聚合，必须直接使用实测总和，不能拿理论 floor
        # 补齐放不下的碎片，否则会系统性高估该候选。
        measured = result.instances
        try:
            raw = result.total_throughput / measured
        except OverflowError:
            return 0.0, 0.0, None
        if getattr(result, "full_host_measured", False):
            per_host = result.total_throughput
        else:
            per_host = raw * _instances_per_host(result.tp_size, gpu_count)
        if not math.isfinite(raw) or raw < 0 or not math.isfinite(per_host) or per_host < 0:
            return 0.0, 0.0, None
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


def _rank_legacy_candidates(
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
        # Any infrastructure/invalid point means this candidate did not finish
        # an exact statistical search.  Keep it in candidate reporting, but do
        # not manufacture an official rank from an older successful point.
        if not results or any(
            result.status not in SEARCH_VERDICT_STATUSES for result in results
        ):
            continue
        raw, per_host, best_concurrency = _best_qualifying(
            results, sla, output_len=output_len, gpu_count=gpu_count
        )
        if best_concurrency is None:
            continue
        # tp_size from the first healthy result for display
        tp_size = 1
        for r in results:
            healthy, _ = data_health_check(r, output_len=output_len)
            if healthy and passes_sla(r, sla):
                tp_size = r.tp_size
                break
        instances_per_host = _instances_per_host(tp_size, gpu_count)
        for result in reversed(results):
            healthy, _ = data_health_check(result, output_len=output_len)
            if (
                result.concurrency == best_concurrency
                and healthy
                and passes_sla(result, sla)
                and getattr(result, "full_host_measured", False)
            ):
                instances_per_host = float(result.instances)
                break
        ranking.append(
            {
                "candidate_id": candidate_id,
                "tp_size": tp_size,
                "instances_per_host": instances_per_host,
                "goodput_raw": raw,
                "goodput_per_host": per_host,
                "best_concurrency": best_concurrency,
                **_best_point_metrics(results, best_concurrency, sla, output_len),
            }
        )
    ranking.sort(key=lambda row: row["goodput_per_host"], reverse=True)
    return ranking


def measurement_validation_error(
    candidate_id: str,
    measurement: CandidateMeasurement,
    *,
    output_len: int,
    gpu_count: int,
    required_measurement_mode: MeasurementMode | str,
) -> str | None:
    required_mode = MeasurementMode(required_measurement_mode)
    try:
        mode = MeasurementMode(measurement.measurement_mode)
    except (TypeError, ValueError):
        return "invalid measurement mode"
    if measurement.round_number != 2:
        return "ranking requires round 2 evidence"
    if measurement.complete is not True or measurement.certainty != "exact":
        return "ranking requires exact complete evidence"
    if mode != required_mode:
        return "measurement mode mismatch"
    if not _is_representable_positive_int(gpu_count):
        return "invalid host GPU count"
    if not _is_representable_positive_int(measurement.expected_instances):
        return "invalid expected instance count"
    if not measurement.sample_groups:
        return "missing fresh sample groups"
    results_by_c = {result.concurrency: result for result in measurement.results}
    if len(results_by_c) != len(measurement.results):
        return "duplicate representative concurrency"
    if set(results_by_c) != set(measurement.sample_groups):
        return "representatives and sample groups do not match"

    expected_tp: int | None = None
    expected_instances: int | None = None
    median_fields = (
        "num_prompts",
        "completed",
        "success_rate",
        "request_throughput",
        "output_throughput",
        "total_throughput",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "total_output_tokens",
        "avg_output_tokens",
        "duration",
    )
    for concurrency, group in measurement.sample_groups.items():
        representative = results_by_c[concurrency]
        if concurrency != group.concurrency or len(group.samples) != 3:
            return "each concurrency requires exactly three fresh samples"
        if representative != group.representative:
            return "sample-group representative mismatch"
        all_records = (*group.samples, representative)
        if any(record.candidate_id != candidate_id for record in all_records):
            return "candidate identity mismatch"
        if any(record.concurrency != concurrency for record in all_records):
            return "sample concurrency mismatch"
        if any(
            record.status not in SEARCH_VERDICT_STATUSES
            or _metric_sanity_error(record, output_len=output_len) is not None
            for record in all_records
        ):
            return "invalid statistical sample"
        group_tp = representative.tp_size
        group_instances = representative.instances
        if any(
            sample.tp_size != group_tp or sample.instances != group_instances
            for sample in group.samples
        ):
            return "sample topology mismatch"
        if expected_tp is None:
            expected_tp = group_tp
            expected_instances = group_instances
        elif group_tp != expected_tp or group_instances != expected_instances:
            return "candidate topology changed across concurrencies"
        physical_capacity = gpu_count // group_tp
        if physical_capacity < 1:
            return "candidate cannot fit on host"
        if mode == MeasurementMode.FULL_HOST:
            if (
                group_instances != measurement.expected_instances
                or measurement.expected_instances > physical_capacity
                or any(not record.full_host_measured for record in all_records)
            ):
                return "invalid full-host topology evidence"
        elif measurement.expected_instances != 1 or any(
            record.full_host_measured or record.instances != 1 for record in all_records
        ):
            return "estimated mode requires single-instance evidence"
        pass_votes = sum(sample.status == ProbeStatus.OK for sample in group.samples)
        majority_qualifies = pass_votes >= 2
        expected_status = (
            ProbeStatus.OK if majority_qualifies else ProbeStatus.SLA_FAILED
        )
        if (
            group.qualifies is not majority_qualifies
            or representative.status != expected_status
        ):
            return "sample majority verdict mismatch"
        for field_name in median_fields:
            if getattr(representative, field_name) != median(
                getattr(sample, field_name) for sample in group.samples
            ):
                return f"representative {field_name} is not the sample median"
    return None


def _measured_candidate_row(
    candidate_id: str,
    measurement: CandidateMeasurement,
    sla: SLA,
    *,
    output_len: int,
    gpu_count: int,
    required_mode: MeasurementMode,
) -> dict | None:
    try:
        mode = MeasurementMode(measurement.measurement_mode)
    except (TypeError, ValueError):
        return None
    if measurement_validation_error(
        candidate_id,
        measurement,
        output_len=output_len,
        gpu_count=gpu_count,
        required_measurement_mode=required_mode,
    ) is not None:
        return None
    best: dict | None = None
    for concurrency, group in measurement.sample_groups.items():
        if not group.qualifies:
            continue
        representative = group.representative
        healthy, _ = data_health_check(representative, output_len=output_len)
        if not healthy or not passes_sla(representative, sla):
            continue
        sample_per_host: list[float] = []
        sample_raw: list[float] = []
        actual_instances: int | None = None
        invalid_group = False
        for sample in group.samples:
            if (
                sample.concurrency != concurrency
                or sample.status not in SEARCH_VERDICT_STATUSES
                or _metric_sanity_error(sample, output_len=output_len) is not None
            ):
                invalid_group = True
                break
            if mode == MeasurementMode.FULL_HOST:
                if not sample.full_host_measured:
                    invalid_group = True
                    break
                if actual_instances is None:
                    actual_instances = sample.instances
                elif sample.instances != actual_instances:
                    invalid_group = True
                    break
                per_host = sample.total_throughput
                raw = sample.total_throughput / sample.instances
            else:
                if sample.full_host_measured or sample.instances != 1:
                    invalid_group = True
                    break
                actual_instances = 1
                raw = sample.total_throughput
                per_host = raw * _instances_per_host(sample.tp_size, gpu_count)
            if not math.isfinite(raw) or not math.isfinite(per_host):
                invalid_group = True
                break
            sample_raw.append(raw)
            sample_per_host.append(per_host)
        if invalid_group or actual_instances is None:
            continue
        row = {
            "candidate_id": candidate_id,
            "tp_size": representative.tp_size,
            "instances_per_host": (
                float(actual_instances)
                if mode == MeasurementMode.FULL_HOST
                else _instances_per_host(representative.tp_size, gpu_count)
            ),
            "actual_instances": actual_instances,
            "measurement_mode": str(mode),
            "goodput_raw": median(sample_raw),
            "goodput_per_host_min": min(sample_per_host),
            "goodput_per_host_median": median(sample_per_host),
            "goodput_per_host_max": max(sample_per_host),
            "goodput_per_host": median(sample_per_host),
            "sample_count": len(sample_per_host),
            "best_concurrency": concurrency,
            "ranking_eligible": True,
            "ranking_eligibility_reason": None,
            **_best_point_metrics([representative], concurrency, sla, output_len),
        }
        if best is None or (
            row["goodput_per_host_median"], -concurrency
        ) > (
            best["goodput_per_host_median"], -best["best_concurrency"]
        ):
            best = row
    return best


def measurement_ranking_eligibility_reason(
    candidate_id: str,
    measurement: CandidateMeasurement,
    sla: SLA,
    *,
    output_len: int,
    gpu_count: int,
    required_measurement_mode: MeasurementMode | str,
) -> str | None:
    """Explain why valid Round-2 evidence cannot produce a ranking row."""

    required_mode = MeasurementMode(required_measurement_mode)
    validation_error = measurement_validation_error(
        candidate_id,
        measurement,
        output_len=output_len,
        gpu_count=gpu_count,
        required_measurement_mode=required_mode,
    )
    if validation_error is not None:
        return validation_error
    if _measured_candidate_row(
        candidate_id,
        measurement,
        sla,
        output_len=output_len,
        gpu_count=gpu_count,
        required_mode=required_mode,
    ) is None:
        return "no qualifying SLA point"
    return None


def rank_candidates(
    results_by_candidate: Mapping[
        str, list[RunResult] | CandidateMeasurement
    ],
    sla: SLA,
    *,
    output_len: int,
    gpu_count: int = 1,
    required_measurement_mode: MeasurementMode | str | None = None,
) -> list[dict]:
    """Rank legacy result lists or strict fresh Round-2 measurements."""

    if not results_by_candidate:
        return []
    if all(isinstance(value, list) for value in results_by_candidate.values()):
        legacy = {
            candidate_id: value
            for candidate_id, value in results_by_candidate.items()
            if isinstance(value, list)
        }
        return _rank_legacy_candidates(
            legacy, sla, output_len=output_len, gpu_count=gpu_count
        )
    if not all(
        isinstance(value, CandidateMeasurement)
        for value in results_by_candidate.values()
    ):
        raise TypeError("ranking inputs must not mix legacy and measured candidates")
    if required_measurement_mode is None:
        raise ValueError("strict measured ranking requires a measurement mode")
    required_mode = MeasurementMode(required_measurement_mode)
    ranking = [
        row
        for candidate_id, value in results_by_candidate.items()
        if isinstance(value, CandidateMeasurement)
        for row in [
            _measured_candidate_row(
                candidate_id,
                value,
                sla,
                output_len=output_len,
                gpu_count=gpu_count,
                required_mode=required_mode,
            )
        ]
        if row is not None
    ]
    ranking.sort(
        key=lambda row: (-row["goodput_per_host_median"], row["candidate_id"])
    )
    parents = list(range(len(ranking)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(ranking)):
        for right in range(left + 1, len(ranking)):
            if (
                ranking[left]["goodput_per_host_min"]
                <= ranking[right]["goodput_per_host_max"]
                and ranking[right]["goodput_per_host_min"]
                <= ranking[left]["goodput_per_host_max"]
            ):
                union(left, right)
    rank_groups: dict[int, int] = {}
    for index, row in enumerate(ranking):
        component = find(index)
        if component not in rank_groups:
            rank_groups[component] = len(rank_groups) + 1
        row["rank_group"] = rank_groups[component]
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
