"""Measured three-sample ranking and interval semantics."""

from __future__ import annotations

from dataclasses import replace

import pytest

from runners.concurrency_search import SampleGroup
from runners.metrics import ProbeStatus, RunResult
from runners.ranker import (
    CandidateMeasurement,
    MeasurementMode,
    measurement_ranking_eligibility_reason,
    measurement_validation_error,
    rank_candidates,
)
from runners.reporting import annotate_baseline_threshold
from schemas.job_spec import SLA


def _sample(candidate_id: str, throughput: float) -> RunResult:
    return RunResult(
        candidate_id=candidate_id,
        concurrency=4,
        num_prompts=16,
        completed=16,
        success_rate=1.0,
        request_throughput=throughput / 100.0,
        output_throughput=throughput,
        total_throughput=throughput,
        mean_ttft_ms=100.0,
        p99_ttft_ms=200.0,
        mean_tpot_ms=10.0,
        p99_tpot_ms=20.0,
        total_output_tokens=16_000,
        avg_output_tokens=1000.0,
        duration=10.0,
        tp_size=2,
        instances=4,
        full_host_measured=True,
        status=ProbeStatus.OK,
    )


def _measurement(candidate_id: str, throughputs: list[float]) -> CandidateMeasurement:
    samples = tuple(_sample(candidate_id, throughput) for throughput in throughputs)
    representative = sorted(samples, key=lambda sample: sample.total_throughput)[1]
    group = SampleGroup(
        concurrency=4,
        samples=samples,
        representative=representative,
        qualifies=True,
    )
    return CandidateMeasurement(
        results=[representative],
        sample_groups={4: group},
        round_number=2,
        complete=True,
        certainty="exact",
        measurement_mode=MeasurementMode.FULL_HOST,
        expected_instances=4,
    )


def test_measured_ranking_uses_three_sample_median_not_single_peak() -> None:
    sla = SLA(max_avg_ttft_ms=1000.0, max_avg_tpot_ms=100.0)
    ranking = rank_candidates(
        {
            "a": _measurement("a", [100.0, 200.0, 99.0]),
            "b": _measurement("b", [110.0, 111.0, 109.0]),
        },
        sla,
        output_len=1000,
        gpu_count=8,
        required_measurement_mode=MeasurementMode.FULL_HOST,
    )

    assert [row["candidate_id"] for row in ranking] == ["b", "a"]
    assert ranking[0]["goodput_per_host_min"] == 109.0
    assert ranking[0]["goodput_per_host_median"] == 110.0
    assert ranking[0]["goodput_per_host_max"] == 111.0
    assert ranking[0]["goodput_per_host"] == 110.0
    assert ranking[0]["sample_count"] == 3
    assert ranking[0]["actual_instances"] == 4
    assert ranking[0]["measurement_mode"] == "full_host"
    assert ranking[0]["ranking_eligible"] is True
    assert ranking[0]["ranking_eligibility_reason"] is None


def test_overlapping_intervals_form_transitive_deterministic_rank_groups() -> None:
    sla = SLA(max_avg_ttft_ms=1000.0, max_avg_tpot_ms=100.0)
    ranking = rank_candidates(
        {
            "a": _measurement("a", [100.0, 110.0, 120.0]),
            "b": _measurement("b", [115.0, 120.0, 125.0]),
            "c": _measurement("c", [124.0, 130.0, 140.0]),
            "d": _measurement("d", [70.0, 80.0, 90.0]),
        },
        sla,
        output_len=1000,
        gpu_count=8,
        required_measurement_mode=MeasurementMode.FULL_HOST,
    )

    assert [row["candidate_id"] for row in ranking] == ["c", "b", "a", "d"]
    assert [row["rank_group"] for row in ranking] == [1, 1, 1, 2]


def test_closed_intervals_that_touch_at_endpoint_share_rank_group() -> None:
    sla = SLA(max_avg_ttft_ms=1000.0, max_avg_tpot_ms=100.0)
    ranking = rank_candidates(
        {
            "lower": _measurement("lower", [80.0, 90.0, 100.0]),
            "upper": _measurement("upper", [100.0, 110.0, 120.0]),
        },
        sla,
        output_len=1000,
        gpu_count=8,
        required_measurement_mode=MeasurementMode.FULL_HOST,
    )

    assert [row["candidate_id"] for row in ranking] == ["upper", "lower"]
    assert [row["rank_group"] for row in ranking] == [1, 1]


def test_estimated_seed_cannot_contaminate_measured_candidate() -> None:
    sla = SLA(max_avg_ttft_ms=1000.0, max_avg_tpot_ms=100.0)
    measurement = _measurement("measured", [400.0, 405.0, 410.0])
    estimated_seed = replace(
        _sample("measured", 480.0),
        concurrency=8,
        num_prompts=32,
        completed=32,
        total_output_tokens=32_000,
        instances=1,
        full_host_measured=False,
    )
    contaminated = replace(
        measurement,
        results=[*measurement.results, estimated_seed],
    )

    ranking = rank_candidates(
        {"measured": contaminated},
        sla,
        output_len=1000,
        gpu_count=8,
        required_measurement_mode=MeasurementMode.FULL_HOST,
    )

    assert ranking == []


def test_estimated_ranking_rejects_candidate_that_cannot_fit_on_host() -> None:
    sla = SLA(max_avg_ttft_ms=1000.0, max_avg_tpot_ms=100.0)
    measurement = _measurement("too-wide", [100.0, 101.0, 102.0])
    group = measurement.sample_groups[4]
    samples = tuple(
        replace(sample, tp_size=9, instances=1, full_host_measured=False)
        for sample in group.samples
    )
    representative = replace(
        group.representative,
        tp_size=9,
        instances=1,
        full_host_measured=False,
    )
    estimated = replace(
        measurement,
        results=[representative],
        sample_groups={
            4: replace(group, samples=samples, representative=representative)
        },
        measurement_mode=MeasurementMode.ESTIMATED,
        expected_instances=1,
    )

    assert rank_candidates(
        {"too-wide": estimated},
        sla,
        output_len=1000,
        gpu_count=8,
        required_measurement_mode=MeasurementMode.ESTIMATED,
    ) == []


@pytest.mark.parametrize(
    "corruption",
    [
        "candidate_id",
        "tp_size",
        "physical_instances",
        "planned_instances",
        "majority",
        "representative_median",
        "sample_count",
    ],
)
def test_strict_measured_ranking_rejects_corrupted_fresh_evidence(
    corruption: str,
) -> None:
    sla = SLA(max_avg_ttft_ms=1000.0, max_avg_tpot_ms=100.0)
    measurement = _measurement("strict", [400.0, 405.0, 410.0])
    group = measurement.sample_groups[4]
    samples = list(group.samples)
    representative = group.representative
    qualifies = group.qualifies
    if corruption == "candidate_id":
        samples[0] = replace(samples[0], candidate_id="other")
    elif corruption == "tp_size":
        samples[0] = replace(samples[0], tp_size=4)
    elif corruption == "physical_instances":
        samples = [replace(sample, instances=99) for sample in samples]
        representative = replace(representative, instances=99)
    elif corruption == "planned_instances":
        samples = [replace(sample, instances=3) for sample in samples]
        representative = replace(representative, instances=3)
    elif corruption == "majority":
        samples[0] = replace(samples[0], status=ProbeStatus.SLA_FAILED)
        samples[1] = replace(samples[1], status=ProbeStatus.SLA_FAILED)
    elif corruption == "representative_median":
        representative = replace(representative, total_throughput=999.0)
    elif corruption == "sample_count":
        samples.pop()
    corrupted_group = replace(
        group,
        samples=tuple(samples),
        representative=representative,
        qualifies=qualifies,
    )
    corrupted = replace(
        measurement,
        results=[representative],
        sample_groups={4: corrupted_group},
    )

    assert rank_candidates(
        {"strict": corrupted},
        sla,
        output_len=1000,
        gpu_count=8,
        required_measurement_mode=MeasurementMode.FULL_HOST,
    ) == []


def test_exact_c1_failure_is_valid_but_not_ranking_eligible() -> None:
    sla = SLA(max_avg_ttft_ms=1000.0, max_avg_tpot_ms=100.0)
    measurement = _measurement("saturated", [100.0, 101.0, 102.0])
    group = measurement.sample_groups[4]
    samples = tuple(
        replace(sample, status=ProbeStatus.SLA_FAILED) for sample in group.samples
    )
    representative = replace(group.representative, status=ProbeStatus.SLA_FAILED)
    no_pass = replace(
        measurement,
        results=[representative],
        sample_groups={
            4: replace(
                group,
                samples=samples,
                representative=representative,
                qualifies=False,
            )
        },
    )

    assert measurement_validation_error(
        "saturated",
        no_pass,
        output_len=1000,
        gpu_count=8,
        required_measurement_mode=MeasurementMode.FULL_HOST,
    ) is None
    assert measurement_ranking_eligibility_reason(
        "saturated",
        no_pass,
        sla,
        output_len=1000,
        gpu_count=8,
        required_measurement_mode=MeasurementMode.FULL_HOST,
    ) == "no qualifying SLA point"


def test_baseline_threshold_uses_interval_bounds_and_three_states() -> None:
    annotated = annotate_baseline_threshold(
        [
            {
                "candidate_id": "yes",
                "measurement_mode": "full_host",
                "ranking_eligible": True,
                "goodput_per_host_min": 120.0,
                "goodput_per_host_median": 125.0,
                "goodput_per_host_max": 130.0,
                "goodput_per_host": 125.0,
            },
            {
                "candidate_id": "crosses",
                "measurement_mode": "full_host",
                "ranking_eligible": True,
                "goodput_per_host_min": 119.0,
                "goodput_per_host_median": 120.0,
                "goodput_per_host_max": 121.0,
                "goodput_per_host": 120.0,
            },
            {
                "candidate_id": "baseline",
                "measurement_mode": "full_host",
                "ranking_eligible": True,
                "goodput_per_host_min": 90.0,
                "goodput_per_host_median": 100.0,
                "goodput_per_host_max": 110.0,
                "goodput_per_host": 100.0,
            },
            {
                "candidate_id": "no",
                "measurement_mode": "full_host",
                "ranking_eligible": True,
                "goodput_per_host_min": 100.0,
                "goodput_per_host_median": 110.0,
                "goodput_per_host_max": 119.999,
                "goodput_per_host": 110.0,
            },
        ],
        threshold_pct=20,
    )

    by_id = {row["candidate_id"]: row for row in annotated}
    assert by_id["yes"]["baseline_threshold_status"] == "yes"
    assert by_id["yes"]["beats_baseline_threshold"] is True
    assert by_id["crosses"]["baseline_threshold_status"] == "unknown"
    assert by_id["crosses"]["beats_baseline_threshold"] is False
    assert by_id["no"]["baseline_threshold_status"] == "no"
    assert by_id["no"]["beats_baseline_threshold"] is False
    assert by_id["baseline"]["baseline_threshold_status"] == "unknown"


@pytest.mark.parametrize(
    "corruption",
    [
        "baseline_order",
        "baseline_ineligible",
        "baseline_zero",
        "missing_median",
        "mixed_mode",
    ],
)
def test_strict_threshold_fails_closed_for_invalid_interval_evidence(
    corruption: str,
) -> None:
    rows = [
        {
            "candidate_id": "candidate",
            "measurement_mode": "full_host",
            "ranking_eligible": True,
            "goodput_per_host_min": 120.0,
            "goodput_per_host_median": 125.0,
            "goodput_per_host_max": 130.0,
            "goodput_per_host": 125.0,
        },
        {
            "candidate_id": "baseline",
            "measurement_mode": "full_host",
            "ranking_eligible": True,
            "goodput_per_host_min": 90.0,
            "goodput_per_host_median": 100.0,
            "goodput_per_host_max": 110.0,
            "goodput_per_host": 100.0,
        },
    ]
    if corruption == "baseline_order":
        rows[1]["goodput_per_host_min"] = 110.0
        rows[1]["goodput_per_host_max"] = 90.0
    elif corruption == "baseline_ineligible":
        rows[1]["ranking_eligible"] = False
    elif corruption == "baseline_zero":
        rows[1]["goodput_per_host_min"] = 0.0
        rows[1]["goodput_per_host_median"] = 0.0
        rows[1]["goodput_per_host_max"] = 0.0
        rows[1]["goodput_per_host"] = 0.0
    elif corruption == "missing_median":
        rows[0]["goodput_per_host_median"] = None
    elif corruption == "mixed_mode":
        rows[0]["measurement_mode"] = "estimated"

    annotated = annotate_baseline_threshold(rows, threshold_pct=20)

    assert all(row["baseline_threshold_status"] == "unknown" for row in annotated)
    assert all(row["beats_baseline_threshold"] is False for row in annotated)
