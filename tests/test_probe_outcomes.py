"""Fail-closed probe outcome contracts (offline; no SSH/GPU)."""

from __future__ import annotations

import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from runners.concurrency_search import search_saturation
from runners.executor import _classify_probe_for_search, _make_evaluate
from runners.metrics import (
    ProbeStatus,
    RunResult,
    classify_known_runtime_issue,
    failed_probe_result,
    parse_bench_text,
)
from runners.ranker import (
    candidate_goodput,
    data_health_check,
    passes_sla,
    rank_candidates,
)
from runners.remote import CommandFailureKind, CommandResult, RemoteRunner
from runners.reporting import build_candidate_rows, render_candidate_preview
from schemas.job_spec import SLA


def _bench_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "max_concurrency": 8,
        "completed": 32,
        "total_output_tokens": 32768,
        "request_throughput": 8.0,
        "output_throughput": 8192.0,
        "total_throughput": 40960.0,
        "mean_ttft_ms": 200.0,
        "p99_ttft_ms": 400.0,
        "mean_tpot_ms": 10.0,
        "p99_tpot_ms": 20.0,
        "duration": 4.0,
    }
    record.update(overrides)
    return record


_VALID_KNOWN_TRACE = """
Traceback (most recent call last):
  File "/sglang/srt/layers/attention/triton_backend.py", line 987, in _update_target_verify_buffers
    custom_mask = custom_mask.expand(expected_shape)
RuntimeError: The expanded size of the tensor (64) must match the existing size (32)
"""


def test_parser_rejects_a_record_for_the_wrong_requested_concurrency() -> None:
    result = parse_bench_text(
        json.dumps(_bench_record(max_concurrency=16)),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        tp_size=4,
        output_len=1024,
    )

    assert result.status == "invalid_result"
    assert result.candidate_id == "c001"
    assert result.tp_size == 4
    assert result.concurrency == 8
    assert result.num_prompts == 32
    assert "max_concurrency" in (result.failure_reason or "")


def test_parser_rejects_a_record_missing_a_required_metric() -> None:
    record = _bench_record()
    del record["mean_tpot_ms"]

    result = parse_bench_text(
        json.dumps(record),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        tp_size=4,
        output_len=1024,
    )

    assert result.status == "invalid_result"
    assert "mean_tpot_ms" in (result.failure_reason or "")


def test_parser_rejects_duplicate_json_keys() -> None:
    record = json.dumps(_bench_record())
    duplicate = record.replace(
        '"max_concurrency": 8',
        '"max_concurrency": 8, "max_concurrency": 8',
        1,
    )

    result = parse_bench_text(
        duplicate,
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )

    assert result.status == ProbeStatus.INVALID_RESULT
    assert "duplicate JSON key" in (result.failure_reason or "")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_concurrency", True),
        ("completed", True),
        ("total_output_tokens", True),
        ("max_concurrency", 8.0),
        ("completed", 32.0),
        ("total_output_tokens", 32768.0),
    ],
)
def test_parser_rejects_boolean_or_integral_float_integer_fields(
    field_name: str,
    value: object,
) -> None:
    result = parse_bench_text(
        json.dumps(_bench_record(**{field_name: value})),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )

    assert result.status == ProbeStatus.INVALID_RESULT
    assert field_name in (result.failure_reason or "")


def test_parser_rejects_malformed_noise_instead_of_falling_back_to_valid_line() -> None:
    text = "{not-json\n" + json.dumps(_bench_record())

    result = parse_bench_text(
        text,
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )

    assert result.status == ProbeStatus.INVALID_RESULT
    assert "malformed JSON" in (result.failure_reason or "")


def test_parser_accepts_one_complete_record_surrounded_by_blank_lines() -> None:
    result = parse_bench_text(
        "\n  \n" + json.dumps(_bench_record()) + "\n\t\n",
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )

    assert result.status == ProbeStatus.OK


@pytest.mark.parametrize(
    "overrides",
    [
        {"concurrency": 0},
        {"concurrency": True},
        {"num_prompts": -1},
        {"output_len": 0},
        {"tp_size": False},
    ],
)
def test_parser_rejects_invalid_requested_probe_coordinates(
    overrides: dict[str, object],
) -> None:
    kwargs: dict[str, object] = {
        "candidate_id": "c001",
        "concurrency": 8,
        "num_prompts": 32,
        "output_len": 1024,
        "tp_size": 4,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match="positive integer"):
        parse_bench_text(json.dumps(_bench_record()), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0])
def test_parser_rejects_nonfinite_or_negative_required_metrics(value: float) -> None:
    result = parse_bench_text(
        json.dumps(_bench_record(total_throughput=value)),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )

    assert result.status == "invalid_result"
    assert "total_throughput" in (result.failure_reason or "")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("request_throughput", 10**1000),
        ("total_output_tokens", 10**1000),
    ],
)
def test_parser_types_unrepresentable_required_numbers_as_invalid_result(
    field_name: str,
    value: int,
) -> None:
    result = parse_bench_text(
        json.dumps(_bench_record(**{field_name: value})),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )

    assert result.status == ProbeStatus.INVALID_RESULT
    assert field_name in (result.failure_reason or "") or "output tokens" in (
        result.failure_reason or ""
    )


def test_parser_types_excessively_nested_json_as_invalid_result() -> None:
    nested = '{"x":' + "[" * 10_000 + "0" + "]" * 10_000 + "}"

    result = parse_bench_text(
        nested,
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )

    assert result.status == ProbeStatus.INVALID_RESULT
    assert "JSON" in (result.failure_reason or "")


def test_parser_ignores_nonfinite_unconsumed_metadata() -> None:
    result = parse_bench_text(
        json.dumps(_bench_record(request_rate=math.inf)),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )

    assert result.status == "ok"


def test_parser_rejects_partial_request_completion() -> None:
    result = parse_bench_text(
        json.dumps(_bench_record(completed=31, total_output_tokens=31744)),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        tp_size=4,
        output_len=1024,
    )

    assert result.status == "invalid_result"
    assert result.completed == 0
    assert result.tp_size == 4
    assert result.concurrency == 8
    assert result.num_prompts == 32
    assert "completed" in (result.failure_reason or "")


def test_parser_rejects_incomplete_output_tokens() -> None:
    result = parse_bench_text(
        json.dumps(_bench_record(total_output_tokens=32 * 800)),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )

    assert result.status == "invalid_result"
    assert "output tokens" in (result.failure_reason or "")


def test_ranker_rejects_non_ok_probe_even_when_metrics_look_healthy() -> None:
    result = parse_bench_text(
        json.dumps(_bench_record()),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )
    result.status = ProbeStatus.RUNTIME_FAILED
    sla = SLA(max_avg_ttft_ms=1000, max_avg_tpot_ms=100, min_success_rate=0.99)

    assert data_health_check(result, output_len=1024)[0] is False
    assert passes_sla(result, sla) is False
    assert candidate_goodput([result], sla, output_len=1024) == 0.0


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("total_throughput", math.nan),
        ("request_throughput", math.inf),
        ("total_throughput", -1.0),
        ("avg_output_tokens", math.nan),
        ("duration", -1.0),
        ("completed", True),
        ("success_rate", 2.0),
    ],
)
def test_ranker_defensively_rejects_invalid_ok_metrics(
    field_name: str,
    value: object,
) -> None:
    result = parse_bench_text(
        json.dumps(_bench_record()),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )
    setattr(result, field_name, value)
    sla = SLA(max_avg_ttft_ms=1000, max_avg_tpot_ms=100, min_success_rate=0.99)

    healthy, _reason = data_health_check(result, output_len=1024)
    assert healthy is False
    assert passes_sla(result, sla) is False
    assert candidate_goodput([result], sla, output_len=1024) == 0.0


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("total_throughput", 10**1000),
        ("full_host_measured", "yes"),
    ],
)
def test_ranker_never_emits_a_row_for_adversarial_ok_metrics(
    field_name: str,
    value: object,
) -> None:
    result = parse_bench_text(
        json.dumps(_bench_record()),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )
    setattr(result, field_name, value)
    sla = SLA(max_avg_ttft_ms=1000, max_avg_tpot_ms=100, min_success_rate=0.99)

    assert rank_candidates(
        {"c001": [result]}, sla, output_len=1024, gpu_count=8
    ) == []


def test_ranker_requires_at_least_one_healthy_sla_passing_ok_point() -> None:
    result = parse_bench_text(
        json.dumps(_bench_record(mean_ttft_ms=2000.0)),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )
    result.status = ProbeStatus.SLA_FAILED
    sla = SLA(max_avg_ttft_ms=1000, max_avg_tpot_ms=100, min_success_rate=0.99)

    assert rank_candidates({"c001": [result]}, sla, output_len=1024) == []


def test_invalid_sla_failed_point_invalidates_older_ok_rank() -> None:
    ok_result = parse_bench_text(
        json.dumps(_bench_record(max_concurrency=1, completed=4, total_output_tokens=4096)),
        candidate_id="c001",
        concurrency=1,
        num_prompts=4,
        output_len=1024,
    )
    invalid_fail = parse_bench_text(
        json.dumps(_bench_record(mean_ttft_ms=2000.0)),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )
    invalid_fail.status = ProbeStatus.SLA_FAILED
    invalid_fail.total_throughput = -1.0
    sla = SLA(max_avg_ttft_ms=1000, max_avg_tpot_ms=100, min_success_rate=0.99)

    assert rank_candidates(
        {"c001": [ok_result, invalid_fail]}, sla, output_len=1024
    ) == []


def test_ranker_rejects_nonfinite_derived_per_host_goodput() -> None:
    result = parse_bench_text(
        json.dumps(_bench_record(total_throughput=sys.float_info.max)),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )
    sla = SLA(max_avg_ttft_ms=1000, max_avg_tpot_ms=100, min_success_rate=0.99)

    assert result.status == ProbeStatus.OK
    assert rank_candidates(
        {"c001": [result]}, sla, output_len=1024, gpu_count=8
    ) == []


def test_ranker_rejects_physically_impossible_full_host_instance_count() -> None:
    result = parse_bench_text(
        json.dumps(_bench_record()),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
        tp_size=4,
    )
    result.full_host_measured = True
    result.instances = 10**1000
    sla = SLA(max_avg_ttft_ms=1000, max_avg_tpot_ms=100, min_success_rate=0.99)

    assert rank_candidates(
        {"c001": [result]}, sla, output_len=1024, gpu_count=8
    ) == []


@pytest.mark.parametrize("gpu_count", [True, 0, -1, 10**1000])
def test_ranker_rejects_invalid_or_unrepresentable_gpu_count(
    gpu_count: object,
) -> None:
    result = parse_bench_text(
        json.dumps(_bench_record()),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )
    sla = SLA(max_avg_ttft_ms=1000, max_avg_tpot_ms=100, min_success_rate=0.99)

    assert rank_candidates(
        {"c001": [result]},
        sla,
        output_len=1024,
        gpu_count=cast(Any, gpu_count),
    ) == []


@pytest.mark.parametrize("output_len", [True, 0, -1, 10**1000])
def test_ranker_rejects_invalid_or_unrepresentable_output_len(
    output_len: object,
) -> None:
    result = parse_bench_text(
        json.dumps(
            _bench_record(
                max_concurrency=1,
                completed=1,
                total_output_tokens=1024,
            )
        ),
        candidate_id="c001",
        concurrency=1,
        num_prompts=1,
        output_len=1024,
    )
    if output_len == 10**1000:
        result.total_output_tokens = 10**1000
    sla = SLA(max_avg_ttft_ms=1000, max_avg_tpot_ms=100, min_success_rate=0.99)

    assert rank_candidates(
        {"c001": [result]},
        sla,
        output_len=cast(Any, output_len),
        gpu_count=1,
    ) == []


def test_run_result_rejects_an_invented_probe_status() -> None:
    with pytest.raises(ValueError, match="unknown probe status"):
        RunResult(
            candidate_id="c001",
            concurrency=8,
            num_prompts=32,
            completed=32,
            success_rate=1.0,
            request_throughput=8.0,
            output_throughput=8192.0,
            total_throughput=40960.0,
            mean_ttft_ms=200.0,
            p99_ttft_ms=400.0,
            mean_tpot_ms=10.0,
            p99_tpot_ms=20.0,
            total_output_tokens=32768,
            avg_output_tokens=1024.0,
            duration=4.0,
            status="looks_good_to_me",
        )


def test_infrastructure_probe_stops_search_without_becoming_a_boundary() -> None:
    def evaluate(concurrency: int) -> RunResult:
        result = parse_bench_text(
            json.dumps(
                _bench_record(
                    max_concurrency=concurrency,
                    completed=concurrency * 4,
                    total_output_tokens=concurrency * 4 * 1024,
                )
            ),
            candidate_id="c001",
            concurrency=concurrency,
            num_prompts=concurrency * 4,
            tp_size=4,
            output_len=1024,
        )
        if concurrency == 2:
            result.status = ProbeStatus.RUNTIME_FAILED
            result.failure_reason = "server died after benchmark"
        return result

    outcome = search_saturation(
        evaluate,
        lambda result: result.status == ProbeStatus.OK,
        max_probes=8,
        confirm=1,
    )

    assert outcome.stop_reason == "runtime_failed"
    assert outcome.complete is False
    assert outcome.certainty == "unknown"
    assert outcome.c_star is None
    assert outcome.last_pass == 1
    assert outcome.first_fail is None
    assert [result.concurrency for result in outcome.results] == [1, 2]


def test_evaluator_exception_adapter_preserves_full_probe_coordinates() -> None:
    def evaluate(concurrency: int) -> RunResult:
        if concurrency == 2:
            raise ConnectionError("ssh vanished")
        return parse_bench_text(
            json.dumps(
                _bench_record(
                    max_concurrency=concurrency,
                    completed=concurrency * 4,
                    total_output_tokens=concurrency * 4 * 1024,
                )
            ),
            candidate_id="c001",
            concurrency=concurrency,
            num_prompts=concurrency * 4,
            tp_size=4,
            output_len=1024,
        )

    def on_exception(concurrency: int, exc: Exception) -> RunResult:
        assert isinstance(exc, ConnectionError)
        return failed_probe_result(
            "c001",
            status=ProbeStatus.TRANSPORT_FAILED,
            reason=str(exc),
            concurrency=concurrency,
            num_prompts=concurrency * 4,
            tp_size=4,
        )

    outcome = search_saturation(
        evaluate,
        lambda result: result.status == ProbeStatus.OK,
        max_probes=8,
        confirm=1,
        on_evaluate_exception=on_exception,
    )

    failed = outcome.results[-1]
    assert outcome.complete is False
    assert outcome.c_star is None
    assert failed.candidate_id == "c001"
    assert failed.tp_size == 4
    assert failed.concurrency == 2
    assert failed.num_prompts == 8


def test_nonverdict_seed_is_remeasured_instead_of_poisoning_fresh_search() -> None:
    seed = failed_probe_result(
        "c001",
        status=ProbeStatus.RUNTIME_FAILED,
        reason="old round crashed",
        concurrency=1,
        num_prompts=4,
        tp_size=4,
    )
    calls: list[int] = []

    def evaluate(concurrency: int) -> RunResult:
        calls.append(concurrency)
        return parse_bench_text(
            json.dumps(
                _bench_record(
                    max_concurrency=concurrency,
                    completed=concurrency * 4,
                    total_output_tokens=concurrency * 4 * 1024,
                    mean_ttft_ms=200.0 if concurrency == 1 else 2000.0,
                )
            ),
            candidate_id="c001",
            concurrency=concurrency,
            num_prompts=concurrency * 4,
            output_len=1024,
        )

    outcome = search_saturation(
        evaluate,
        lambda result: result.mean_ttft_ms <= 500,
        seeds=[seed],
        confirm=1,
    )

    assert calls[:2] == [1, 2]
    assert outcome.complete is True
    assert outcome.c_star == 1


def test_incomplete_candidate_has_no_official_rank_and_reports_failed() -> None:
    ok_result = parse_bench_text(
        json.dumps(_bench_record(max_concurrency=1, completed=4, total_output_tokens=4096)),
        candidate_id="c001",
        concurrency=1,
        num_prompts=4,
        output_len=1024,
    )
    failed_result = failed_probe_result(
        "c001",
        status=ProbeStatus.RUNTIME_FAILED,
        reason="server died at C=2",
        concurrency=2,
        num_prompts=8,
        tp_size=4,
    )
    sla = SLA(max_avg_ttft_ms=1000, max_avg_tpot_ms=100, min_success_rate=0.99)

    assert candidate_goodput(
        [ok_result, failed_result], sla, output_len=1024
    ) == 0.0
    ranking = rank_candidates(
        {"c001": [ok_result, failed_result]},
        sla,
        output_len=1024,
    )
    assert ranking == []

    failure = {
        "failed_at": "2026-08-31T12:00:00+08:00",
        "round": 2,
        "concurrency": 2,
        "num_prompts": 8,
        "tp_size": 4,
        "status": "runtime_failed",
        "known_issue": "known-runtime-defect",
        "reason": "server died at C=2",
    }
    rows = build_candidate_rows(
        [{"id": "c001", "params": {"tp_size": 4}}],
        {
            "c001": {
                "attempts": 1,
                "round2": {
                    "complete": False,
                    "certainty": "unknown",
                    "stop_reason": "runtime_failed",
                },
                "failures": [failure],
            }
        },
        {"c001": {2: [ok_result, failed_result]}},
        ranking,
        output_len=1024,
    )
    assert rows[0]["status"] == "incomplete"
    assert "rank" not in rows[0]
    assert rows[0]["failed_concurrency"] == 2
    assert rows[0]["failed_num_prompts"] == 8
    assert rows[0]["failed_tp_size"] == 4
    assert rows[0]["failure_status"] == "runtime_failed"
    assert rows[0]["known_issue"] == "known-runtime-defect"
    preview = render_candidate_preview(rows)[0]
    assert "r2/C1:status=ok" in preview
    assert "r2/C2:status=runtime_failed" in preview
    assert "failure_status=runtime_failed" in preview
    assert "failed_tp=4" in preview
    assert "failed_C=2" in preview
    assert "failed_N=8" in preview
    assert "known_issue=known-runtime-defect" in preview


def test_reporting_output_health_handles_unrepresentable_output_len() -> None:
    result = parse_bench_text(
        json.dumps(
            _bench_record(
                max_concurrency=1,
                completed=1,
                total_output_tokens=1024,
            )
        ),
        candidate_id="c001",
        concurrency=1,
        num_prompts=1,
        output_len=1024,
    )
    result.total_output_tokens = 10**1000
    rows = build_candidate_rows(
        [{"id": "c001", "params": {"tp_size": 1}}],
        {
            "c001": {
                "round2": {"complete": False, "certainty": "unknown"},
                "failures": [],
            }
        },
        {"c001": {2: [result]}},
        [],
        output_len=10**1000,
    )

    assert rows[0]["concurrency_points"][0]["output_healthy"] is True


def test_valid_metrics_over_sla_become_a_statistical_sla_failed_boundary() -> None:
    def evaluate(concurrency: int) -> RunResult:
        return parse_bench_text(
            json.dumps(
                _bench_record(
                    max_concurrency=concurrency,
                    completed=concurrency * 4,
                    total_output_tokens=concurrency * 4 * 1024,
                    mean_ttft_ms=200.0 if concurrency == 1 else 2000.0,
                )
            ),
            candidate_id="c001",
            concurrency=concurrency,
            num_prompts=concurrency * 4,
            output_len=1024,
        )

    outcome = search_saturation(
        evaluate,
        lambda result: result.mean_ttft_ms <= 500,
        max_probes=8,
        confirm=1,
    )

    assert outcome.complete is True
    assert outcome.c_star == 1
    assert outcome.first_fail == 2
    assert outcome.stop_reason == "found_boundary"
    assert outcome.results[-1].status == ProbeStatus.SLA_FAILED


def test_executor_assigns_sla_failed_only_after_data_health_passes() -> None:
    sla = SLA(max_avg_ttft_ms=500, max_avg_tpot_ms=100, min_success_rate=0.99)
    over_sla = parse_bench_text(
        json.dumps(_bench_record(mean_ttft_ms=2000.0)),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )
    truncated = parse_bench_text(
        json.dumps(_bench_record()),
        candidate_id="c001",
        concurrency=8,
        num_prompts=32,
        output_len=1024,
    )
    truncated.avg_output_tokens = 100.0

    assert _classify_probe_for_search(over_sla, output_len=1024, sla=sla) is False
    assert over_sla.status == ProbeStatus.SLA_FAILED
    assert _classify_probe_for_search(truncated, output_len=1024, sla=sla) is False
    assert truncated.status == ProbeStatus.INVALID_RESULT


class _ProbeContainer:
    def __init__(
        self,
        *,
        bench_returncode: int = 0,
        cat_result: CommandResult | None = None,
        healthy_after: bool = True,
        server_log: str = "",
        bench_writes_result: bool = True,
        stale_record: str = "",
    ) -> None:
        self.bench_returncode = bench_returncode
        self.cat_result = cat_result
        self.healthy_after = healthy_after
        self.server_log = server_log
        self.bench_writes_result = bench_writes_result
        self.output_record = stale_record
        self.result_exists = bool(stale_record)

    def exec(self, command: str, *, timeout: int | None = None) -> CommandResult:
        del timeout
        if "sglang.bench_serving" in command:
            parts = shlex.split(command)
            concurrency = int(parts[parts.index("--max-concurrency") + 1])
            num_prompts = int(parts[parts.index("--num-prompts") + 1])
            fresh_record = json.dumps(
                _bench_record(
                    max_concurrency=concurrency,
                    completed=num_prompts,
                    total_output_tokens=num_prompts * 1024,
                )
            )
            if self.bench_writes_result:
                self.output_record = fresh_record
                self.result_exists = True
            return CommandResult(
                returncode=self.bench_returncode,
                stdout=fresh_record,
                stderr="synthetic benchmark failure" if self.bench_returncode else "",
            )
        if command.startswith("cat "):
            if command.endswith("server.log"):
                return CommandResult(0, self.server_log, "")
            if self.cat_result is not None:
                return self.cat_result
            if self.result_exists:
                return CommandResult(0, self.output_record, "")
            return CommandResult(1, "", "no such file")
        if "/health" in command:
            return CommandResult(0 if self.healthy_after else 7, "", "")
        if command.startswith("rm -f -- "):
            self.result_exists = False
            self.output_record = ""
            return CommandResult(0, "", "")
        return CommandResult(0, "", "")


def _evaluate_once(
    container: _ProbeContainer,
    tmp_path,
    *,
    candidate: dict[str, object] | None = None,
    engine_version: str = "0.5.16",
) -> RunResult:
    context = SimpleNamespace(
        container=container,
        bench_template=(
            "python -m sglang.bench_serving --max-concurrency 1 --num-prompts 4 "
            "--host ${BENCHMARK_HOST} --port ${BENCHMARK_PORT} "
            "--model ${MODEL_PATH}"
        ),
        multiplier=4,
        outputs_container_path="/workspace/outputs",
        port=30000,
        config=SimpleNamespace(
            model_container_path="/models/qwen",
        ),
        job=SimpleNamespace(job_id="job"),
        engine_version=engine_version,
    )
    evaluate, _warmup = _make_evaluate(
        cast(Any, context),
        "c001",
        tmp_path,
        tp_size=4,
        output_len=1024,
        candidate=candidate,
    )
    return evaluate(8)


def test_nonzero_benchmark_exit_cannot_reuse_a_stale_valid_result(tmp_path) -> None:
    result = _evaluate_once(_ProbeContainer(bench_returncode=1), tmp_path)

    assert result.status == ProbeStatus.BENCHMARK_FAILED
    assert result.tp_size == 4
    assert result.concurrency == 8
    assert result.num_prompts == 32
    assert "bench exit 1" in (result.failure_reason or "")


def test_nonzero_result_read_cannot_parse_valid_stdout(tmp_path) -> None:
    result = _evaluate_once(
        _ProbeContainer(
            cat_result=CommandResult(
                1,
                json.dumps(_bench_record()),
                "synthetic read failure",
            )
        ),
        tmp_path,
    )

    assert result.status == ProbeStatus.BENCHMARK_FAILED
    assert "result read failed" in (result.failure_reason or "")


def test_local_evidence_write_oserror_is_not_mislabeled_transport(
    tmp_path,
    monkeypatch,
) -> None:
    original_write_text = Path.write_text

    def fail_log_write(path: Path, *args, **kwargs):
        if path.suffix == ".log":
            raise OSError("synthetic local disk full")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_log_write)

    result = _evaluate_once(_ProbeContainer(), tmp_path)

    assert result.status == ProbeStatus.BENCHMARK_FAILED
    assert result.concurrency == 8
    assert result.num_prompts == 32


def test_result_read_transport_failure_cannot_be_hidden_by_valid_stdout(tmp_path) -> None:
    stale_record = json.dumps(_bench_record())
    result = _evaluate_once(
        _ProbeContainer(
            cat_result=CommandResult(
                255,
                stale_record,
                "connection lost",
                failure_kind=CommandFailureKind.TRANSPORT,
            )
        ),
        tmp_path,
    )

    assert result.status == ProbeStatus.TRANSPORT_FAILED
    assert result.tp_size == 4
    assert result.concurrency == 8
    assert result.num_prompts == 32


def test_successful_bench_that_writes_no_file_cannot_reuse_stale_same_c_json(
    tmp_path,
) -> None:
    result = _evaluate_once(
        _ProbeContainer(
            bench_writes_result=False,
            stale_record=json.dumps(_bench_record()),
        ),
        tmp_path,
    )

    assert result.status == ProbeStatus.BENCHMARK_FAILED
    assert "result read failed" in (result.failure_reason or "")


def test_server_death_after_bench_is_runtime_failure_not_sla_failure(tmp_path) -> None:
    result = _evaluate_once(
        _ProbeContainer(bench_returncode=1, healthy_after=False),
        tmp_path,
    )

    assert result.status == ProbeStatus.RUNTIME_FAILED
    assert result.tp_size == 4
    assert result.concurrency == 8
    assert result.num_prompts == 32
    assert "server unhealthy" in (result.failure_reason or "")


def test_known_0516_triton_eagle_shape_trace_is_classified_exactly() -> None:
    assert classify_known_runtime_issue(
        engine_version="0.5.16",
        attention_backend="triton",
        speculative_algorithm="EAGLE",
        traceback_text=_VALID_KNOWN_TRACE,
    ) == "sglang-0.5.16-triton-eagle-custom-mask-shape"


def test_known_issue_requires_the_triton_file_and_function_in_one_frame() -> None:
    split_trace = """
Traceback (most recent call last):
  File "/sglang/triton_backend.py", line 1, in unrelated
    unrelated()
RuntimeError: elsewhere in _update_target_verify_buffers: custom_mask expanded size existing size
"""

    assert classify_known_runtime_issue(
        engine_version="0.5.16",
        attention_backend="triton",
        speculative_algorithm="EAGLE",
        traceback_text=split_trace,
    ) is None


@pytest.mark.parametrize(
    ("version", "attention", "algorithm", "traceback_text"),
    [
        (
            "0.5.17",
            "triton",
            "EAGLE",
            _VALID_KNOWN_TRACE,
        ),
        (
            "0.5.16",
            "flashinfer",
            "EAGLE",
            _VALID_KNOWN_TRACE,
        ),
        (
            "0.5.16",
            "triton",
            "NONE",
            _VALID_KNOWN_TRACE,
        ),
        ("0.5.16", "triton", "EAGLE", "RuntimeError: unrelated crash"),
    ],
)
def test_known_issue_requires_all_four_exact_factors(
    version: str,
    attention: str,
    algorithm: str,
    traceback_text: str,
) -> None:
    assert classify_known_runtime_issue(
        engine_version=version,
        attention_backend=attention,
        speculative_algorithm=algorithm,
        traceback_text=traceback_text,
    ) is None


@pytest.mark.parametrize(
    "version",
    ["0.5.16rc1", "0.5.16.post1", "0.5.16.dev1", "0.5.16.0", "x0.5.16"],
)
def test_known_issue_rejects_non_exact_0516_release_versions(version: str) -> None:
    assert classify_known_runtime_issue(
        engine_version=version,
        attention_backend="triton",
        speculative_algorithm="EAGLE",
        traceback_text=_VALID_KNOWN_TRACE,
    ) is None


def test_known_issue_accepts_exact_0516_local_build_suffix() -> None:
    assert classify_known_runtime_issue(
        engine_version="0.5.16+cu129",
        attention_backend="triton",
        speculative_algorithm="EAGLE",
        traceback_text=_VALID_KNOWN_TRACE,
    ) == "sglang-0.5.16-triton-eagle-custom-mask-shape"


@pytest.mark.parametrize(
    "missing_fragment",
    ["custom_mask", "expanded size", "existing size"],
)
def test_known_issue_requires_each_shape_diagnostic_fragment(
    missing_fragment: str,
) -> None:
    trace = _VALID_KNOWN_TRACE.lower().replace(missing_fragment, "removed-fragment")

    assert classify_known_runtime_issue(
        engine_version="0.5.16",
        attention_backend="triton",
        speculative_algorithm="EAGLE",
        traceback_text=trace,
    ) is None


def test_known_issue_does_not_join_unrelated_log_lines_into_a_shape_error() -> None:
    trace = """
Traceback (most recent call last):
  File "/sglang/triton_backend.py", line 99, in _update_target_verify_buffers
    verify_buffers()
ValueError: unrelated failure
INFO custom_mask
INFO expanded size
INFO existing size
"""

    assert classify_known_runtime_issue(
        engine_version="0.5.16",
        attention_backend="triton",
        speculative_algorithm="EAGLE",
        traceback_text=trace,
    ) is None


def test_known_issue_ignores_custom_mask_logged_after_the_exception() -> None:
    trace = """
Traceback (most recent call last):
  File "/sglang/triton_backend.py", line 99, in _update_target_verify_buffers
    verify_buffers()
RuntimeError: expanded size must match the existing size
INFO custom_mask
"""

    assert classify_known_runtime_issue(
        engine_version="0.5.16",
        attention_backend="triton",
        speculative_algorithm="EAGLE",
        traceback_text=trace,
    ) is None


def test_runtime_failure_path_attaches_exact_known_issue_id(tmp_path) -> None:
    traceback_text = """
Traceback (most recent call last):
  File "/sglang/triton_backend.py", line 99, in _update_target_verify_buffers
    custom_mask = custom_mask.expand(shape)
RuntimeError: expanded size must match the existing size
"""
    result = _evaluate_once(
        _ProbeContainer(
            bench_returncode=1,
            healthy_after=False,
            server_log=traceback_text,
        ),
        tmp_path,
        candidate={
            "params": {
                "attention_backend": "triton",
                "speculative_algorithm": "EAGLE",
            }
        },
    )

    assert result.status == ProbeStatus.RUNTIME_FAILED
    assert result.known_issue == "sglang-0.5.16-triton-eagle-custom-mask-shape"


def test_remote_runner_distinguishes_ssh_255_from_local_child_255() -> None:
    def exit_255(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 255, stdout="", stderr="exit 255")

    runner = RemoteRunner("user@host", runner=exit_255)

    assert runner.run("true").failure_kind == CommandFailureKind.TRANSPORT
    assert runner.run_local(["local-child"]).failure_kind is None


def test_remote_runner_types_timeout_and_local_oserror() -> None:
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output=b"partial")

    timed_out = RemoteRunner("user@host", runner=timeout).run("slow", timeout=3)
    assert timed_out.failure_kind == CommandFailureKind.TIMEOUT
    assert timed_out.returncode == 124

    def missing_binary(argv, **kwargs):
        raise OSError("synthetic missing executable")

    transport = RemoteRunner("user@host", runner=missing_binary).run_local(["missing"])
    assert transport.failure_kind == CommandFailureKind.TRANSPORT
    assert transport.returncode == 127
