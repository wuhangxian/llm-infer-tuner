from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import runners.reporting as reporting
from runners.reporting import REPORT_SCHEMA_VERSION, write_reports


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _probe_base(
    probe_id: str, *, round_number: int, repeat: int, concurrency: int = 4
) -> dict[str, Any]:
    return {
        "probe_id": probe_id,
        "candidate_id": "a",
        "round": round_number,
        "batch": "1/1",
        "concurrency": concurrency,
        "repeat": repeat,
        "recovery": 0,
        "measurement_mode": "estimated",
        "started_at": "2026-09-01T00:00:00+00:00",
        "ended_at": "2026-09-01T00:00:01+00:00",
        "failed_at": None,
        "status": "ok",
        "failure_reason": None,
        "known_issue": None,
        "raw": {"request_rate": 4},
        "normalized": {
            "total_throughput": 100.0,
            "mean_ttft_ms": 10.0,
            "mean_tpot_ms": 2.0,
            "success_rate": 1.0,
        },
        "instances": 1,
        "output_healthy": True,
        "server_health": {"before": "healthy", "after": "healthy"},
        "artifacts": [
            {
                "path": f"artifacts/{probe_id}.json",
                "sha256": "a" * 64,
            }
        ],
    }


def _valid_final_payloads() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    probes: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for round_number, concurrency, repeats, qualifies in (
        (1, 4, range(1), True),
        (2, 4, range(3), True),
        (2, 5, range(3), False),
    ):
        aggregate_ids: list[str] = []
        for repeat in repeats:
            physical_id = f"r{round_number}-c{concurrency}-rep{repeat}-physical"
            aggregate_id = f"r{round_number}-c{concurrency}-rep{repeat}-aggregate"
            physical = {
                **_probe_base(
                    physical_id,
                    round_number=round_number,
                    repeat=repeat,
                    concurrency=concurrency,
                ),
                "record_type": "physical_probe",
                "replica_index": 0,
                "port": 30000,
                "statistical_vote": False,
            }
            aggregate = {
                **_probe_base(
                    aggregate_id,
                    round_number=round_number,
                    repeat=repeat,
                    concurrency=concurrency,
                ),
                "record_type": "aggregate_sample",
                "replica_probe_ids": [physical_id],
                "statistical_vote": True,
            }
            if not qualifies:
                aggregate["status"] = "sla_failed"
                aggregate["failure_reason"] = "SLA failed"
            probes.extend((physical, aggregate))
            aggregate_ids.append(aggregate_id)
        groups.append(
            {
                "group_id": f"r{round_number}-c{concurrency}",
                "round": round_number,
                "concurrency": concurrency,
                "aggregate_probe_ids": aggregate_ids,
                "representative": {
                    "status": "ok" if qualifies else "sla_failed",
                    "total_throughput": 100.0,
                    "mean_ttft_ms": 10.0,
                    "mean_tpot_ms": 2.0,
                    "success_rate": 1.0,
                },
                "qualifies": qualifies,
            }
        )
    candidate = {
        "candidate_id": "a",
        "status": "completed",
        "completion_state": "completed",
        "measurement_mode": "estimated",
        "actual_instances": 1,
        "measurement_valid": True,
        "ranking_eligible": True,
        "ranking_eligibility_reason": None,
        "rank": 1,
        "rank_group": 1,
        "probe_ids": [str(probe["probe_id"]) for probe in probes],
        "sample_groups": groups,
        "incomplete_groups": [],
        "round1": {
            "stop_reason": "max_probes",
            "num_evals": 1,
            "newly_probed": [4],
            "complete": False,
            "certainty": "lower_bound",
        },
        "round2": {
            "stop_reason": "found_boundary",
            "num_evals": 6,
            "newly_probed": [4, 5],
            "c_star": 4,
            "last_pass": 4,
            "first_fail": 5,
            "complete": True,
            "certainty": "exact",
        },
        "failures": [],
        "recovery_count": 0,
        "final_failure": None,
    }
    task_status = {
        "task_status": "COMPLETED",
        "ranking_status": "FINAL",
        "expected_candidate_ids": ["a"],
        "interrupted": False,
        "cleanup_failures": [],
    }
    provenance = {
        "git": {"sha": "a" * 40, "dirty": False},
        "inputs": {"job_sha256": "b" * 64, "config_sha256": "c" * 64},
        "image": {
            "reference": "example/image:tag",
            "digest": None,
            "unavailable_reason": "image inspection not available in dry run",
        },
        "actual_gpu": {
            "count": None,
            "model": None,
            "memory_bytes": None,
            "topology": None,
            "unavailable_reason": "remote hardware not inspected in dry run",
        },
        "engine": {
            "version": None,
            "unavailable_reason": "server did not publish a version",
        },
        "run_started_at": "2026-09-01T00:00:00+00:00",
        "run_ended_at": "2026-09-01T00:01:00+00:00",
    }
    ranking = [
        {
            "candidate_id": "a",
            "rank": 1,
            "rank_group": 1,
            "ranking_eligible": True,
            "measurement_mode": "estimated",
            "actual_instances": 1,
            "instances_per_host": 1.0,
            "best_concurrency": 4,
            "sample_count": 3,
            "goodput_raw": 100.0,
            "goodput_per_host_min": 100.0,
            "goodput_per_host_median": 100.0,
            "goodput_per_host_max": 100.0,
            "goodput_per_host": 100.0,
            "baseline_threshold_status": "unknown",
            "beats_baseline_threshold": False,
        }
    ]
    return ranking, [candidate], probes, task_status, provenance


def test_v2_report_set_shares_run_identity_across_every_payload(tmp_path: Path) -> None:
    run_id = "run-20260901-fixed"

    report = write_reports(
        tmp_path,
        ranking=[{"candidate_id": "c001", "rank": 1, "ranking_eligible": True}],
        candidate_rows=[{"candidate_id": "c001", "status": "completed"}],
        probe_rows=[{"probe_id": "probe-001", "candidate_id": "c001"}],
        task_status={"task_status": "COMPLETED", "ranking_status": "FINAL"},
        provenance={"git_sha": "a" * 40},
        run_id=run_id,
    )

    ranking = json.loads((tmp_path / "ranking.json").read_text(encoding="utf-8"))
    candidates = _read_jsonl(tmp_path / "candidate_results.jsonl")
    probes = _read_jsonl(tmp_path / "probe_results.jsonl")
    task_status = json.loads(
        (tmp_path / "task_status.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (tmp_path / "provenance.json").read_text(encoding="utf-8")
    )

    assert REPORT_SCHEMA_VERSION == 2
    payloads = [*ranking, *candidates, *probes, task_status, provenance]
    assert payloads
    assert {payload["run_id"] for payload in payloads} == {run_id}
    assert {payload["report_schema_version"] for payload in payloads} == {2}
    assert report == reporting.load_report_generation(tmp_path)


def test_failed_multifile_write_does_not_publish_a_mixed_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_reports(
        tmp_path,
        ranking=[{"candidate_id": "old", "rank": 1, "ranking_eligible": True}],
        candidate_rows=[{"candidate_id": "old", "status": "completed"}],
        probe_rows=[],
        task_status={"task_status": "COMPLETED", "ranking_status": "FINAL"},
        provenance={"git_sha": "a" * 40},
        run_id="run-old",
    )
    old_manifest = (tmp_path / "report_manifest.json").read_bytes()
    real_atomic_write = reporting._atomic_write

    def fail_on_candidates(path: Path, text: str) -> None:
        if path.name == "candidate_results.jsonl":
            raise OSError("synthetic interrupted report write")
        real_atomic_write(path, text)

    monkeypatch.setattr(reporting, "_atomic_write", fail_on_candidates)
    with pytest.raises(OSError, match="synthetic interrupted"):
        write_reports(
            tmp_path,
            ranking=[{"candidate_id": "new", "rank": 1, "ranking_eligible": True}],
            candidate_rows=[{"candidate_id": "new", "status": "completed"}],
            probe_rows=[],
            task_status={"task_status": "COMPLETED", "ranking_status": "FINAL"},
            provenance={"git_sha": "b" * 40},
            run_id="run-new",
        )

    assert (tmp_path / "report_manifest.json").read_bytes() == old_manifest
    authoritative = reporting.load_report_generation(tmp_path)
    assert authoritative["manifest"]["run_id"] == "run-old"
    assert authoritative["ranking"][0]["candidate_id"] == "old"


@pytest.mark.parametrize("run_id", ["../escape", "/tmp/escape", "bad\nrun"])
def test_writer_rejects_unsafe_run_id_before_creating_a_generation(
    tmp_path: Path, run_id: str
) -> None:
    with pytest.raises(ValueError, match="run_id"):
        write_reports(
            tmp_path,
            ranking=[],
            candidate_rows=[],
            probe_rows=[],
            task_status={"task_status": "INCOMPLETE", "ranking_status": "PROVISIONAL"},
            provenance={},
            run_id=run_id,
        )

    assert not (tmp_path / ".report_generations").exists()


def test_manifest_failure_cannot_promote_loose_final_task_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_reports(
        tmp_path,
        ranking=[],
        candidate_rows=[{"candidate_id": "c001", "status": "incomplete"}],
        probe_rows=[],
        task_status={"task_status": "INCOMPLETE", "ranking_status": "PROVISIONAL"},
        provenance={},
        run_id="same-run",
    )
    old_manifest = (tmp_path / "report_manifest.json").read_bytes()
    real_atomic_write = reporting._atomic_write

    def fail_root_manifest(path: Path, text: str) -> None:
        if path == tmp_path / "report_manifest.json":
            raise OSError("synthetic manifest commit failure")
        real_atomic_write(path, text)

    monkeypatch.setattr(reporting, "_atomic_write", fail_root_manifest)
    with pytest.raises(OSError, match="manifest commit"):
        write_reports(
            tmp_path,
            ranking=[],
            candidate_rows=[{"candidate_id": "c001", "status": "completed"}],
            probe_rows=[],
            task_status={"task_status": "COMPLETED", "ranking_status": "FINAL"},
            provenance={},
            run_id="same-run",
        )

    assert (tmp_path / "report_manifest.json").read_bytes() == old_manifest
    loose_status = json.loads(
        (tmp_path / "task_status.json").read_text(encoding="utf-8")
    )
    assert loose_status["ranking_status"] == "PROVISIONAL"
    authoritative = reporting.load_report_generation(tmp_path)
    assert authoritative["task_status"]["ranking_status"] == "PROVISIONAL"


@pytest.mark.parametrize(
    "candidate_ids",
    [
        ["a", "a", "b"],
        ["a"],
        ["a", "b", "extra"],
    ],
)
def test_candidate_rows_are_canonicalized_to_the_exact_expected_set(
    tmp_path: Path, candidate_ids: list[str]
) -> None:
    report = write_reports(
        tmp_path,
        ranking=[],
        candidate_rows=[
            {"candidate_id": candidate_id, "status": "completed"}
            for candidate_id in candidate_ids
        ],
        probe_rows=[],
        task_status={
            "task_status": "COMPLETED",
            "ranking_status": "FINAL",
            "expected_candidate_ids": ["a", "b"],
        },
        provenance={},
        run_id="candidate-set-run",
    )

    assert report == reporting.load_report_generation(tmp_path)
    assert [row["candidate_id"] for row in report["candidate_rows"]] == ["a", "b"]
    assert report["task_status"]["expected_candidate_ids"] == ["a", "b"]
    assert report["task_status"]["expected_candidate_count"] == 2
    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert report["task_status"]["task_status"] == "INCOMPLETE"
    assert report["task_status"]["invariant_errors"]


def test_duplicate_and_foreign_probe_and_ranking_references_fail_closed(
    tmp_path: Path,
) -> None:
    report = write_reports(
        tmp_path,
        ranking=[
            {"candidate_id": "a", "rank": 1, "ranking_eligible": True},
            {"candidate_id": "a", "rank": 2, "ranking_eligible": True},
            {"candidate_id": "foreign", "rank": 3, "ranking_eligible": True},
        ],
        candidate_rows=[
            {
                "candidate_id": "a",
                "status": "completed",
                "probe_ids": ["p-good", "p-missing"],
            }
        ],
        probe_rows=[
            {"probe_id": "p-good", "candidate_id": "a"},
            {"probe_id": "p-good", "candidate_id": "a"},
            {"probe_id": "p-foreign", "candidate_id": "foreign"},
        ],
        task_status={
            "task_status": "COMPLETED",
            "ranking_status": "FINAL",
            "expected_candidate_ids": ["a"],
        },
        provenance={},
        run_id="foreign-key-run",
    )

    assert report == reporting.load_report_generation(tmp_path)
    assert [row["candidate_id"] for row in report["ranking"]] == ["a"]
    assert [row["probe_id"] for row in report["probe_rows"]] == ["p-good"]
    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert any(
        "probe" in error or "ranking" in error
        for error in report["task_status"]["invariant_errors"]
    )


def test_final_is_recomputed_from_round_and_linked_evidence_not_trusted(
    tmp_path: Path,
) -> None:
    write_reports(
        tmp_path,
        ranking=[],
        candidate_rows=[{"candidate_id": "a", "status": "completed"}],
        probe_rows=[{"probe_id": "orphan", "candidate_id": "a"}],
        task_status={
            "task_status": "COMPLETED",
            "ranking_status": "FINAL",
            "expected_candidate_ids": ["a"],
        },
        provenance={},
        run_id="false-final-run",
    )

    status = reporting.load_report_generation(tmp_path)["task_status"]
    assert status["task_status"] == "INCOMPLETE"
    assert status["ranking_status"] == "PROVISIONAL"
    assert any("round 1" in error for error in status["invariant_errors"])
    assert any("round 2" in error for error in status["invariant_errors"])
    assert any("does not link probes" in error for error in status["invariant_errors"])


def test_malformed_foreign_keys_become_invariant_errors_instead_of_crashing(
    tmp_path: Path,
) -> None:
    write_reports(
        tmp_path,
        ranking=[{"candidate_id": {}, "rank": 1}],
        candidate_rows=[
            {
                "candidate_id": "a",
                "status": "incomplete",
                "probe_ids": [{}],
            }
        ],
        probe_rows=[{"probe_id": "p", "candidate_id": {}}],
        task_status={
            "task_status": "INCOMPLETE",
            "ranking_status": "PROVISIONAL",
            "expected_candidate_ids": ["a"],
            "invariant_errors": "not-a-list",
        },
        provenance={},
        run_id="malformed-fk-run",
    )

    report = reporting.load_report_generation(tmp_path)
    assert report["ranking"] == []
    assert report["probe_rows"] == []
    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert "invariant_errors must be a list" in report["task_status"][
        "invariant_errors"
    ]


def test_complete_hierarchical_evidence_can_publish_final(tmp_path: Path) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="valid-final",
    )

    assert report == reporting.load_report_generation(tmp_path)
    assert report["task_status"]["task_status"] == "COMPLETED"
    assert report["task_status"]["ranking_status"] == "FINAL"
    assert report["task_status"]["invariant_errors"] == []


def test_manifest_rejects_boolean_counts_schema_and_duplicate_keys(
    tmp_path: Path,
) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="strict-manifest",
    )
    manifest_path = tmp_path / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["ranking.json"]["row_count"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        reporting.load_report_generation(tmp_path)

    manifest["files"]["ranking.json"]["row_count"] = 0
    manifest["report_schema_version"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        reporting.load_report_generation(tmp_path)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        reporting._strict_json_loads('{"run_id":"a","run_id":"b"}')


def test_final_requires_complete_provenance(tmp_path: Path) -> None:
    ranking, candidates, probes, task_status, _ = _valid_final_payloads()

    write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance={},
        run_id="missing-provenance",
    )

    status = reporting.load_report_generation(tmp_path)["task_status"]
    assert status["ranking_status"] == "PROVISIONAL"
    assert any("provenance" in error for error in status["invariant_errors"])


def test_physical_only_probe_cannot_certify_round2_exact(tmp_path: Path) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    candidates[0]["probe_ids"] = [
        probe["probe_id"] for probe in probes if probe["record_type"] == "physical_probe"
    ]
    candidates[0]["sample_groups"] = []
    probes = [probe for probe in probes if probe["record_type"] == "physical_probe"]

    write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="physical-only",
    )

    status = reporting.load_report_generation(tmp_path)["task_status"]
    assert status["ranking_status"] == "PROVISIONAL"
    assert any("sample group" in error for error in status["invariant_errors"])


def test_duplicate_group_and_unclosed_terminal_attempt_block_final(tmp_path: Path) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    candidates[0]["sample_groups"].append(
        deepcopy(candidates[0]["sample_groups"][1])
    )
    failed_physical = {
        **_probe_base("terminal-physical", round_number=2, repeat=3),
        "record_type": "physical_probe",
        "recovery": 0,
        "status": "transport_failed",
        "failed_at": "2026-09-01T00:00:02+00:00",
        "failure_reason": "ssh lost",
        "replica_index": 0,
        "port": 30000,
        "statistical_vote": False,
    }
    failed_aggregate = {
        **_probe_base("terminal-aggregate", round_number=2, repeat=3),
        "record_type": "aggregate_sample",
        "recovery": 0,
        "status": "transport_failed",
        "failed_at": "2026-09-01T00:00:02+00:00",
        "failure_reason": "ssh lost",
        "replica_probe_ids": ["terminal-physical"],
        "statistical_vote": False,
    }
    probes.extend((failed_physical, failed_aggregate))
    candidates[0]["probe_ids"].extend(["terminal-physical", "terminal-aggregate"])

    write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="bad-hierarchy",
    )

    status = reporting.load_report_generation(tmp_path)["task_status"]
    assert status["ranking_status"] == "PROVISIONAL"
    assert any("duplicate" in error for error in status["invariant_errors"])
    assert any("terminal" in error for error in status["invariant_errors"])


def test_ranking_eligible_candidate_requires_one_matching_rank_row(tmp_path: Path) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    ranking = []

    write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="missing-rank",
    )

    status = reporting.load_report_generation(tmp_path)["task_status"]
    assert status["ranking_status"] == "PROVISIONAL"
    assert any("ranking" in error for error in status["invariant_errors"])


def test_optional_raw_nonfinite_values_are_sanitized_but_metrics_remain_strict(
    tmp_path: Path,
) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    probes[0]["raw"] = {"nested": [1, math.inf, {"rate": math.nan}]}

    write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="sanitized-raw",
    )

    text = (tmp_path / "probe_results.jsonl").read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    report = reporting.load_report_generation(tmp_path)
    first = report["probe_rows"][0]
    assert first["raw"]["nested"] == [1, None, {"rate": None}]
    assert len(first["raw_sanitization"]) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "unhashable_probe_link",
        "failures_none",
        "aggregate_ids_int",
        "unhashable_newly_probed",
        "unhashable_status",
        "huge_metric",
        "record_type_object",
        "round_object",
        "probe_mode_object",
        "repeat_object",
        "candidate_mode_object",
        "round1_stop_object",
        "task_status_list",
        "ranking_status_list",
    ],
)
def test_untrusted_report_shapes_fail_closed_without_crashing(
    tmp_path: Path, mutation: str
) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    if mutation == "unhashable_probe_link":
        candidates[0]["probe_ids"] = [{}]
    elif mutation == "failures_none":
        candidates[0]["failures"] = None
    elif mutation == "aggregate_ids_int":
        candidates[0]["sample_groups"][0]["aggregate_probe_ids"] = 1
    elif mutation == "unhashable_newly_probed":
        candidates[0]["round2"]["newly_probed"] = [{}]
    elif mutation == "unhashable_status":
        probes[0]["status"] = {}
    elif mutation == "huge_metric":
        probes[0]["normalized"]["total_throughput"] = 10**10000
    elif mutation == "record_type_object":
        probes[0]["record_type"] = {}
    elif mutation == "round_object":
        probes[0]["round"] = {}
    elif mutation == "probe_mode_object":
        probes[0]["measurement_mode"] = {}
    elif mutation == "repeat_object":
        probes[1]["repeat"] = {}
    elif mutation == "candidate_mode_object":
        candidates[0]["measurement_mode"] = {}
    elif mutation == "round1_stop_object":
        candidates[0]["round1"]["stop_reason"] = {}
    elif mutation == "task_status_list":
        task_status["task_status"] = []
    elif mutation == "ranking_status_list":
        task_status["ranking_status"] = []

    write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id=f"malformed-{mutation.replace('_', '-')}",
    )

    status = reporting.load_report_generation(tmp_path)["task_status"]
    assert status["task_status"] == "INCOMPLETE"
    assert status["ranking_status"] == "PROVISIONAL"
    assert status["invariant_errors"]


@pytest.mark.parametrize(
    "mutation",
    [
        "replica_index",
        "port",
        "child_mode",
        "output_health",
        "server_health",
        "artifact",
        "failure_time",
        "representative",
        "qualifies",
        "round_count",
        "actual_instances",
        "image_reference",
    ],
)
def test_final_rejects_single_field_evidence_fabrication(
    tmp_path: Path, mutation: str
) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    if mutation == "replica_index":
        probes[0]["replica_index"] = -1
    elif mutation == "port":
        probes[0]["port"] = 70000
    elif mutation == "child_mode":
        probes[0]["measurement_mode"] = "full_host"
    elif mutation == "output_health":
        probes[0]["output_healthy"] = False
    elif mutation == "server_health":
        probes[1]["server_health"]["after"] = "dead"
    elif mutation == "artifact":
        probes[0]["artifacts"] = []
    elif mutation == "failure_time":
        probes[0]["status"] = "transport_failed"
        probes[0]["failure_reason"] = "ssh lost"
    elif mutation == "representative":
        candidates[0]["sample_groups"][0]["representative"][
            "total_throughput"
        ] = 999999.0
    elif mutation == "qualifies":
        candidates[0]["sample_groups"][0]["qualifies"] = False
    elif mutation == "round_count":
        candidates[0]["round2"]["num_evals"] = 5
    elif mutation == "actual_instances":
        candidates[0]["actual_instances"] = 2
    elif mutation == "image_reference":
        provenance["image"]["reference"] = ""

    write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id=f"fabricated-{mutation.replace('_', '-')}",
    )

    status = reporting.load_report_generation(tmp_path)["task_status"]
    assert status["ranking_status"] == "PROVISIONAL"
    assert status["invariant_errors"]


def test_final_rejects_aggregate_ok_when_physical_child_failed(
    tmp_path: Path,
) -> None:
    """An aggregate verdict cannot certify a failed physical replica."""

    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    for probe in probes:
        if probe["probe_id"] == "r2-c4-rep0-physical":
            probe["status"] = "sla_failed"
            probe["failure_reason"] = "physical replica missed SLA"
            probe["failed_at"] = None

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="aggregate-physical-status-mismatch",
    )

    assert report["task_status"]["task_status"] == "INCOMPLETE"
    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert any(
        "aggregate probe" in error and "status" in error
        for error in report["task_status"]["invariant_errors"]
    )


def test_unrecovered_warmup_failure_blocks_final(tmp_path: Path) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    physical = {
        **_probe_base("warmup-physical", round_number=2, repeat=-1),
        "record_type": "physical_probe",
        "status": "transport_failed",
        "failed_at": "2026-09-01T00:00:01+00:00",
        "failure_reason": "ssh lost",
        "replica_index": 0,
        "port": 30000,
        "statistical_vote": False,
    }
    aggregate = {
        **_probe_base("warmup-aggregate", round_number=2, repeat=-1),
        "record_type": "aggregate_sample",
        "status": "transport_failed",
        "failed_at": "2026-09-01T00:00:01+00:00",
        "failure_reason": "ssh lost",
        "replica_probe_ids": ["warmup-physical"],
        "statistical_vote": False,
    }
    probes.extend((physical, aggregate))
    candidates[0]["probe_ids"].extend(["warmup-physical", "warmup-aggregate"])

    write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="terminal-warmup",
    )

    status = reporting.load_report_generation(tmp_path)["task_status"]
    assert status["ranking_status"] == "PROVISIONAL"
    assert any("terminal" in error for error in status["invariant_errors"])


def _append_failed_recovery_attempt(
    candidates: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    *,
    suffix: str,
    recovery: int,
) -> None:
    physical_id = f"failed-{suffix}-physical"
    aggregate_id = f"failed-{suffix}-aggregate"
    physical = {
        **_probe_base(physical_id, round_number=2, repeat=0),
        "record_type": "physical_probe",
        "recovery": recovery,
        "status": "transport_failed",
        "failed_at": "2026-09-01T00:00:01+00:00",
        "failure_reason": "ssh lost",
        "replica_index": 0,
        "port": 30000,
        "statistical_vote": False,
    }
    aggregate = {
        **_probe_base(aggregate_id, round_number=2, repeat=0),
        "record_type": "aggregate_sample",
        "recovery": recovery,
        "status": "transport_failed",
        "failed_at": "2026-09-01T00:00:01+00:00",
        "failure_reason": "ssh lost",
        "replica_probe_ids": [physical_id],
        "statistical_vote": False,
    }
    probes.extend((physical, aggregate))
    candidates[0]["probe_ids"].extend((physical_id, aggregate_id))


def test_recovery_chain_must_be_contiguous(tmp_path: Path) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    for probe in probes:
        if probe["round"] == 2 and probe["concurrency"] == 4 and probe["repeat"] == 0:
            probe["recovery"] = 2
    _append_failed_recovery_attempt(candidates, probes, suffix="r0", recovery=0)

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="recovery-gap",
    )

    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert any("recovery" in error for error in report["task_status"]["invariant_errors"])


def test_duplicate_logical_recovery_attempt_is_rejected(tmp_path: Path) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    _append_failed_recovery_attempt(candidates, probes, suffix="dup-a", recovery=0)
    _append_failed_recovery_attempt(candidates, probes, suffix="dup-b", recovery=0)

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="duplicate-recovery",
    )

    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert any("duplicate" in error for error in report["task_status"]["invariant_errors"])


def test_recovered_attempt_must_be_linked_from_failure_summary(tmp_path: Path) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    for probe in probes:
        if probe["round"] == 2 and probe["concurrency"] == 4 and probe["repeat"] == 0:
            probe["recovery"] = 1
    _append_failed_recovery_attempt(candidates, probes, suffix="unlinked", recovery=0)

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="unlinked-recovery",
    )

    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert any("failure summary" in error for error in report["task_status"]["invariant_errors"])


def test_contiguous_recovered_attempt_with_summary_can_remain_final(tmp_path: Path) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    for probe in probes:
        if probe["round"] == 2 and probe["concurrency"] == 4 and probe["repeat"] == 0:
            probe["recovery"] = 1
    _append_failed_recovery_attempt(candidates, probes, suffix="linked", recovery=0)
    candidates[0]["failures"] = [
        {"probe_id": "failed-linked-aggregate", "resolved": True}
    ]
    candidates[0]["recovery_count"] = 1

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="linked-recovery",
    )

    assert report["task_status"]["ranking_status"] == "FINAL"


@pytest.mark.parametrize("shape", ["duplicate", "gap"])
def test_infrastructure_recovery_identity_is_contiguous_and_unique(
    tmp_path: Path, shape: str
) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()

    def infrastructure(probe_id: str, recovery: int, status: str) -> dict[str, Any]:
        row = {
            **_probe_base(probe_id, round_number=2, repeat=0, concurrency=0),
            "record_type": "infrastructure_attempt",
            "recovery": recovery,
            "status": status,
            "statistical_vote": False,
        }
        if status != "ok":
            row["failed_at"] = "2026-09-01T00:00:01+00:00"
            row["failure_reason"] = "server startup failed"
        return row

    added_ids = ["infra-fail-a"]
    probes.append(infrastructure("infra-fail-a", 0, "runtime_failed"))
    if shape == "duplicate":
        added_ids.extend(("infra-fail-b", "infra-ok"))
        probes.append(infrastructure("infra-fail-b", 0, "runtime_failed"))
        probes.append(infrastructure("infra-ok", 1, "ok"))
        failure_ids = ["infra-fail-a", "infra-fail-b"]
    else:
        added_ids.append("infra-ok")
        probes.append(infrastructure("infra-ok", 2, "ok"))
        failure_ids = ["infra-fail-a"]
    candidates[0]["probe_ids"].extend(added_ids)
    candidates[0]["failures"] = [
        {"probe_id": probe_id, "resolved": True} for probe_id in failure_ids
    ]
    candidates[0]["recovery_count"] = len(failure_ids)

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id=f"infra-{shape}",
    )

    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert any("recovery" in error for error in report["task_status"]["invariant_errors"])


def test_recovery_closure_matches_batch_and_measurement_mode(tmp_path: Path) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    failed_physical = {
        **_probe_base("batch-a-failed-physical", round_number=2, repeat=0, concurrency=9),
        "record_type": "physical_probe",
        "batch": "A",
        "status": "transport_failed",
        "failed_at": "2026-09-01T00:00:01+00:00",
        "failure_reason": "batch A lost",
        "replica_index": 0,
        "port": 30000,
        "statistical_vote": False,
    }
    failed_aggregate = {
        **_probe_base("batch-a-failed-aggregate", round_number=2, repeat=0, concurrency=9),
        "record_type": "aggregate_sample",
        "batch": "A",
        "status": "transport_failed",
        "failed_at": "2026-09-01T00:00:01+00:00",
        "failure_reason": "batch A lost",
        "replica_probe_ids": ["batch-a-failed-physical"],
        "statistical_vote": False,
    }
    probes.extend((failed_physical, failed_aggregate))
    aggregate_ids: list[str] = []
    for repeat in range(3):
        physical_id = f"batch-b-ok-physical-{repeat}"
        aggregate_id = f"batch-b-ok-aggregate-{repeat}"
        probes.append(
            {
                **_probe_base(physical_id, round_number=2, repeat=repeat, concurrency=9),
                "record_type": "physical_probe",
                "batch": "B",
                "recovery": 1,
                "replica_index": 0,
                "port": 30000,
                "statistical_vote": False,
            }
        )
        probes.append(
            {
                **_probe_base(aggregate_id, round_number=2, repeat=repeat, concurrency=9),
                "record_type": "aggregate_sample",
                "batch": "B",
                "recovery": 1,
                "replica_probe_ids": [physical_id],
                "statistical_vote": True,
            }
        )
        aggregate_ids.append(aggregate_id)
    candidates[0]["sample_groups"].append(
        {
            "group_id": "r2-c9",
            "round": 2,
            "concurrency": 9,
            "aggregate_probe_ids": aggregate_ids,
            "representative": {
                "status": "ok",
                "total_throughput": 100.0,
                "mean_ttft_ms": 10.0,
                "mean_tpot_ms": 2.0,
                "success_rate": 1.0,
            },
            "qualifies": True,
        }
    )
    candidates[0]["probe_ids"].extend(
        ["batch-a-failed-physical", "batch-a-failed-aggregate", *aggregate_ids]
    )
    candidates[0]["failures"] = [
        {"probe_id": "batch-a-failed-aggregate", "resolved": True}
    ]
    candidates[0]["recovery_count"] = 1
    candidates[0]["round2"]["newly_probed"] = [4, 5, 9]
    candidates[0]["round2"]["num_evals"] = 9

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="batch-recovery-closure",
    )

    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert any("terminal" in error for error in report["task_status"]["invariant_errors"])


@pytest.mark.parametrize("value", [True, 1, -1, 1.5, float("nan"), float("inf")])
def test_truthy_scalar_sample_groups_fail_closed(tmp_path: Path, value: object) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    candidates[0]["sample_groups"] = value

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id=f"scalar-groups-{type(value).__name__}",
    )

    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert report["task_status"]["invariant_errors"]


@pytest.mark.parametrize("value", [None, True, 0, 1.5, "bad", []])
def test_aggregate_normalized_scalar_fails_closed(tmp_path: Path, value: object) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    probes[1]["normalized"] = value

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id=f"scalar-normalized-{type(value).__name__}",
    )

    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert report["task_status"]["invariant_errors"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("best_concurrency", 999),
        ("measurement_mode", "full_host"),
        ("actual_instances", 99),
        ("sample_count", 999),
        ("goodput_per_host_min", 1.0e100),
        ("goodput_per_host_median", 1.0e100),
        ("goodput_per_host_max", 1.0e100),
        ("baseline_threshold_status", "yes"),
        ("beats_baseline_threshold", True),
        ("rank_group", 999),
        ("request_throughput", 999.0),
        ("mean_ttft_ms", 999.0),
    ],
)
def test_ranking_row_must_match_authoritative_r2_evidence(
    tmp_path: Path, field: str, value: object
) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    ranking[0][field] = value

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id=f"bad-ranking-{field.replace('_', '-')}",
    )

    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert any("ranking" in error for error in report["task_status"]["invariant_errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [("best_concurrency", 999), ("goodput_raw", 999.0), ("rank_group", 999)],
)
def test_candidate_summary_fields_must_match_ranking_evidence(
    tmp_path: Path, field: str, value: object
) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    candidates[0][field] = value

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id=f"bad-candidate-summary-{field}",
    )

    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert report["task_status"]["invariant_errors"]


def test_threshold_yes_without_authoritative_baseline_is_unknown(tmp_path: Path) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    ranking[0]["baseline_threshold_status"] = "yes"
    ranking[0]["beats_baseline_threshold"] = True
    ranking[0]["baseline_threshold_pct"] = 999.0

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="missing-baseline-threshold",
    )

    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert any("baseline" in error for error in report["task_status"]["invariant_errors"])


def test_ranking_order_must_follow_authoritative_round2_scores(tmp_path: Path) -> None:
    ranking_a, candidates_a, probes_a, task_status, provenance = _valid_final_payloads()
    candidate_b = deepcopy(candidates_a[0])
    candidate_b["candidate_id"] = "b"
    candidate_b["rank"] = 2
    candidate_b["rank_group"] = 2
    candidate_b_probe_ids: list[str] = []
    probes_b: list[dict[str, Any]] = []
    for probe in probes_a:
        copied = deepcopy(probe)
        old_id = copied["probe_id"]
        new_id = f"b-{old_id}"
        copied["probe_id"] = new_id
        copied["candidate_id"] = "b"
        if isinstance(copied.get("normalized"), dict):
            copied["normalized"]["total_throughput"] = 90.0
        if copied["record_type"] == "aggregate_sample":
            copied["replica_probe_ids"] = [f"b-{value}" for value in copied["replica_probe_ids"]]
        probes_b.append(copied)
        candidate_b_probe_ids.append(new_id)
    candidate_b["probe_ids"] = candidate_b_probe_ids
    for group in candidate_b["sample_groups"]:
        group["aggregate_probe_ids"] = [f"b-{value}" for value in group["aggregate_probe_ids"]]
        group["representative"]["total_throughput"] = 90.0
    candidate_b["rank"] = 2
    candidate_b["rank_group"] = 2
    ranking_b = deepcopy(ranking_a[0])
    ranking_b.update(
        {
            "candidate_id": "b",
            "rank": 1,
            "rank_group": 2,
            "goodput_raw": 90.0,
            "goodput_per_host_min": 90.0,
            "goodput_per_host_median": 90.0,
            "goodput_per_host_max": 90.0,
            "goodput_per_host": 90.0,
        }
    )
    candidates_a[0]["candidate_id"] = "a"
    candidates_a[0]["rank"] = 2
    candidates_a[0]["rank_group"] = 1
    task_status["expected_candidate_ids"] = ["a", "b"]

    report = write_reports(
        tmp_path,
        ranking=[ranking_b, {**ranking_a[0], "rank": 2, "rank_group": 1}],
        candidate_rows=[candidate_b, candidates_a[0]],
        probe_rows=[*probes_a, *probes_b],
        task_status=task_status,
        provenance=provenance,
        run_id="reversed-ranking",
    )

    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert any(
        "ordered" in error or "rank" in error
        for error in report["task_status"]["invariant_errors"]
    )


@pytest.mark.parametrize("mutation", [
    "round1_num_evals_bool",
    "ranking_rank_bool",
    "sample_group_round_bool",
    "probe_normalized_nan",
    "ranking_interval_inf",
])
def test_required_numeric_evidence_is_strict_and_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    """Malformed numeric evidence must never become FINAL or crash the writer."""
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    if mutation == "round1_num_evals_bool":
        candidates[0]["round1"]["num_evals"] = True
    elif mutation == "ranking_rank_bool":
        ranking[0]["rank"] = True
        candidates[0]["rank"] = True
    elif mutation == "sample_group_round_bool":
        candidates[0]["sample_groups"][0]["round"] = True
    elif mutation == "probe_normalized_nan":
        probes[1]["normalized"]["total_throughput"] = math.nan
    elif mutation == "ranking_interval_inf":
        ranking[0]["goodput_per_host_median"] = math.inf

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id=f"strict-numeric-{mutation}",
    )

    assert report["task_status"]["task_status"] != "COMPLETED"
    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert report["task_status"]["invariant_errors"]


@pytest.mark.parametrize(
    "mutation",
    ["round1_c_star", "round1_last_pass", "round1_first_fail", "round1_certainty", "group_batch"],
)
def test_diagnostic_fields_and_group_coordinates_are_authoritative(
    tmp_path: Path, mutation: str
) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    round1 = candidates[0]["round1"]
    assert isinstance(round1, dict)
    if mutation == "round1_c_star":
        round1["c_star"] = 99
    elif mutation == "round1_last_pass":
        round1["last_pass"] = 99
    elif mutation == "round1_first_fail":
        round1["first_fail"] = 99
    elif mutation == "round1_certainty":
        round1["certainty"] = "garbage"
    else:
        candidates[0]["sample_groups"][0]["batch"] = "wrong-batch"

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id=f"diagnostic-{mutation}",
    )
    assert report["task_status"]["task_status"] != "COMPLETED"
    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert report["task_status"]["invariant_errors"]


@pytest.mark.parametrize("field", ["ranking", "candidate_rows", "probe_rows"])
@pytest.mark.parametrize("value", [None, True, 1, 1.5, "bad", {"x": 1}, [1]])
def test_report_writer_malformed_top_level_rows_fail_closed_without_crashing(
    tmp_path: Path, field: str, value: object
) -> None:
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    if field == "ranking":
        ranking = value  # type: ignore[assignment]
    elif field == "candidate_rows":
        candidates = value  # type: ignore[assignment]
    else:
        probes = value  # type: ignore[assignment]

    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id=f"shape-{field}-{type(value).__name__}",
    )
    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert report["task_status"]["invariant_errors"]
