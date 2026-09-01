"""Offline compatibility checks for sanitized historical report extracts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from runners.reporting import load_report_generation, write_reports

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "replay"


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _probe(
    probe_id: str,
    candidate_id: str,
    *,
    round_number: int,
    concurrency: int,
    repeat: int,
    status: str = "ok",
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "probe_id": probe_id,
        "candidate_id": candidate_id,
        "record_type": "aggregate_sample",
        "round": round_number,
        "batch": "replay/1",
        "concurrency": concurrency,
        "repeat": repeat,
        "recovery": 0,
        "measurement_mode": "estimated",
        "started_at": "2026-08-01T00:00:00+00:00",
        "ended_at": "2026-08-01T00:00:01+00:00",
        # SLA failures are valid statistical verdicts, so they retain the
        # measurement timestamps and use a textual reason rather than a
        # terminal ``failed_at`` marker.  Infrastructure failures below use
        # ``failed_at`` to remain distinguishable from voting samples.
        "failed_at": (
            None
            if status in {"ok", "sla_failed"}
            else "2026-08-01T00:00:01+00:00"
        ),
        "status": status,
        "failure_reason": reason,
        "known_issue": None,
        "raw": {},
        "normalized": {
            "total_throughput": 100.0,
            "mean_ttft_ms": 10.0,
            "mean_tpot_ms": 2.0,
            "success_rate": 1.0,
        },
        "instances": 1,
        "output_healthy": status in {"ok", "sla_failed"},
        "server_health": {"before": "healthy", "after": "healthy"},
        "artifacts": [{"path": f"artifacts/{probe_id}.json", "sha256": "a" * 64}],
        "statistical_vote": status in {"ok", "sla_failed"},
    }


def _valid_payload(fixture: dict[str, Any]) -> tuple[list[dict[str, Any]], ...]:
    cid = fixture["candidate_id"]
    metrics = fixture["metrics"]
    probes: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for round_number, concurrency, repeats, qualifies in (
        (1, fixture["round1_concurrency"], 1, True),
        (2, fixture["round2_last_pass"], 3, True),
        (2, fixture["round2_first_fail"], 3, False),
    ):
        ids: list[str] = []
        for repeat in range(repeats):
            probe_id = f"{cid}:r{round_number}:c{concurrency}:rep{repeat}"
            status = "ok" if qualifies else "sla_failed"
            physical_id = f"{probe_id}:physical"
            physical = _probe(
                physical_id,
                cid,
                round_number=round_number,
                concurrency=concurrency,
                repeat=repeat,
                status=status,
                reason=None if qualifies else "SLA failed",
            )
            physical["record_type"] = "physical_probe"
            physical["replica_index"] = 0
            physical["port"] = 30000
            physical["statistical_vote"] = False
            physical["normalized"] = deepcopy(metrics)
            aggregate = _probe(
                probe_id,
                cid,
                round_number=round_number,
                concurrency=concurrency,
                repeat=repeat,
                status=status,
                reason=None if qualifies else "SLA failed",
            )
            aggregate["record_type"] = "aggregate_sample"
            aggregate["replica_probe_ids"] = [physical_id]
            aggregate["statistical_vote"] = True
            aggregate["normalized"] = deepcopy(metrics)
            probes.extend((physical, aggregate))
            ids.append(probe_id)
        groups.append(
            {
                "group_id": f"{cid}:r{round_number}:c{concurrency}",
                "round": round_number,
                "concurrency": concurrency,
                "aggregate_probe_ids": ids,
                "representative": {"status": "ok" if qualifies else "sla_failed", **metrics},
                "qualifies": qualifies,
            }
        )
    candidate = {
        "candidate_id": cid,
        "status": "completed",
        "completion_state": "completed",
        "measurement_mode": fixture["measurement_mode"],
        "actual_instances": 1,
        "measurement_valid": True,
        "ranking_eligible": True,
        "ranking_eligibility_reason": None,
        "rank": 1,
        "rank_group": 1,
        "probe_ids": [row["probe_id"] for row in probes],
        "sample_groups": groups,
        "incomplete_groups": [],
        "round1": {
            "stop_reason": "max_probes",
            "num_evals": 1,
            "newly_probed": [fixture["round1_concurrency"]],
            "complete": False,
            "certainty": "lower_bound",
        },
        "round2": {
            "stop_reason": "found_boundary",
            "num_evals": 6,
            "newly_probed": [fixture["round2_last_pass"], fixture["round2_first_fail"]],
            "c_star": fixture["round2_last_pass"],
            "last_pass": fixture["round2_last_pass"],
            "first_fail": fixture["round2_first_fail"],
            "complete": True,
            "certainty": "exact",
        },
        "failures": [],
        "recovery_count": 0,
        "final_failure": None,
    }
    ranking = [{
        "candidate_id": cid,
        "rank": 1,
        "rank_group": 1,
        "ranking_eligible": True,
        "measurement_mode": fixture["measurement_mode"],
        "actual_instances": 1,
        "instances_per_host": 1.0,
        "best_concurrency": fixture["round2_last_pass"],
        "sample_count": 3,
        "goodput_raw": metrics["total_throughput"],
        "goodput_per_host_min": metrics["total_throughput"],
        "goodput_per_host_median": metrics["total_throughput"],
        "goodput_per_host_max": metrics["total_throughput"],
        "goodput_per_host": metrics["total_throughput"],
        "baseline_threshold_status": "unknown",
        "beats_baseline_threshold": False,
    }]
    status = {
        "task_status": "COMPLETED",
        "ranking_status": "FINAL",
        "expected_candidate_ids": [cid],
        "interrupted": False,
        "cleanup_failures": [],
    }
    provenance = {
        "git": {"sha": "a" * 40, "dirty": False},
        "inputs": {"job_sha256": "b" * 64, "config_sha256": "c" * 64},
        "image": {"reference": "replay/image:tag", "digest": None, "unavailable_reason": "offline"},
        "actual_gpu": {
            "count": None,
            "model": None,
            "memory_bytes": None,
            "topology": None,
            "unavailable_reason": "offline",
        },
        "engine": {"version": None, "unavailable_reason": "offline"},
        "run_started_at": "2026-08-01T00:00:00+00:00",
        "run_ended_at": "2026-08-01T00:01:00+00:00",
    }
    return ranking, [candidate], probes, status, provenance


def test_valid_historical_metrics_survive_offline_replay(tmp_path: Path) -> None:
    fixture = _load_fixture("valid_metrics.json")
    ranking, candidates, probes, status, provenance = _valid_payload(fixture)
    report = write_reports(
        tmp_path,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=status,
        provenance=provenance,
        run_id="replay-valid-run",
    )
    loaded = load_report_generation(tmp_path)
    assert loaded == report
    assert loaded["task_status"]["ranking_status"] == "FINAL"
    assert loaded["ranking"][0]["goodput_per_host_median"] == fixture["metrics"]["total_throughput"]
    assert loaded["probe_rows"][0]["normalized"] == fixture["metrics"]


def test_historical_runtime_failures_remain_incomplete_and_retain_requested_tp(
    tmp_path: Path,
) -> None:
    fixture = _load_fixture("runtime_failures.json")
    cid = fixture["candidate_id"]
    statuses = fixture["failure_statuses"]
    probes = [
        _probe(
            f"{cid}:failure:{index}",
            cid,
            round_number=2,
            concurrency=1,
            repeat=index,
            status=probe_status,
            reason=f"replayed {probe_status}",
        )
        for index, probe_status in enumerate(statuses)
    ]
    candidate = {
        "candidate_id": cid,
        "status": "incomplete",
        "completion_state": fixture["expected_completion_state"],
        "measurement_mode": "estimated",
        "requested_params": {"tp_size": fixture["requested_tp"]},
        "requested_tp": fixture["requested_tp"],
        "actual_instances": 1,
        "ranking_eligible": False,
        "ranking_eligibility_reason": "replayed infrastructure failures",
        "probe_ids": [row["probe_id"] for row in probes],
        "sample_groups": [],
        "incomplete_groups": [],
        "round1": None,
        "round2": None,
        "failures": [
            {"probe_id": row["probe_id"], "status": row["status"], "reason": row["failure_reason"]}
            for row in probes
        ],
        "recovery_count": len(probes),
        "final_failure": "replayed infrastructure failures",
    }
    report = write_reports(
        tmp_path,
        ranking=[],
        candidate_rows=[candidate],
        probe_rows=probes,
        task_status={
            "task_status": fixture["expected_task_status"],
            "ranking_status": "FINAL",
            "expected_candidate_ids": [cid],
            "interrupted": False,
            "cleanup_failures": [],
        },
        provenance={},
        run_id="replay-failures-run",
    )
    assert report["task_status"]["task_status"] == "INCOMPLETE"
    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert report["candidate_rows"][0]["requested_tp"] == fixture["requested_tp"]
    assert len(report["probe_rows"]) == len(statuses)
    assert report["candidate_rows"][0]["recovery_count"] == len(statuses)
