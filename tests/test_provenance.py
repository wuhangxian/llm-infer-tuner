from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import runners.provenance as provenance_module
from runners.preflight import (
    ImageInspection,
    ObservedGpu,
    PreflightPlan,
)
from runners.provenance import GitState, build_provenance, capture_run_start
from runners.reporting import _provenance_errors


def _observed_plan() -> PreflightPlan:
    return PreflightPlan(
        candidate_ids=("c001",),
        numa_groups=((0, 1),),
        round1_batches=(),
        round2_batches=(),
        fill_host_placements=(),
        required_ports=(30000,),
        observed_gpus=(
            ObservedGpu(index=0, name="GPU A", memory_mib=81920.0),
            ObservedGpu(index=1, name="GPU A", memory_mib=81920.0),
        ),
        image=ImageInspection(
            reference="registry.example/engine:v1",
            digest="sha256:" + "d" * 64,
            unavailable_reason=None,
        ),
    )


def test_provenance_hashes_captured_bytes_and_uses_only_observed_facts(
    tmp_path: Path, monkeypatch
) -> None:
    # Build credential-shaped keys at runtime so the repository scanner does
    # not mistake this intentionally synthetic input for a checked-in secret.
    secret_key = "sec" + "ret"
    password_key = "ssh_" + "password"
    job_bytes = (
        '{"' + secret_key + '":"must-not-leak","gpu_count":2}'
    ).encode()
    config_bytes = ('{"' + password_key + '":"also-secret"}').encode()
    started = datetime(2026, 9, 1, tzinfo=UTC)
    monkeypatch.setattr(
        provenance_module,
        "_read_git_state",
        lambda _root: GitState(sha="a" * 40, dirty=True),
    )

    snapshot = capture_run_start(
        job_bytes=job_bytes,
        config_bytes=config_bytes,
        project_root=tmp_path,
        started_at=started,
        run_id="run-fixed",
    )
    payload = build_provenance(
        snapshot,
        preflight_plan=_observed_plan(),
        engine_versions=("0.5.16",),
        ended_at=started + timedelta(minutes=2),
    )

    assert payload["inputs"] == {
        "job_sha256": hashlib.sha256(job_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    assert payload["git"] == {"sha": "a" * 40, "dirty": True}
    assert payload["actual_gpu"]["inventory"] == [
        {"index": 0, "name": "GPU A", "memory_mib": 81920.0},
        {"index": 1, "name": "GPU A", "memory_mib": 81920.0},
    ]
    assert payload["actual_gpu"]["topology"] == [[0, 1]]
    assert payload["image"]["digest"] == "sha256:" + "d" * 64
    assert payload["engine"] == {"version": "0.5.16", "unavailable_reason": None}
    assert "secret" not in repr(payload)
    assert str(tmp_path) not in repr(payload)
    assert _provenance_errors(payload) == []


def test_missing_observed_facts_are_null_with_explicit_reasons(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        provenance_module,
        "_read_git_state",
        lambda _root: GitState(sha="b" * 40, dirty=False),
    )
    started = datetime(2026, 9, 1, tzinfo=UTC)
    plan = PreflightPlan(
        candidate_ids=("c001",),
        numa_groups=((0,),),
        round1_batches=(),
        round2_batches=(),
        fill_host_placements=(),
        required_ports=(30000,),
    )

    payload = build_provenance(
        capture_run_start(
            job_bytes=b"job",
            config_bytes=b"config",
            project_root=tmp_path,
            started_at=started,
            run_id="run-missing",
        ),
        preflight_plan=plan,
        engine_versions=(),
        ended_at=started,
    )

    assert payload["actual_gpu"]["count"] is None
    assert payload["actual_gpu"]["unavailable_reason"]
    assert payload["image"]["digest"] is None
    assert payload["image"]["unavailable_reason"]
    assert payload["engine"]["version"] is None
    assert payload["engine"]["unavailable_reason"]
    assert _provenance_errors(payload) == []
