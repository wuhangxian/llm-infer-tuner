"""Run-start provenance snapshots built from local bytes and read-only facts.

The provenance builder intentionally accepts an already captured input snapshot.
It never re-reads mutable job/config files at report time and never serializes
credentials or local paths into the report.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runners.preflight import ImageInspection, PreflightPlan


@dataclass(frozen=True)
class GitState:
    sha: str | None
    dirty: bool | None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class RunStartSnapshot:
    run_id: str
    job_sha256: str
    config_sha256: str
    git: GitState
    started_at: datetime


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("provenance timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _read_git_state(project_root: Path) -> GitState:
    try:
        sha_result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = sha_result.stdout.strip()
        if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
            return GitState(None, None, "git revision was not a canonical SHA-1")
        dirty_result = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return GitState(sha=sha, dirty=bool(dirty_result.stdout.strip()))
    except (OSError, subprocess.SubprocessError) as exc:
        return GitState(None, None, f"git metadata unavailable: {type(exc).__name__}")


def capture_run_start(
    *,
    job_bytes: bytes,
    config_bytes: bytes,
    project_root: Path,
    started_at: datetime,
    run_id: str,
) -> RunStartSnapshot:
    """Capture immutable input hashes and Git state before execution starts."""

    if not isinstance(job_bytes, bytes) or not isinstance(config_bytes, bytes):
        raise TypeError("job_bytes and config_bytes must be bytes")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    _utc_iso(started_at)
    return RunStartSnapshot(
        run_id=run_id,
        job_sha256=hashlib.sha256(job_bytes).hexdigest(),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        git=_read_git_state(project_root),
        started_at=started_at,
    )


def provisional_provenance(snapshot: RunStartSnapshot) -> dict[str, Any]:
    """Build a complete, explicitly-unavailable provenance envelope.

    A lifecycle can be interrupted while the executor is still publishing its
    candidate placeholder, before remote preflight has observed GPUs, an image
    digest, or an engine version.  Persisting only ``{"run_id": ...}`` loses
    the immutable input identity and makes the interrupted generation
    impossible to audit.  Keep the run-start facts and represent the not-yet-
    observed values as null with reasons; the normal final builder replaces
    those sections once preflight has completed.
    """

    git_payload: dict[str, Any] = {
        "sha": snapshot.git.sha,
        "dirty": snapshot.git.dirty,
    }
    if snapshot.git.unavailable_reason:
        git_payload["unavailable_reason"] = snapshot.git.unavailable_reason
    started = _utc_iso(snapshot.started_at)
    return {
        "run_id": snapshot.run_id,
        "git": git_payload,
        "inputs": {
            "job_sha256": snapshot.job_sha256,
            "config_sha256": snapshot.config_sha256,
        },
        "image": {
            "reference": "unavailable",
            "digest": None,
            "unavailable_reason": "image inspection has not run",
        },
        "actual_gpu": {
            "count": None,
            "model": None,
            "memory_bytes": None,
            "inventory": None,
            "topology": None,
            "unavailable_reason": "GPU inventory has not been collected",
        },
        "engine": {
            "version": None,
            "unavailable_reason": "engine version has not been observed",
        },
        "run_started_at": started,
        # A provisional snapshot has no distinct end event yet.  Using the
        # start instant keeps the timestamp schema valid without inventing a
        # future/end time; the final checkpoint replaces it with the true end.
        "run_ended_at": started,
    }


def _image_payload(plan: PreflightPlan) -> dict[str, Any]:
    image: ImageInspection | None = plan.image
    if image is None:
        return {
            "reference": plan.image_reference or "unavailable",
            "digest": None,
            "unavailable_reason": "image inspection was unavailable",
        }
    return {
        "reference": image.reference or plan.image_reference or "unavailable",
        "digest": image.digest,
        "unavailable_reason": image.unavailable_reason,
    }


def _gpu_payload(plan: PreflightPlan) -> dict[str, Any]:
    if not plan.observed_gpus:
        return {
            "count": None,
            "model": None,
            "memory_bytes": None,
            "inventory": None,
            "topology": None,
            "unavailable_reason": "GPU inventory was not collected",
        }
    inventory = [
        {
            "index": gpu.index,
            "name": gpu.name,
            "memory_mib": gpu.memory_mib,
        }
        for gpu in plan.observed_gpus
    ]
    names = {gpu.name for gpu in plan.observed_gpus}
    return {
        "count": len(inventory),
        "model": next(iter(names)) if len(names) == 1 else "mixed",
        "memory_bytes": int(
            round(sum(gpu.memory_mib for gpu in plan.observed_gpus) * 1024 * 1024)
        ),
        "inventory": inventory,
        "topology": [list(group) for group in plan.numa_groups],
        "unavailable_reason": None,
    }


def build_provenance(
    snapshot: RunStartSnapshot,
    *,
    preflight_plan: PreflightPlan,
    engine_versions: tuple[str, ...] | list[str],
    ended_at: datetime,
) -> dict[str, Any]:
    """Build a schema-compatible, secret-free provenance payload."""

    ended_iso = _utc_iso(ended_at)
    if ended_at < snapshot.started_at:
        raise ValueError("provenance end timestamp precedes start")
    versions = tuple(
        version for version in engine_versions if isinstance(version, str) and version
    )
    engine = (
        {"version": versions[0], "unavailable_reason": None}
        if versions
        else {
            "version": None,
            "unavailable_reason": "engine version was not observed",
        }
    )
    git_payload: dict[str, Any] = {
        "sha": snapshot.git.sha,
        "dirty": snapshot.git.dirty,
    }
    if snapshot.git.unavailable_reason:
        git_payload["unavailable_reason"] = snapshot.git.unavailable_reason
    return {
        "run_id": snapshot.run_id,
        "git": git_payload,
        "inputs": {
            "job_sha256": snapshot.job_sha256,
            "config_sha256": snapshot.config_sha256,
        },
        "image": _image_payload(preflight_plan),
        "actual_gpu": _gpu_payload(preflight_plan),
        "engine": engine,
        "run_started_at": _utc_iso(snapshot.started_at),
        "run_ended_at": ended_iso,
    }
