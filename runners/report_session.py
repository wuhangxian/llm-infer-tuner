"""Thread-safe report/evidence session for one executor generation."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from runners.reporting import write_reports

_SAFE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,200}\Z")


class EvidenceJournal:
    """Append-only in-memory evidence journal with consistent snapshots."""

    def __init__(self, *, run_id: str) -> None:
        self.run_id = run_id
        self._lock = threading.RLock()
        self._rows: list[dict[str, Any]] = []
        self._sequence = 0

    def append(self, row: dict[str, Any]) -> str:
        if not isinstance(row, dict):
            raise TypeError("evidence row must be an object")
        with self._lock:
            copied = deepcopy(row)
            probe_id = copied.get("probe_id")
            if not isinstance(probe_id, str) or not probe_id:
                self._sequence += 1
                probe_id = f"{self.run_id}:probe:{self._sequence:08d}"
                copied["probe_id"] = probe_id
            if any(existing.get("probe_id") == probe_id for existing in self._rows):
                raise ValueError(f"duplicate evidence probe_id: {probe_id}")
            copied["run_id"] = self.run_id
            self._rows.append(copied)
            return probe_id

    def extend(self, rows: list[dict[str, Any]]) -> list[str]:
        return [self.append(row) for row in rows]

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._rows)


class ReportSession:
    """Own expected IDs, run identity, evidence and atomic report checkpoints."""

    def __init__(
        self,
        results_dir: Path,
        *,
        expected_candidate_ids: list[str] | tuple[str, ...],
        run_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        expected = list(expected_candidate_ids)
        if (
            not expected
            or any(not isinstance(value, str) or not value for value in expected)
            or len(set(expected)) != len(expected)
        ):
            raise ValueError("expected_candidate_ids must be non-empty and unique")
        resolved_run_id = uuid.uuid4().hex if run_id is None else run_id
        if (
            not isinstance(resolved_run_id, str)
            or _SAFE_ID.fullmatch(resolved_run_id) is None
            or ".." in resolved_run_id
        ):
            raise ValueError("run_id must be a safe non-empty identifier")
        self.results_dir = results_dir
        self.expected_candidate_ids = tuple(expected)
        self.run_id = resolved_run_id
        self.provenance = deepcopy(provenance or {})
        self.journal = EvidenceJournal(run_id=resolved_run_id)
        self._lock = threading.RLock()
        self._last_report: dict[str, Any] | None = None
        self._draft: dict[str, Any] | None = None
        # Keep the payload that produced ``_last_report`` so an interrupt that
        # arrives after a final callback has committed can still revoke that
        # generation.  A loose status update alone is not authoritative: the
        # immutable manifest must be downgraded as well.
        self._last_payload: dict[str, Any] | None = None
        self._placeholder_job_id: str | None = None
        self._placeholder_measurement_mode: str | None = None

    def prepare_placeholder(self, *, job_id: str, measurement_mode: str) -> None:
        """Remember placeholder metadata before installing lifecycle hooks.

        There is a tiny but real signal window between callback registration
        and the first disk checkpoint.  Keeping these values ahead of that
        window lets ``abort`` synthesize a complete expected-ID placeholder if
        a signal arrives before :meth:`begin_placeholder` can publish it.
        """

        if not isinstance(job_id, str) or not job_id:
            raise ValueError("placeholder job_id must be a non-empty string")
        if not isinstance(measurement_mode, str) or not measurement_mode:
            raise ValueError("placeholder measurement_mode must be a non-empty string")
        with self._lock:
            self._placeholder_job_id = job_id
            self._placeholder_measurement_mode = measurement_mode

    def _placeholder_rows(self) -> list[dict[str, Any]]:
        mode = self._placeholder_measurement_mode or "estimated"
        return [
            {
                "candidate_id": candidate_id,
                "status": "incomplete",
                "completion_state": "missing",
                "measurement_mode": mode,
                "measurement_valid": False,
                "ranking_eligible": False,
                "ranking_eligibility_reason": "candidate has not started",
                "rank": None,
                "rank_group": None,
                "baseline_threshold_status": "unknown",
                "beats_baseline_threshold": False,
                "requested_params": {},
                "effective_params": {},
                "requested_command": None,
                "round1": None,
                "round2": None,
                "round1_batch": None,
                "round2_batch": None,
                "attempts": 0,
                "recovery_count": 0,
                "failures": [],
                "final_failure": None,
                "concurrency_points": [],
                "sample_groups": [],
                "incomplete_groups": [],
                "probe_ids": [],
                "actual_instances": 1,
            }
            for candidate_id in self.expected_candidate_ids
        ]

    def begin_placeholder(
        self,
        *,
        job_id: str,
        measurement_mode: str,
    ) -> dict[str, Any]:
        """Publish a provisional, one-row-per-expected-candidate snapshot.

        This is called before the first remote mutation.  It immediately makes
        the new run's generation authoritative, so an older FINAL manifest can
        never be mistaken for the in-flight invocation after a signal or
        preflight failure.
        """

        self.prepare_placeholder(
            job_id=job_id,
            measurement_mode=measurement_mode,
        )
        rows = self._placeholder_rows()
        return self.checkpoint(
            ranking=[],
            candidate_rows=rows,
            task_status={
                "job_id": job_id,
                "task_status": "INCOMPLETE",
                "ranking_status": "PROVISIONAL",
                "measurement_mode": measurement_mode,
                "interrupted": False,
                "cleanup_failures": [],
            },
            # Keep the immutable run-start hashes and explicit unavailable
            # observations captured by the session.  Falling back to only a
            # run_id made an interrupt during placeholder publication lose the
            # provenance needed to audit that partial generation.
            provenance={**deepcopy(self.provenance), "run_id": self.run_id},
            final=False,
        )

    def append_probe(self, row: dict[str, Any]) -> str:
        return self.journal.append(row)

    def checkpoint(
        self,
        *,
        ranking: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        task_status: dict[str, Any],
        provenance: dict[str, Any] | None = None,
        final: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            status = deepcopy(task_status)
            status["expected_candidate_ids"] = list(self.expected_candidate_ids)
            payload = {
                "ranking": deepcopy(ranking),
                "candidate_rows": deepcopy(candidate_rows),
                "task_status": deepcopy(status),
                "provenance": deepcopy(
                    self.provenance if provenance is None else provenance
                ),
            }
            write_status = deepcopy(status)
            if not final:
                self._draft = payload
                write_status["task_status"] = "INCOMPLETE"
                write_status["ranking_status"] = "PROVISIONAL"
                write_status["interrupted"] = False
            report = write_reports(
                self.results_dir,
                ranking=payload["ranking"],
                candidate_rows=payload["candidate_rows"],
                probe_rows=self.journal.snapshot(),
                task_status=write_status,
                provenance=payload["provenance"],
                run_id=self.run_id,
            )
            if final:
                self._draft = None
            self._last_payload = payload
            self._last_report = report
            return deepcopy(report)

    def commit_final(self) -> dict[str, Any] | None:
        """Commit the staged draft after lifecycle cleanup has succeeded."""

        with self._lock:
            if self._draft is None:
                return deepcopy(self._last_report)
            draft = deepcopy(self._draft)
            report = write_reports(
                self.results_dir,
                ranking=draft["ranking"],
                candidate_rows=draft["candidate_rows"],
                probe_rows=self.journal.snapshot(),
                task_status=draft["task_status"],
                provenance=draft["provenance"],
                run_id=self.run_id,
            )
            self._draft = None
            self._last_payload = draft
            self._last_report = report
            return deepcopy(report)

    def abort(
        self,
        *,
        interrupted: bool,
        cleanup_failures: list[str] | None = None,
        reason: str | None = None,
        signal_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Persist the staged evidence as INTERRUPTED/INCOMPLETE."""

        with self._lock:
            source = self._draft
            if source is None and self._last_payload is not None:
                source = deepcopy(self._last_payload)
            if source is None and self._last_report is not None:
                # This path is needed when a signal lands after commit_final.
                # Reconstruct a writeable payload from the verified generation
                # instead of leaving its FINAL manifest authoritative.
                source = {
                    "ranking": deepcopy(self._last_report.get("ranking", [])),
                    "candidate_rows": deepcopy(
                        self._last_report.get("candidate_rows", [])
                    ),
                    "task_status": deepcopy(
                        self._last_report.get("task_status", {})
                    ),
                    "provenance": deepcopy(
                        self._last_report.get("provenance", self.provenance)
                    ),
                }
            if source is None and self._placeholder_job_id is not None:
                # A signal can arrive immediately after callbacks are
                # registered but before ``begin_placeholder`` gets to its
                # first write.  Synthesize the same expected-ID shape here so
                # the interrupted generation is still auditable.
                source = {
                    "ranking": [],
                    "candidate_rows": self._placeholder_rows(),
                    "task_status": {
                        "job_id": self._placeholder_job_id,
                        "task_status": "INCOMPLETE",
                        "ranking_status": "PROVISIONAL",
                        "measurement_mode": self._placeholder_measurement_mode,
                        "interrupted": False,
                        "cleanup_failures": [],
                    },
                    "provenance": {
                        **deepcopy(self.provenance),
                        "run_id": self.run_id,
                    },
                }
            if source is None:
                return deepcopy(self._last_report)
            draft = deepcopy(source)
            status = deepcopy(draft["task_status"])
            status["task_status"] = "INTERRUPTED" if interrupted else "INCOMPLETE"
            status["ranking_status"] = "PROVISIONAL"
            status["interrupted"] = interrupted
            if signal_name is not None:
                status["signal"] = signal_name
            status["cleanup_failures"] = list(cleanup_failures or [])
            if reason:
                status["failure_reason"] = reason
            try:
                report = write_reports(
                    self.results_dir,
                    ranking=draft["ranking"],
                    candidate_rows=draft["candidate_rows"],
                    probe_rows=self.journal.snapshot(),
                    task_status=status,
                    provenance=draft["provenance"],
                    run_id=self.run_id,
                )
            except BaseException:
                # ``write_reports`` stages all payloads before committing its
                # manifest, but a fault injected after the manifest rename
                # (or during a later compatibility-copy step) can still leave
                # a FINAL manifest while this revocation path is unwinding.
                # Move the active pointer out of the way as a last-resort
                # fail-closed action.  The immutable generation remains
                # recoverable under the unique stale name, while readers can
                # never mistake it for the current run.
                self._revoke_manifest_pointer()
                self._draft = None
                self._last_payload = draft
                self._last_report = None
                raise
            self._draft = None
            self._last_payload = draft
            self._last_report = report
            return deepcopy(report)

    @property
    def last_report(self) -> dict[str, Any] | None:
        with self._lock:
            return deepcopy(self._last_report)

    def _revoke_manifest_pointer(self) -> None:
        """Hide this run's active manifest when a downgrade cannot be written."""

        manifest = self.results_dir / "report_manifest.json"
        if not manifest.exists():
            return
        # A report directory may be shared by concurrently running wrappers.
        # Before moving the pointer, verify that it still names this session's
        # generation.  In particular, ``write_reports`` can fail *after* a
        # different run has atomically published its manifest; blindly moving
        # that pointer would revoke a healthy foreign FINAL generation.
        try:
            pointer = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            # A malformed pointer is not authoritative evidence.  Hiding it
            # is the safer fail-closed action, while the immutable generation
            # directories remain recoverable for diagnosis.
            pointer = None
        if isinstance(pointer, dict):
            owner = pointer.get("run_id")
            if isinstance(owner, str) and owner != self.run_id:
                return
        stale = self.results_dir / (
            f".report_manifest.revoked.{os.getpid()}.{time.time_ns()}"
        )
        try:
            os.replace(manifest, stale)
        except OSError:
            # The original publication error is more actionable than a
            # best-effort pointer cleanup failure; callers still record the
            # downgrade failure and must not claim FINAL.
            return
