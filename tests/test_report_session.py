from __future__ import annotations

import os
import signal
import sys
import threading
from pathlib import Path

import pytest

from runners.lifecycle import ExecutorLifecycle, LifecycleCleanupError, LifecycleInterrupted
from runners.report_session import EvidenceJournal, ReportSession
from runners.reporting import load_report_generation


def test_session_owns_run_id_expected_set_and_checkpoint(tmp_path: Path) -> None:
    session = ReportSession(
        tmp_path,
        expected_candidate_ids=("a", "b"),
        run_id="session-run",
        provenance={},
    )
    first = session.append_probe({"candidate_id": "a"})
    assert first == "session-run:probe:00000001"
    report = session.checkpoint(
        ranking=[],
        candidate_rows=[
            {"candidate_id": "a", "status": "incomplete"},
            {"candidate_id": "b", "status": "incomplete"},
        ],
        task_status={"task_status": "INCOMPLETE", "ranking_status": "PROVISIONAL"},
    )
    assert report["manifest"]["run_id"] == "session-run"
    assert report["task_status"]["expected_candidate_ids"] == ["a", "b"]
    assert session.last_report == report


def test_journal_append_is_thread_safe_and_rejects_duplicate_ids() -> None:
    journal = EvidenceJournal(run_id="thread-run")
    barrier = threading.Barrier(8)

    def append(index: int) -> None:
        barrier.wait()
        journal.append({"candidate_id": "a", "probe_id": f"p-{index}"})

    threads = [threading.Thread(target=append, args=(index,)) for index in range(80)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows = journal.snapshot()
    assert len(rows) == 80
    assert {row["probe_id"] for row in rows} == {f"p-{index}" for index in range(80)}
    with pytest.raises(ValueError, match="duplicate"):
        journal.append({"probe_id": "p-0"})


@pytest.mark.parametrize("ids", [[], ["a", "a"], ["", "b"], [1]])
def test_session_rejects_invalid_expected_ids(tmp_path: Path, ids: list[object]) -> None:
    with pytest.raises(ValueError, match="expected_candidate_ids"):
        ReportSession(tmp_path, expected_candidate_ids=ids)  # type: ignore[arg-type]


def test_begin_placeholder_publishes_provisional_expected_rows(tmp_path: Path) -> None:
    session = ReportSession(tmp_path, expected_candidate_ids=("a", "b"), run_id="run")
    report = session.begin_placeholder(job_id="job", measurement_mode="estimated")
    assert report["task_status"]["task_status"] == "INCOMPLETE"
    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert [row["candidate_id"] for row in report["candidate_rows"]] == ["a", "b"]
    assert report["manifest"]["run_id"] == "run"


def test_checkpoint_stages_final_intent_but_publishes_provisional(tmp_path: Path) -> None:
    session = ReportSession(tmp_path, expected_candidate_ids=("a",), run_id="run")
    session.begin_placeholder(job_id="job", measurement_mode="estimated")
    report = session.checkpoint(
        ranking=[],
        candidate_rows=[
            {
                "candidate_id": "a",
                "status": "incomplete",
                "completion_state": "incomplete",
                "measurement_mode": "estimated",
                "measurement_valid": False,
                "ranking_eligible": False,
                "rank": None,
                "probe_ids": [],
                "sample_groups": [],
                "incomplete_groups": [],
            }
        ],
        task_status={
            "task_status": "COMPLETED",
            "ranking_status": "FINAL",
            "interrupted": False,
        },
        final=False,
    )
    assert report["task_status"]["task_status"] == "INCOMPLETE"
    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    # The callback can still commit the original FINAL intent after cleanup.
    assert session._draft is not None
    assert session._draft["task_status"]["task_status"] == "COMPLETED"
    assert session._draft["task_status"]["ranking_status"] == "FINAL"


def test_lifecycle_report_callbacks_finalize_only_after_cleanup(tmp_path: Path) -> None:
    events: list[str] = []
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job")
    lifecycle.__enter__()
    lifecycle.register_report_callbacks(
        on_interrupt=lambda _failures: events.append("interrupt"),
        on_failure=lambda _failures, _exc: events.append("failure"),
        on_finalize=lambda: events.append("final"),
    )
    lifecycle.register_possible("resource", lambda: events.append("cleanup"))
    lifecycle.__exit__(None, None, None)
    assert events == ["cleanup", "final"]


def test_lifecycle_callback_error_does_not_mask_interrupt(tmp_path: Path) -> None:
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job")
    lifecycle.__enter__()
    lifecycle.register_report_callbacks(
        on_interrupt=lambda _failures: (_ for _ in ()).throw(RuntimeError("callback")),
    )
    with pytest.raises(LifecycleInterrupted):
        lifecycle.interrupt(2)
    lifecycle.__exit__(LifecycleInterrupted, LifecycleInterrupted(2), None)
    assert lifecycle.status_write_failures == []


def test_interrupt_notifies_report_callback_before_unwind(tmp_path: Path) -> None:
    events: list[str] = []
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job")
    lifecycle.__enter__()
    lifecycle.register_report_callbacks(on_interrupt=lambda _failures: events.append("interrupt"))
    with pytest.raises(LifecycleInterrupted):
        lifecycle.interrupt(2)
    assert events == ["interrupt"]
    lifecycle.__exit__(LifecycleInterrupted, LifecycleInterrupted(2), None)
    assert events == ["interrupt", "interrupt"]


def test_report_callback_keeps_schema2_status_authoritative_on_interrupt(
    tmp_path: Path,
) -> None:
    session = ReportSession(tmp_path, expected_candidate_ids=("a",), run_id="run")
    session.begin_placeholder(job_id="job", measurement_mode="estimated")
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job")
    lifecycle.__enter__()
    lifecycle.register_report_callbacks(
        on_interrupt=lambda failures: session.abort(
            interrupted=True,
            cleanup_failures=failures,
        )
    )
    with pytest.raises(LifecycleInterrupted):
        lifecycle.interrupt(2)
    lifecycle.__exit__(LifecycleInterrupted, LifecycleInterrupted(2), None)

    report = load_report_generation(tmp_path)
    assert report["manifest"]["run_id"] == "run"
    assert report["task_status"]["report_schema_version"] == 2
    assert report["task_status"]["task_status"] == "INTERRUPTED"
    assert report["task_status"]["interrupted"] is True


def test_lifecycle_entry_revokes_previous_manifest_before_local_setup(
    tmp_path: Path,
) -> None:
    old_session = ReportSession(tmp_path, expected_candidate_ids=("old",), run_id="old")
    old_session.begin_placeholder(job_id="old-job", measurement_mode="estimated")
    assert (tmp_path / "report_manifest.json").exists()

    lifecycle = ExecutorLifecycle(tmp_path, job_id="new-job")
    lifecycle.__enter__()
    try:
        tombstone = load_report_generation(tmp_path)
        assert tombstone["manifest"]["run_id"] == lifecycle.run_id
        assert tombstone["task_status"]["ranking_status"] == "PROVISIONAL"
        assert list(tmp_path.glob(".report_manifest.stale.*"))
    finally:
        lifecycle.__exit__(None, None, None)


def test_lifecycle_entry_isolates_previous_manifest_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient pointer-rename failure must not expose the old FINAL run."""

    old_session = ReportSession(tmp_path, expected_candidate_ids=("old",), run_id="old")
    old_session.begin_placeholder(job_id="old-job", measurement_mode="estimated")
    manifest = tmp_path / "report_manifest.json"
    original_replace = os.replace

    def fail_old_pointer_replace(source, destination):
        if Path(source) == manifest:
            raise OSError("injected pointer rename failure")
        return original_replace(source, destination)

    monkeypatch.setattr("runners.lifecycle.os.replace", fail_old_pointer_replace)
    lifecycle = ExecutorLifecycle(tmp_path, job_id="new-job")
    lifecycle.__enter__()
    try:
        report = load_report_generation(tmp_path)
        assert report["manifest"]["run_id"] == lifecycle.run_id
        assert report["task_status"]["ranking_status"] == "PROVISIONAL"
        assert '"run_id": "old"' not in manifest.read_text(encoding="utf-8")
        assert list(tmp_path.glob(".report_manifest.stale.*"))
        assert lifecycle.status_write_failures
    finally:
        lifecycle.__exit__(None, None, None)


def test_early_interrupt_updates_tombstone_manifest_status(tmp_path: Path) -> None:
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job")
    lifecycle.__enter__()
    with pytest.raises(LifecycleInterrupted):
        lifecycle.interrupt(15)
    lifecycle.__exit__(LifecycleInterrupted, LifecycleInterrupted(15), None)

    report = load_report_generation(tmp_path)
    assert report["task_status"]["task_status"] == "INTERRUPTED"
    assert report["task_status"]["ranking_status"] == "PROVISIONAL"
    assert report["task_status"]["interrupted"] is True


def test_signal_during_finalize_revokes_final_callback(tmp_path: Path) -> None:
    events: list[str] = []
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job")
    lifecycle.__enter__()

    def finalize() -> None:
        events.append("final")
        lifecycle.interrupt(2)

    lifecycle.register_report_callbacks(
        on_interrupt=lambda _failures: events.append("interrupt"),
        on_finalize=finalize,
    )
    with pytest.raises(LifecycleInterrupted):
        lifecycle.__exit__(None, None, None)
    assert events[:2] == ["final", "interrupt"]


def test_finalize_callback_failure_invokes_revoke_failure_hook(tmp_path: Path) -> None:
    events: list[str] = []
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job")
    lifecycle.__enter__()

    def finalize() -> None:
        events.append("final")
        raise RuntimeError("post-commit failure")

    lifecycle.register_report_callbacks(
        on_failure=lambda failures, failure: events.append("failure"),
        on_finalize=finalize,
    )
    with pytest.raises(LifecycleCleanupError):
        lifecycle.__exit__(None, None, None)
    assert events == ["final", "failure"]


def test_signal_between_exit_gate_and_cleanup_is_deferred(tmp_path: Path, monkeypatch) -> None:
    """A signal after gate-close must not skip the cleanup pass."""

    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-exit-race")
    lifecycle.__enter__()
    cleaned: list[str] = []
    lifecycle.register_possible("resource", lambda: cleaned.append("resource"))
    original_cleanup_all = lifecycle.cleanup_all
    injected = False

    def interrupt_before_cleanup() -> list[str]:
        nonlocal injected
        if not injected:
            injected = True
            # __exit__ has already closed the start gate and marked its
            # cleanup-unwind section active, but cleanup_all has not started.
            lifecycle.interrupt(signal.SIGTERM)
        return original_cleanup_all()

    monkeypatch.setattr(lifecycle, "cleanup_all", interrupt_before_cleanup)
    with pytest.raises(LifecycleInterrupted) as raised:
        lifecycle.__exit__(None, None, None)

    assert raised.value.exit_code == 143
    assert cleaned == ["resource"]
    assert lifecycle._exiting is False


def test_signal_during_handler_restore_is_deferred_until_restore_finishes(
    tmp_path: Path, monkeypatch
) -> None:
    """A signal in final unwind must not interrupt handler restoration."""

    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-restore-race")
    old_term_handler = signal.getsignal(signal.SIGTERM)
    lifecycle.__enter__()
    original_restore = lifecycle._restore_handlers
    injected = False

    def interrupt_before_restore() -> None:
        nonlocal injected
        if not injected:
            injected = True
            lifecycle.interrupt(signal.SIGTERM)
        original_restore()

    monkeypatch.setattr(lifecycle, "_restore_handlers", interrupt_before_restore)
    with pytest.raises(LifecycleInterrupted) as raised:
        lifecycle.__exit__(None, None, None)

    assert raised.value.exit_code == 143
    assert signal.getsignal(signal.SIGTERM) is old_term_handler
    assert lifecycle._exiting is False


def test_signal_after_enter_return_still_runs_exit_transaction(tmp_path: Path) -> None:
    """Cover CPython's __enter__ return -> BEFORE_WITH handoff window."""

    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-enter-handoff")
    cleaned: list[str] = []
    lifecycle.register_possible("resource", lambda: cleaned.append("resource"))
    old_term_handler = signal.getsignal(signal.SIGTERM)
    body_entered = False
    injected = False

    def profile(frame, event, _arg):
        nonlocal injected
        if (
            not injected
            and event == "return"
            and frame.f_code is ExecutorLifecycle.__enter__.__code__
            and frame.f_locals.get("self") is lifecycle
        ):
            injected = True
            os.kill(os.getpid(), signal.SIGTERM)
        return profile

    sys.setprofile(profile)
    try:
        with pytest.raises(LifecycleInterrupted) as raised:
            with lifecycle:
                body_entered = True
    finally:
        sys.setprofile(None)

    assert raised.value.exit_code == 143
    assert body_entered is False
    assert cleaned == ["resource"]
    assert lifecycle.entered is False
    assert lifecycle._entering is False
    assert lifecycle._exiting is False
    assert lifecycle._old_handlers == {}
    assert signal.getsignal(signal.SIGTERM) is old_term_handler
    report = load_report_generation(tmp_path)
    assert report["task_status"]["task_status"] == "INTERRUPTED"
    assert report["task_status"]["ranking_status"] == "PROVISIONAL"


def test_failed_revoke_callback_hides_active_manifest(
    tmp_path: Path,
) -> None:
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job")
    lifecycle.__enter__()
    lifecycle.register_report_callbacks(
        on_finalize=lambda: (_ for _ in ()).throw(RuntimeError("commit failed")),
        on_failure=lambda *_: (_ for _ in ()).throw(RuntimeError("revoke failed")),
    )

    with pytest.raises(LifecycleCleanupError):
        lifecycle.__exit__(None, None, None)

    assert not (tmp_path / "report_manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        load_report_generation(tmp_path)


def test_abort_write_failure_revokes_active_manifest_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed downgrade must never leave a FINAL manifest authoritative."""

    import runners.report_session as report_session_module

    session = ReportSession(tmp_path, expected_candidate_ids=("a",), run_id="run")
    session.begin_placeholder(job_id="job", measurement_mode="estimated")
    manifest = tmp_path / "report_manifest.json"
    assert manifest.exists()

    def fail_write(*_args, **_kwargs):
        raise OSError("injected report write failure")

    monkeypatch.setattr(report_session_module, "write_reports", fail_write)
    with pytest.raises(OSError, match="injected"):
        session.abort(interrupted=True, cleanup_failures=["cleanup failed"])

    assert not manifest.exists()
    assert list(tmp_path.glob(".report_manifest.revoked.*"))
    with pytest.raises((FileNotFoundError, ValueError)):
        load_report_generation(tmp_path)


def test_abort_does_not_revoke_a_foreign_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed downgrade must not hide a concurrent run's active pointer."""

    import runners.report_session as report_session_module

    first = ReportSession(tmp_path, expected_candidate_ids=("a",), run_id="run-a")
    first.begin_placeholder(job_id="job", measurement_mode="estimated")
    second = ReportSession(tmp_path, expected_candidate_ids=("b",), run_id="run-b")
    second.begin_placeholder(job_id="job", measurement_mode="estimated")

    def fail_write(*_args, **_kwargs):
        raise OSError("injected report write failure")

    monkeypatch.setattr(report_session_module, "write_reports", fail_write)
    with pytest.raises(OSError, match="injected"):
        first.abort(interrupted=True)

    pointer = (tmp_path / "report_manifest.json").read_text(encoding="utf-8")
    assert '"run_id": "run-b"' in pointer
