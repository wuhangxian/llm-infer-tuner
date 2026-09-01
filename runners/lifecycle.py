"""Process-wide executor lifecycle, signal state, and checked resource cleanup."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, TypeVar

Cleanup = Callable[[], object]
T = TypeVar("T")

# Keep an emergency reference to the interpreter's signal setter.  Tests and
# embedders may wrap ``signal.signal`` to inject failures; cleanup must still
# be able to restore the process dispositions without allowing that wrapper to
# mask the original lifecycle exception.
_ORIGINAL_SIGNAL = signal.signal


class LifecycleInterrupted(KeyboardInterrupt):
    """Control-flow exception used to unwind starts after SIGINT/SIGTERM."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        self.exit_code = 128 + signum
        super().__init__(signal.Signals(signum).name)


class LifecycleCleanupError(RuntimeError):
    """One or more possibly-started resources could not be proven cleaned."""

    def __init__(self, failures: list[str]) -> None:
        self.failures = failures
        super().__init__("executor lifecycle cleanup failed: " + "; ".join(failures))


class LifecycleAborted(RuntimeError):
    """New starts are forbidden after an unsafe non-signal lifecycle failure."""


@dataclass
class _Resource:
    name: str
    cleanup: Cleanup


class ExecutorLifecycle:
    """One outer guard for every resource start made by an executor invocation.

    Callers register cleanup *before* issuing a remote start.  The registration
    therefore deliberately means "possibly started", including the ambiguous
    interval after a remote command returned but before its postcheck ran.
    """

    def __init__(self, results_dir: Path, *, job_id: str) -> None:
        self.results_dir = Path(results_dir)
        self.job_id = job_id
        # One lifecycle invocation owns one report generation identity.  The
        # executor reuses this value so externally managed contexts (including
        # the CLI) return the same run_id that callbacks publish.
        self.run_id = uuid.uuid4().hex
        self._resources: dict[str, _Resource] = {}
        self._lock = threading.RLock()
        self._start_allowed = True
        self._signal: int | None = None
        self._cleanup_failures: list[str] = []
        self._old_handlers: dict[int, Any] = {}
        self._entered = False
        self._entering = False
        # ``__exit__`` closes the start gate before it enters ``cleanup_all``.
        # A synchronous signal can arrive in that tiny hand-off window; mark
        # the unwind as active first so ``interrupt`` records the signal but
        # does not raise through cleanup or handler restoration.
        self._exiting = False
        self._cleaning = False
        self._inflight_starts = 0
        self._condition = threading.Condition(self._lock)
        self._cleanup_owner: int | None = None
        self._cleanup_done = False
        self._cleanup_result: list[str] = []
        self._cleanup_waiters = 0
        self._status_write_failures: list[str] = []
        self._report_on_interrupt: Callable[[list[str]], object] | None = None
        self._report_on_failure: (
            Callable[[list[str], BaseException | None], object] | None
        ) = None
        self._report_on_finalize: Callable[[], object] | None = None
        self._report_interrupt_active = False
        self._tombstone_published = False
        self._defer_interrupt_depth = 0

    @property
    def interrupted(self) -> bool:
        with self._lock:
            return self._signal is not None

    @property
    def signal_name(self) -> str | None:
        """Name of the first signal observed by this lifecycle, if any."""

        with self._lock:
            signum = self._signal
        if signum is None:
            return None
        try:
            return signal.Signals(signum).name
        except ValueError:
            return f"SIG{signum}"

    @property
    def entered(self) -> bool:
        """Whether this lifecycle context currently owns its signal handlers.

        ``run_executor(..., lifecycle=...)`` is an externally managed API: the
        caller must enter the context first and exit it afterwards.  Exposing
        the state lets that boundary fail fast instead of silently running with
        an unregistered cleanup registry and leaking resources.
        """
        with self._lock:
            return self._entered

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return not self._start_allowed

    @property
    def cleanup_failures(self) -> list[str]:
        with self._lock:
            return list(self._cleanup_failures)

    @property
    def status_write_failures(self) -> list[str]:
        with self._lock:
            return list(self._status_write_failures)

    def __enter__(self) -> ExecutorLifecycle:
        try:
            with self._lock:
                self._entering = True
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    self._old_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, self._handle_signal)
            # A new invocation must not leave an older immutable FINAL
            # manifest authoritative while local parsing or session setup is
            # still in progress.  Keep the old manifest recoverable under a
            # unique name; the report session publishes the new generation
            # once its expected candidate set is known.
            manifest = self.results_dir / "report_manifest.json"
            if manifest.exists():
                stale = self.results_dir / (
                    f".report_manifest.stale.{os.getpid()}.{time.time_ns()}"
                )
                try:
                    # ``os.replace`` is the normal atomic path.  Keep a
                    # pointer-only fallback for a transient/monkeypatched
                    # replace failure: the immutable generation directory is
                    # still retained, while the active pointer must not leave
                    # an older FINAL report authoritative for this run.
                    os.replace(manifest, stale)
                except OSError as replace_exc:
                    self._record_status_write_failure(replace_exc)
                    try:
                        # ``Path.rename`` uses the platform rename primitive
                        # and can succeed when an injected ``os.replace``
                        # failure is transient.  It remains atomic within
                        # the results directory and preserves the old pointer.
                        manifest.rename(stale)
                    except OSError as rename_exc:
                        self._record_status_write_failure(rename_exc)
                        try:
                            # Last resort: remove only the pointer (never the
                            # generation payload).  This is fail-closed if
                            # both rename mechanisms are unavailable.
                            manifest.unlink()
                        except OSError as unlink_exc:
                            self._record_status_write_failure(unlink_exc)
                            raise RuntimeError(
                                "cannot isolate previous report manifest"
                            ) from unlink_exc
            # Publish a minimal schema-v2 provisional generation even when
            # local job/config parsing has not yet produced the authoritative
            # candidate set.  This closes the early-failure window where an
            # old FINAL manifest could be revoked to a loose status only.  The
            # executor's ReportSession replaces this tombstone with explicit
            # expected rows as soon as candidates are loaded.
            try:
                from runners.reporting import write_reports

                write_reports(
                    self.results_dir,
                    ranking=[],
                    candidate_rows=[],
                    probe_rows=[],
                    task_status={
                        "task_status": "INCOMPLETE",
                        "ranking_status": "PROVISIONAL",
                        "expected_candidate_ids": [],
                        "interrupted": False,
                        "cleanup_failures": [],
                    },
                    provenance={"run_id": self.run_id},
                    run_id=self.run_id,
                )
                self._tombstone_published = True
            except Exception as tombstone_exc:
                self._record_status_write_failure(tombstone_exc)
            # Invalidate a stale FINAL report before the first remote mutation.
            # Handlers are already active so a signal during the atomic write
            # can replace it with INTERRUPTED instead of taking the OS default.
            with self._lock:
                pending_signal = self._signal
            if pending_signal is not None:
                self._try_write_interrupted_status()
                raise LifecycleInterrupted(pending_signal)
            self._write_status("RUNNING", interrupted=False)
            # Keep the final state transition and return inside the protected
            # ``try``.  A signal in either bytecode must still restore the
            # handlers instead of escaping with ``_entered`` half-published.
            with self._lock:
                self._entered = True
                self._entering = False
                pending_signal = self._signal
            if pending_signal is not None:
                self._try_write_interrupted_status()
                self._restore_handlers()
                with self._lock:
                    self._entered = False
                raise LifecycleInterrupted(pending_signal)
            return self
        except BaseException:
            if self._signal is not None and not self._active_report_is_interrupted():
                self._try_write_interrupted_status()
            try:
                self._restore_handlers()
            finally:
                with self._lock:
                    self._entered = False
                    self._entering = False
                    self._exiting = False
            raise

    def __exit__(self, exc_type, exc, traceback) -> bool:
        """Finish the lifecycle while making cleanup signal-safe.

        The private implementation below predates a very small but important
        race: a synchronous signal can arrive between entering ``__exit__``
        and the first assignment that marks the unwind.  Keeping an outer
        ``try/finally`` here means even that first-bytecode interruption gets a
        cleanup pass and restores the process signal handlers.
        """

        try:
            with self._lock:
                self._exiting = True
                self._start_allowed = False
            return self._exit_impl(exc_type, exc, traceback)
        finally:
            self._ensure_exit_safety()

    def _exit_impl(self, exc_type, exc, traceback) -> bool:
        del exc_type, traceback
        with self._lock:
            self._exiting = True
            self._start_allowed = False
        failures: list[str] = []
        cleanup_exception: BaseException | None = None
        try:
            try:
                failures = self.cleanup_all()
            except BaseException as cleanup_exc:
                cleanup_exception = cleanup_exc
                detail = (
                    "lifecycle cleanup_all raised "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
                failures.append(detail)
                with self._lock:
                    self._cleanup_failures.append(detail)
        finally:
            try:
                def combined_failures() -> list[str]:
                    # The cleanup list is copied into ``failures`` in some
                    # paths and appended again by callbacks.  Preserve order
                    # while removing duplicates so repeated signal/cleanup
                    # passes do not manufacture misleading evidence counts.
                    return list(dict.fromkeys([*failures, *self.cleanup_failures]))

                effective_exc = (
                    exc if exc is not None else cleanup_exception
                )
                interrupted = self.interrupted or isinstance(
                    effective_exc, LifecycleInterrupted
                )
                with self._lock:
                    on_interrupt = self._report_on_interrupt
                    on_failure = self._report_on_failure
                    on_finalize = self._report_on_finalize
                callback: Callable[..., object] | None
                callback_args: tuple[object, ...]
                callback_kind: str
                callback_succeeded = False
                failure_callback_failed = False
                callback_failure_type: str | None = None
                if interrupted:
                    callback = on_interrupt
                    callback_args = (
                        combined_failures(),
                    )
                    callback_kind = "interrupt"
                elif failures or effective_exc is not None:
                    callback = on_failure
                    callback_args = (list(failures), effective_exc)
                    callback_kind = "failure"
                else:
                    callback = on_finalize
                    callback_args = ()
                    callback_kind = "finalize"
                if callback is not None:
                    try:
                        callback(*callback_args)
                        callback_succeeded = True
                    except BaseException as callback_exc:
                        callback_succeeded = False
                        callback_failure_type = type(callback_exc).__name__
                        if isinstance(callback_exc, LifecycleInterrupted):
                            # A callback may surface a deferred signal itself
                            # (for example, an injected write hook).  Treat it
                            # as lifecycle control flow rather than converting
                            # it into a generic cleanup failure.
                            with self._lock:
                                if self._signal is None:
                                    self._signal = callback_exc.signum
                        detail = (
                            "report callback raised "
                            f"{type(callback_exc).__name__}: {callback_exc}"
                        )
                        failures.append(detail)
                        with self._lock:
                            self._cleanup_failures.append(detail)
                        # ``on_finalize`` may have published its immutable
                        # FINAL generation before a post-write hook raised.
                        # Ask the failure callback to revoke that generation;
                        # otherwise the loose INCOMPLETE status below would
                        # disagree with the manifest-selected FINAL report.
                        if callback_kind == "finalize" and on_failure is not None:
                            try:
                                on_failure(
                                    combined_failures(),
                                    callback_exc,
                                )
                            except BaseException as failure_callback_exc:
                                failure_callback_failed = True
                                failure_detail = (
                                    "report failure callback raised "
                                    f"{type(failure_callback_exc).__name__}: "
                                    f"{failure_callback_exc}"
                                )
                                failures.append(failure_detail)
                                with self._lock:
                                    self._cleanup_failures.append(failure_detail)
                        elif callback_kind == "finalize":
                            # No downgrade hook exists. Revoke the active
                            # immutable pointer rather than leaving a FINAL
                            # generation authoritative after a failed commit.
                            failure_callback_failed = True
                # If the failing finalize hook also surfaced a lifecycle
                # signal, let the signal-specific downgrade below preserve a
                # loadable INTERRUPTED snapshot.  Revoking here first would
                # discard that active pointer and leave only a loose status.
                if failure_callback_failed and self._signal is None:
                    self._revoke_active_manifest()
                # A signal can be delivered while a report finalizer is
                # writing its immutable generation.  Never leave that FINAL
                # generation authoritative: immediately run the interrupt
                # callback as a revocation pass, then write INTERRUPTED below.
                with self._lock:
                    signal_after_callback = self._signal
                    interrupt_callback = self._report_on_interrupt
                interrupt_callback_succeeded = (
                    callback_kind == "interrupt"
                    and callback is not None
                    and callback_succeeded
                )
                if signal_after_callback is not None and callback_kind != "interrupt":
                    interrupted = True
                    if interrupt_callback is not None:
                        interrupt_callback_succeeded = self._invoke_report_interrupt(
                            interrupt_callback,
                            combined_failures(),
                        )

                # A callback's normal return is not evidence that it actually
                # changed the immutable report.  Any exception, cleanup
                # failure, or signal makes a FINAL generation unsafe.  Check
                # the generation selected by the manifest after *all* hooks,
                # including no-op compatibility hooks, and downgrade it (or
                # revoke the pointer) before publishing the loose status.
                unsafe_exit = (
                    interrupted
                    or bool(failures)
                    or effective_exc is not None
                    or callback_kind == "failure"
                    or (callback_kind == "finalize" and not callback_succeeded)
                )
                downgrade_failed = False
                downgrade_succeeded = False
                if unsafe_exit and self._manifest_run_id() == self.run_id:
                    # Re-publish for every unsafe exit, not only when the
                    # pointer still says FINAL.  A compatibility callback may
                    # have returned while dropping cleanup/failure metadata;
                    # the canonical generation must retain those diagnostics
                    # even when it was already provisional.
                    needs_downgrade = True
                    if needs_downgrade:
                        downgraded = self._downgrade_active_report(
                            interrupted=interrupted,
                            cleanup_failures=combined_failures(),
                            signal_name=(
                                signal.Signals(self._signal).name
                                if self._signal is not None
                                else None
                            ),
                            reason=(
                                str(effective_exc)
                                if effective_exc is not None
                                else None
                            ),
                            failure_type=(
                                type(effective_exc).__name__
                                if effective_exc is not None
                                else callback_failure_type
                            ),
                        )
                        if not downgraded:
                            downgrade_failed = True
                            self._revoke_active_manifest()
                        else:
                            downgrade_succeeded = True
                if interrupted:
                    if (
                        not interrupt_callback_succeeded
                        or (downgrade_failed and not downgrade_succeeded)
                    ):
                        self._revoke_active_manifest()
                        self._try_write_interrupted_status()
                elif (
                    failures
                    and (not callback_succeeded or downgrade_failed)
                    and not downgrade_succeeded
                ):
                    self._try_write_status(
                        "INCOMPLETE",
                        interrupted=False,
                        cleanup_failures=self.cleanup_failures,
                        failure=effective_exc,
                    )
                elif (
                    effective_exc is not None
                    and (not callback_succeeded or downgrade_failed)
                    and not downgrade_succeeded
                ):
                    self._try_write_status(
                        "INCOMPLETE",
                        interrupted=False,
                        failure=effective_exc,
                    )
            finally:
                # Keep our handlers installed until cleanup and the final
                # provisional status attempt are both complete.
                self._restore_handlers()
                with self._lock:
                    self._entered = False
                    self._exiting = False
        # A first signal received while cleanup owns the lifecycle is recorded
        # immediately but deliberately does not asynchronously interrupt a
        # callback.  Restore its control flow only after every callback, the
        # final status write, and handler restoration have completed.
        with self._lock:
            deferred_signal = self._signal
        effective_exc = exc if exc is not None else cleanup_exception
        if deferred_signal is not None and not isinstance(
            effective_exc, LifecycleInterrupted
        ):
            raise LifecycleInterrupted(deferred_signal)
        if failures:
            cleanup_error = LifecycleCleanupError(failures)
            if effective_exc is not None and not isinstance(
                effective_exc, LifecycleInterrupted
            ):
                raise cleanup_error from effective_exc
            if effective_exc is None:
                raise cleanup_error
        return False

    def _ensure_exit_safety(self) -> None:
        """Finish cleanup if an exception interrupted the exit prologue.

        Signal delivery is synchronous on the main thread and can happen
        before the private exit implementation has installed its own
        ``try/finally``.  This outer safety net is deliberately idempotent:
        normal exits have already cleared ``_entered``/``_old_handlers`` and
        therefore return immediately, while an interrupted prologue gets one
        last cleanup, report downgrade, and handler-restore pass.
        """

        pending_exc = sys.exc_info()[1]
        with self._lock:
            active = (
                self._entered
                or bool(self._old_handlers)
                or not self._cleanup_done
                or bool(self._resources)
            )
            if not active:
                return
            self._exiting = True
            self._start_allowed = False

        failures: list[str] = []
        if not self._cleanup_done:
            try:
                failures = self.cleanup_all()
            except BaseException as cleanup_exc:
                detail = (
                    "lifecycle cleanup_all raised "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
                failures.append(detail)
                with self._lock:
                    self._cleanup_failures.append(detail)

        signum = self._signal
        interrupted = signum is not None or isinstance(
            pending_exc, LifecycleInterrupted
        )
        if (interrupted or pending_exc is not None) and (
            self._manifest_run_id() == self.run_id
        ):
            signal_name = self.signal_name
            downgraded = False
            if interrupted:
                downgraded = self._active_report_is_interrupted()
            if not downgraded:
                downgraded = self._downgrade_active_report(
                    interrupted=interrupted,
                    cleanup_failures=list(
                        dict.fromkeys([*failures, *self.cleanup_failures])
                    ),
                    signal_name=signal_name,
                    reason=(str(pending_exc) if pending_exc is not None else None),
                    failure_type=(
                        type(pending_exc).__name__
                        if pending_exc is not None
                        else None
                    ),
                )
            if not downgraded:
                self._revoke_active_manifest()
                if interrupted:
                    self._try_write_interrupted_status()
                else:
                    self._try_write_status(
                        "INCOMPLETE",
                        interrupted=False,
                        cleanup_failures=self.cleanup_failures,
                        failure=pending_exc,
                    )

        # A signal can arrive while restoring the handlers too.  Keep this
        # operation best-effort and preserve the original control-flow error.
        try:
            self._restore_handlers()
        except BaseException as restore_exc:
            self._record_status_write_failure(restore_exc)
        finally:
            with self._lock:
                self._entered = False
                self._exiting = False

    def register_report_callbacks(
        self,
        *,
        on_interrupt: Callable[[list[str]], object] | None = None,
        on_failure: Callable[[list[str], BaseException | None], object] | None = None,
        on_finalize: Callable[[], object] | None = None,
    ) -> None:
        """Attach report-generation hooks to the lifecycle transaction.

        Hooks run only after every registered resource cleanup callback has
        completed.  Callback failures are recorded as cleanup failures; they
        are never allowed to mask a pending :class:`LifecycleInterrupted`.
        """

        with self._lock:
            if self._cleaning or self._cleanup_done:
                raise RuntimeError("cannot register report callbacks after cleanup")
            self._report_on_interrupt = on_interrupt
            self._report_on_failure = on_failure
            self._report_on_finalize = on_finalize
            pending_signal = self._signal
            gate_closed = not self._start_allowed
        if pending_signal is not None:
            # Registration may race with a signal immediately after a session
            # publishes its placeholder.  Keep the callbacks installed so
            # ``__exit__`` can provide the post-cleanup evidence pass, but
            # unwind now instead of allowing a remote start through a closed
            # gate.
            if on_interrupt is not None:
                self._invoke_report_interrupt(on_interrupt, self.cleanup_failures)
            raise LifecycleInterrupted(pending_signal)
        if gate_closed:
            raise LifecycleAborted("executor lifecycle start gate is closed")

    def assert_start_allowed(self) -> None:
        with self._lock:
            if not self._start_allowed:
                if self._signal is not None:
                    raise LifecycleInterrupted(self._signal)
                raise LifecycleAborted("executor lifecycle start gate is closed")

    def register_possible(self, name: str, cleanup: Cleanup) -> None:
        """Register a possibly-started resource before its start command."""
        self.assert_start_allowed()
        with self._lock:
            # Re-check while holding the registration lock: a signal must never
            # slip between the gate check and insertion.
            if not self._start_allowed:
                if self._signal is not None:
                    raise LifecycleInterrupted(self._signal)
                raise LifecycleAborted("executor lifecycle start gate is closed")
            if name in self._resources:
                raise ValueError(f"lifecycle resource already registered: {name}")
            self._resources[name] = _Resource(name=name, cleanup=cleanup)

    def start_resource(
        self,
        name: str,
        *,
        starter: Callable[[], T],
        cleanup: Cleanup,
    ) -> T:
        """Atomically lease the start gate, pre-register, and invoke ``starter``.

        Cleanup closes the gate and waits for all leased starters to return (or
        raise) before draining the registry.  This prevents a worker from
        starting a resource after another thread already ran its cleanup.
        """
        with self._condition:
            if not self._start_allowed:
                if self._signal is not None:
                    raise LifecycleInterrupted(self._signal)
                raise LifecycleAborted("executor lifecycle start gate is closed")
            if name in self._resources:
                raise ValueError(f"lifecycle resource already registered: {name}")
            self._resources[name] = _Resource(name=name, cleanup=cleanup)
            self._inflight_starts += 1
        try:
            return starter()
        finally:
            with self._condition:
                self._inflight_starts -= 1
                self._condition.notify_all()

    def abort(self, reason: str) -> None:
        """Close the global start gate after an unsafe cleanup/proof failure."""
        with self._lock:
            self._start_allowed = False
        self._try_write_status(
            "INCOMPLETE",
            interrupted=False,
            cleanup_failures=self.cleanup_failures,
            failure=RuntimeError(reason),
        )

    def release(self, name: str) -> None:
        """Forget a resource only after the caller proved it absent."""
        with self._lock:
            self._resources.pop(name, None)

    def is_registered(self, name: str) -> bool:
        with self._lock:
            return name in self._resources

    def interrupt(self, signum: int) -> None:
        """Close the start gate, persist interruption immediately, then unwind."""
        with self._lock:
            self._start_allowed = False
            first_signal = self._signal is None
            if first_signal:
                self._signal = int(signum)
            # A later signal must not asynchronously abort a cleanup callback
            # or the narrow unwind-before-cleanup window.
            suppress_raise = (
                self._exiting
                or self._entering
                or self._cleaning
                or self._defer_interrupt_depth > 0
                or not first_signal
            )
            callback = self._report_on_interrupt if first_signal else None
        # Revoke any previous immutable generation as soon as the first signal
        # arrives.  ``__exit__`` invokes the callback again with complete
        # cleanup failures, so this first pass is intentionally best effort.
        if callback is not None:
            callback_succeeded = self._invoke_report_interrupt(callback, [])
        else:
            callback_succeeded = False
        # A report session owns the schema-v2 task-status view.  Do not let the
        # legacy schema-v1 lifecycle fallback overwrite that canonical status
        # after the callback has successfully revoked the generation.
        if callback is None or not callback_succeeded:
            self._try_write_interrupted_status()
        elif not self._active_report_is_interrupted():
            # A callback may return normally without actually downgrading the
            # immutable generation (for example, a no-op compatibility hook).
            # The signal handler cannot treat a return value as proof of a
            # durable state transition: if a FINAL pointer is still active,
            # preserve the evidence when possible and force an interrupted,
            # provisional snapshot before allowing the process to unwind.
            if not self._downgrade_active_report(
                interrupted=True,
                cleanup_failures=self.cleanup_failures,
                signal_name=signal.Signals(signum).name,
            ):
                self._revoke_active_manifest()
                self._try_write_interrupted_status()
        if suppress_raise:
            return
        raise LifecycleInterrupted(int(signum))

    @contextmanager
    def defer_interrupt(self):
        """Defer raising the first signal until a journal checkpoint completes.

        The signal is still recorded and the report interrupt callback still
        runs immediately; only the control-flow exception is delayed.  This
        closes the narrow handoff between a batch returning outcomes and the
        caller appending those outcomes to the durable evidence journal.
        """
        with self._lock:
            self._defer_interrupt_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._defer_interrupt_depth -= 1
                pending_signal = (
                    self._signal if self._defer_interrupt_depth == 0 else None
                )
            if pending_signal is not None:
                raise LifecycleInterrupted(pending_signal)

    def cleanup_all(self) -> list[str]:
        """Attempt all callbacks in reverse registration order exactly once."""
        with self._condition:
            self._start_allowed = False
            owner = threading.get_ident()
            if self._cleanup_done:
                return list(self._cleanup_result)
            if self._cleanup_owner is not None:
                if self._cleanup_owner == owner:
                    return []
                self._cleanup_waiters += 1
                self._condition.notify_all()
                try:
                    while not self._cleanup_done:
                        self._condition.wait()
                finally:
                    self._cleanup_waiters -= 1
                return list(self._cleanup_result)
            self._cleanup_owner = owner
            self._cleaning = True
            while self._inflight_starts:
                self._condition.wait()
            resources = list(reversed(self._resources.values()))
            self._resources.clear()
        failures: list[str] = []
        for resource in resources:
            try:
                result = resource.cleanup()
                if isinstance(result, str) and result:
                    failures.append(f"{resource.name}: {result}")
                elif isinstance(result, (list, tuple)):
                    failures.extend(
                        f"{resource.name}: {item}" for item in result if item
                    )
            except BaseException as cleanup_exc:  # cleanup must not mask later resources
                failures.append(f"{resource.name}: cleanup raised {cleanup_exc!r}")
        with self._condition:
            self._cleanup_failures.extend(failures)
            self._cleanup_result = list(failures)
            self._cleanup_done = True
            self._cleanup_owner = None
            self._cleaning = False
            self._condition.notify_all()
        return failures

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        # The interpreter may deliver a signal before the first ``try`` in
        # ``__enter__``/``__exit__`` has installed its exception table.  The
        # frame identity lets us mark the transition as deferred before
        # delegating to ``interrupt``; the transition method then checks the
        # pending signal and performs the normal cleanup/restore path.
        transition: str | None = None
        frame = _frame
        # A tracing/profiling hook can itself be the frame supplied by Python's
        # signal machinery.  Walk its callers so deterministic fault-injection
        # tests (and debuggers) receive the same transition-safe behavior as a
        # normal delivery directly into the lifecycle method.
        while frame is not None:
            if frame.f_locals.get("self") is self:
                if frame.f_code.co_name in {"__exit__", "_exit_impl"}:
                    transition = "exit"
                    break
                if frame.f_code.co_name == "__enter__":
                    transition = "enter"
                    break
            frame = frame.f_back
        enter_handoff = False
        if transition == "exit":
            with self._lock:
                self._exiting = True
        elif transition == "enter":
            with self._lock:
                if not self._entered:
                    self._entering = True
                else:
                    # ``with`` does not install its implicit ``__exit__``
                    # handler until after ``__enter__`` returns.  A signal in
                    # that tiny protocol handoff would otherwise raise past
                    # the caller without ever cleaning resources or restoring
                    # our process handlers.  Mark an emergency exit before
                    # recording the signal so ``interrupt`` defers its raise;
                    # then run the same idempotent exit transaction here.
                    self._exiting = True
                    enter_handoff = True
        self.interrupt(signum)
        if enter_handoff:
            interrupted = LifecycleInterrupted(int(signum))
            try:
                self.__exit__(LifecycleInterrupted, interrupted, None)
            finally:
                # ``__exit__`` returns False for the supplied control-flow
                # exception; the signal handler must still unwind the caller.
                raise interrupted

    def _invoke_report_interrupt(
        self,
        callback: Callable[[list[str]], object],
        failures: list[str],
    ) -> bool:
        """Run the interrupt hook without masking lifecycle control flow."""
        with self._lock:
            if self._report_interrupt_active:
                return True
            self._report_interrupt_active = True
        try:
            try:
                callback(list(failures))
            except BaseException as callback_exc:
                detail = (
                    "report callback raised "
                    f"{type(callback_exc).__name__}: {callback_exc}"
                )
                with self._lock:
                    self._cleanup_failures.append(detail)
                return False
            return True
        finally:
            with self._lock:
                self._report_interrupt_active = False

    def _restore_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        old_handlers = dict(self._old_handlers)
        # Clear before invoking user/interpreter hooks so a re-entrant signal
        # cannot observe a half-owned handler map and recursively retry an
        # already-restored disposition.
        self._old_handlers.clear()
        for signum, handler in old_handlers.items():
            try:
                signal.signal(signum, handler)
            except BaseException as restore_exc:
                # A wrapped setter may fail, but the real primitive can still
                # restore the original process disposition.  If that fallback
                # also fails, record the loss and continue restoring the other
                # signal; callers retain their original control-flow error.
                try:
                    _ORIGINAL_SIGNAL(signum, handler)
                except BaseException as fallback_exc:
                    self._record_status_write_failure(fallback_exc)
                self._record_status_write_failure(restore_exc)

    def _write_interrupted_status(self) -> None:
        signum = self._signal or signal.SIGINT
        # If no report session has registered a callback yet (for example a
        # SIGTERM during ``load_job``), update the tombstone generation itself
        # so the manifest-selected status is INTERRUPTED rather than merely
        # changing the loose compatibility file.
        if self._tombstone_published and self._report_on_interrupt is None:
            try:
                from runners.reporting import write_reports

                write_reports(
                    self.results_dir,
                    ranking=[],
                    candidate_rows=[],
                    probe_rows=[],
                    task_status={
                        "task_status": "INTERRUPTED",
                        "ranking_status": "PROVISIONAL",
                        "expected_candidate_ids": [],
                        "interrupted": True,
                        "signal": signal.Signals(signum).name,
                        "cleanup_failures": self.cleanup_failures,
                    },
                    provenance={"run_id": self.run_id},
                    run_id=self.run_id,
                )
            except Exception as report_exc:
                self._record_status_write_failure(report_exc)
        self._write_status(
            "INTERRUPTED",
            interrupted=True,
            signal_name=signal.Signals(signum).name,
            cleanup_failures=self.cleanup_failures,
        )

    def _manifest_run_id(self) -> str | None:
        """Read the active pointer owner without trusting arbitrary fields."""

        try:
            payload = json.loads(
                (self.results_dir / "report_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        owner = payload.get("run_id")
        return owner if isinstance(owner, str) else None

    def _active_report_status(self) -> tuple[str, str, bool] | None:
        """Return the active generation's status tuple when it is ours.

        This deliberately uses the verified loader instead of trusting the
        loose compatibility ``task_status.json``.  A malformed or foreign
        pointer is treated as unavailable; callers then fail closed rather
        than making a decision from untrusted text.
        """

        owner = self._manifest_run_id()
        if owner != self.run_id:
            return None
        try:
            from runners.reporting import load_report_generation

            report = load_report_generation(self.results_dir)
        except Exception:
            return None
        status = report.get("task_status")
        if not isinstance(status, dict):
            return None
        task_status = status.get("task_status")
        ranking_status = status.get("ranking_status")
        interrupted = status.get("interrupted")
        if not isinstance(task_status, str) or not isinstance(ranking_status, str):
            return None
        if type(interrupted) is not bool:
            return None
        return task_status, ranking_status, interrupted

    def _active_manifest_is_final(self) -> bool:
        state = self._active_report_status()
        return state == ("COMPLETED", "FINAL", False)

    def _active_report_is_interrupted(self) -> bool:
        state = self._active_report_status()
        return state == ("INTERRUPTED", "PROVISIONAL", True)

    def _downgrade_active_report(
        self,
        *,
        interrupted: bool,
        cleanup_failures: list[str] | None = None,
        signal_name: str | None = None,
        reason: str | None = None,
        failure_type: str | None = None,
    ) -> bool:
        """Publish a provisional copy of the active report for an unsafe exit.

        Callback success is not an integrity proof.  This helper reloads the
        immutable payload (when available), changes only the lifecycle status,
        and republishes it through the transactional writer.  Keeping the
        evidence rows avoids turning a recoverable interrupted run into an
        empty tombstone.  If loading or writing fails, the caller can revoke
        the pointer as a last-resort fail-closed action.
        """

        owner = self._manifest_run_id()
        if owner is not None and owner != self.run_id:
            # Never revoke or overwrite another concurrent invocation's
            # generation merely because this lifecycle is unwinding.
            return False
        try:
            from runners.reporting import load_report_generation, write_reports

            try:
                report = load_report_generation(self.results_dir)
            except Exception:
                report = None
            if isinstance(report, dict):
                ranking = report.get("ranking", [])
                candidate_rows = report.get("candidate_rows", [])
                probe_rows = report.get("probe_rows", [])
                provenance = report.get("provenance", {"run_id": self.run_id})
                status = report.get("task_status", {})
            else:
                ranking = []
                candidate_rows = []
                probe_rows = []
                provenance = {"run_id": self.run_id}
                status = {
                    "job_id": self.job_id,
                    "expected_candidate_ids": [],
                }
            if not isinstance(ranking, list):
                ranking = []
            if not isinstance(candidate_rows, list):
                candidate_rows = []
            if not isinstance(probe_rows, list):
                probe_rows = []
            if not isinstance(provenance, dict):
                provenance = {"run_id": self.run_id}
            if not isinstance(status, dict):
                status = {"job_id": self.job_id}
            status_payload: dict[str, Any] = dict(status)
            status_payload["task_status"] = "INTERRUPTED" if interrupted else "INCOMPLETE"
            status_payload["ranking_status"] = "PROVISIONAL"
            status_payload["interrupted"] = bool(interrupted)
            status_payload["signal"] = signal_name
            status_payload["cleanup_failures"] = list(cleanup_failures or [])
            if reason:
                status_payload["failure_reason"] = reason
            if failure_type:
                status_payload["failure_type"] = failure_type
            write_reports(
                self.results_dir,
                ranking=ranking,
                candidate_rows=candidate_rows,
                probe_rows=probe_rows,
                task_status=status_payload,
                provenance=provenance,
                run_id=self.run_id,
            )
            self._tombstone_published = True
            return True
        except BaseException as report_exc:
            self._record_status_write_failure(report_exc)
            return False

    def _revoke_active_manifest(self) -> None:
        """Move the current report pointer aside after a failed downgrade."""

        manifest = self.results_dir / "report_manifest.json"
        if not manifest.exists():
            return
        owner = self._manifest_run_id()
        if owner is not None and owner != self.run_id:
            # A concurrent run owns the pointer; do not hide its generation.
            return
        stale = self.results_dir / (
            f".report_manifest.revoked.{os.getpid()}.{time.time_ns()}"
        )
        try:
            os.replace(manifest, stale)
        except OSError as exc:
            with self._lock:
                self._cleanup_failures.append(
                    f"report manifest revoke failed: {exc}"
                )

    def _record_status_write_failure(self, exc: BaseException) -> None:
        with self._lock:
            self._status_write_failures.append(
                f"status write raised {type(exc).__name__}: {exc}"
            )

    def _try_write_interrupted_status(self) -> None:
        try:
            self._write_interrupted_status()
        except Exception as exc:  # signal control flow must still unwind
            self._record_status_write_failure(exc)

    def _try_write_status(self, task_status: str, **kwargs: Any) -> None:
        try:
            self._write_status(task_status, **kwargs)
        except Exception as exc:  # preserve the body/cleanup exception
            self._record_status_write_failure(exc)

    def _write_status(
        self,
        task_status: str,
        *,
        interrupted: bool,
        signal_name: str | None = None,
        cleanup_failures: list[str] | None = None,
        failure: BaseException | None = None,
    ) -> None:
        payload = {
            "report_schema_version": 2,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "task_status": task_status,
            "ranking_status": "PROVISIONAL",
            "interrupted": interrupted,
            "signal": signal_name,
            "cleanup_failures": list(cleanup_failures or []),
            "failure_type": type(failure).__name__ if failure is not None else None,
            "failure_reason": str(failure) if failure is not None else None,
        }
        path = self.results_dir / "task_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
        )
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
