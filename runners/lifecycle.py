"""Process-wide executor lifecycle, signal state, and checked resource cleanup."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, TypeVar

Cleanup = Callable[[], object]
T = TypeVar("T")


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
        self._resources: dict[str, _Resource] = {}
        self._lock = threading.RLock()
        self._start_allowed = True
        self._signal: int | None = None
        self._cleanup_failures: list[str] = []
        self._old_handlers: dict[int, Any] = {}
        self._entered = False
        self._cleaning = False
        self._inflight_starts = 0
        self._condition = threading.Condition(self._lock)
        self._cleanup_owner: int | None = None
        self._cleanup_done = False
        self._cleanup_result: list[str] = []
        self._cleanup_waiters = 0
        self._status_write_failures: list[str] = []

    @property
    def interrupted(self) -> bool:
        with self._lock:
            return self._signal is not None

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
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._old_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
        try:
            # Invalidate a stale FINAL report before the first remote mutation.
            # Handlers are already active so a signal during the atomic write
            # can replace it with INTERRUPTED instead of taking the OS default.
            self._write_status("RUNNING", interrupted=False)
        except BaseException:
            self._restore_handlers()
            raise
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, traceback
        with self._lock:
            self._start_allowed = False
        failures: list[str] = []
        try:
            failures = self.cleanup_all()
        finally:
            try:
                interrupted = self.interrupted or isinstance(exc, LifecycleInterrupted)
                if interrupted:
                    self._try_write_interrupted_status()
                elif failures:
                    self._try_write_status(
                        "INCOMPLETE",
                        interrupted=False,
                        cleanup_failures=self.cleanup_failures,
                        failure=exc,
                    )
                elif exc is not None:
                    self._try_write_status(
                        "INCOMPLETE",
                        interrupted=False,
                        failure=exc,
                    )
            finally:
                # Keep our handlers installed until cleanup and the final
                # provisional status attempt are both complete.
                self._restore_handlers()
        # A first signal received while cleanup owns the lifecycle is recorded
        # immediately but deliberately does not asynchronously interrupt a
        # callback.  Restore its control flow only after every callback, the
        # final status write, and handler restoration have completed.
        with self._lock:
            deferred_signal = self._signal
        if deferred_signal is not None and not isinstance(exc, LifecycleInterrupted):
            raise LifecycleInterrupted(deferred_signal)
        if failures:
            cleanup_error = LifecycleCleanupError(failures)
            if exc is not None and not isinstance(exc, LifecycleInterrupted):
                raise cleanup_error from exc
            if exc is None:
                raise cleanup_error
        return False

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
            suppress_raise = self._cleaning or not first_signal
        self._try_write_interrupted_status()
        if suppress_raise:
            return
        raise LifecycleInterrupted(int(signum))

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
        self.interrupt(signum)

    def _restore_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum, handler in self._old_handlers.items():
            signal.signal(signum, handler)
        self._old_handlers.clear()

    def _write_interrupted_status(self) -> None:
        signum = self._signal or signal.SIGINT
        self._write_status(
            "INTERRUPTED",
            interrupted=True,
            signal_name=signal.Signals(signum).name,
            cleanup_failures=self.cleanup_failures,
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
            "report_schema_version": 1,
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
