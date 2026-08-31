from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

from runners.bench_runner import (
    benchmark_artifact_paths,
    cleanup_monitored_benchmark,
    run_benchmark_monitored,
)
from runners.container import Container, process_group_state_command
from runners.lifecycle import (
    ExecutorLifecycle,
    LifecycleCleanupError,
    LifecycleInterrupted,
)
from runners.remote import CommandFailureKind, CommandResult


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class _StalledBenchmarkContainer:
    def __init__(self) -> None:
        self.signals: list[tuple[int, str]] = []
        self.killed = False
        self.timeouts: list[int] = []

    def start_monitored(self, command, log_path, status_path, pid_path, *, timeout):
        self.timeouts.append(timeout)
        return CommandResult(0, "4242\n", "")

    def monitored_state(self, pid, status_path, *, timeout):
        self.timeouts.append(timeout)
        return CommandResult(0, "MISSING\n" if self.killed else "RUNNING\n", "")

    def process_group_state(self, pid, *, timeout):
        self.timeouts.append(timeout)
        return CommandResult(0, "MISSING\n" if self.killed else "RUNNING\n", "")

    def file_progress(self, paths, *, timeout):
        self.timeouts.append(timeout)
        return CommandResult(
            0,
            json.dumps({path: "missing" for path in paths}) + "\n",
            "",
        )

    def health(self, port, *, timeout):
        self.timeouts.append(timeout)
        return CommandResult(0, "ok\n", "")

    def signal_process_group(self, pid, signal_name, *, timeout):
        self.timeouts.append(timeout)
        self.signals.append((pid, signal_name))
        if signal_name == "KILL":
            self.killed = True
        return CommandResult(0, "", "")


class _ProgressingBenchmarkContainer(_StalledBenchmarkContainer):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self.clock = clock
        self.progress_counter = 0

    def monitored_state(self, pid, status_path, *, timeout):
        self.timeouts.append(timeout)
        if self.clock.value >= 350:
            return CommandResult(0, "EXIT 0\n", "")
        return CommandResult(0, "RUNNING\n", "")

    def file_progress(self, paths, *, timeout):
        self.timeouts.append(timeout)
        self.progress_counter += 1
        return CommandResult(
            0,
            json.dumps({path: str(self.progress_counter) for path in paths}) + "\n",
            "",
        )


class _HealthTransportContainer(_StalledBenchmarkContainer):
    def health(self, port, *, timeout):
        self.timeouts.append(timeout)
        return CommandResult(
            255,
            "",
            "ssh health lost",
            failure_kind=CommandFailureKind.TRANSPORT,
        )


def test_health_transport_preserves_root_diagnostic_through_checked_termination():
    clock = FakeClock()
    container = _HealthTransportContainer()

    result = run_benchmark_monitored(
        container,
        "bench",
        log_path="/tmp/bench.log",
        result_path="/tmp/result.jsonl",
        status_path="/tmp/bench.status",
        server_ports=[30000],
        now=clock.monotonic,
        sleep=clock.sleep,
        poll_interval_s=1,
        terminate_grace_s=10,
        poll_timeout_s=3,
    )

    assert result.failure_kind == CommandFailureKind.TRANSPORT
    assert "ssh health lost" in result.stderr
    assert clock.value == 10
    assert container.signals == [(4242, "TERM"), (4242, "KILL")]


def test_interrupt_closes_start_gate_and_cleans_possibly_started_resource(tmp_path):
    (tmp_path / "task_status.json").write_text(
        '{"task_status":"COMPLETED","ranking_status":"FINAL"}', encoding="utf-8"
    )
    cleaned: list[str] = []
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-1")

    with pytest.raises(LifecycleInterrupted) as raised:
        with lifecycle:
            running = json.loads(
                (tmp_path / "task_status.json").read_text(encoding="utf-8")
            )
            assert running["task_status"] == "RUNNING"
            assert running["ranking_status"] == "PROVISIONAL"
            lifecycle.register_possible(
                "container:candidate-a", lambda: cleaned.append("container:candidate-a")
            )
            lifecycle.interrupt(signal.SIGINT)

    assert raised.value.exit_code == 130
    assert cleaned == ["container:candidate-a"]
    with pytest.raises(LifecycleInterrupted):
        lifecycle.assert_start_allowed()

    status = json.loads((tmp_path / "task_status.json").read_text(encoding="utf-8"))
    assert status["task_status"] == "INTERRUPTED"
    assert status["ranking_status"] == "PROVISIONAL"
    assert status["interrupted"] is True
    assert status["signal"] == "SIGINT"
    assert status["cleanup_failures"] == []


def test_second_signal_during_cleanup_does_not_skip_reverse_checked_cleanup(tmp_path):
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-2")
    cleaned: list[str] = []

    def second_signal() -> None:
        cleaned.append("third")
        lifecycle.interrupt(signal.SIGINT)

    with pytest.raises(LifecycleInterrupted) as raised:
        with lifecycle:
            lifecycle.register_possible("first", lambda: cleaned.append("first"))
            lifecycle.register_possible(
                "second", lambda: (cleaned.append("second"), "remove rc=1")[1]
            )
            lifecycle.register_possible("third", second_signal)
            lifecycle.interrupt(signal.SIGTERM)

    assert raised.value.exit_code == 143
    assert cleaned == ["third", "second", "first"]
    status = json.loads((tmp_path / "task_status.json").read_text(encoding="utf-8"))
    assert status["signal"] == "SIGTERM"
    assert status["cleanup_failures"] == ["second: remove rc=1"]


def test_managed_start_is_registered_before_start_and_cleanup_waits_for_it(tmp_path):
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-3")
    start_entered = threading.Event()
    allow_start_return = threading.Event()
    events: list[str] = []

    def starter() -> str:
        start_entered.set()
        assert allow_start_return.wait(timeout=2)
        events.append("start-returned")
        return "started"

    lifecycle.__enter__()
    worker = threading.Thread(
        target=lambda: lifecycle.start_resource(
            "container:candidate-a",
            starter=starter,
            cleanup=lambda: events.append("cleanup"),
        )
    )
    worker.start()
    assert start_entered.wait(timeout=2)
    with pytest.raises(LifecycleInterrupted) as interrupted:
        lifecycle.interrupt(signal.SIGINT)

    cleanup = threading.Thread(target=lifecycle.cleanup_all)
    cleanup.start()
    allow_start_return.set()
    worker.join(timeout=2)
    cleanup.join(timeout=2)
    lifecycle.__exit__(LifecycleInterrupted, interrupted.value, None)

    assert not worker.is_alive()
    assert not cleanup.is_alive()
    assert events == ["start-returned", "cleanup"]


def test_body_and_cleanup_failure_are_both_preserved_and_status_is_not_running(tmp_path):
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-4")

    with pytest.raises(LifecycleCleanupError) as raised:
        with lifecycle:
            lifecycle.register_possible(
                "container:candidate-a",
                lambda: (_ for _ in ()).throw(OSError("remove failed")),
            )
            raise ValueError("body failed")

    assert isinstance(raised.value.__cause__, ValueError)
    status = json.loads((tmp_path / "task_status.json").read_text(encoding="utf-8"))
    assert status["task_status"] == "INCOMPLETE"
    assert status["ranking_status"] == "PROVISIONAL"
    assert "remove failed" in status["cleanup_failures"][0]


def test_body_failure_with_successful_cleanup_atomically_invalidates_running(tmp_path):
    with pytest.raises(ValueError, match="body failed"):
        with ExecutorLifecycle(tmp_path, job_id="job-5"):
            raise ValueError("body failed")

    status = json.loads((tmp_path / "task_status.json").read_text(encoding="utf-8"))
    assert status["task_status"] == "INCOMPLETE"
    assert status["ranking_status"] == "PROVISIONAL"
    assert status["failure_type"] == "ValueError"
    assert status["failure_reason"] == "body failed"


def test_second_signal_before_cleanup_is_idempotent(tmp_path):
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-6")
    lifecycle.__enter__()
    with pytest.raises(LifecycleInterrupted) as first:
        lifecycle.interrupt(signal.SIGTERM)

    # A nested signal can arrive after the first handler closed the gate but
    # before context unwinding reaches cleanup_all.  It must not raise again.
    lifecycle.interrupt(signal.SIGINT)
    lifecycle.__exit__(LifecycleInterrupted, first.value, None)

    status = json.loads((tmp_path / "task_status.json").read_text(encoding="utf-8"))
    assert status["signal"] == "SIGTERM"


def test_signal_during_inflight_start_wait_is_persisted_immediately(tmp_path):
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-7")
    start_entered = threading.Event()
    allow_start_return = threading.Event()
    lifecycle.__enter__()

    def starter() -> None:
        start_entered.set()
        assert allow_start_return.wait(timeout=2)

    worker = threading.Thread(
        target=lambda: lifecycle.start_resource(
            "container:candidate-a", starter=starter, cleanup=lambda: None
        )
    )
    worker.start()
    assert start_entered.wait(timeout=2)
    cleaner = threading.Thread(target=lifecycle.cleanup_all)
    cleaner.start()
    with lifecycle._condition:  # deterministic observation of the wait transition
        assert lifecycle._cleaning is True

    # cleanup_all is waiting for the bounded managed start.  The signal is
    # suppressed as control flow in this phase, but its state must be durable now.
    lifecycle.interrupt(signal.SIGTERM)
    status = json.loads((tmp_path / "task_status.json").read_text(encoding="utf-8"))
    assert status["task_status"] == "INTERRUPTED"
    assert status["signal"] == "SIGTERM"

    allow_start_return.set()
    worker.join(timeout=2)
    cleaner.join(timeout=2)
    with pytest.raises(LifecycleInterrupted) as interrupted:
        lifecycle.__exit__(None, None, None)
    assert interrupted.value.exit_code == 143


def test_concurrent_cleanup_has_one_owner_and_does_not_reopen_signal_window(tmp_path):
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-8")
    callback_entered = threading.Event()
    allow_callback_return = threading.Event()
    second_done = threading.Event()
    callbacks: list[str] = []
    lifecycle.__enter__()

    def checked_cleanup() -> None:
        callbacks.append("cleanup")
        callback_entered.set()
        assert allow_callback_return.wait(timeout=2)

    lifecycle.register_possible("container:candidate-a", checked_cleanup)
    first = threading.Thread(target=lifecycle.cleanup_all)
    first.start()
    assert callback_entered.wait(timeout=2)
    second = threading.Thread(
        target=lambda: (lifecycle.cleanup_all(), second_done.set())
    )
    second.start()
    with lifecycle._condition:
        assert lifecycle._condition.wait_for(
            lambda: lifecycle._cleanup_waiters == 1, timeout=2
        )

    lifecycle.interrupt(signal.SIGTERM)
    assert not second_done.is_set()
    status = json.loads((tmp_path / "task_status.json").read_text(encoding="utf-8"))
    assert status["task_status"] == "INTERRUPTED"

    allow_callback_return.set()
    first.join(timeout=2)
    second.join(timeout=2)
    with pytest.raises(LifecycleInterrupted) as interrupted:
        lifecycle.__exit__(None, None, None)
    assert interrupted.value.exit_code == 143
    assert callbacks == ["cleanup"]


def test_status_write_failure_cannot_replace_signal_unwind_or_skip_cleanup(
    tmp_path, monkeypatch
):
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-9")
    cleaned: list[str] = []
    lifecycle.__enter__()
    lifecycle.register_possible("container:candidate-a", lambda: cleaned.append("cleanup"))
    monkeypatch.setattr(
        lifecycle,
        "_write_interrupted_status",
        lambda: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(LifecycleInterrupted) as interrupted:
        lifecycle.interrupt(signal.SIGTERM)
    lifecycle.__exit__(LifecycleInterrupted, interrupted.value, None)

    assert interrupted.value.exit_code == 143
    assert cleaned == ["cleanup"]
    assert "disk full" in lifecycle.status_write_failures[0]


def test_signal_during_status_write_is_never_swallowed(tmp_path, monkeypatch):
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-status-signal")
    lifecycle.__enter__()
    original_write = lifecycle._write_status
    injected = False

    def interrupting_write(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            lifecycle.interrupt(signal.SIGTERM)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(lifecycle, "_write_status", interrupting_write)
    with pytest.raises(LifecycleInterrupted) as interrupted:
        lifecycle.abort("synthetic cleanup proof failure")
    lifecycle.__exit__(LifecycleInterrupted, interrupted.value, None)

    assert interrupted.value.exit_code == 143


def test_signal_during_initial_running_write_persists_interrupted_and_restores_handlers(
    tmp_path, monkeypatch
):
    lifecycle = ExecutorLifecycle(tmp_path, job_id="job-enter-signal")
    original_write = lifecycle._write_status
    old_term_handler = signal.getsignal(signal.SIGTERM)
    injected = False

    def interrupting_write(task_status, **kwargs):
        nonlocal injected
        if task_status == "RUNNING" and not injected:
            injected = True
            lifecycle.interrupt(signal.SIGTERM)
        return original_write(task_status, **kwargs)

    monkeypatch.setattr(lifecycle, "_write_status", interrupting_write)
    with pytest.raises(LifecycleInterrupted) as interrupted:
        lifecycle.__enter__()

    assert interrupted.value.exit_code == 143
    assert signal.getsignal(signal.SIGTERM) is old_term_handler
    status = json.loads((tmp_path / "task_status.json").read_text(encoding="utf-8"))
    assert status["task_status"] == "INTERRUPTED"
    assert status["signal"] == "SIGTERM"


def test_monitored_benchmark_stall_waits_full_grace_then_kills_and_proves_absent():
    clock = FakeClock()
    container = _StalledBenchmarkContainer()

    result = run_benchmark_monitored(
        container,
        "python -m sglang.bench_serving --max-concurrency 48",
        log_path="/outputs/r1_c48_repeat0_recovery0_123_uuid.log",
        result_path="/outputs/r1_c48_repeat0_recovery0_123_uuid.jsonl",
        status_path="/outputs/r1_c48_repeat0_recovery0_123_uuid.status",
        server_ports=[30000],
        now=clock.monotonic,
        sleep=clock.sleep,
        poll_interval_s=1,
        stall_timeout_s=300,
        terminate_grace_s=10,
        poll_timeout_s=5,
    )

    assert result.returncode != 0
    assert "stalled" in result.stderr
    assert clock.value == 310
    assert container.signals == [(4242, "TERM"), (4242, "KILL")]
    assert max(container.timeouts) <= 5


def test_monitored_benchmark_has_no_runtime_ceiling_while_evidence_grows():
    clock = FakeClock()
    container = _ProgressingBenchmarkContainer(clock)

    result = run_benchmark_monitored(
        container,
        "bench",
        log_path="/outputs/growing.log",
        result_path="/outputs/growing.jsonl",
        status_path="/outputs/growing.status",
        server_ports=[30000, 30001],
        now=clock.monotonic,
        sleep=clock.sleep,
        poll_interval_s=1,
        stall_timeout_s=300,
        terminate_grace_s=10,
        poll_timeout_s=3,
    )

    assert result.ok
    assert clock.value == 350
    assert container.signals == []
    assert max(container.timeouts) <= 3


def test_benchmark_artifacts_are_unique_and_encode_full_attempt_identity():
    first = benchmark_artifact_paths(
        "/outputs/candidate-a",
        round_label="r2",
        concurrency=48,
        repeat=2,
        recovery_attempt=3,
        replica_index=1,
    )
    second = benchmark_artifact_paths(
        "/outputs/candidate-a",
        round_label="r2",
        concurrency=48,
        repeat=2,
        recovery_attempt=3,
        replica_index=1,
    )

    assert first != second
    for path in first:
        assert "r2_c48_repeat2_recovery3_replica1_" in path


def test_process_group_probe_ignores_zombies_but_detects_live_members(tmp_path):
    marker = tmp_path / "exited"
    zombie = subprocess.Popen(
        ["setsid", "sh", "-c", f"touch {marker}; exit 0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            observed = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(zombie.pid)],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if observed.startswith("Z"):
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"child never became zombie: {observed!r}")
        absent = subprocess.run(
            ["bash", "-lc", process_group_state_command(zombie.pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert absent.returncode == 0
        assert absent.stdout == "MISSING\n"
    finally:
        zombie.wait(timeout=2)

    live = subprocess.Popen(
        ["setsid", "sleep", "30"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        present = subprocess.run(
            ["bash", "-lc", process_group_state_command(live.pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert present.returncode == 0
        assert present.stdout == "RUNNING\n"
    finally:
        os.killpg(live.pid, signal.SIGKILL)
        live.wait(timeout=2)


@pytest.mark.parametrize("kind", ["benchmark", "server"])
def test_monitored_start_uses_trusted_setsid_and_returns_group_ready(tmp_path, kind):
    real_setsid = shutil.which("setsid", path=os.defpath)
    assert real_setsid is not None
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "setsid"
    shim.write_text(
        "#!/bin/bash\nsleep 0.2\nexec \"$REAL_SETSID\" \"$@\"\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    env = os.environ.copy()
    env.update({"PATH": f"{bin_dir}:{env['PATH']}", "REAL_SETSID": real_setsid})

    class LocalContainer(Container):
        def exec(self, command, *, timeout=None):
            completed = subprocess.run(
                ["bash", "-c", command],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return CommandResult(
                completed.returncode, completed.stdout, completed.stderr
            )

    container = object.__new__(LocalContainer)
    pid_path = tmp_path / f"{kind}.pid"
    log_path = tmp_path / f"{kind}.log"
    status_path = tmp_path / f"{kind}.status"
    started_at = time.monotonic()
    if kind == "benchmark":
        result = container.start_monitored(
            "sleep 30",
            str(log_path),
            str(status_path),
            str(pid_path),
            timeout=3,
        )
    else:
        result = container.start_server_monitored(
            "sleep 30", str(log_path), str(pid_path), timeout=3
        )
    assert result.ok, result.stderr
    pid = int(result.stdout.strip())
    try:
        assert time.monotonic() - started_at < 0.18
        assert os.getpgid(pid) == pid
    finally:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize("kind", ["benchmark", "server"])
def test_monitored_start_pid_publication_failure_kills_started_group(tmp_path, kind):
    real_mv = shutil.which("mv", path=os.defpath)
    assert real_mv is not None
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mv_shim = bin_dir / "mv"
    mv_shim.write_text(
        "#!/bin/bash\nsleep 0.2\nexit 1\n",
        encoding="utf-8",
    )
    mv_shim.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    class LocalContainer(Container):
        def exec(self, command, *, timeout=None):
            completed = subprocess.run(
                ["bash", "-c", command],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return CommandResult(
                completed.returncode, completed.stdout, completed.stderr
            )

    container = object.__new__(LocalContainer)
    marker = tmp_path / f"{kind}.started-pid"
    pid_path = tmp_path / f"{kind}.pid"
    log_path = tmp_path / f"{kind}.log"
    command = f"bash -c 'echo $$ > {marker}; sleep 30'"
    if kind == "benchmark":
        result = container.start_monitored(
            command,
            str(log_path),
            str(tmp_path / "benchmark.status"),
            str(pid_path),
            timeout=3,
        )
    else:
        result = container.start_server_monitored(
            command, str(log_path), str(pid_path), timeout=3
        )

    assert not result.ok
    assert not pid_path.exists()
    assert marker.exists(), result.stderr
    started_pid = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(started_pid, 0)


def test_checked_process_group_cleanup_kills_parent_and_different_cmdline_child(
    tmp_path,
):
    child_pid_path = tmp_path / "child.pid"
    script = (
        "import pathlib, signal, subprocess, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "child=subprocess.Popen(['sleep', '30']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    class LocalProcessGroupContainer:
        def __init__(self) -> None:
            self.signals: list[str] = []

        def monitored_pid(self, pid_path, *, timeout):
            del pid_path, timeout
            return CommandResult(0, f"{parent.pid}\n", "")

        def process_group_state(self, pid, *, timeout):
            del timeout
            completed = subprocess.run(
                ["bash", "-lc", process_group_state_command(pid)],
                capture_output=True,
                text=True,
                check=False,
            )
            return CommandResult(
                completed.returncode, completed.stdout, completed.stderr
            )

        def signal_process_group(self, pid, signal_name, *, timeout):
            del timeout
            self.signals.append(signal_name)
            try:
                os.killpg(pid, getattr(signal, f"SIG{signal_name}"))
            except ProcessLookupError:
                pass
            return CommandResult(0, "", "")

    container = LocalProcessGroupContainer()
    try:
        for _ in range(100):
            if child_pid_path.exists():
                break
            time.sleep(0.01)
        else:
            pytest.fail("server parent did not spawn its differently-named child")

        failures = cleanup_monitored_benchmark(
            container,
            pid_path="/remote/server.pid",
            status_path="/remote/server.status",
            poll_interval_s=0.01,
            terminate_grace_s=0.05,
            poll_timeout_s=1,
        )

        assert failures == []
        assert container.signals == ["TERM", "KILL"]
        state = subprocess.run(
            ["bash", "-lc", process_group_state_command(parent.pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert state.stdout == "MISSING\n"
    finally:
        try:
            os.killpg(parent.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        parent.wait(timeout=2)
