"""Offline consistency tests for the phase-2 executor runners (no ssh/docker/network)."""

from __future__ import annotations

import json
import shlex

from runners.container import Container, ContainerConfig
from runners.metrics import RunResult, parse_bench_text
from runners.ranker import candidate_goodput, data_health_check, passes_sla
from runners.readiness import wait_until_ready
from runners.remote import DEFAULT_SSH_OPTIONS, CommandResult, RemoteRunner
from schemas.job_spec import SLA

# --- metrics.parse_bench_text -------------------------------------------------


def _bench_line(**overrides) -> str:
    record = {
        "max_concurrency": 32,
        "completed": 100,
        "total_output_tokens": 51200,  # 512 tokens/request on average
        "request_throughput": 12.5,
        "output_throughput": 6400.0,
        "total_throughput": 7000.0,
        "mean_ttft_ms": 250.0,
        "p99_ttft_ms": 900.0,
        "mean_tpot_ms": 20.0,
        "p99_tpot_ms": 55.0,
        "duration": 8.0,
    }
    record.update(overrides)
    return json.dumps(record)


def test_parse_bench_text_maps_real_sglang_fields() -> None:
    text = "\n".join(["not-json-noise", _bench_line()])
    result = parse_bench_text(
        text, candidate_id="cand-a", concurrency=32, num_prompts=100
    )

    assert isinstance(result, RunResult)
    assert result.candidate_id == "cand-a"
    assert result.concurrency == 32
    assert result.status == "ok"
    assert result.completed == 100
    assert result.total_output_tokens == 51200
    # real field names carried straight through
    assert result.total_throughput == 7000.0
    assert result.mean_ttft_ms == 250.0
    assert result.p99_ttft_ms == 900.0
    assert result.mean_tpot_ms == 20.0
    assert result.p99_tpot_ms == 55.0
    assert result.duration == 8.0
    # derived values
    assert result.success_rate == 1.0  # 100 / 100
    assert result.avg_output_tokens == 512.0  # 51200 / 100


def test_parse_bench_text_success_rate_and_avg_with_partial_completion() -> None:
    text = _bench_line(completed=90, total_output_tokens=27000)
    result = parse_bench_text(
        text, candidate_id="cand-b", concurrency=32, num_prompts=100
    )
    assert result.success_rate == 0.9  # 90 / 100
    assert result.avg_output_tokens == 300.0  # 27000 / 90


def test_parse_bench_text_selects_record_matching_concurrency() -> None:
    text = "\n".join(
        [
            _bench_line(max_concurrency=1, total_throughput=1000.0),
            _bench_line(max_concurrency=32, total_throughput=7000.0),
            _bench_line(max_concurrency=64, total_throughput=5000.0),
        ]
    )
    result = parse_bench_text(
        text, candidate_id="cand-c", concurrency=32, num_prompts=100
    )
    assert result.total_throughput == 7000.0


def test_parse_bench_text_empty_is_bad_args() -> None:
    result = parse_bench_text("", candidate_id="cand-d", concurrency=8, num_prompts=50)
    assert result.status == "bad_args"
    assert result.completed == 0
    assert result.success_rate == 0.0


# --- ranker: SLA, health, goodput ---------------------------------------------


def _sla() -> SLA:
    return SLA(max_avg_ttft_ms=500.0, max_avg_tpot_ms=40.0, min_success_rate=0.99)


def _result(**overrides) -> RunResult:
    base = dict(
        candidate_id="c",
        concurrency=32,
        num_prompts=100,
        completed=100,
        success_rate=1.0,
        request_throughput=10.0,
        output_throughput=6000.0,
        total_throughput=7000.0,
        mean_ttft_ms=250.0,
        p99_ttft_ms=800.0,
        mean_tpot_ms=20.0,
        p99_tpot_ms=50.0,
        total_output_tokens=51200,
        avg_output_tokens=512.0,
        duration=8.0,
    )
    base.update(overrides)
    return RunResult(**base)


def test_passes_sla_true_when_within_all_gates() -> None:
    assert passes_sla(_result(), _sla()) is True


def test_passes_sla_false_on_ttft_tpot_or_success_rate() -> None:
    sla = _sla()
    assert passes_sla(_result(mean_ttft_ms=600.0), sla) is False
    assert passes_sla(_result(mean_tpot_ms=41.0), sla) is False
    assert passes_sla(_result(success_rate=0.5), sla) is False


def test_data_health_check_flags_truncated_and_empty() -> None:
    # output_len target 512; 0.9 * 512 = 460.8 threshold
    ok, reason = data_health_check(_result(avg_output_tokens=512.0), output_len=512)
    assert ok is True and reason is None

    bad, why = data_health_check(_result(avg_output_tokens=300.0), output_len=512)
    assert bad is False and "truncated" in why

    none_done, why2 = data_health_check(_result(completed=0), output_len=512)
    assert none_done is False and "no_completed" in why2


def test_candidate_goodput_takes_max_over_qualifying_runs() -> None:
    sla = _sla()
    runs = [
        _result(concurrency=8, total_throughput=2000.0),
        _result(concurrency=32, total_throughput=7000.0),  # best qualifying
        _result(concurrency=64, total_throughput=9000.0, mean_tpot_ms=99.0),  # fails SLA
        _result(concurrency=16, total_throughput=8500.0, avg_output_tokens=10.0),  # truncated
    ]
    assert candidate_goodput(runs, sla, output_len=512) == 7000.0


def test_candidate_goodput_normalizes_by_gpu_count_and_tp_size() -> None:
    """Per-GPU goodput = raw_throughput * (gpu_count / tp_size).

    A TP2 instance doing 1000 tok/s on an 8-GPU host -> 4 instances -> 4000
    A TP8 instance doing 1000 tok/s on an 8-GPU host -> 1 instance -> 1000
    """
    sla = _sla()
    # tp_size=2, raw=1000, gpu_count=8 -> per_host = 1000 * (8/2) = 4000
    runs_tp2 = [_result(concurrency=16, total_throughput=1000.0, tp_size=2)]
    assert candidate_goodput(runs_tp2, sla, output_len=512, gpu_count=8) == 4000.0

    # tp_size=8, raw=1000, gpu_count=8 -> per_host = 1000 * (8/8) = 1000
    runs_tp8 = [_result(concurrency=16, total_throughput=1000.0, tp_size=8)]
    assert candidate_goodput(runs_tp8, sla, output_len=512, gpu_count=8) == 1000.0

    # TP2 beats TP8 at same raw throughput (4000 > 1000)
    assert candidate_goodput(runs_tp2, sla, output_len=512, gpu_count=8) > candidate_goodput(
        runs_tp8, sla, output_len=512, gpu_count=8
    )


def test_candidate_goodput_zero_when_nothing_qualifies() -> None:
    sla = _sla()
    runs = [_result(success_rate=0.1), _result(avg_output_tokens=5.0)]
    assert candidate_goodput(runs, sla, output_len=512) == 0.0


# --- remote.RemoteRunner.build_ssh_argv ---------------------------------------


def test_build_ssh_argv_structure() -> None:
    calls: list = []

    def fake_runner(argv, **kwargs):  # never actually shells out
        calls.append((argv, kwargs))
        raise AssertionError("runner should not be invoked by build_ssh_argv")

    runner = RemoteRunner("user@host", runner=fake_runner)
    argv = runner.build_ssh_argv("nvidia-smi -L")

    assert argv[0] == "ssh"
    assert argv[-2] == "user@host"
    assert argv[-1] == "nvidia-smi -L"
    # default options preserved in order between "ssh" and the target
    assert tuple(argv[1:-2]) == DEFAULT_SSH_OPTIONS
    assert calls == []  # build only, no execution


def test_password_ssh_uses_protected_fd_and_never_places_secret_in_argv() -> None:
    import os
    import subprocess

    sentinel = "remote-password-sentinel"
    observed: dict[str, object] = {}

    def fake_runner(argv, **kwargs):
        observed["argv"] = list(argv)
        observed["password"] = os.read(kwargs["pass_fds"][0], 8192).decode()
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    runner = RemoteRunner("user@host", ssh_password=sentinel, runner=fake_runner)
    result = runner.run("echo ok")

    assert result.ok
    argv = observed["argv"]
    assert isinstance(argv, list)
    assert sentinel not in argv
    assert argv[:2] == ["sshpass", "-d"]
    assert observed["password"] == sentinel


def test_container_start_preserves_paths_with_spaces_unicode_and_single_quotes() -> None:
    commands: list[str] = []

    class RecordingRemote:
        def run(self, command: str, *, timeout=None) -> CommandResult:
            commands.append(command)
            return CommandResult(returncode=0, stdout="container-id\n", stderr="")

    model_host = "/models/模型 family's copy"
    model_container = "/container/模型 family's copy"
    outputs_host = "/outputs/本次 run's evidence"
    container = Container(
        RecordingRemote(),  # type: ignore[arg-type]
        ContainerConfig(
            image_ref="registry.example/sglang:test",
            name="candidate-c001",
            model_host_dir=model_host,
            model_container_path=model_container,
            outputs_host_dir=outputs_host,
        ),
    )

    result = container.start()

    assert result.ok
    argv = shlex.split(commands[0])
    mounts = [argv[index + 1] for index, token in enumerate(argv[:-1]) if token == "-v"]
    assert f"{model_host}:{model_container}" in mounts
    assert f"{outputs_host}:/workspace/outputs" in mounts


def test_run_local_uses_injected_runner_and_wraps_result() -> None:
    import subprocess

    def fake_runner(argv, **kwargs):
        assert argv == ["echo", "hi"]
        return subprocess.CompletedProcess(argv, 0, stdout="hi\n", stderr="")

    runner = RemoteRunner("user@host", runner=fake_runner)
    result = runner.run_local(["echo", "hi"])
    assert isinstance(result, CommandResult)
    assert result.ok is True
    assert result.returncode == 0
    assert result.stdout == "hi\n"


def test_remote_timeout_zero_means_no_deadline() -> None:
    import subprocess

    observed = {}

    def fake_runner(argv, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    runner = RemoteRunner("user@host", runner=fake_runner)
    result = runner.run_local(["long-benchmark"], timeout=0)

    assert result.ok
    assert observed["timeout"] is None


def test_readiness_stops_when_log_makes_no_progress() -> None:
    clock = {"now": 0.0}

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    ready = wait_until_ready(
        lambda: False,
        is_alive=lambda: True,
        timeout_s=100,
        stall_timeout_s=5,
        progress=lambda: "same-log-size",
        interval_s=1,
        sleep=sleep,
        now=lambda: clock["now"],
    )

    assert ready is False
    assert clock["now"] == 5
