from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from runners import executor as executor_module
from runners.preflight import build_preflight_plan
from runners.reporting import write_reports

REPO_ROOT = Path(__file__).resolve().parents[1]


def _job_payload(*, max_candidates: int = 1, baseline: bool = False) -> dict[str, object]:
    return {
        "job_id": "executor-cli-test",
        "engine": "sglang",
        "gpu_model": "gpu-test",
        "gpu_count": 2,
        "gpu_memory_gb": 16.0,
        "model": "model-test",
        "image": "image-test",
        "workload": "workload-test",
        "benchmark_method": "benchmark-test",
        "sla": {"max_avg_ttft_ms": 100.0, "max_avg_tpot_ms": 20.0},
        "search": {
            "max_candidates": max_candidates,
            **({"baseline": {}} if baseline else {}),
        },
    }


def _target_payload() -> dict[str, object]:
    return {
        "gpu_model": "gpu-test",
        "gpu_count": 2,
        "gpu_memory_gb": 16.0,
        "ssh_target": "runner@example.test",
        "model_host_dir": "/models/example",
        "model_container_path": "/container/models/example",
        "image_ref": "registry.example/sglang:test",
        "port": 30000,
    }


def _candidate(candidate_id: str, *, baseline: bool = False) -> dict[str, object]:
    return {
        "id": candidate_id,
        "params": {
            "tp_size": 1,
            "mem_fraction_static": 0.8 if baseline else 0.81,
            **({"is_baseline": True} if baseline else {}),
        },
        "reasons": ["test"],
    }


@dataclass
class ExecutorScriptProject:
    root: Path
    job: Path
    target: Path
    configs: Path
    argv_file: Path
    snapshot_dir: Path
    env: dict[str, str]

    def invoke(
        self, *args: str, env_overrides: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        if self.argv_file.exists():
            self.argv_file.unlink()
        env = self.env.copy()
        env.update(env_overrides or {})
        return subprocess.run(
            ["./run_executor.sh", *args],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    @property
    def runner_called(self) -> bool:
        return self.argv_file.exists()


@pytest.fixture
def executor_script_project(tmp_path: Path) -> ExecutorScriptProject:
    shutil.copy2(REPO_ROOT / "run_executor.sh", tmp_path / "run_executor.sh")
    (tmp_path / "run_executor.sh").chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_file = tmp_path / "uv.argv"
    snapshot_dir = tmp_path / "uv-snapshot"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" > "$FAKE_UV_ARGV_FILE"\n'
        'mkdir -p "$FAKE_UV_SNAPSHOT_DIR"\n'
        'previous=""\n'
        'for arg in "$@"; do\n'
        '  case "$previous" in\n'
        '    --job) label=job ;;\n'
        '    --target) label=target ;;\n'
        '    --configs) label=configs ;;\n'
        '    *) label="" ;;\n'
        '  esac\n'
        '  if [ -n "$label" ]; then\n'
        '    printf "%s" "$arg" > "$FAKE_UV_SNAPSHOT_DIR/$label.path"\n'
        '    cp -- "$arg" "$FAKE_UV_SNAPSHOT_DIR/$label.content"\n'
        '    stat -c %a -- "$(dirname -- "$arg")" > "$FAKE_UV_SNAPSHOT_DIR/$label.dir_mode"\n'
        '  fi\n'
        '  previous="$arg"\n'
        'done\n'
        'if [ "${FAKE_UV_MODE:-}" = "block" ]; then\n'
        '  printf "%s" "$$" > "$FAKE_UV_PID_FILE"\n'
        '  : > "$FAKE_UV_READY_FILE"\n'
        '  trap "exit 130" INT\n'
        '  trap "exit 143" TERM\n'
        '  sleep 30\n'
        'fi\n'
        'if [ "${FAKE_UV_MODE:-}" = "early-unhandled" ]; then\n'
        '  printf "%s" "$$" > "$FAKE_UV_PID_FILE"\n'
        '  : > "$FAKE_UV_READY_FILE"\n'
        '  exec sleep 30\n'
        'fi\n'
        'if [ "${FAKE_UV_MODE:-}" = "stubborn-child" ]; then\n'
        '  printf "%s" "$$" > "$FAKE_UV_PID_FILE"\n'
        '  trap "exit 130" INT\n'
        '  trap "exit 143" TERM\n'
        '  (\n'
        '    trap "" INT TERM\n'
        '    printf "%s" "$BASHPID" > "$FAKE_UV_CHILD_PID_FILE"\n'
        '    : > "$FAKE_UV_READY_FILE"\n'
        '    while :; do sleep 30; done\n'
        '  ) &\n'
        '  wait\n'
        'fi\n'
        'if [ "${FAKE_UV_MODE:-}" = "leader-exits-slow-child" ]; then\n'
        '  printf "%s" "$$" > "$FAKE_UV_PID_FILE"\n'
        '  trap "exit 143" TERM\n'
        '  (\n'
        "    trap 'sleep 3.2; exit 0' TERM\n"
        '    printf "%s" "$BASHPID" > "$FAKE_UV_CHILD_PID_FILE"\n'
        '    : > "$FAKE_UV_READY_FILE"\n'
        '    while :; do sleep 30; done\n'
        '  ) &\n'
        '  wait\n'
        'fi\n'
        'if [ "${FAKE_UV_MODE:-}" = "slow-cleanup" ]; then\n'
        '  printf "%s" "$$" > "$FAKE_UV_PID_FILE"\n'
        '  : > "$FAKE_UV_READY_FILE"\n'
        "  trap 'printf done > \"$FAKE_UV_CLEANUP_MARKER\"; "
        "if [ -n \"${FAKE_UV_SIGNAL_MANIFEST:-}\" ]; then "
        "printf '{\"run_id\":\"after-first-signal\",\"ranking_status\":\"PROVISIONAL\"}\\n' "
        "> \"$FAKE_UV_SIGNAL_MANIFEST\"; fi; sleep 3.2; exit 143' TERM\n"
        "  trap 'printf done > \"$FAKE_UV_CLEANUP_MARKER\"; sleep 3.2; exit 130' INT\n"
        '  while :; do sleep 30; done\n'
        'fi\n'
        'exit "${FAKE_UV_EXIT_CODE:-0}"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    job = tmp_path / "job.json"
    target = tmp_path / "target.json"
    configs = tmp_path / "configs.jsonl"
    job.write_text(json.dumps(_job_payload()), encoding="utf-8")
    target.write_text(json.dumps(_target_payload()), encoding="utf-8")
    configs.write_text(json.dumps(_candidate("c001")) + "\n", encoding="utf-8")
    default_configs = tmp_path / "outputs/executor-cli-test/configs.jsonl"
    default_configs.parent.mkdir(parents=True)
    shutil.copy2(configs, default_configs)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "FAKE_UV_ARGV_FILE": str(argv_file),
            "FAKE_UV_SNAPSHOT_DIR": str(snapshot_dir),
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    return ExecutorScriptProject(tmp_path, job, target, configs, argv_file, snapshot_dir, env)


def test_extra_positional_argument_is_rejected_before_runner(
    executor_script_project: ExecutorScriptProject,
) -> None:
    project = executor_script_project

    completed = project.invoke(
        str(project.job),
        str(project.target),
        str(project.configs),
        str(project.root / "results"),
        "unexpected-fifth-argument",
    )

    assert completed.returncode != 0
    assert not project.runner_called
    assert "用法" in completed.stderr


@pytest.mark.parametrize(
    "args",
    [(), ("unknown-mode.txt",)],
    ids=["zero-arguments", "unknown-single-file-mode"],
)
def test_unknown_modes_are_rejected_before_runner(
    executor_script_project: ExecutorScriptProject, args: tuple[str, ...]
) -> None:
    completed = executor_script_project.invoke(*args)

    assert completed.returncode != 0
    assert not executor_script_project.runner_called
    assert "用法" in completed.stderr


@pytest.mark.parametrize("argument_count", [2, 3, 4])
def test_split_mode_accepts_exactly_two_to_four_arguments(
    executor_script_project: ExecutorScriptProject, argument_count: int
) -> None:
    project = executor_script_project
    args = [str(project.job), str(project.target)]
    if argument_count >= 3:
        args.append(str(project.configs))
    if argument_count == 4:
        args.append(str(project.root / "custom results"))

    completed = project.invoke(*args)

    assert completed.returncode == 0, completed.stderr
    assert project.runner_called


@pytest.mark.parametrize(
    ("environment", "expected_mode"),
    [
        ({}, "full_host"),
        ({"MEASUREMENT_MODE": "estimated"}, "estimated"),
        ({"FILL_HOST": "1"}, "full_host"),
        ({"FILL_HOST": "true"}, "full_host"),
        ({"FILL_HOST": "0"}, "estimated"),
        ({"FILL_HOST": "false"}, "estimated"),
    ],
)
def test_shell_resolves_default_explicit_and_legacy_measurement_modes(
    executor_script_project: ExecutorScriptProject,
    environment: dict[str, str],
    expected_mode: str,
) -> None:
    project = executor_script_project

    completed = project.invoke(
        str(project.job),
        str(project.target),
        env_overrides=environment,
    )

    assert completed.returncode == 0, completed.stderr
    argv = project.argv_file.read_text(encoding="utf-8").splitlines()
    mode_index = argv.index("--measurement-mode")
    assert argv[mode_index + 1] == expected_mode


@pytest.mark.parametrize(
    "environment",
    [
        {"MEASUREMENT_MODE": "estimated", "FILL_HOST": "true"},
        {"MEASUREMENT_MODE": "full_host", "FILL_HOST": "false"},
        {"MEASUREMENT_MODE": "invalid"},
        {"FILL_HOST": "invalid"},
    ],
)
def test_shell_rejects_invalid_or_conflicting_measurement_modes_before_runner(
    executor_script_project: ExecutorScriptProject,
    environment: dict[str, str],
) -> None:
    completed = executor_script_project.invoke(
        str(executor_script_project.job),
        str(executor_script_project.target),
        env_overrides=environment,
    )

    assert completed.returncode != 0
    assert executor_script_project.runner_called is False


def test_jsonl_bundle_infers_baseline_count_and_never_exposes_plaintext_secret(
    executor_script_project: ExecutorScriptProject,
) -> None:
    project = executor_script_project
    sentinel = "executor-cli-password-sentinel"
    bundle = project.root / "bundle.jsonl"
    meta = {
        "job_id": "bundle-test",
        "gpu_model": "gpu-test",
        "gpu_count": 2,
        "gpu_memory_gb": 16.0,
        "workload": "workload-test",
        "benchmark_method": "benchmark-test",
        "sla": {"max_avg_ttft_ms": 100.0, "max_avg_tpot_ms": 20.0},
        **_target_payload(),
        "allow_cross_numa": True,
        "ssh_password": sentinel,
    }
    candidates = [
        _candidate("baseline", baseline=True),
        _candidate("c001"),
        {
            **_candidate("c002"),
            "params": {"tp_size": 1, "mem_fraction_static": 0.82},
        },
    ]
    bundle.write_text(
        json.dumps({"_meta": meta, "candidates": candidates}), encoding="utf-8"
    )

    completed = project.invoke(str(bundle))

    assert completed.returncode == 0, completed.stderr
    argv_text = project.argv_file.read_text(encoding="utf-8")
    assert "--target" in argv_text
    assert "--ssh-password" not in argv_text
    assert sentinel not in argv_text + completed.stdout + completed.stderr
    generated_job = json.loads(
        (project.snapshot_dir / "job.content").read_text(encoding="utf-8")
    )
    assert generated_job["search"]["max_candidates"] == 2
    assert generated_job["search"]["baseline"] == {}
    generated_target = json.loads(
        (project.snapshot_dir / "target.content").read_text(encoding="utf-8")
    )
    assert generated_target["allow_cross_numa"] is True
    for label in ("job", "target", "configs"):
        content = (project.snapshot_dir / f"{label}.content").read_text(encoding="utf-8")
        assert sentinel not in content
        assert (project.snapshot_dir / f"{label}.dir_mode").read_text().strip() == "700"
        original_path = Path(
            (project.snapshot_dir / f"{label}.path").read_text(encoding="utf-8")
        )
        assert not original_path.exists()


def test_bundle_temporary_files_are_removed_when_runner_fails(
    executor_script_project: ExecutorScriptProject,
) -> None:
    project = executor_script_project
    bundle = project.root / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "_meta": {
                    "job_id": "bundle-failure",
                    "workload": "workload-test",
                    "benchmark_method": "benchmark-test",
                    "sla": {"max_avg_ttft_ms": 100.0, "max_avg_tpot_ms": 20.0},
                    **_target_payload(),
                },
                "candidates": [_candidate("c001")],
            }
        ),
        encoding="utf-8",
    )

    completed = project.invoke(str(bundle), env_overrides={"FAKE_UV_EXIT_CODE": "42"})

    assert completed.returncode == 42
    for label in ("job", "target", "configs"):
        original_path = Path((project.snapshot_dir / f"{label}.path").read_text())
        assert not original_path.exists()


@pytest.mark.parametrize(
    ("signal_number", "expected_code"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
    ids=["interrupt", "terminate"],
)
def test_bundle_temporary_files_are_removed_on_signal(
    executor_script_project: ExecutorScriptProject,
    signal_number: signal.Signals,
    expected_code: int,
) -> None:
    project = executor_script_project
    bundle = project.root / "signal-bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "_meta": {
                    "job_id": "signal-bundle",
                    "workload": "workload-test",
                    "benchmark_method": "benchmark-test",
                    "sla": {"max_avg_ttft_ms": 100.0, "max_avg_tpot_ms": 20.0},
                    **_target_payload(),
                },
                "candidates": [_candidate("c001")],
            }
        ),
        encoding="utf-8",
    )
    ready = project.root / "uv.ready"
    pid_file = project.root / "uv.pid"
    env = project.env.copy()
    env.update(
        {
            "FAKE_UV_MODE": "block",
            "FAKE_UV_READY_FILE": str(ready),
            "FAKE_UV_PID_FILE": str(pid_file),
        }
    )
    process = subprocess.Popen(
        ["./run_executor.sh", str(bundle)],
        cwd=project.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.02)
    assert ready.exists(), "fake runner did not start"
    temp_paths = [
        Path((project.snapshot_dir / f"{label}.path").read_text())
        for label in ("job", "target", "configs")
    ]
    assert all(path.exists() for path in temp_paths)

    runner_pid = int(pid_file.read_text())
    os.kill(process.pid, signal_number)
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)
        pytest.fail("wrapper did not forward the signal to its runner")

    assert process.returncode == expected_code, (stdout, stderr)
    assert all(not path.exists() for path in temp_paths)
    with pytest.raises(ProcessLookupError):
        os.kill(runner_pid, 0)


def test_signal_kills_runner_descendants_that_ignore_term(
    executor_script_project: ExecutorScriptProject,
) -> None:
    project = executor_script_project
    bundle = project.root / "stubborn-signal-bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "_meta": {
                    "job_id": "stubborn-signal-bundle",
                    "workload": "workload-test",
                    "benchmark_method": "benchmark-test",
                    "sla": {"max_avg_ttft_ms": 100.0, "max_avg_tpot_ms": 20.0},
                    **_target_payload(),
                },
                "candidates": [_candidate("c001")],
            }
        ),
        encoding="utf-8",
    )
    ready = project.root / "uv-stubborn.ready"
    runner_pid_file = project.root / "uv-stubborn.pid"
    child_pid_file = project.root / "uv-stubborn-child.pid"
    env = project.env.copy()
    env.update(
        {
            "FAKE_UV_MODE": "stubborn-child",
            "FAKE_UV_READY_FILE": str(ready),
            "FAKE_UV_PID_FILE": str(runner_pid_file),
            "FAKE_UV_CHILD_PID_FILE": str(child_pid_file),
            "LLM_INFER_TUNER_SHUTDOWN_GRACE_SECONDS": "0.2",
        }
    )
    process = subprocess.Popen(
        ["./run_executor.sh", str(bundle)],
        cwd=project.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.02)
    assert ready.exists(), "stubborn fake runner did not start"
    runner_pid = int(runner_pid_file.read_text())
    child_pid = int(child_pid_file.read_text())
    temp_paths = [
        Path((project.snapshot_dir / f"{label}.path").read_text())
        for label in ("job", "target", "configs")
    ]

    os.kill(process.pid, signal.SIGTERM)
    try:
        stdout, stderr = process.communicate(timeout=7)
    except subprocess.TimeoutExpired:
        for process_group in (runner_pid, process.pid):
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.communicate(timeout=5)
        pytest.fail("wrapper leaked a runner descendant that ignored SIGTERM")

    assert process.returncode == 143, (stdout, stderr)
    assert all(not path.exists() for path in temp_paths)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and Path(f"/proc/{child_pid}").exists():
        time.sleep(0.02)
    assert not Path(f"/proc/{child_pid}").exists()


def test_wrapper_allows_cleanup_past_three_seconds_and_ignores_second_signal(
    executor_script_project: ExecutorScriptProject,
) -> None:
    project = executor_script_project
    results = project.root / "slow-signal-results"
    results.mkdir()
    (results / "report_manifest.json").write_text(
        '{"run_id":"old-final","ranking_status":"FINAL"}\n', encoding="utf-8"
    )
    ready = project.root / "uv-slow.ready"
    runner_pid_file = project.root / "uv-slow.pid"
    cleanup_marker = project.root / "uv-slow.cleanup"
    signal_manifest = results / "report_manifest.json"
    env = project.env.copy()
    env.update(
        {
            "FAKE_UV_MODE": "slow-cleanup",
            "FAKE_UV_READY_FILE": str(ready),
            "FAKE_UV_PID_FILE": str(runner_pid_file),
            "FAKE_UV_CLEANUP_MARKER": str(cleanup_marker),
            "FAKE_UV_SIGNAL_MANIFEST": str(signal_manifest),
            "LLM_INFER_TUNER_SHUTDOWN_GRACE_SECONDS": "20",
        }
    )
    process = subprocess.Popen(
        [
            "./run_executor.sh",
            str(project.job),
            str(project.target),
            str(project.configs),
            str(results),
        ],
        cwd=project.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.02)
    assert ready.exists(), "slow-cleanup runner did not start"

    started = time.monotonic()
    os.kill(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not cleanup_marker.exists():
        time.sleep(0.02)
    assert cleanup_marker.exists(), "runner did not enter its cleanup handler"
    os.kill(process.pid, signal.SIGINT)
    stdout, stderr = process.communicate(timeout=7)

    assert process.returncode == 143, (stdout, stderr)
    elapsed = time.monotonic() - started
    assert 3.0 <= elapsed < 5.0
    runner_pid = int(runner_pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(runner_pid, 0)
    # The fake runner publishes a fresh provisional pointer while handling the
    # first TERM. A repeated signal must not move that current generation.
    assert signal_manifest.exists()
    stale = list(results.glob(".report_manifest.stale.wrapper.*"))
    assert len(stale) == 1


def test_wrapper_returns_when_slow_descendant_cleanup_finishes_before_grace(
    executor_script_project: ExecutorScriptProject,
) -> None:
    project = executor_script_project
    ready = project.root / "uv-slow-child.ready"
    runner_pid_file = project.root / "uv-slow-child.pid"
    child_pid_file = project.root / "uv-slow-child-child.pid"
    env = project.env.copy()
    env.update(
        {
            "FAKE_UV_MODE": "leader-exits-slow-child",
            "FAKE_UV_READY_FILE": str(ready),
            "FAKE_UV_PID_FILE": str(runner_pid_file),
            "FAKE_UV_CHILD_PID_FILE": str(child_pid_file),
            "LLM_INFER_TUNER_SHUTDOWN_GRACE_SECONDS": "20",
        }
    )
    process = subprocess.Popen(
        ["./run_executor.sh", str(project.job), str(project.target), str(project.configs)],
        cwd=project.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.02)
    assert ready.exists()

    started = time.monotonic()
    os.kill(process.pid, signal.SIGTERM)
    try:
        stdout, stderr = process.communicate(timeout=6)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)
        pytest.fail("wrapper waited for the full shutdown grace after its group exited")

    assert process.returncode == 143, (stdout, stderr)
    assert 3.0 <= time.monotonic() - started < 5.0
    for pid_file in (runner_pid_file, child_pid_file):
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)


def test_early_sigint_before_python_handler_is_not_inherited_as_ignored(
    executor_script_project: ExecutorScriptProject,
) -> None:
    project = executor_script_project
    ready = project.root / "uv-early-int.ready"
    runner_pid_file = project.root / "uv-early-int.pid"
    env = project.env.copy()
    env.update(
        {
            "FAKE_UV_MODE": "early-unhandled",
            "FAKE_UV_READY_FILE": str(ready),
            "FAKE_UV_PID_FILE": str(runner_pid_file),
        }
    )
    process = subprocess.Popen(
        ["./run_executor.sh", str(project.job), str(project.target), str(project.configs)],
        cwd=project.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.02)
    assert ready.exists()

    started = time.monotonic()
    os.kill(process.pid, signal.SIGINT)
    stdout, stderr = process.communicate(timeout=3)

    assert process.returncode == 130, (stdout, stderr)
    assert time.monotonic() - started < 2
    runner_pid = int(runner_pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(runner_pid, 0)


def test_early_sigint_revokes_previous_report_manifest(
    executor_script_project: ExecutorScriptProject,
) -> None:
    project = executor_script_project
    results = project.root / "early-signal-results"
    results.mkdir()
    old_manifest = results / "report_manifest.json"
    old_manifest.write_text(
        json.dumps(
            {
                "report_schema_version": 2,
                "run_id": "old-final",
                "ranking_status": "FINAL",
            }
        ),
        encoding="utf-8",
    )
    ready = project.root / "uv-early-manifest.ready"
    runner_pid_file = project.root / "uv-early-manifest.pid"
    env = project.env.copy()
    env.update(
        {
            "FAKE_UV_MODE": "early-unhandled",
            "FAKE_UV_READY_FILE": str(ready),
            "FAKE_UV_PID_FILE": str(runner_pid_file),
        }
    )
    process = subprocess.Popen(
        [
            "./run_executor.sh",
            str(project.job),
            str(project.target),
            str(project.configs),
            str(results),
        ],
        cwd=project.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.02)
    assert ready.exists(), "fake runner did not start"

    os.kill(process.pid, signal.SIGINT)
    stdout, stderr = process.communicate(timeout=3)

    assert process.returncode == 130, (stdout, stderr)
    assert not old_manifest.exists()
    stale = list(results.glob(".report_manifest.stale.wrapper.*"))
    assert len(stale) == 1
    assert "old-final" in stale[0].read_text(encoding="utf-8")
    status = json.loads((results / "task_status.json").read_text(encoding="utf-8"))
    assert status["task_status"] == "INTERRUPTED"
    assert status["ranking_status"] == "PROVISIONAL"
    assert status["interrupted"] is True


def test_runner_exits_before_process_group_probe_publishes_incomplete_status(
    executor_script_project: ExecutorScriptProject,
) -> None:
    """A pre-Python crash must not leave the wrapper status at RUNNING."""
    project = executor_script_project
    results = project.root / "early-exit-results"
    results.mkdir()
    old_manifest = results / "report_manifest.json"
    old_manifest.write_text(
        json.dumps(
            {
                "report_schema_version": 2,
                "run_id": "old-final",
                "ranking_status": "FINAL",
            }
        ),
        encoding="utf-8",
    )

    completed = project.invoke(
        str(project.job),
        str(project.target),
        str(project.configs),
        str(results),
        env_overrides={"FAKE_UV_EXIT_CODE": "7"},
    )

    assert completed.returncode == 7, (completed.stdout, completed.stderr)
    status = json.loads(
        (results / "task_status.json").read_text(encoding="utf-8")
    )
    assert status["task_status"] == "INCOMPLETE"
    assert status["ranking_status"] == "PROVISIONAL"
    assert not old_manifest.exists()
    assert list(results.glob(".report_manifest.stale.wrapper.*"))


def test_nonzero_runner_cannot_leave_child_published_final_manifest(
    executor_script_project: ExecutorScriptProject,
) -> None:
    """A post-report child failure must revoke its newly published FINAL pointer."""

    project = executor_script_project
    results = project.root / "child-final-results"
    fake_uv = project.root / "bin" / "uv"
    fake_uv.write_text(
        "#!/bin/bash\n"
        "results=''\n"
        "previous=''\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$previous\" = \"--results\" ]; then results=\"$arg\"; fi\n"
        "  previous=\"$arg\"\n"
        "done\n"
        "mkdir -p -- \"$results\"\n"
        "printf '%s\\n' "
        "'{\"report_schema_version\":2,\"run_id\":\"child-final\","
        "\"task_status\":\"COMPLETED\",\"ranking_status\":\"FINAL\","
        "\"interrupted\":false}' > \"$results/task_status.json\"\n"
        "printf '%s\\n' "
        "'{\"report_schema_version\":2,\"run_id\":\"child-final\","
        "\"generation_id\":\"00000000000000000000000000000000\","
        "\"snapshot_id\":\"00000000000000000000000000000000\","
        "\"files\":{}}' > \"$results/report_manifest.json\"\n"
        "exit 7\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    completed = project.invoke(
        str(project.job),
        str(project.target),
        str(project.configs),
        str(results),
    )

    assert completed.returncode == 7, completed.stderr
    assert not (results / "report_manifest.json").exists()
    assert list(results.glob(".report_manifest.stale.wrapper.*"))
    status = json.loads((results / "task_status.json").read_text(encoding="utf-8"))
    assert status["task_status"] == "INCOMPLETE"
    assert status["ranking_status"] == "PROVISIONAL"


def test_nonzero_runner_uses_manifest_generation_not_loose_status(
    executor_script_project: ExecutorScriptProject,
) -> None:
    """A lagging loose status must not hide a FINAL immutable generation."""

    from tests.test_report_invariants import _valid_final_payloads

    project = executor_script_project
    source = project.root / "source-final"
    source.mkdir()
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    write_reports(
        source,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="source-final",
    )
    results = project.root / "child-final-lagging-status-results"
    fake_uv = project.root / "bin" / "uv"
    fake_uv.write_text(
        "#!/bin/bash\n"
        "results=''\n"
        "previous=''\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$previous\" = \"--results\" ]; then results=\"$arg\"; fi\n"
        "  previous=\"$arg\"\n"
        "done\n"
        "mkdir -p -- \"$results\"\n"
        "cp -a -- \"$FAKE_FINAL_SOURCE/.report_generations\" \"$results/\"\n"
        "cp -- \"$FAKE_FINAL_SOURCE/report_manifest.json\" \"$results/\"\n"
        "printf '%s\\n' '{\"report_schema_version\":2,\"run_id\":\"source-final\","
        "\"task_status\":\"INCOMPLETE\",\"ranking_status\":\"PROVISIONAL\","
        "\"interrupted\":false}' > \"$results/task_status.json\"\n"
        "exit 7\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    completed = project.invoke(
        str(project.job),
        str(project.target),
        str(project.configs),
        str(results),
        env_overrides={"FAKE_FINAL_SOURCE": str(source)},
    )

    assert completed.returncode == 7, completed.stderr
    assert not (results / "report_manifest.json").exists()
    assert list(results.glob(".report_manifest.stale.wrapper.*"))
    status = json.loads((results / "task_status.json").read_text(encoding="utf-8"))
    assert status["task_status"] == "INCOMPLETE"
    assert status["ranking_status"] == "PROVISIONAL"


def test_signal_with_manifest_jq_read_failure_revokes_active_final(
    executor_script_project: ExecutorScriptProject,
) -> None:
    """A jq/manifest read error must fail closed instead of preserving FINAL."""

    from tests.test_report_invariants import _valid_final_payloads

    project = executor_script_project
    source = project.root / "jq-failure-source"
    source.mkdir()
    ranking, candidates, probes, task_status, provenance = _valid_final_payloads()
    write_reports(
        source,
        ranking=ranking,
        candidate_rows=candidates,
        probe_rows=probes,
        task_status=task_status,
        provenance=provenance,
        run_id="jq-failure-final",
    )

    results = project.root / "jq-failure-results"
    ready = project.root / "jq-failure.ready"
    fail_marker = project.root / "jq-failure.trigger"
    # jq is installed in the developer toolchain's user bin on some hosts,
    # so resolve it from the current PATH rather than only os.defpath.
    real_jq = shutil.which("jq")
    assert real_jq is not None
    fake_jq = project.root / "bin" / "jq"
    fake_jq.write_text(
        "#!/bin/bash\n"
        'if [ -e "$FAKE_JQ_FAIL_MARKER" ]; then\n'
        '  for arg in "$@"; do\n'
        '    [ "$arg" = "$FAKE_JQ_MANIFEST" ] && exit 9\n'
        "  done\n"
        "fi\n"
        f"exec {shlex.quote(real_jq)} \"$@\"\n",
        encoding="utf-8",
    )
    fake_jq.chmod(0o755)
    fake_uv = project.root / "bin" / "uv"
    fake_uv.write_text(
        "#!/bin/bash\n"
        'results=""; previous=""\n'
        'for arg in "$@"; do\n'
        '  if [ "$previous" = "--results" ]; then results="$arg"; fi\n'
        '  previous="$arg"\n'
        "done\n"
        'mkdir -p -- "$results"\n'
        'cp -a -- "$FAKE_FINAL_SOURCE/.report_generations" "$results/"\n'
        'cp -- "$FAKE_FINAL_SOURCE/report_manifest.json" "$results/"\n'
        'cp -- "$FAKE_FINAL_SOURCE/task_status.json" "$results/"\n'
        'touch -- "$FAKE_UV_READY"\n'
        "while :; do sleep 30; done\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = project.env.copy()
    env.update(
        {
            "FAKE_FINAL_SOURCE": str(source),
            "FAKE_UV_READY": str(ready),
            "FAKE_JQ_FAIL_MARKER": str(fail_marker),
            "FAKE_JQ_MANIFEST": str(results / "report_manifest.json"),
            "LLM_INFER_TUNER_SHUTDOWN_GRACE_SECONDS": "0.2",
        }
    )
    process = subprocess.Popen(
        [
            "./run_executor.sh",
            str(project.job),
            str(project.target),
            str(project.configs),
            str(results),
        ],
        cwd=project.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.02)
    assert ready.exists(), "fake runner did not publish a ready marker"
    fail_marker.touch()
    os.kill(process.pid, signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 143, (stdout, stderr)
    assert not (results / "report_manifest.json").exists()
    assert list(results.glob(".report_manifest.stale.wrapper.*"))


def test_manifest_move_failure_unlinks_active_pointer_fail_closed(
    executor_script_project: ExecutorScriptProject,
) -> None:
    """A failed ``mv`` must not leave an old active FINAL pointer behind."""

    project = executor_script_project
    results = project.root / "move-failure-results"
    results.mkdir()
    old_manifest = results / "report_manifest.json"
    old_manifest.write_text(
        json.dumps(
            {
                "report_schema_version": 2,
                "run_id": "old-final",
                "generation_id": "a" * 32,
                "snapshot_id": "a" * 32,
            }
        ),
        encoding="utf-8",
    )
    real_mv = shutil.which("mv", path=os.defpath)
    assert real_mv is not None
    mv_shim = project.root / "bin" / "mv"
    mv_shim.write_text(
        "#!/bin/bash\n"
        'for arg in "$@"; do\n'
        '  case "$arg" in *report_manifest.json) exit 1 ;;\n'
        "  esac\n"
        "done\n"
        f"exec {shlex.quote(real_mv)} \"$@\"\n",
        encoding="utf-8",
    )
    mv_shim.chmod(0o755)

    completed = project.invoke(
        str(project.job),
        str(project.target),
        str(project.configs),
        str(results),
        env_overrides={"FAKE_UV_EXIT_CODE": "7"},
    )

    assert completed.returncode == 7, completed.stderr
    assert not old_manifest.exists()
    assert list(results.glob(".report_manifest.stale.wrapper.*"))


def test_runner_success_without_report_manifest_is_not_left_running(
    executor_script_project: ExecutorScriptProject,
) -> None:
    """A zero exit from a crashed/mocked runner cannot masquerade as RUNNING."""
    project = executor_script_project
    results = project.root / "success-without-manifest-results"

    completed = project.invoke(
        str(project.job),
        str(project.target),
        str(project.configs),
        str(results),
    )

    # The shell retains the inner process exit code for compatibility, but the
    # status is fail-closed because no schema-v2 report was published.
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    status = json.loads(
        (results / "task_status.json").read_text(encoding="utf-8")
    )
    assert status["task_status"] == "INCOMPLETE"
    assert status["ranking_status"] == "PROVISIONAL"


def test_malformed_job_revoke_does_not_leave_previous_final_authoritative(
    executor_script_project: ExecutorScriptProject,
) -> None:
    project = executor_script_project
    results = project.root / "malformed-job-results"
    results.mkdir()
    old_manifest = results / "report_manifest.json"
    old_manifest.write_text(
        json.dumps(
            {
                "report_schema_version": 2,
                "run_id": "old-final",
                "ranking_status": "FINAL",
            }
        ),
        encoding="utf-8",
    )
    malformed_job = project.root / "malformed-job.json"
    malformed_job.write_text("{not valid json\n", encoding="utf-8")

    completed = project.invoke(
        str(malformed_job),
        str(project.target),
        str(project.configs),
        str(results),
    )

    assert completed.returncode != 0
    assert not old_manifest.exists()
    assert list(results.glob(".report_manifest.stale.wrapper.*"))
    status = json.loads(
        (results / "task_status.json").read_text(encoding="utf-8")
    )
    assert status["task_status"] == "INCOMPLETE"
    assert status["ranking_status"] == "PROVISIONAL"


def test_preparse_job_id_revokes_default_manifest_before_target_validation(
    executor_script_project: ExecutorScriptProject,
) -> None:
    """A job id different from its filename cannot hide an old FINAL report."""
    project = executor_script_project
    renamed_job = project.root / "renamed-input.json"
    payload = json.loads(project.job.read_text(encoding="utf-8"))
    payload["job_id"] = "preparse-id"
    renamed_job.write_text(json.dumps(payload), encoding="utf-8")
    results = project.root / "outputs/preparse-id/results"
    results.mkdir(parents=True)
    old_manifest = results / "report_manifest.json"
    old_manifest.write_text(
        json.dumps(
            {
                "report_schema_version": 2,
                "run_id": "old-final",
                "ranking_status": "FINAL",
            }
        ),
        encoding="utf-8",
    )
    missing_target = project.root / "missing-target.json"

    completed = project.invoke(str(renamed_job), str(missing_target))

    assert completed.returncode != 0
    assert not old_manifest.exists()
    assert list(results.glob(".report_manifest.stale.wrapper.*"))
    status = json.loads((results / "task_status.json").read_text(encoding="utf-8"))
    assert status["task_status"] == "INCOMPLETE"
    assert status["ranking_status"] == "PROVISIONAL"


def test_unsafe_job_id_cannot_escape_default_output_directory(
    executor_script_project: ExecutorScriptProject,
) -> None:
    project = executor_script_project
    unsafe_job = project.root / "unsafe-job.json"
    payload = json.loads(project.job.read_text(encoding="utf-8"))
    payload["job_id"] = "../escaped-output"
    unsafe_job.write_text(json.dumps(payload), encoding="utf-8")
    escaped_results = project.root / "escaped-output" / "results"

    completed = project.invoke(str(unsafe_job), str(project.target), str(project.configs))

    assert completed.returncode != 0
    assert not project.runner_called
    assert not (escaped_results / "report_manifest.json").exists()
    # The filename-derived safe fallback may receive a provisional wrapper
    # status, but no path outside ``outputs`` may be created.
    assert not escaped_results.exists()


def test_signal_during_fork_to_process_group_publication_is_deferred_and_cleaned(
    executor_script_project: ExecutorScriptProject,
) -> None:
    project = executor_script_project
    real_setsid = shutil.which("setsid", path=os.defpath)
    assert real_setsid is not None
    setsid_shim = project.root / "bin" / "setsid"
    setsid_shim.write_text(
        "#!/bin/bash\n"
        'kill -TERM "$PPID"\n'
        "sleep 0.1\n"
        'exec "$REAL_SETSID" "$@"\n',
        encoding="utf-8",
    )
    setsid_shim.chmod(0o755)
    ready = project.root / "uv-publication.ready"
    runner_pid_file = project.root / "uv-publication.pid"
    env = project.env.copy()
    env.update(
        {
            "REAL_SETSID": real_setsid,
            "FAKE_UV_MODE": "block",
            "FAKE_UV_READY_FILE": str(ready),
            "FAKE_UV_PID_FILE": str(runner_pid_file),
            "LLM_INFER_TUNER_SHUTDOWN_GRACE_SECONDS": "0.5",
        }
    )
    results = project.root / "early-signal-results"
    results.mkdir()
    (results / "task_status.json").write_text(
        json.dumps(
            {
                "task_status": "COMPLETED",
                "ranking_status": "FINAL",
                "interrupted": False,
                "cleanup_failures": ["stale cleanup failure"],
                "failure_type": "OldError",
                "job_id": "old-job",
                "sentinel": "stale",
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "./run_executor.sh",
            str(project.job),
            str(project.target),
            str(project.configs),
            str(results),
        ],
        cwd=project.root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=7,
    )

    assert completed.returncode == 143, completed.stderr
    status = json.loads((results / "task_status.json").read_text(encoding="utf-8"))
    assert status["task_status"] == "INTERRUPTED"
    assert status["ranking_status"] == "PROVISIONAL"
    assert status["interrupted"] is True
    assert status["signal"] == "SIGTERM"
    assert status["job_id"] == "executor-cli-test"
    assert status["cleanup_failures"] == []
    assert "sentinel" not in status
    if runner_pid_file.exists():
        runner_pid = int(runner_pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(runner_pid, 0)


@pytest.mark.parametrize("failing_tool", ["ps", "awk"])
def test_process_group_probe_error_still_uses_bounded_watchdog_and_kills_runner(
    executor_script_project: ExecutorScriptProject, failing_tool: str
) -> None:
    project = executor_script_project
    real_tool = shutil.which(failing_tool, path=os.defpath)
    assert real_tool is not None
    failure_marker = project.root / f"fail-{failing_tool}"
    shim = project.root / "bin" / failing_tool
    shim.write_text(
        "#!/bin/bash\n"
        'if [ -e "$FAKE_PROBE_FAILURE_MARKER" ]; then exit 2; fi\n'
        'exec "$REAL_PROBE_TOOL" "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    ready = project.root / f"uv-{failing_tool}.ready"
    runner_pid_file = project.root / f"uv-{failing_tool}.pid"
    env = project.env.copy()
    env.update(
        {
            "REAL_PROBE_TOOL": real_tool,
            "FAKE_PROBE_FAILURE_MARKER": str(failure_marker),
            "FAKE_UV_MODE": "block",
            "FAKE_UV_READY_FILE": str(ready),
            "FAKE_UV_PID_FILE": str(runner_pid_file),
            "LLM_INFER_TUNER_SHUTDOWN_GRACE_SECONDS": "0.2",
        }
    )
    process = subprocess.Popen(
        ["./run_executor.sh", str(project.job), str(project.target), str(project.configs)],
        cwd=project.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.02)
    assert ready.exists()
    runner_pid = int(runner_pid_file.read_text())
    failure_marker.touch()
    os.kill(process.pid, signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=4)

    assert process.returncode == 143, (stdout, stderr)
    with pytest.raises(ProcessLookupError):
        os.kill(runner_pid, 0)
    status_path = project.root / "outputs/executor-cli-test/results/task_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["task_status"] == "INTERRUPTED"
    assert any(
        "absence could not be proven" in failure
        for failure in status["cleanup_failures"]
    )


def test_python_executor_cli_loads_target_and_resolves_password_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job.json"
    configs = tmp_path / "configs.jsonl"
    target = tmp_path / "target.json"
    results = tmp_path / "results"
    job.write_text(json.dumps(_job_payload()), encoding="utf-8")
    configs.write_text(json.dumps(_candidate("c001")) + "\n", encoding="utf-8")
    password_env = "LLM_TUNER_TEST_RUNTIME_PASSWORD"
    password = "target-env-password-sentinel"
    target.write_text(
        json.dumps(
            {
                **_target_payload(),
                "ssh_password_env": password_env,
                "allow_cross_numa": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(password_env, password)
    captured: list[object] = []

    def fake_run(config, *, lifecycle=None):
        captured.append(config)
        assert lifecycle is not None
        return {"task_status": "COMPLETED"}

    monkeypatch.setattr(executor_module, "run_executor", fake_run)

    rc = executor_module.main(
        [
            "--job",
            str(job),
            "--target",
            str(target),
            "--configs",
            str(configs),
            "--results",
            str(results),
        ]
    )

    assert rc == 0
    assert len(captured) == 1
    config = captured[0]
    assert isinstance(config, executor_module.ExecutorConfig)
    assert config.ssh_target == "runner@example.test"
    assert config.ssh_password == password
    assert config.port == 30000
    assert config.allow_cross_numa is True
    assert config.measurement_mode == "full_host"
    assert config.fill_host is True
    assert password not in repr(config)


def test_python_executor_cli_accepts_explicit_estimated_mode_and_rejects_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job.json"
    configs = tmp_path / "configs.jsonl"
    target = tmp_path / "target.json"
    results = tmp_path / "results"
    job.write_text(json.dumps(_job_payload()), encoding="utf-8")
    configs.write_text(json.dumps(_candidate("c001")) + "\n", encoding="utf-8")
    target.write_text(json.dumps(_target_payload()), encoding="utf-8")
    captured: list[object] = []

    def fake_run(config, *, lifecycle=None):
        captured.append(config)
        return {"task_status": "COMPLETED"}

    monkeypatch.setattr(executor_module, "run_executor", fake_run)
    base_args = [
        "--job",
        str(job),
        "--target",
        str(target),
        "--configs",
        str(configs),
        "--results",
        str(results),
    ]

    assert executor_module.main([*base_args, "--measurement-mode", "estimated"]) == 0
    config = captured.pop()
    assert isinstance(config, executor_module.ExecutorConfig)
    assert config.measurement_mode == "estimated"
    assert config.fill_host is False
    with pytest.raises(SystemExit) as exc_info:
        executor_module.main(
            [*base_args, "--measurement-mode", "estimated", "--fill-host"]
        )
    assert exc_info.value.code == 2
    assert captured == []


def test_executor_config_defaults_to_full_host_and_rejects_legacy_conflicts(
    tmp_path: Path,
) -> None:
    required = {
        "job_path": tmp_path / "job.json",
        "configs_path": tmp_path / "configs.jsonl",
        "results_dir": tmp_path / "results",
        "ssh_target": "runner@example.test",
        "image_ref": "registry.example/sglang:test",
        "model_host_dir": "/models/example",
        "model_container_path": "/container/models/example",
        "project_root": REPO_ROOT,
    }

    default = executor_module.ExecutorConfig(**required)
    explicit_estimated = executor_module.ExecutorConfig(
        **required, measurement_mode="estimated"
    )
    legacy_estimated = executor_module.ExecutorConfig(**required, fill_host=False)

    assert default.measurement_mode == "full_host"
    assert default.fill_host is True
    assert explicit_estimated.measurement_mode == "estimated"
    assert explicit_estimated.fill_host is False
    assert legacy_estimated.measurement_mode == "estimated"
    assert legacy_estimated.fill_host is False
    with pytest.raises(ValueError, match="conflict"):
        executor_module.ExecutorConfig(
            **required,
            measurement_mode="estimated",
            fill_host=True,
        )


def test_python_cli_early_signal_invalidates_stale_final_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job.json"
    target = tmp_path / "target.json"
    configs = tmp_path / "configs.jsonl"
    results = tmp_path / "results"
    results.mkdir()
    stale_status = results / "task_status.json"
    stale_status.write_text(
        json.dumps({"task_status": "COMPLETED", "ranking_status": "FINAL"}),
        encoding="utf-8",
    )
    job.write_text(json.dumps(_job_payload()), encoding="utf-8")
    target.write_text(json.dumps(_target_payload()), encoding="utf-8")
    configs.write_text(json.dumps(_candidate("c001")) + "\n", encoding="utf-8")

    def interrupt_while_loading(_path):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("signal control flow must leave the loader")

    monkeypatch.setattr(executor_module, "load_job", interrupt_while_loading)

    rc = executor_module.main(
        [
            "--job",
            str(job),
            "--target",
            str(target),
            "--configs",
            str(configs),
            "--results",
            str(results),
        ]
    )

    assert rc == 143
    status = json.loads(stale_status.read_text(encoding="utf-8"))
    assert status["task_status"] == "INTERRUPTED"
    assert status["ranking_status"] == "PROVISIONAL"
    assert status["signal"] == "SIGTERM"


@pytest.mark.parametrize("password_value", [None, ""], ids=["missing", "empty"])
def test_python_executor_cli_rejects_missing_password_environment_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    password_value: str | None,
) -> None:
    job = tmp_path / "job.json"
    target = tmp_path / "target.json"
    configs = tmp_path / "configs.jsonl"
    job.write_text(json.dumps(_job_payload()), encoding="utf-8")
    configs.write_text(json.dumps(_candidate("c001")) + "\n", encoding="utf-8")
    variable = "LLM_TUNER_MISSING_PASSWORD"
    target.write_text(
        json.dumps({**_target_payload(), "ssh_password_env": variable}),
        encoding="utf-8",
    )
    if password_value is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, password_value)
    called = False

    def fake_run(config, *, lifecycle=None):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(executor_module, "run_executor", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        executor_module.main(
            [
                "--job",
                str(job),
                "--target",
                str(target),
                "--configs",
                str(configs),
                "--results",
                str(tmp_path / "results"),
            ]
        )

    assert exc_info.value.code == 2
    assert called is False


@pytest.mark.parametrize("base_port", [0, -1, 65536])
def test_port_span_rejects_invalid_base_port(base_port: int) -> None:
    with pytest.raises(ValueError, match="base port"):
        executor_module.validate_port_span(base_port, 1)


def test_port_span_rejects_overflow_and_accepts_last_single_port() -> None:
    with pytest.raises(ValueError, match="exceeds 65535"):
        executor_module.validate_port_span(65535, 2)

    assert executor_module.validate_port_span(65535, 1) == (65535, 65535)


def test_executor_rejects_exact_fill_host_port_overflow_before_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job.json"
    configs = tmp_path / "configs.jsonl"
    job.write_text(json.dumps(_job_payload(max_candidates=2)), encoding="utf-8")
    first = _candidate("c001")
    second = _candidate("c002")
    second["params"]["mem_fraction_static"] = 0.82  # type: ignore[index]
    configs.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )
    constructed = False

    class RecordingRemote:
        def __init__(self, *args, **kwargs) -> None:
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(executor_module, "RemoteRunner", RecordingRemote)
    monkeypatch.setattr(
        executor_module,
        "prepare_remote_host",
        lambda _remote, request: build_preflight_plan(
            request,
            numa_groups=((0, 1),),
        ),
    )
    monkeypatch.setattr(executor_module, "_load_output_len", lambda _workload: 1)
    monkeypatch.setattr(
        executor_module,
        "_load_num_prompts_multiplier",
        lambda _method, *, project_root: 1,
    )
    monkeypatch.setattr(
        executor_module,
        "_resolve_bench_template",
        lambda _job, *, config: "python -m sglang.bench_serving",
    )
    config = executor_module.ExecutorConfig(
        job_path=job,
        configs_path=configs,
        results_dir=tmp_path / "results",
        ssh_target="runner@example.test",
        image_ref="registry.example/sglang:test",
        model_host_dir="/models/example",
        model_container_path="/container/models/example",
        project_root=REPO_ROOT,
        port=65535,
        target_gpu_model="gpu-test",
        target_gpu_count=2,
        target_gpu_memory_gb=16.0,
        fill_host=True,
    )

    with pytest.raises(ValueError, match="exceeds 65535"):
        executor_module.run_executor(config)

    assert constructed is True
    assert config.results_dir.is_dir()
    # Task10 publishes a schema-v2 provisional tombstone before local
    # validation, so an early preflight failure leaves an auditable generation
    # rather than only the legacy loose status file.
    assert {
        path.name for path in config.results_dir.iterdir()
    } >= {
        "task_status.json",
        "ranking.json",
        "candidate_results.jsonl",
        "probe_results.jsonl",
        "provenance.json",
        "report_manifest.json",
        ".report_generations",
    }
    status = json.loads(
        (config.results_dir / "task_status.json").read_text(encoding="utf-8")
    )
    assert status["task_status"] == "INCOMPLETE"
    assert status["ranking_status"] == "PROVISIONAL"
    assert (config.results_dir / "ranking.json").exists()


def test_executor_rejects_impossible_tp_before_constructing_remote_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job.json"
    configs = tmp_path / "configs.jsonl"
    job.write_text(json.dumps(_job_payload()), encoding="utf-8")
    candidate = _candidate("c001")
    candidate["params"]["tp_size"] = 3  # type: ignore[index]
    configs.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    constructed = False

    class MustNotConstructRemote:
        def __init__(self, *args, **kwargs) -> None:
            nonlocal constructed
            constructed = True
            raise AssertionError("RemoteRunner must not be constructed")

    monkeypatch.setattr(executor_module, "RemoteRunner", MustNotConstructRemote)
    monkeypatch.setattr(executor_module, "_load_output_len", lambda _workload: 1)
    monkeypatch.setattr(
        executor_module,
        "_load_num_prompts_multiplier",
        lambda _method, *, project_root: 1,
    )
    monkeypatch.setattr(
        executor_module,
        "_resolve_bench_template",
        lambda _job, *, config: "python -m sglang.bench_serving",
    )
    config = executor_module.ExecutorConfig(
        job_path=job,
        configs_path=configs,
        results_dir=tmp_path / "results",
        ssh_target="runner@example.test",
        image_ref="registry.example/sglang:test",
        model_host_dir="/models/example",
        model_container_path="/container/models/example",
        project_root=REPO_ROOT,
        target_gpu_model="gpu-test",
        target_gpu_count=2,
        target_gpu_memory_gb=16.0,
    )

    with pytest.raises(ValueError, match="tp_size"):
        executor_module.run_executor(config)

    assert constructed is False


def test_executor_prepares_local_results_before_constructing_remote_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job.json"
    configs = tmp_path / "configs.jsonl"
    blocked_results = tmp_path / "results"
    job.write_text(json.dumps(_job_payload()), encoding="utf-8")
    configs.write_text(json.dumps(_candidate("c001")) + "\n", encoding="utf-8")
    blocked_results.write_text("not a directory", encoding="utf-8")
    constructed = False

    class MustNotConstructRemote:
        def __init__(self, *args, **kwargs) -> None:
            nonlocal constructed
            constructed = True
            raise AssertionError("RemoteRunner must not be constructed")

    monkeypatch.setattr(executor_module, "RemoteRunner", MustNotConstructRemote)
    monkeypatch.setattr(executor_module, "_load_output_len", lambda _workload: 1)
    monkeypatch.setattr(
        executor_module,
        "_load_num_prompts_multiplier",
        lambda _method, *, project_root: 1,
    )
    monkeypatch.setattr(
        executor_module,
        "_resolve_bench_template",
        lambda _job, *, config: "python -m sglang.bench_serving",
    )
    config = executor_module.ExecutorConfig(
        job_path=job,
        configs_path=configs,
        results_dir=blocked_results,
        ssh_target="runner@example.test",
        image_ref="registry.example/sglang:test",
        model_host_dir="/models/example",
        model_container_path="/container/models/example",
        project_root=REPO_ROOT,
        target_gpu_model="gpu-test",
        target_gpu_count=2,
        target_gpu_memory_gb=16.0,
    )

    with pytest.raises(RuntimeError, match="local results directory"):
        executor_module.run_executor(config)

    assert constructed is False
