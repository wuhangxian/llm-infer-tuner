from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from runners import executor as executor_module
from runners.preflight import build_preflight_plan

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

    def fake_run(config):
        captured.append(config)
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
    assert password not in repr(config)


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

    def fake_run(config):
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
    assert list(config.results_dir.iterdir()) == []


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
