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

REPO_ROOT = Path(__file__).resolve().parents[1]


def _process_is_alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    fields = stat.read_text().split()
    return len(fields) >= 3 and fields[2] != "Z"


def _wait_for_file(path: Path, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def _wait_for_pids_to_exit(pids: list[int], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_process_is_alive(pid) for pid in pids):
            return
        time.sleep(0.02)
    assert not [pid for pid in pids if _process_is_alive(pid)]


@dataclass
class ScriptProject:
    root: Path
    job: Path
    args_file: Path
    count_file: Path
    models_file: Path
    env: dict[str, str]

    def invoke(
        self, *argv: str, env_overrides: dict[str, str] | None = None
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        env = self.env.copy()
        env.update(env_overrides or {})
        for recorded_file in (self.args_file, self.count_file, self.models_file):
            if recorded_file.exists():
                recorded_file.unlink()
        completed = subprocess.run(
            ["./gen_configs.sh", *argv],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        recorded = self.args_file.read_text().splitlines() if self.args_file.exists() else []
        return completed, recorded

    def run(
        self,
        *extra_args: str,
        output_name: str = "result.jsonl",
        env_overrides: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        output = self.root / output_name
        return self.invoke(
            *extra_args,
            str(self.job),
            str(output),
            env_overrides=env_overrides,
        )


@pytest.fixture
def script_project(tmp_path: Path) -> ScriptProject:
    shutil.copy2(REPO_ROOT / "gen_configs.sh", tmp_path / "gen_configs.sh")
    runners_dir = tmp_path / "runners"
    runners_dir.mkdir()
    shutil.copy2(REPO_ROOT / "runners/tclaude_guard.py", runners_dir / "tclaude_guard.py")

    skill_dir = tmp_path / ".claude/skills/sglang-server-config-gen"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# test skill\n")

    job = tmp_path / "job.json"
    job.write_text(json.dumps({"job_id": "tclaude-cli-test"}))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "tclaude.args"
    count_file = tmp_path / "tclaude.count"
    models_file = tmp_path / "tclaude.models"


    real_jq = shutil.which("jq")
    assert real_jq is not None
    fake_jq = bin_dir / "jq"
    fake_jq.write_text(
        "#!/bin/bash\n"
        'if [ "${FAKE_JQ_MODE:-}" = "fail-render" ]; then\n'
        '  for arg in "$@"; do\n'
        '    if [ "$arg" = "--unbuffered" ]; then exit 55; fi\n'
        '  done\n'
        'fi\n'
        'if [ "${FAKE_JQ_MODE:-}" = "block-parse" ]; then\n'
        '  for arg in "$@"; do\n'
        '    case "$arg" in\n'
        '      *claude-raw-outputs/*.stdout.json)\n'
        '        : > "$FAKE_JQ_PARSE_MARKER"\n'
        '        sleep 30\n'
        '        exit 99\n'
        '        ;;\n'
        '    esac\n'
        '  done\n'
        'fi\n'
        f'exec {real_jq!r} "$@"\n'
    )
    fake_jq.chmod(0o755)

    candidate = {
        "id": "c001",
        "params": {"tp_size": 1, "mem_fraction_static": 0.8},
        "cmd": "python3 -m sglang.launch_server --model-path ${MODEL_PATH}",
        "reasons": ["test"],
    }
    payload = json.dumps({"structured_output": {"candidates": [candidate]}})
    stream_payload = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "structured_output": {"candidates": [candidate]},
        }
    )
    stream_init = json.dumps({"type": "system", "subtype": "init"})
    fake_tclaude = bin_dir / "tclaude"
    fake_tclaude.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" > "$FAKE_TCLAUDE_ARGS_FILE"\n'
        'attempt=1\n'
        'if [ -f "$FAKE_TCLAUDE_COUNT_FILE" ]; then '
        'attempt=$(( $(cat "$FAKE_TCLAUDE_COUNT_FILE") + 1 )); fi\n'
        'printf "%s" "$attempt" > "$FAKE_TCLAUDE_COUNT_FILE"\n'
        'previous=""\n'
        'for arg in "$@"; do\n'
        '  if [ "$previous" = "--model" ]; then '
        'printf "%s\\n" "$arg" >> "$FAKE_TCLAUDE_MODELS_FILE"; fi\n'
        '  previous="$arg"\n'
        'done\n'
        'mode="${FAKE_TCLAUDE_MODE:-success}"\n'
        'if [ "$mode" = "exit-42" ]; then exit 42; fi\n'
        'if [ "$mode" = "ignore-signals" ]; then\n'
        '  trap "" INT TERM\n'
        "  python3 -c 'import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)' &\n"
        '  child_pid=$!\n'
        '  printf "%s %s" "$$" "$child_pid" > "$FAKE_TCLAUDE_PIDS_FILE"\n'
        '  wait "$child_pid"\n'
        'fi\n'
        'if { [ "$mode" = "timeout-first" ] && [ "$attempt" = 1 ]; } '
        '|| [ "$mode" = "always-timeout" ]; then sleep 3; fi\n'
        'if [ "$mode" = "malformed" ]; then printf "%s\\n" "{not-json"; exit 0; fi\n'
        "stream=0\n"
        'for arg in "$@"; do\n'
        '  [ "$arg" = "stream-json" ] && stream=1\n'
        "done\n"
        "if [ \"$stream\" = 1 ]; then\n"
        f"  printf '%s\\n' '{stream_init}'\n"
        f"  printf '%s\\n' '{stream_payload}'\n"
        "else\n"
        f"  printf '%s\\n' '{payload}'\n"
        "fi\n"
    )
    fake_tclaude.chmod(0o755)

    # 公开 claude CLI 的假实现:与 tclaude 契约一致,记录参数以便断言 --agent claude 路径。
    fake_claude = bin_dir / "claude"
    fake_claude.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" > "$FAKE_TCLAUDE_ARGS_FILE"\n'
        'previous=""\n'
        'for arg in "$@"; do\n'
        '  if [ "$previous" = "--model" ]; then '
        'printf "%s\\n" "$arg" >> "$FAKE_TCLAUDE_MODELS_FILE"; fi\n'
        '  previous="$arg"\n'
        'done\n'
        "stream=0\n"
        'for arg in "$@"; do\n'
        '  [ "$arg" = "stream-json" ] && stream=1\n'
        "done\n"
        "if [ \"$stream\" = 1 ]; then\n"
        f"  printf '%s\\n' '{stream_init}'\n"
        f"  printf '%s\\n' '{stream_payload}'\n"
        "else\n"
        f"  printf '%s\\n' '{payload}'\n"
        "fi\n"
    )
    fake_claude.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "FAKE_TCLAUDE_ARGS_FILE": str(args_file),
            "FAKE_TCLAUDE_COUNT_FILE": str(count_file),
            "FAKE_TCLAUDE_MODELS_FILE": str(models_file),
            "GEN_STREAM": "0",
        }
    )
    return ScriptProject(tmp_path, job, args_file, count_file, models_file, env)


def test_defaults_to_tclaude_hy3(script_project: ScriptProject) -> None:
    completed, argv = script_project.run()

    assert completed.returncode == 0, completed.stderr
    assert argv[argv.index("--model") + 1] == "claude-hy3"
    assert "tclaude 模型 → claude-hy3 (默认)" in completed.stderr


def test_model_option_preserves_gateway_model_name(script_project: ScriptProject) -> None:
    completed, argv = script_project.run(
        "--model", "claude-glm-5.2[1m]", output_name="custom.jsonl"
    )

    assert completed.returncode == 0, completed.stderr
    assert argv[argv.index("--model") + 1] == "claude-glm-5.2[1m]"
    assert (script_project.root / "custom.jsonl").is_file()
    assert "tclaude 模型 → claude-glm-5.2[1m] (命令行)" in completed.stderr


def test_agent_claude_omits_model_by_default(script_project: ScriptProject) -> None:
    """--agent claude 且不指定 --model:不硬塞腾讯别名 claude-hy3,交给 claude 自身默认。"""
    completed, argv = script_project.run("--agent", "claude", output_name="pub.jsonl")

    assert completed.returncode == 0, completed.stderr
    assert "--model" not in argv
    assert "claude-hy3" not in argv
    assert "claude 模型 →" in completed.stderr
    assert (script_project.root / "pub.jsonl").is_file()


def test_agent_claude_with_explicit_model(script_project: ScriptProject) -> None:
    """--agent claude --model X:把 X 原样传给公开 claude CLI。"""
    completed, argv = script_project.run(
        "--agent", "claude", "--model", "claude-opus-4-8", output_name="pub2.jsonl"
    )

    assert completed.returncode == 0, completed.stderr
    assert argv[argv.index("--model") + 1] == "claude-opus-4-8"


def test_unknown_agent_rejected(script_project: ScriptProject) -> None:
    """未支持的 agent(如 codex)在调用任何 CLI 前就被挡下。"""
    completed, recorded = script_project.invoke(
        "--agent", "codex", str(script_project.job)
    )

    assert completed.returncode == 2
    assert "未知 --agent" in completed.stderr
    assert recorded == []


@pytest.mark.parametrize(
    "argv",
    [
        ("--model=claude-hy3",),
        ("after-job",),
    ],
)
def test_model_option_is_order_independent(
    script_project: ScriptProject, argv: tuple[str, ...]
) -> None:
    output = script_project.root / "ordered.jsonl"
    if argv == ("after-job",):
        command = (
            str(script_project.job),
            "--model",
            "claude-opus-4-8",
            str(output),
        )
        expected = "claude-opus-4-8"
    else:
        command = (*argv, str(script_project.job), str(output))
        expected = "claude-hy3"

    completed, recorded = script_project.invoke(*command)

    assert completed.returncode == 0, completed.stderr
    assert recorded[recorded.index("--model") + 1] == expected
    assert output.is_file()


@pytest.mark.parametrize(
    "argv",
    [
        ("--model=",),
        ("job-before-bare-model", "--model"),
    ],
)
def test_model_requires_value(script_project: ScriptProject, argv: tuple[str, ...]) -> None:
    if argv[0] == "job-before-bare-model":
        command = (str(script_project.job), "--model")
    else:
        command = (*argv, str(script_project.job))

    completed, recorded = script_project.invoke(*command)

    assert completed.returncode == 2
    assert "--model 需要非空模型名" in completed.stderr
    assert recorded == []


@pytest.mark.parametrize(
    "argv",
    [
        ("--unknown",),
        ("job", "out", "extra"),
    ],
)
def test_invalid_arguments_fail_before_tclaude(
    script_project: ScriptProject, argv: tuple[str, ...]
) -> None:
    completed, recorded = script_project.invoke(*argv)

    assert completed.returncode == 2
    assert "用法:" in completed.stderr
    assert recorded == []


def test_stream_mode_uses_tclaude_model_and_parses_result(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "stream.jsonl"

    completed, argv = script_project.invoke(
        "--model",
        "claude-opus-4-8",
        str(script_project.job),
        str(output),
        env_overrides={"GEN_STREAM": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    assert argv[argv.index("--model") + 1] == "claude-opus-4-8"
    assert argv.count("--model") == 1
    assert json.loads(output.read_text())["candidates"][0]["id"] == "c001"


def test_stream_timeout_retries_once_then_succeeds(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "stream-retry.jsonl"

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={
            "GEN_STREAM": "1",
            "GEN_TIMEOUT_SECONDS": "1",
            "GEN_TIMEOUT_GRACE_SECONDS": "1",
            "FAKE_TCLAUDE_MODE": "timeout-first",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert script_project.count_file.read_text() == "2"
    assert script_project.models_file.read_text().splitlines() == [
        "claude-hy3",
        "claude-hy3",
    ]
    assert json.loads(output.read_text())["candidates"][0]["id"] == "c001"


def test_non_stream_timeout_retries_once_then_succeeds(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "non-stream-retry.jsonl"

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={
            "GEN_STREAM": "0",
            "GEN_TIMEOUT_SECONDS": "1",
            "GEN_TIMEOUT_GRACE_SECONDS": "1",
            "FAKE_TCLAUDE_MODE": "timeout-first",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert script_project.count_file.read_text() == "2"
    assert script_project.models_file.read_text().splitlines() == [
        "claude-hy3",
        "claude-hy3",
    ]
    assert json.loads(output.read_text())["candidates"][0]["id"] == "c001"
    raw_files = list((script_project.root / "claude-raw-outputs").glob("*.stdout.json"))
    assert len(raw_files) == 2
    assert ".attempt-2." in completed.stderr


@pytest.mark.parametrize("stream_mode", ["0", "1"])
def test_exhausted_timeouts_return_124_and_preserve_existing_output(
    script_project: ScriptProject, stream_mode: str
) -> None:
    output = script_project.root / f"timeout-{stream_mode}.jsonl"
    output.write_text("previous-good-output\n")

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={
            "GEN_STREAM": stream_mode,
            "GEN_TIMEOUT_SECONDS": "1",
            "GEN_TIMEOUT_GRACE_SECONDS": "1",
            "FAKE_TCLAUDE_MODE": "always-timeout",
        },
    )

    assert completed.returncode == 124
    assert script_project.count_file.read_text() == "2"
    assert output.read_text() == "previous-good-output\n"
    assert "已有输出未修改" in completed.stderr
    assert ".attempt-1.stdout." in completed.stderr
    assert ".attempt-2.stdout." in completed.stderr


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GEN_TIMEOUT_SECONDS", "08"),
        ("GEN_TIMEOUT_GRACE_SECONDS", "0"),
        ("GEN_MAX_RETRIES", "-1"),
        ("GEN_MAX_RETRIES", "11"),
    ],
)
def test_invalid_guard_config_fails_before_tclaude(
    script_project: ScriptProject, name: str, value: str
) -> None:
    completed, _ = script_project.run(env_overrides={name: value})

    assert completed.returncode == 2
    assert not script_project.count_file.exists()


def test_ordinary_tclaude_failure_is_not_retried(script_project: ScriptProject) -> None:
    completed, _ = script_project.run(
        env_overrides={"FAKE_TCLAUDE_MODE": "exit-42"}
    )

    assert completed.returncode == 42
    assert script_project.count_file.read_text() == "1"


def test_startup_prints_timeout_protection(script_project: ScriptProject) -> None:
    completed, _ = script_project.run(
        env_overrides={
            "GEN_TIMEOUT_SECONDS": "900",
            "GEN_TIMEOUT_GRACE_SECONDS": "15",
            "GEN_MAX_RETRIES": "1",
        }
    )

    assert completed.returncode == 0, completed.stderr
    assert "单次软超时 900s" in completed.stderr
    assert "TERM→KILL 宽限 15s" in completed.stderr
    assert "最多尝试 2 次" in completed.stderr


def test_malformed_raw_preserves_existing_output_and_cleans_temp(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "existing.jsonl"
    output.write_text("previous-good-output\n")

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={"GEN_STREAM": "0", "FAKE_TCLAUDE_MODE": "malformed"},
    )

    assert completed.returncode != 0
    assert output.read_text() == "previous-good-output\n"
    assert list(script_project.root.glob(".existing.jsonl.tmp.*")) == []


def test_progress_renderer_failure_does_not_retry_or_replace_output(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "renderer.jsonl"
    output.write_text("previous-good-output\n")

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={"GEN_STREAM": "1", "FAKE_JQ_MODE": "fail-render"},
    )

    assert completed.returncode != 0
    assert script_project.count_file.read_text() == "1"
    assert output.read_text() == "previous-good-output\n"


def test_public_entry_sigint_exits_130_and_cleans_tclaude_tree(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "interrupt.jsonl"
    output.write_text("previous-good-output\n")
    pid_file = script_project.root / "fake-tclaude.pids"
    env = script_project.env.copy()
    env.update(
        {
            "GEN_STREAM": "1",
            "GEN_TIMEOUT_SECONDS": "30",
            "GEN_TIMEOUT_GRACE_SECONDS": "1",
            "FAKE_TCLAUDE_MODE": "ignore-signals",
            "FAKE_TCLAUDE_PIDS_FILE": str(pid_file),
        }
    )
    process = subprocess.Popen(
        ["./gen_configs.sh", str(script_project.job), str(output)],
        cwd=script_project.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pids: list[int] = []
    try:
        _wait_for_file(pid_file)
        pids = [int(value) for value in pid_file.read_text().split()]
        os.killpg(process.pid, signal.SIGINT)
        _, stderr = process.communicate(timeout=5)

        assert process.returncode == 130, stderr
        assert script_project.count_file.read_text() == "1"
        assert output.read_text() == "previous-good-output\n"
        _wait_for_pids_to_exit(pids)
        assert list(script_project.root.glob(".interrupt.jsonl.tmp.*")) == []
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if pids and _process_is_alive(pids[0]):
            os.killpg(pids[0], signal.SIGKILL)


def test_parse_stage_sigint_preserves_output_and_cleans_temp(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "parse.jsonl"
    output.write_text("previous-good-output\n")
    marker = script_project.root / "parse-started"
    env = script_project.env.copy()
    env.update(
        {
            "GEN_STREAM": "0",
            "FAKE_JQ_MODE": "block-parse",
            "FAKE_JQ_PARSE_MARKER": str(marker),
        }
    )
    process = subprocess.Popen(
        ["./gen_configs.sh", str(script_project.job), str(output)],
        cwd=script_project.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _wait_for_file(marker)
        os.killpg(process.pid, signal.SIGINT)
        _, stderr = process.communicate(timeout=3)

        assert process.returncode == 130, stderr
        assert output.read_text() == "previous-good-output\n"
        assert list(script_project.root.glob(".parse.jsonl.tmp.*")) == []
        state_files = (script_project.root / "claude-raw-outputs").glob(".*.success.*")
        assert list(state_files) == []
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
