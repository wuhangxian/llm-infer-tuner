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


def _agent_response(candidates: list[dict[str, object]]) -> str:
    return json.dumps({"structured_output": {"candidates": candidates}})


def _agent_stream_response(candidates: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "structured_output": {"candidates": candidates},
        }
    )


def _candidate_payload(candidate_id: str, *, tp_size: int = 1) -> dict[str, object]:
    return {
        "id": candidate_id,
        "params": {"tp_size": tp_size, "mem_fraction_static": 0.8},
        "cmd": (
            "python -m sglang.launch_server --model-path ${MODEL_PATH} "
            f"--tp-size {tp_size} --mem-fraction-static 0.8"
        ),
        "reasons": ["test"],
    }


def _candidate_with_mamba(candidate_id: str, strategy: str) -> dict[str, object]:
    candidate = _candidate_payload(candidate_id)
    params = candidate["params"]
    assert isinstance(params, dict)
    candidate["params"] = {
        **params,
        "mamba_radix_cache_strategy": strategy,
    }
    candidate["cmd"] = (
        f"{candidate['cmd']} --mamba-radix-cache-strategy {strategy}"
    )
    return candidate


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
    prompt_file: Path
    env: dict[str, str]

    def invoke(
        self, *argv: str, env_overrides: dict[str, str] | None = None
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        env = self.env.copy()
        env.update(env_overrides or {})
        for recorded_file in (
            self.args_file,
            self.count_file,
            self.models_file,
            self.prompt_file,
        ):
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
    (skill_dir / "knowledge.md").write_text("# test knowledge\n")
    catalogs_dir = tmp_path / "catalogs"
    catalogs_dir.mkdir()
    for catalog_name in (
        "gpu.yaml",
        "models.yaml",
        "sglang-images.yaml",
        "workloads.yaml",
    ):
        (catalogs_dir / catalog_name).write_text("test: true\n")

    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "job_id": "tclaude-cli-test",
                "engine": "sglang",
                "gpu_model": "test-gpu",
                "gpu_count": 1,
                "gpu_memory_gb": 16.0,
                "model": "test-model",
                "image": "test-image",
                "workload": "test-workload",
                "benchmark_method": "test-method",
                "sla": {"max_avg_ttft_ms": 100.0, "max_avg_tpot_ms": 20.0},
                "search": {"max_candidates": 1},
            }
        )
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "tclaude.args"
    count_file = tmp_path / "tclaude.count"
    models_file = tmp_path / "tclaude.models"
    prompt_file = tmp_path / "tclaude.prompt"


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
        'if [ "${FAKE_JQ_MODE:-}" = "fail-preview" ]; then\n'
        '  for arg in "$@"; do\n'
        '    case "$arg" in *requested_mamba*) exit 56 ;; esac\n'
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

    candidate = _candidate_payload("c001")
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
        'if [ -n "${FAKE_TCLAUDE_PWD_FILE:-}" ]; then pwd -P > "$FAKE_TCLAUDE_PWD_FILE"; fi\n'
        'attempt=1\n'
        'if [ -f "$FAKE_TCLAUDE_COUNT_FILE" ]; then '
        'attempt=$(( $(cat "$FAKE_TCLAUDE_COUNT_FILE") + 1 )); fi\n'
        'printf "%s" "$attempt" > "$FAKE_TCLAUDE_COUNT_FILE"\n'
        'previous=""\n'
        'available_tools=""\n'
        'add_dirs=()\n'
        'saw_tools=0\n'
        'dangerous_permissions=0\n'
        'for arg in "$@"; do\n'
        '  if [ "$previous" = "-p" ]; then printf "%s" "$arg" > "$FAKE_TCLAUDE_PROMPT_FILE"; fi\n'
        '  if [ "$previous" = "--tools" ]; then available_tools="$arg"; saw_tools=1; fi\n'
        '  if [ "$previous" = "--add-dir" ]; then add_dirs+=("$arg"); fi\n'
        '  if [ "$previous" = "--model" ]; then '
        'printf "%s\\n" "$arg" >> "$FAKE_TCLAUDE_MODELS_FILE"; fi\n'
        '  if [ "$arg" = "--dangerously-skip-permissions" ]; then dangerous_permissions=1; fi\n'
        '  if [ "$arg" = "--restricted" ]; then '
        'echo "error: unknown option --restricted" >&2; exit 64; fi\n'
        '  previous="$arg"\n'
        'done\n'
        'mode="${FAKE_TCLAUDE_MODE:-success}"\n'
        'if [ -n "${FAKE_TCLAUDE_ADD_DIRS_FILE:-}" ]; then '
        'printf "%s\\n" "${add_dirs[@]}" > "$FAKE_TCLAUDE_ADD_DIRS_FILE"; fi\n'
        'if [ "$mode" = "exfiltrate-if-visible" ]; then\n'
        '  discovered="not-visible"\n'
        '  probe_roots=(".")\n'
        '  probe_roots+=("${add_dirs[@]}")\n'
        '  for probe_root in "${probe_roots[@]}"; do\n'
        '    if [ -f "$probe_root/$FAKE_SENTINEL_NAME" ]; then '
        'discovered="$(cat "$probe_root/$FAKE_SENTINEL_NAME")"; break; fi\n'
        '  done\n'
        '  response="${FAKE_TCLAUDE_RESPONSE//__PROBE__/$discovered}"\n'
        '  printf "%s\\n" "$response"\n'
        '  exit 0\n'
        'fi\n'
        'if [ "$mode" = "model-policy-overwrite" ]; then\n'
        '  policy_can_write="$dangerous_permissions"\n'
        '  if [ "$saw_tools" = 0 ]; then policy_can_write=1; fi\n'
        '  case ",$available_tools," in '
        '*",Bash,"*|*",Edit,"*|*",Write,"*) policy_can_write=1 ;; esac\n'
        '  if [ "$policy_can_write" = 1 ]; then '
        'printf "%s\\n" "agent-overwrite" > "$FAKE_OFFICIAL_OUTPUT"; fi\n'
        '  exit 42\n'
        'fi\n'
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
        'if [ -n "${FAKE_TCLAUDE_RESPONSE:-}" ]; then\n'
        '  if [ "$stream" = 1 ]; then '
        f"printf '%s\\n' '{stream_init}'; fi\n"
        '  printf "%s\\n" "$FAKE_TCLAUDE_RESPONSE"\n'
        '  if [ -n "${FAKE_TCLAUDE_TRAILING_LINE:-}" ]; then '
        'printf "%s\\n" "$FAKE_TCLAUDE_TRAILING_LINE"; fi\n'
        '  exit 0\n'
        'fi\n'
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
            "FAKE_TCLAUDE_PROMPT_FILE": str(prompt_file),
            "GEN_STREAM": "0",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    return ScriptProject(tmp_path, job, args_file, count_file, models_file, prompt_file, env)


def test_invalid_job_fails_before_invoking_agent_and_preserves_output(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "invalid-job.jsonl"
    original = b'{"candidates":[{"id":"old"}]}\n'
    output.write_bytes(original)
    invalid_job = script_project.root / "invalid-job.json"
    invalid_job.write_text(json.dumps({"job_id": "missing-required-fields"}))

    completed, recorded = script_project.invoke(str(invalid_job), str(output))

    assert completed.returncode != 0
    assert recorded == []
    assert output.read_bytes() == original
    assert "job validation failed" in completed.stderr


def test_prompt_is_literal_and_never_executes_markdown_backticks(
    script_project: ScriptProject,
) -> None:
    completed, _ = script_project.run(output_name="literal-prompt.jsonl")

    assert completed.returncode == 0, completed.stderr
    assert "command not found" not in completed.stderr
    prompt = script_project.prompt_file.read_text()
    assert "`--disable-radix-cache`" in prompt
    assert "`mamba_radix_cache_strategy`" in prompt
    assert "`${MODEL_PATH}`" in prompt


def test_agent_is_restricted_to_read_only_tools_and_cannot_overwrite_output(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "policy-protected.jsonl"
    original = b"previous-good-output\n"
    output.write_bytes(original)

    completed, argv = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={
            "FAKE_TCLAUDE_MODE": "model-policy-overwrite",
            "FAKE_OFFICIAL_OUTPUT": str(output),
        },
    )

    assert completed.returncode == 42
    assert output.read_bytes() == original
    assert "--dangerously-skip-permissions" not in argv
    assert argv[argv.index("--tools") + 1] == "Read,Grep,Glob"
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in argv
    assert "--disable-slash-commands" in argv


def test_agent_can_read_only_staged_knowledge_not_repository_files(
    script_project: ScriptProject,
) -> None:
    sentinel_value = "repository-sentinel-must-not-reach-model-output"
    sentinel = script_project.root / "sentinel-secret.txt"
    sentinel.write_text(sentinel_value)
    pwd_file = script_project.root / "agent.pwd"
    add_dirs_file = script_project.root / "agent.add-dirs"
    candidate = _candidate_payload("c001")
    candidate["reasons"] = ["probe=__PROBE__"]

    completed, argv = script_project.run(
        output_name="isolated-generation.jsonl",
        env_overrides={
            "FAKE_TCLAUDE_MODE": "exfiltrate-if-visible",
            "FAKE_TCLAUDE_RESPONSE": _agent_response([candidate]),
            "FAKE_SENTINEL_NAME": sentinel.name,
            "FAKE_TCLAUDE_PWD_FILE": str(pwd_file),
            "FAKE_TCLAUDE_ADD_DIRS_FILE": str(add_dirs_file),
        },
    )

    assert completed.returncode == 0, completed.stderr
    output = (script_project.root / "isolated-generation.jsonl").read_text()
    assert sentinel_value not in output
    agent_cwd = Path(pwd_file.read_text().strip())
    assert agent_cwd != script_project.root
    assert not agent_cwd.exists()
    assert add_dirs_file.read_text().splitlines() == ["."]
    assert "--restricted" not in argv


def test_zero_candidates_fail_validation_and_preserve_existing_output(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "zero-candidates.jsonl"
    original = b'{"candidates":[{"id":"previous"}]}\n'
    output.write_bytes(original)

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={"FAKE_TCLAUDE_RESPONSE": _agent_response([])},
    )

    assert completed.returncode != 0
    assert script_project.count_file.read_text() == "1"
    assert output.read_bytes() == original
    assert "candidate validation failed" in completed.stderr
    assert list(script_project.root.glob(".zero-candidates.jsonl.tmp.*")) == []


@pytest.mark.parametrize(
    ("job_search", "candidates", "expected_error"),
    [
        (
            {"max_candidates": 2},
            [_candidate_payload("c001"), _candidate_payload("c001", tp_size=2)],
            "candidate IDs must be unique",
        ),
        (
            {"max_candidates": 2},
            [
                _candidate_with_mamba("c001", "extra_buffer"),
                _candidate_with_mamba("c002", "no_buffer"),
            ],
            "duplicates c001",
        ),
        (
            {"max_candidates": 2},
            [_candidate_payload("c001")],
            "expected 2 candidates",
        ),
        (
            {"max_candidates": 1, "baseline": {"tp_size": 1}},
            [_candidate_payload("c001")],
            "configured baseline requires",
        ),
        (
            {"max_candidates": 1},
            [
                {
                    **_candidate_payload("c001"),
                    "params": {"tp_size": "1", "mem_fraction_static": 0.8},
                }
            ],
            "tp_size",
        ),
    ],
    ids=[
        "duplicate-id",
        "duplicate-effective-config",
        "wrong-count",
        "baseline-mismatch",
        "malformed-candidate",
    ],
)
def test_semantically_invalid_candidate_sets_preserve_existing_output(
    script_project: ScriptProject,
    job_search: dict[str, object],
    candidates: list[dict[str, object]],
    expected_error: str,
) -> None:
    job = json.loads(script_project.job.read_text())
    job["search"] = job_search
    script_project.job.write_text(json.dumps(job))
    output = script_project.root / "invalid-candidate-set.jsonl"
    original = b'{"candidates":[{"id":"previous"}]}\n'
    output.write_bytes(original)

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={"FAKE_TCLAUDE_RESPONSE": _agent_response(candidates)},
    )

    assert completed.returncode != 0
    assert output.read_bytes() == original
    assert expected_error in completed.stderr
    assert list(script_project.root.glob(".invalid-candidate-set.jsonl.tmp.*")) == []


def test_successful_generation_atomically_replaces_existing_output(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "atomic-success.jsonl"
    output.write_bytes(b"previous-good-output\n")

    completed, _ = script_project.invoke(str(script_project.job), str(output))

    assert completed.returncode == 0, completed.stderr
    document = json.loads(output.read_text())
    assert document["candidates"][0]["id"] == "c001"
    assert list(script_project.root.glob(".atomic-success.jsonl.tmp.*")) == []
    assert output.stat().st_mode & 0o077 == 0


def test_stream_candidate_validation_failure_preserves_existing_output(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "invalid-stream.jsonl"
    original = b"previous-good-output\n"
    output.write_bytes(original)

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={
            "GEN_STREAM": "1",
            "FAKE_TCLAUDE_RESPONSE": _agent_stream_response([]),
        },
    )

    assert completed.returncode != 0
    assert output.read_bytes() == original
    assert "candidate validation failed" in completed.stderr
    assert list(script_project.root.glob(".invalid-stream.jsonl.tmp.*")) == []


def test_non_stream_error_result_never_replaces_existing_output(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "non-stream-error.jsonl"
    original = b"previous-good-output\n"
    output.write_bytes(original)
    error_result = json.dumps(
        {
            "is_error": True,
            "subtype": "error_during_execution",
            "structured_output": {"candidates": [_candidate_payload("c001")]},
        }
    )

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={"GEN_STREAM": "0", "FAKE_TCLAUDE_RESPONSE": error_result},
    )

    assert completed.returncode != 0
    assert output.read_bytes() == original
    assert "is_error" in completed.stderr
    assert list(script_project.root.glob(".non-stream-error.jsonl.tmp.*")) == []


def test_stream_trailing_malformed_event_never_replaces_existing_output(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "stream-trailing-malformed.jsonl"
    original = b"previous-good-output\n"
    output.write_bytes(original)

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={
            "GEN_STREAM": "1",
            "FAKE_TCLAUDE_RESPONSE": _agent_stream_response(
                [_candidate_payload("c001")]
            ),
            "FAKE_TCLAUDE_TRAILING_LINE": "{not-json",
        },
    )

    assert completed.returncode != 0
    assert output.read_bytes() == original
    assert "解析 candidates 失败" in completed.stderr
    assert list(script_project.root.glob(".stream-trailing-malformed.jsonl.tmp.*")) == []


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
    output = script_project.root / "agent-failure.jsonl"
    original = b"previous-good-output\n"
    output.write_bytes(original)

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={"FAKE_TCLAUDE_MODE": "exit-42"},
    )

    assert completed.returncode == 42
    assert script_project.count_file.read_text() == "1"
    assert output.read_bytes() == original


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


def test_preview_distinguishes_requested_and_effective_mamba_cache(
    script_project: ScriptProject,
) -> None:
    completed, _ = script_project.run()

    assert completed.returncode == 0, completed.stderr
    assert "requested_mamba=no_buffer(default)" in completed.stderr
    assert "effective_mamba=inactive(radix_off)" in completed.stderr
    assert "radix=off" in completed.stderr


@pytest.mark.parametrize("stream_mode", ["0", "1"])
def test_malformed_raw_preserves_existing_output_and_cleans_temp(
    script_project: ScriptProject, stream_mode: str
) -> None:
    output = script_project.root / f"malformed-{stream_mode}.jsonl"
    output.write_text("previous-good-output\n")

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={"GEN_STREAM": stream_mode, "FAKE_TCLAUDE_MODE": "malformed"},
    )

    assert completed.returncode != 0
    assert output.read_text() == "previous-good-output\n"
    assert list(script_project.root.glob(f".malformed-{stream_mode}.jsonl.tmp.*")) == []


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


def test_preview_failure_does_not_replace_output_and_cleans_temp(
    script_project: ScriptProject,
) -> None:
    output = script_project.root / "preview-failure.jsonl"
    original = b"previous-good-output\n"
    output.write_bytes(original)

    completed, _ = script_project.invoke(
        str(script_project.job),
        str(output),
        env_overrides={"GEN_STREAM": "0", "FAKE_JQ_MODE": "fail-preview"},
    )

    assert completed.returncode != 0
    assert output.read_bytes() == original
    assert list(script_project.root.glob(".preview-failure.jsonl.tmp.*")) == []


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
