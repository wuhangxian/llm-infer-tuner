# tclaude Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gen_configs.sh` invoke tclaude with `claude-hy3` by default and accept a safe command-line `--model` override.

**Architecture:** Keep the existing shell entry point and both output modes intact. Parse the new option before the current positional-argument logic, construct one shared Bash argument array for tclaude, and exercise the public script interface with a fake tclaude in an isolated temporary project copy.

**Tech Stack:** Bash, pytest, Python `subprocess`, jq, tclaude/Claude Code CLI

---

### Task 1: Default to tclaude and HY3

**Files:**
- Create: `tests/test_gen_configs_cli.py`
- Modify: `gen_configs.sh:20-40,157-223`

- [ ] **Step 1: Write the failing default-model test**

Create a pytest fixture that copies `gen_configs.sh` plus a placeholder skill file into `tmp_path`, installs fake `claude` and `tclaude` executables in a temporary `bin/`, and runs the copied script with `GEN_STREAM=0`. The fake tclaude records each argument and returns a minimal valid structured response. Assert that the invocation contains adjacent arguments `--model`, `claude-hy3`; the fake claude must exit 99 so the test also proves the old executable was not used.

```python
def test_defaults_to_tclaude_hy3(script_project):
    completed, argv = script_project.run()
    assert completed.returncode == 0, completed.stderr
    assert argv[argv.index("--model") + 1] == "claude-hy3"
```

- [ ] **Step 2: Run the test to verify RED**

Run: `uv run pytest tests/test_gen_configs_cli.py::test_defaults_to_tclaude_hy3 -v`

Expected: FAIL because the current script checks/calls `claude`, causing the sentinel exit code 99.

- [ ] **Step 3: Implement the minimal default behavior**

In `gen_configs.sh`, replace the executable preflight with `command -v tclaude`. Define:

```bash
MODEL="claude-hy3"
MODEL_SOURCE="默认"
TCLAUDE_ARGS=(--model "$MODEL")
```

Use `tclaude -p "$PROMPT" "${TCLAUDE_ARGS[@]}" ...` in both stream and non-stream branches. Update user-facing comments and messages that name the executable, without rewriting unrelated user changes.

- [ ] **Step 4: Run the test to verify GREEN**

Run: `uv run pytest tests/test_gen_configs_cli.py::test_defaults_to_tclaude_hy3 -v`

Expected: PASS.

### Task 2: Add the command-line model override

**Files:**
- Modify: `tests/test_gen_configs_cli.py`
- Modify: `gen_configs.sh:20-50`

- [ ] **Step 1: Write the failing override test**

```python
def test_model_option_preserves_gateway_model_name(script_project):
    completed, argv = script_project.run(
        "--model", "claude-glm-5.2[1m]", output_name="custom.jsonl"
    )
    assert completed.returncode == 0, completed.stderr
    assert argv[argv.index("--model") + 1] == "claude-glm-5.2[1m]"
    assert script_project.path("custom.jsonl").is_file()
```

- [ ] **Step 2: Run the override test to verify RED**

Run: `uv run pytest tests/test_gen_configs_cli.py::test_model_option_preserves_gateway_model_name -v`

Expected: FAIL because `--model` is currently interpreted as the job path.

- [ ] **Step 3: Implement minimal option parsing**

Parse `--model VALUE`, `--model=VALUE`, and `--` while collecting no more than two positional arguments. Reject an empty value, unknown option, or too many positionals with usage text and exit code 2. Set `MODEL_SOURCE="命令行"` on override, then assign `JOB` and optional `OUT_ARG` from the parsed positionals. Replace the old `$2` lookup with `OUT="${OUT_ARG:-outputs/${JOB_ID}/configs.jsonl}"`, so options never shift the output path. Keep the old `<job.json> [out.json]` interface valid and print `ℹ️  tclaude 模型 → $MODEL ($MODEL_SOURCE)` before invocation.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run: `uv run pytest tests/test_gen_configs_cli.py -v`

Expected: all current CLI tests PASS.

### Task 3: Cover errors and the default streaming path

**Files:**
- Modify: `tests/test_gen_configs_cli.py`
- Modify: `gen_configs.sh:150-215` only if the stream test exposes a defect
- Modify: `README.md:10-45`

- [ ] **Step 1: Add one error test**

Verify `--model ''` and bare `--model` both fail with exit code 2 before either fake CLI is invoked. Also assert the default and override runs print the selected model plus `默认`/`命令行` source labels.

- [ ] **Step 2: Run the error test and verify behavior**

Run: `uv run pytest tests/test_gen_configs_cli.py -k model_requires_value -v`

Expected: PASS after Task 2 parsing; if it fails, make only the validation change required.

- [ ] **Step 3: Add a stream-mode integration test**

Teach the fake tclaude to emit a valid NDJSON `result` event when it receives `--output-format stream-json`. Run without `GEN_STREAM=0`, assert successful output parsing, and assert the selected model is still passed exactly once.

Add focused parser cases for `--model=claude-hy3`, an option placed after `job.json`, an unknown option, and excessive positional arguments. These cases ensure the documented order-independent interface is enforced rather than accidentally tied to one argv layout.

- [ ] **Step 4: Run all CLI tests**

Run: `uv run pytest tests/test_gen_configs_cli.py -v`

Expected: PASS.

- [ ] **Step 5: Update README examples**

Replace the prerequisite `claude` with `tclaude`, document that HY3 is the default, and add quoted GLM 5.2 plus Opus override examples.

### Task 4: Full verification

**Files:**
- Verify only; no planned production changes

- [ ] **Step 1: Validate shell syntax and formatting**

Run: `bash -n gen_configs.sh && uv run ruff check tests/test_gen_configs_cli.py`

Expected: exit code 0.

- [ ] **Step 2: Run the complete test suite**

Run: `uv run pytest -q`

Expected: all tests PASS.

- [ ] **Step 3: Smoke-test the installed tclaude without a costly generation**

Run: `tclaude --version` and `tclaude -- --help | rg -- '--model|--json-schema|--output-format'`

Expected: installed wrapper/upstream versions print and all required forwarded options are present.

- [ ] **Step 4: Review the final diff**

Run: `git diff --check && git diff -- gen_configs.sh tests/test_gen_configs_cli.py README.md`

Expected: no whitespace errors; diff contains only the approved CLI/model-selection changes plus the user's pre-existing `gen_configs.sh` edits.
