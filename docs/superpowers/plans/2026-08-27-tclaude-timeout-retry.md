# tclaude Timeout Retry Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent `gen_configs.sh` from waiting forever on a stalled tclaude request, retry the identical invocation once after timeout, and leave no child processes or partial formal output.

**Architecture:** Add a focused Python supervisor that owns timeout, retry, signal forwarding, process-group cleanup, and attempt-level raw logs. Keep prompt construction and candidate parsing in Bash, but route both modes through the supervisor and publish candidates with a same-directory temporary file plus atomic rename.

**Tech Stack:** Python 3.11 standard library, Bash 5.2, GNU/Linux process groups, pytest, jq.

---

## File structure

- Create `runners/tclaude_guard.py`: public CLI for guarded tclaude execution.
- Create `tests/test_tclaude_guard.py`: real-process tests for timeouts and signals.
- Modify `gen_configs.sh`: guard integration and atomic output publication.
- Modify `tests/test_gen_configs_cli.py`: public-entry regression tests.
- Modify `README.md`: operational documentation.

### Task 1: Guard CLI validation and successful invocation

**Files:**
- Create: `runners/tclaude_guard.py`
- Create: `tests/test_tclaude_guard.py`

- [ ] **Step 1: Write one failing successful-invocation test**

Run the guard through `subprocess.run` with a fake command that prints to stdout and stderr. Assert exit `0`, one attempt raw pair, exact bytes in each file, and an atomically written success-path pointing to stdout raw.

- [ ] **Step 2: Run it to verify RED**

```bash
uv run pytest tests/test_tclaude_guard.py::test_success_records_attempt_and_success_path -q
```

Expected: FAIL because the guard does not exist.

- [ ] **Step 3: Implement the minimal successful path**

Expose this public CLI:

```text
python3 runners/tclaude_guard.py
  --timeout-seconds N --grace-seconds N --max-retries N
  --raw-dir DIR --job-id ID --stdout-suffix jsonl|json
  --success-path-file PATH [--forward-stdout]
  -- COMMAND [ARG ...]
```

Implement a bounded decimal parser that rejects leading zeros. Enforce timeout `1..86400`, grace `1..300`, and retries `0..10`. Generate a run ID from UTC microseconds, parent PID, and a random suffix.

Start the command with `stdout=PIPE`, `stderr=PIPE`, `start_new_session=True`. Two reader threads write binary stdout/stderr to separate attempt files; stdout also forwards and flushes when requested. Write success-path through a sibling temp and `os.replace`. Spawn-not-found maps to `127`; guard I/O failures return `1` without success-path.

- [ ] **Step 4: Run tracer test to verify GREEN**

- [ ] **Step 5: Add invalid-config tests one case at a time**

Cover zero/negative/over-limit/nonnumeric/leading-zero values and prove the child was never invoked. Run each RED then GREEN.

- [ ] **Step 6: Run Task 1 tests**

```bash
uv run pytest tests/test_tclaude_guard.py -q
```

### Task 2: Timeout, same-command retry, and raw isolation

**Files:**
- Modify: `runners/tclaude_guard.py`
- Modify: `tests/test_tclaude_guard.py`

- [ ] **Step 1: Write a failing timeout-then-success test**

The fake command records argv, prints partial output, blocks beyond one second on attempt 1, and succeeds on attempt 2. Assert two byte-identical argv/model records, separate raw pairs, and success-path pointing only to attempt 2.

- [ ] **Step 2: Run it to verify RED**

```bash
uv run pytest tests/test_tclaude_guard.py::test_timeout_retries_same_command_once_then_succeeds -q
```

- [ ] **Step 3: Implement deadline and retry**

Use `time.monotonic()`. Before deadline cleanup, set an internal marker; signal the whole child group with TERM, wait `grace_seconds`, then KILL. Reap the direct child and join readers. A guard-triggered deadline always produces logical `124`, regardless of child `-15`/`-9`; retry only logical `124` while attempts remain, reusing the immutable command list.

- [ ] **Step 4: Verify timeout-then-success GREEN**

- [ ] **Step 5: Add exhausted timeout test RED→GREEN**

Always block; assert two calls for one retry, final `124`, no success-path, and two raw pairs.

- [ ] **Step 6: Add ordinary-failure tests RED→GREEN**

Exit `42` must not retry. A child self-signaled with `SIGUSR1` must map to `138` and not retry.

### Task 3: Signal escalation and orphan prevention

**Files:**
- Modify: `runners/tclaude_guard.py`
- Modify: `tests/test_tclaude_guard.py`

- [ ] **Step 1: Write TERM-to-KILL escalation test and verify RED**

Fake parent and child ignore TERM and record both PIDs. With one-second timeout/grace, assert final `124` and every PID disappears from `/proc`.

- [ ] **Step 2: Complete process-group escalation and verify GREEN**

Centralize `os.killpg(proc.pid, signum)`, treat `ProcessLookupError` as already clean, poll grace periods, always reap, and use bounded reader-thread joins so a broken forwarding/write thread cannot violate the four-second SIGINT contract.

- [ ] **Step 3: Write real SIGINT tests and verify RED**

Signal a guarded fake process tree. Assert within four seconds: status `130`, one attempt, no success-path, no descendants. A second SIGINT during cleanup must cause immediate KILL.

- [ ] **Step 4: Implement signal state machine and verify GREEN**

```text
first INT: group INT → 1s → TERM → 2s → KILL → 130
second INT: immediate group KILL
TERM: group TERM → 2s → KILL → 143
HUP: group TERM → 2s → KILL → 129
```

External signals take priority over deadline and never retry.

- [ ] **Step 5: Add downstream-close and I/O tests RED→GREEN**

Cover a closed forwarding consumer, raw/state write failure, and TERM/HUP mappings. Each must clean descendants, avoid retries, and avoid success-path.

- [ ] **Step 6: Run full guard suite**

```bash
uv run pytest tests/test_tclaude_guard.py -q
```

### Task 4: Integrate the guard into gen_configs.sh

**Files:**
- Modify: `gen_configs.sh`
- Modify: `tests/test_gen_configs_cli.py`

- [ ] **Step 1: Extend fake tclaude and write stream retry test**

Make behavior configurable and record invocation count plus argv. Attempt 1 blocks past `GEN_TIMEOUT_SECONDS=1`; attempt 2 emits a valid stream result. Assert two identical model invocations and successful formal output.

- [ ] **Step 2: Run test to verify RED**

```bash
uv run pytest tests/test_gen_configs_cli.py::test_stream_timeout_retries_once_then_succeeds -q
```

- [ ] **Step 3: Route stream mode through guard**

Preflight `python3` and the helper. Set defaults `GEN_TIMEOUT_SECONDS=600`, `GEN_TIMEOUT_GRACE_SECONDS=10`, `GEN_MAX_RETRIES=1`. Create a unique success-path state filename and exact `EXIT` cleanup trap. Build tclaude argv as an array.

Temporarily disable `errexit` around the guard pipeline, pipe guard `--forward-stdout` into the existing progress jq, immediately save every `PIPESTATUS`, restore `errexit`, and prioritize interruption, guard failure, then renderer failure. Read successful `RAW_FILE` only from success-path. The cleanup trap must save the incoming status, remove only exact per-run paths, then return the original status.

Print the soft timeout, TERM grace, and maximum attempt count before invocation. On failure, print every stdout/stderr attempt path emitted for this run plus whether an existing formal output was preserved.

- [ ] **Step 4: Run CLI tests to verify GREEN**

```bash
uv run pytest tests/test_gen_configs_cli.py -q
```

- [ ] **Step 5: Add non-stream retry tests RED→GREEN**

Cover first-timeout-then-success and exhausted timeout through the same guard without forwarding stdout. Wrap the non-stream guard call in `set +e`, capture `$?` immediately, then restore `set -e` so retry/failure handling is reachable. Assert partial raw retention and success-path selection.

- [ ] **Step 6: Add invalid-config and ordinary-error tests RED→GREEN**

Invalid timeout/retry/grace must fail before fake tclaude. Ordinary tclaude failure must not retry. Both attempts must retain the selected model with no fallback. Assert startup output contains soft timeout, grace, and maximum attempts, and failure output lists the exact run-specific attempt raw paths.

### Task 5: Atomic output and full-entry SIGINT

**Files:**
- Modify: `gen_configs.sh`
- Modify: `tests/test_gen_configs_cli.py`

- [ ] **Step 1: Write old-output preservation test and verify RED**

Precreate output with a sentinel, force two timeouts, and assert exit `124`, exact sentinel remains, and no sibling output temp remains.

- [ ] **Step 2: Implement atomic publication and verify GREEN**

Create `OUT_TMP` with `mktemp` in the output directory. Parse into it, then atomically `mv` only after full validation. `EXIT` removes uncommitted temp/state paths. Never delete existing `OUT` on failure.

- [ ] **Step 3: Add parse and renderer failure tests RED→GREEN**

Malformed raw JSON (an existing parser failure path) preserves existing output and cleans temp files. Do not add new candidate-schema validation in this change. A wrapper that fails only progress-render jq must produce nonzero status, one tclaude invocation, and no output replacement.

- [ ] **Step 4: Add public-entry SIGINT test and verify RED**

Launch `gen_configs.sh` in a new session; fake tclaude spawns a recorded child and ignores INT/TERM. Signal the script process group. Within four seconds assert status `130`, one invocation, old output unchanged, no temp, and no script/guard/fake descendants.

- [ ] **Step 5: Implement Shell traps and verify GREEN**

The INT trap records interruption; guard cleans descendants. Wait for the pipeline to settle, use saved statuses, then exit `130`. EXIT removes only exact per-run temp files.

- [ ] **Step 6: Add parse-stage SIGINT test RED→GREEN**

Use a deterministic `jq` wrapper that blocks only during final candidate parsing, records that parsing started, and then waits. Send SIGINT to the script process group at that point. Assert exit `130`, prior formal output unchanged, and output temp/state files removed. Adjust the Shell trap only as needed while preserving the incoming exit code during cleanup.

- [ ] **Step 7: Run CLI suite**

```bash
uv run pytest tests/test_gen_configs_cli.py -q
```

### Task 6: Documentation and full verification

**Files:**
- Modify: `README.md`
- Test: repository checks

- [ ] **Step 1: Document behavior**

Document three environment variables, two-attempt default, exit `124`/`130`, no model fallback, run-ID raw files, and old-output preservation.

- [ ] **Step 2: Run focused tests**

```bash
uv run pytest tests/test_tclaude_guard.py tests/test_gen_configs_cli.py -q
```

- [ ] **Step 3: Run lint and shell syntax checks**

```bash
uv run ruff check runners/tclaude_guard.py tests/test_tclaude_guard.py tests/test_gen_configs_cli.py
bash -n gen_configs.sh
```

- [ ] **Step 4: Run complete suite**

```bash
uv run pytest -q
```

- [ ] **Step 5: Inspect diff and residual processes**

```bash
git diff --check
git diff -- gen_configs.sh runners/tclaude_guard.py tests/test_tclaude_guard.py tests/test_gen_configs_cli.py README.md
ps -eo pid,ppid,args | rg 'tclaude_guard|FAKE_TCLAUDE' || true
```

- [ ] **Step 6: Leave implementation uncommitted for user review**

Do not stage or commit implementation files in the dirty worktree. Report the already committed design-document commit separately from the uncommitted implementation diff.
