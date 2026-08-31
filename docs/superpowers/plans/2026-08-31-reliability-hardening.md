# LLM Infer Tuner Reliability Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make candidate generation and benchmarking fail closed, recover from runtime failures, produce exact/auditable search results, and publish measured rather than misleading rankings.

**Architecture:** Add strict input contracts at the local boundary, introduce typed probe/search states, and order executor preflight so no remote mutation precedes validation. Keep Round 1 coarse, make Round 2 independently verifiable with three-sample decisions and dynamic budgets, then expose candidate summaries plus probe evidence with explicit measurement certainty.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, Bash, Docker/SSH, SGLang bench_serving, Ruff, Pyright.

**Design reference:** `docs/superpowers/specs/2026-08-31-reliability-hardening-design.md`

---

## File responsibility map

- `schemas/job_spec.py`: strict Job/SLA/Search contract.
- `schemas/candidate_spec.py`: candidate IDs, params, legacy launch command validation, candidate-count contract.
- `schemas/target_spec.py`: target/credential/ownership/port contract.
- `runners/metrics.py`: strict benchmark parsing and typed probe result fields.
- `runners/concurrency_search.py`: sample aggregation, seed revalidation, dynamic search, certainty.
- `runners/executor.py`: ordered preflight, GPU/port allocation, lifecycle, recovery, evidence, orchestration.
- `runners/remote.py`: non-leaking SSH credential transport and checked command results.
- `runners/container.py`: argv-safe server/container commands and lifecycle cleanup.
- `runners/ranker.py`: valid-result-only ranking, estimated/measured modes, intervals/ties.
- `runners/reporting.py`: candidate-summary/probe-evidence/task-status contracts.
- `gen_configs.sh`: safe prompt construction and atomic validated generation.
- `run_executor.sh`: strict CLI modes and secret-safe Target loading.
- `tests/`: focused red-green tests and end-to-end dry-run/replay invariants.
- `.github/workflows/ci.yml`: automated pytest/Ruff/Pyright/shell/secret gates.

### Task 1: Strict Job, Target, Candidate, and CandidateSet contracts

**Files:**
- Modify: `schemas/job_spec.py`
- Create: `schemas/target_spec.py`
- Create: `schemas/candidate_spec.py`
- Modify: `schemas/__init__.py`
- Modify: `tests/test_schemas.py`
- Modify: `tests/test_load_candidates.py`

- [ ] **Step 1: Write failing strictness tests**

Add tests that reject numeric strings, booleans-as-integers, NaN/Inf, IDs longer than 128,
unsafe ID characters, invalid ports, unsupported credential combinations, multiple baselines,
empty candidates, duplicate IDs, duplicate effective params, and count mismatch. Assert the
baseline contract is `max_candidates + 1` only when `search.baseline` exists.

- [ ] **Step 2: Run tests and confirm red**

Run: `uv run pytest tests/test_schemas.py tests/test_load_candidates.py -q`

Expected: failures for missing TargetSpec/CandidateSet and permissive coercion.

- [ ] **Step 3: Implement strict models**

Use `ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, str_strip_whitespace=True)`.
Define bounded `Identifier`, `TargetSpec`, `CandidateParams`, `CandidateSpec`, and
`CandidateSet`. CandidateSet validates unique IDs/effective params, exactly one configured
baseline, and expected non-baseline count. Legacy `cmd` parsing accepts only
`python -m sglang.launch_server`, rejects shell operators, and checks params/flags agreement.

- [ ] **Step 4: Replace permissive candidate loading**

Make `_load_candidates()` parse one supported format, reject every malformed line, validate
CandidateSet, and return validated dictionaries without silently truncating input.

- [ ] **Step 5: Run focused and full tests**

Run: `uv run pytest tests/test_schemas.py tests/test_load_candidates.py -q`

Run: `uv run pytest -q`

Expected: all pass.

- [ ] **Step 6: Commit**

`git commit -m "feat: validate jobs targets and candidate sets strictly"`

### Task 2: Safe config generation and atomic output replacement

**Files:**
- Modify: `gen_configs.sh`
- Create: `schemas/validate_cli.py`
- Modify: `tests/test_gen_configs_cli.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write failing CLI tests**

Cover the literal backticks that currently emit `command not found`, invalid JobSpec before AI,
zero candidates, duplicate IDs/configs, bad count, prior valid output preservation, and prompt
retention of `--disable-radix-cache` plus `mamba_radix_cache_strategy`.

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/test_gen_configs_cli.py tests/test_schemas.py -q`

- [ ] **Step 3: Add validation CLI and quoted prompt template**

Expose `python -m schemas.validate_cli job|candidates`. Build the prompt with a single-quoted
heredoc template and controlled placeholder substitution. Remove broad `|| true` masking.

- [ ] **Step 4: Validate temporary output before atomic move**

Write AI output to a restrictive temporary file, validate all semantics, and only then `mv`
over `configs.jsonl`. On failure, leave the old file untouched and return non-zero.

- [ ] **Step 5: Run shell and pytest gates**

Run: `bash -n gen_configs.sh`

Run: `uv run pytest tests/test_gen_configs_cli.py tests/test_schemas.py -q`

Run: `uv run pytest -q`

- [ ] **Step 6: Commit**

`git commit -m "fix: fail closed when generating candidate configs"`

### Task 3: Strict executor CLI, ports, paths, and credential handling

**Files:**
- Modify: `run_executor.sh`
- Modify: `runners/executor.py`
- Modify: `runners/remote.py`
- Modify: `runners/container.py`
- Create: `tests/test_run_executor_cli.py`
- Modify: `tests/test_executor_dryrun.py`

- [ ] **Step 1: Write failing boundary tests**

Cover extra positional arguments, `.json`/`.jsonl` mode detection, temporary directory cleanup,
password environment references, port 0/negative/>65535, replica overflow, duplicate/missing
port flags, and paths containing whitespace/Unicode/single quotes.

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/test_run_executor_cli.py tests/test_executor_dryrun.py -q`

- [ ] **Step 3: Implement strict option parsing and target loading**

Accept only documented argument counts/options. Load TargetSpec through Python. Prefer SSH keys;
resolve `ssh_password_env` at runtime without copying the value into argv or persistent files.
Install EXIT/INT/TERM traps for any unavoidable 0700 temporary directory.

- [ ] **Step 4: Canonicalize ports and argv**

Keep launch arguments as `list[str]`; remove every `--port`, `--port=`, `-p`, and `-p=` instance
only when valid for the launch-server parser, then append one assigned port. Validate the full
port span before SSH. Perform one final `shlex.join` at the remote shell boundary.

- [ ] **Step 5: Run gates and commit**

Run: `bash -n run_executor.sh`

Run: `uv run pytest tests/test_run_executor_cli.py tests/test_executor_dryrun.py -q`

Run: `uv run pytest -q`

Commit: `git commit -m "fix: harden executor cli ports and credentials"`

### Task 4: Local-first preflight and explicit remote ownership

**Files:**
- Create: `runners/preflight.py`
- Modify: `runners/executor.py`
- Modify: `runners/remote.py`
- Create: `tests/test_preflight.py`
- Modify: `tests/test_executor_dryrun.py`

- [ ] **Step 1: Write failing no-mutation tests**

Use a recording fake RemoteRunner. Assert empty/bad candidates, invalid ports, impossible TP,
duplicate allocations, wrong model path, image failure, and actual GPU mismatch produce no
destructive command. Assert zero candidates can never yield COMPLETED.

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/test_preflight.py tests/test_executor_dryrun.py -q`

- [ ] **Step 3: Implement ordered preflight**

Return a `PreflightPlan` after local validation/allocation. Remote read-only checks query
`nvidia-smi`, model directory, image, ports, and job-owned containers. Only then evaluate
`exclusive_host`; default cleanup matches the current job/container prefix, while whole-host
cleanup requires explicit authorization.

- [ ] **Step 4: Check every cleanup result**

Non-zero stop/remove/kill/port-clean results stop execution and enter structured failure data.
Never use `|| true` to convert unknown cleanup state into success.

- [ ] **Step 5: Test and commit**

Run: `uv run pytest tests/test_preflight.py tests/test_executor_dryrun.py -q`

Run: `uv run pytest -q`

Commit: `git commit -m "fix: validate before mutating remote hosts"`

### Task 5: Correct NUMA allocation and property coverage

**Files:**
- Modify: `runners/executor.py`
- Create: `tests/test_gpu_allocator.py`

- [ ] **Step 1: Write allocator property tests**

Generate TP sequences across one, two, and three NUMA groups. Assert exact TP cardinality,
unique GPU membership per batch, in-range IDs, unique ports, deterministic output, and explicit
failure when no placement exists. Include the `[3,3,3,2,1]` 12-GPU regression.

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/test_gpu_allocator.py -q`

- [ ] **Step 3: Implement free-set allocation**

Track available GPU IDs atomically instead of cursor reconstruction. Prefer a single NUMA
group; permit cross-NUMA only under an explicit policy and remove every selected GPU from the
free set.

- [ ] **Step 4: Test and commit**

Run: `uv run pytest tests/test_gpu_allocator.py tests/test_executor_dryrun.py -q`

Commit: `git commit -m "fix: make gpu allocation disjoint and deterministic"`

### Task 6: Fail-closed metrics and typed probe outcomes

**Files:**
- Modify: `runners/metrics.py`
- Modify: `runners/ranker.py`
- Modify: `runners/executor.py`
- Modify: `tests/test_runners_skeleton.py`
- Create: `tests/test_probe_outcomes.py`

- [ ] **Step 1: Write failing parser/status tests**

Cover wrong concurrency, missing required fields, malformed JSON, NaN/Inf, negative values,
partial completion, truncated output, benchmark non-zero exit, server alive/dead distinction,
and known v0.5.16 Triton+EAGLE traceback classification.

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/test_runners_skeleton.py tests/test_probe_outcomes.py -q`

- [ ] **Step 3: Implement strict parse result**

Add ProbeStatus values from the design. Exact record matching and required finite fields are
mandatory. Preserve raw status/failure evidence and actual TP. Ranking accepts only valid `ok`
results; SLA failure is assigned after data health succeeds.

- [ ] **Step 4: Test and commit**

Run: `uv run pytest tests/test_runners_skeleton.py tests/test_probe_outcomes.py -q`

Run: `uv run pytest -q`

Commit: `git commit -m "fix: distinguish invalid probes from sla failures"`

### Task 7: Runtime recovery, probe liveness, and lifecycle cleanup

**Files:**
- Modify: `runners/bench_runner.py`
- Modify: `runners/container.py`
- Modify: `runners/executor.py`
- Create: `runners/lifecycle.py`
- Create: `tests/test_runtime_recovery.py`
- Modify: `tests/test_executor_dryrun.py`

- [ ] **Step 1: Write stateful failure tests**

Cover server death at C48 followed by a valid fresh-server C40, benchmark stall, SSH transport
failure, invalid output, exhausted recovery, Ctrl-C during each container-start position,
cleanup returncode failure, and secondary fill-host replica death/progress.

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/test_runtime_recovery.py tests/test_executor_dryrun.py -q`

- [ ] **Step 3: Move all starts under one lifecycle guard**

Track every started container/process. Always TERM/KILL/stop/remove in `finally`, check results,
and atomically write RUNNING/INTERRUPTED state from signal handlers.

- [ ] **Step 4: Add monitored benchmark execution and recovery**

Poll process state, server health, and evidence growth. Five minutes without progress triggers
TERM then KILL after ten seconds. Infrastructure failures receive up to three recovery attempts
without becoming SLA evidence or consuming statistical/search budgets.

- [ ] **Step 5: Monitor every replica and preserve unique evidence**

Associate each replica with PID/port/log; require all replicas healthy. Name artifacts with
round, concurrency, repeat, recovery attempt, and UUID/nanoseconds.

- [ ] **Step 6: Test and commit**

Run: `uv run pytest tests/test_runtime_recovery.py tests/test_executor_dryrun.py -q`

Run: `uv run pytest -q`

Commit: `git commit -m "fix: recover and report runtime probe failures"`

### Task 8: Exact Round 2 search with seed revalidation

**Files:**
- Modify: `runners/concurrency_search.py`
- Modify: `runners/executor.py`
- Modify: `tests/test_concurrency_search.py`
- Create: `tests/test_search_properties.py`

- [ ] **Step 1: Replace tests that codify seed reuse**

Add tests requiring fresh Round 2 endpoint measurements, three-sample majority, median metrics,
no-bracket fallback from C1, infrastructure failures excluded from verdicts, and certainty
`exact/lower_bound/unknown`.

- [ ] **Step 2: Add exhaustive property test and verify red**

For every true boundary from C*=0 through 256, run production-equivalent two-round search and
assert exact C* when valid evidence is available. Add false-pass/false-fail seeds and alternating
sample order.

Run: `uv run pytest tests/test_concurrency_search.py tests/test_search_properties.py -q`

- [ ] **Step 3: Implement sample groups and dynamic budget**

Represent each concurrency as all raw samples plus a representative median. Round 2 ignores
seed verdict authority, remeasures endpoints, and computes enough evaluations for endpoint
samples, expansion, and bisection through max_cap. Infrastructure retries remain external.

- [ ] **Step 4: Test and commit**

Run: `uv run pytest tests/test_concurrency_search.py tests/test_search_properties.py -q`

Run: `uv run pytest -q`

Commit: `git commit -m "fix: make precise concurrency search statistically robust"`

### Task 9: Measured full-host ranking, intervals, and ties

**Files:**
- Modify: `runners/executor.py`
- Modify: `runners/ranker.py`
- Modify: `run_executor.sh`
- Modify: `tests/test_aggregate_replicas.py`
- Create: `tests/test_ranking_intervals.py`

- [ ] **Step 1: Write failing mode/ranking tests**

Assert Round 2 defaults to one candidate per batch with physical replicas, estimated mode is
explicit, no single-instance seed can win a measured ranking, median/min/max are correct, and
overlapping intervals receive one deterministic rank group.

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/test_aggregate_replicas.py tests/test_ranking_intervals.py -q`

- [ ] **Step 3: Implement explicit measurement modes**

Make full_host the final default and expose `--measurement-mode estimated` as an opt-out.
Aggregate all replicas and all three samples without multiplying a measured total again.

- [ ] **Step 4: Implement interval threshold semantics**

Rank by median. Merge overlapping intervals into rank groups. Emit baseline threshold yes/no/
unknown using interval bounds; baseline missing/nonfinite/nonpositive yields unknown.

- [ ] **Step 5: Test and commit**

Run: `uv run pytest tests/test_aggregate_replicas.py tests/test_ranking_intervals.py -q`

Run: `uv run pytest -q`

Commit: `git commit -m "feat: rank candidates with measured full-host intervals"`

### Task 10: Candidate summary, probe evidence, and final task invariants

**Files:**
- Modify: `runners/reporting.py`
- Modify: `runners/executor.py`
- Modify: `tests/test_reporting.py`
- Create: `tests/test_report_invariants.py`

- [ ] **Step 1: Write failing report tests**

Require one summary row per expected candidate, separate probe evidence, requested TP on failures,
round-result consistency, timestamps, recovery count, boundary certainty, known issue, actual
metrics, failure time, and FINAL only when every candidate is exact/complete.

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/test_reporting.py tests/test_report_invariants.py -q`

- [ ] **Step 3: Implement schema version 2 reports**

Write `candidate_results.jsonl`, `probe_results.jsonl`, `ranking.json`, `task_status.json`, and
`provenance.json` atomically. Retain version-1 compatibility fields while adding explicit mode,
interval, status, and certainty.

- [ ] **Step 4: Improve terminal preview**

Every candidate line includes completion, round/batch, best C, actual throughput/latency,
last failure time/reason, recovery count, and provisional/final certainty.

- [ ] **Step 5: Test and commit**

Run: `uv run pytest tests/test_reporting.py tests/test_report_invariants.py -q`

Run: `uv run pytest -q`

Commit: `git commit -m "feat: publish auditable candidate and probe reports"`

### Task 11: Secret sanitation, CI, static gates, and documentation

**Files:**
- Modify: tracked files under `input/targets/` containing plaintext credentials
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Create: `.github/workflows/ci.yml`
- Create: `scripts/check_no_secrets.py`
- Modify: `README.md`
- Modify: `catalogs/gpu.yaml`
- Create: `tests/test_repository_contracts.py`

- [ ] **Step 1: Write repository-contract tests**

Check no tracked credential values, JobSpec-only files under input/jobs, catalog declared totals,
documentation CLI extensions, and output/archive policy. Expand Pyright include to schemas,
runners, planner, and tests.

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/test_repository_contracts.py -q`

Run: `uv run ruff check .`

Run: `uv run pyright`

- [ ] **Step 3: Sanitize HEAD and add prevention**

Replace plaintext passwords with environment references without printing values. Add CI secret
scan and document required credential rotation/history limitations.

- [ ] **Step 4: Fix changed-surface static errors and docs drift**

Resolve Ruff/Pyright findings rather than excluding production. Correct `.json` mode docs,
baseline strictness wording, catalog totals, and output archive guidance.

- [ ] **Step 5: Run all engineering gates and commit**

Run: `uv run pytest -q`

Run: `uv run ruff check .`

Run: `uv run pyright`

Run: `bash -n gen_configs.sh run_executor.sh`

Commit: `git commit -m "chore: enforce repository reliability gates"`

### Task 12: Offline artifact replay and compatibility verification

**Files:**
- Create: `tests/fixtures/replay/README.md`
- Create: `tests/test_output_replay.py`
- Modify: `runners/reporting.py` as required by discovered compatibility defects

- [ ] **Step 1: Build sanitized minimal replay fixtures**

Extract only necessary metrics/status from qwen36/qwen38 artifacts; exclude credentials and
large logs. Include a stateful engine-crash sequence and c019-style startup failure.

- [ ] **Step 2: Write failing golden assertions**

Assert mechanically valid historical metrics remain equal, six runtime crashes become
incomplete rather than completed, c019 retains requested TP, and all candidate-ID/report-count
invariants hold.

- [ ] **Step 3: Run and fix compatibility only**

Run: `uv run pytest tests/test_output_replay.py tests/test_report_invariants.py -q`

Do not change search policy to match an invalid historical result.

- [ ] **Step 4: Full local verification and commit**

Run: `uv run pytest -q`

Run: `uv run ruff check . && uv run pyright`

Commit: `git commit -m "test: replay historical benchmark failure modes"`

### Task 13: Pro5000 integration validation

**Files:**
- Create: `docs/validation/2026-08-31-pro5000-reliability.md`
- No tracked credential file may be created.

- [ ] **Step 1: Read-only target audit**

Using the authorized `ubuntu@122.51.115.16` target, verify actual GPU count/model/memory,
current containers/processes, model paths, image, disk, and ports. Record sanitized facts only.

- [ ] **Step 2: Run one isolated baseline smoke test**

Use a job-owned container/port and confirm startup, health, one benchmark, checked cleanup, and
no unrelated process mutation.

- [ ] **Step 3: Run four-candidate canary**

Cover TP1/TP2/TP4, all-candidate Round 1/2, three-sample evidence, one controlled invalid/startup
failure, final/provisional semantics, and post-run cleanup. Stop and fix locally on any failure.

- [ ] **Step 4: Run 32-candidate validation**

Only after the canary passes, run the existing 32-candidate job. Verify one summary row per
candidate, separate probe rows, measured full-host mode, no false completed candidates, and
cross-file invariants.

- [ ] **Step 5: Commit sanitized validation report**

Run local gates once more after any integration fix.

Commit: `git commit -m "docs: record pro5000 reliability validation"`

### Task 14: Final review and remote branch publication

**Files:**
- All changed files from Tasks 1–13.

- [ ] **Step 1: Review branch diff and history**

Run: `git diff --check origin/main...HEAD`

Run: `git status --short`

Run: `git log --oneline origin/main..HEAD`

- [ ] **Step 2: Run final verification**

Run: `uv run pytest -q`

Run: `uv run ruff check .`

Run: `uv run pyright`

Run: `bash -n gen_configs.sh run_executor.sh`

Run: `uv run python scripts/check_no_secrets.py`

- [ ] **Step 3: Independent code review**

Review correctness, security, compatibility, state-machine transitions, search properties,
and GPU validation evidence. Resolve every P0/P1 finding with a test and focused commit.

- [ ] **Step 4: Push only the feature branch**

Run: `git push -u origin fix/reliability-hardening`

Do not merge main automatically. Report branch, commits, exact gate outputs, GPU validation
scope, and any remaining known limitations.
