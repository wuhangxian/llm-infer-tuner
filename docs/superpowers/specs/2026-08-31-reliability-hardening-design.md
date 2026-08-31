# LLM Infer Tuner Reliability Hardening Design

**Status:** Proposed for implementation on `fix/reliability-hardening`

**Objective:** Make every published candidate result auditable, complete, and safe to
produce, while preserving full-candidate Round 1/2 coverage and deterministic execution.

## Scope and fixed decisions

This program fixes the tuner-side reliability, search, reporting, CLI, and safety issues
found during the 2026-08-31 audit.

The following decisions are fixed:

- Every input candidate, including baseline, enters both Round 1 and Round 2. There is no
  Top-K pruning.
- Baseline threshold annotates results; it never deletes rows.
- Prefix/Radix cache remains disabled for every measured candidate.
- Phase 2 benchmark command construction remains deterministic and never calls AI.
- There is no whole-job time limit. Startup and individual probe liveness are bounded.
- Restarting the current candidate from the beginning is acceptable after interruption;
  exact request-level resume is not required.
- The confirmed SGLang v0.5.16 Triton+EAGLE tensor-shape defect is an upstream image issue.
  This repository will classify and report it correctly, but will not patch SGLang.
- Existing job/config files remain readable where they are valid. Unsafe or ambiguous
  inputs fail before any remote mutation.

## Non-goals

- Fixing bugs inside the SGLang image.
- Automatically upgrading the serving image during a benchmark.
- Adding a global benchmark deadline.
- Silently deleting or rewriting historical output data.
- Storing or rotating third-party credentials on behalf of the user.

## Safety invariants

1. No SSH connection or destructive command occurs until Job, Target, CandidateSet,
   candidate IDs, commands, ports, and GPU allocation all pass local validation.
2. Empty, partially malformed, duplicate-ID, duplicate-effective-config, or count-mismatched
   candidate sets fail closed and preserve the previous valid output.
   Candidate count means exactly `search.max_candidates` non-baseline candidates, plus exactly
   one baseline when `search.baseline` is present. Without a configured baseline, the total is
   exactly `search.max_candidates`. Multiple baselines are invalid.
3. Remote validation is read-only and verifies actual GPU count/model/memory, model path,
   image availability, port range, and exclusive-host eligibility before cleanup.
4. Cleanup is explicitly authorized by the target and scoped to this job by default.
   Whole-host cleanup requires an explicit `exclusive_host: true` contract.
5. Credentials are not copied into persistent temporary directories, command lines, output
   reports, or Git-tracked examples.

## Architecture

### 1. Typed boundary validation

Add strict Pydantic models for Target, Candidate, and CandidateSet beside `JobSpec`.
Validation uses strict scalar types, finite numeric values, bounded IDs, a safe ID character
set, unique candidate IDs, one optional baseline, legal TP/EP values, safe ports, and an exact
candidate-count contract.

Candidates are represented as structured parameters. Legacy `cmd` remains readable during
the migration, but must parse to the approved `python -m sglang.launch_server` entry point,
must not contain shell control syntax, and must agree with `params`. Runtime code constructs
the final argv from validated values rather than accepting arbitrary shell fragments.

Cache fairness is canonical rather than advisory: every cache-enabling spelling is rejected;
all duplicate disable spellings are removed; the generated argv contains exactly one
`--disable-radix-cache`; effective params always contain `disable_radix_cache: true`.
The user's requested Mamba radix strategy remains in audit metadata, while its effective value
is recorded as inactive under Radix-off.

`gen_configs.sh` validates the Job before invoking AI and validates a temporary CandidateSet
before atomically replacing `configs.jsonl`. The prompt uses a quoted template so Markdown
backticks cannot trigger shell substitution.

### 2. Preflight and ownership

Split executor preparation into ordered stages:

1. Local parse and validation.
2. Local dry-run allocation of every batch, checking disjoint GPUs and unique ports.
3. Read-only remote inspection.
4. Explicit host-ownership/cleanup authorization.
5. Scoped cleanup and execution.

An invalid model path, image, hardware declaration, port, or candidate set therefore cannot
clear the remote machine. Cleanup results are checked; a non-zero cleanup result stops the job
and is recorded.

### 3. Probe and candidate state machine

Replace the current pass/fail collapse with typed probe outcomes:

- `ok`: benchmark completed and metrics are valid.
- `sla_failed`: valid metrics, but an SLA predicate failed.
- `startup_failed`: server never became ready after bounded attempts.
- `runtime_failed`: a previously healthy server died during a probe.
- `benchmark_failed`: benchmark process or output failed independently of server health.
- `transport_failed`: SSH/container transport failed.
- `invalid_result`: output did not match the required benchmark schema/concurrency.
- `interrupted`: local signal or orchestration interruption.

Only `ok` and `sla_failed` are valid search verdicts. Infrastructure/runtime failures never
become boundary evidence.

After every probe, the executor checks the benchmark exit status, exact output record, and
server health. On `runtime_failed`, it preserves the evidence, restarts the candidate under the
startup retry policy, and repeats the current concurrency. If recovery is exhausted, that
candidate becomes `incomplete`; the executor continues with the remaining candidates.

Probe liveness has no fixed successful-runtime ceiling. The benchmark runs under a monitor
that polls process state, server health, and evidence-file/log growth. Five minutes with no
progress is a stall; the monitor sends TERM, waits ten seconds, then sends KILL. Startup,
runtime, transport, benchmark, and invalid-result recovery each use at most three attempts.
Failed infrastructure attempts consume the recovery-attempt budget, but never consume the
three statistical samples or the search probe budget. Exhaustion marks the candidate
`incomplete` and preserves all attempts.

The known upstream image defect is detected only when all of the following match: runtime
engine version `0.5.16`, effective Triton attention plus EAGLE, and a server traceback through
`triton_backend._update_target_verify_buffers` containing the custom-mask expanded-size
mismatch. It is reported as `runtime_failed` with
`known_issue=sglang-0.5.16-triton-eagle-custom-mask-shape`; it receives the same bounded
recovery treatment as any engine failure and is never boundary evidence. Other crashes remain
generic `runtime_failed` rather than being guessed.

All container starts are inside one outer lifecycle `try/finally`. SIGINT/SIGTERM write an
atomic `INTERRUPTED` task status and attempt checked cleanup. The next invocation may restart
the interrupted candidate; completed candidate reuse is optional and is not required in this
program.

### 4. Search correctness

Round 1 remains a cheap exponential grid with one measurement per point. It is explicitly
diagnostic and produces a coarse bracket, not an exact `c_star`.

Round 2:

- remeasures both Round 1 bracket endpoints on a fresh server;
- uses three samples for a boundary decision;
- classifies by majority SLA verdict and stores the median metrics plus all raw samples;
- uses a computed probe budget sufficient for endpoint validation, expansion, and bisection
  through `max_cap`;
- returns `exact`, `lower_bound`, or `unknown` certainty;
- never publishes `max_probes`, `hit_cap`, runtime failure, or invalid evidence as exact.

Every candidate is scheduled in Round 2 even when Round 1 produced no valid bracket. In that
case Round 2 starts fresh at C=1 and performs bounded exponential expansion followed by
bisection. If valid evidence still cannot be obtained, the candidate remains present as
`incomplete/unknown`; it is never skipped.

The complete candidate set is refined. Search tests exhaustively cover every monotone true
boundary from zero through `max_cap`, noisy endpoint seeds, alternating samples, and stateful
server death.

### 5. Strict metrics and ranking

Benchmark parsing fails closed. The selected record must exactly match requested concurrency,
contain every required field, contain only finite non-negative values, complete the expected
request count, and produce the expected output-token range. Ranking additionally requires
`probe_status == ok` and exact candidate completion.

Two measurement modes are explicit:

- `estimated`: single-instance measurement with a clearly named estimated per-host value.
- `full_host`: physically starts `floor(gpu_count / tp_size)` replicas and measures their
  aggregate. This is the default for final Round 2 ranking.

Full-host candidates run one at a time so CPU, NUMA, PCIe, and client contention do not vary
with batch composition. The report preserves both single-instance and measured full-host
metrics. Differences below measured repeat variability are marked as ties instead of implying
false precision.

For each measured point, the three samples produce `goodput_min`, `goodput_median`, and
`goodput_max`; ranking uses the median. Candidates whose `[min,max]` goodput intervals overlap
belong to the same deterministic rank group (overlapping intervals are merged transitively
after median-descending sort). Baseline-threshold status is `yes` only when the candidate's
minimum clears the threshold, `no` only when its maximum is below it, and `unknown` when the
interval crosses it or the baseline is invalid.

### 6. Reporting and evidence

Every candidate has exactly one candidate-summary row, including failed and interrupted
candidates. Repeated measurements are separate probe-evidence records linked by candidate ID,
round, concurrency, and probe ID. A candidate-summary row includes:

- candidate ID and effective parameters;
- requested TP/EP and actual instance count;
- per-round batch, bracket, tested concurrencies, and measurement mode;
- candidate start/end/final-failure timestamps;
- best-point median throughput, interval, request rate, TTFT, TPOT, success rate, and
  output-token health;
- final typed status, failure reason, recovery count, boundary certainty, and completion state;
- baseline delta and threshold result (`yes`, `no`, or `unknown`).

Each probe-evidence record includes round, batch, concurrency, repeat index, recovery attempt,
start/end/failure timestamps, raw metrics, status, failure reason, server-health evidence, and
artifact filenames.

Task status is `COMPLETED/FINAL` only when every expected candidate has an exact, complete
Round 2 result. Otherwise it is `INCOMPLETE/PROVISIONAL` or `INTERRUPTED`.

Evidence filenames include round, concurrency, repeat/attempt number, and a collision-proof
identifier. Candidate diagnostics and task/ranking files obey cross-file invariants verified by
tests. Provenance records Git SHA, job/config hashes, image reference/digest when available,
actual GPU information, engine version, and timestamps.

### 7. CLI, secrets, and repository hygiene

- `run_executor.sh` accepts only documented modes/options and rejects extra arguments.
- Port rewriting removes every prior port spelling and appends one validated assigned port.
- Paths remain argv tokens until the final shell boundary, supporting whitespace, Unicode,
  and quotes.
- Single-file mode parses directly or guarantees trap-based removal of restrictive temporary
  files.
- Tracked targets containing plaintext credentials are sanitized in this branch and migrated
  to `ssh_password_env` references. SSH keys are preferred; password compatibility reads from
  that environment variable or a protected file descriptor without exposing argv. CI scans
  tracked files for credential patterns. Because old commits cannot be made secret by deleting
  HEAD content, history rewriting is not automatic: the user must rotate every exposed
  password/token, and any coordinated history purge is a separate repository-owner action.
- CI runs pytest, Ruff, Pyright over production packages, and shell syntax/static checks.
- Raw benchmark artifacts remain available, but summaries/manifests are separated from raw
  evidence so an optional archive/LFS policy can be adopted without changing result semantics.

## Compatibility and migration

Strict validation is introduced with clear errors naming the file, candidate, field, and
expected value. Valid current JobSpec and CandidateSet files continue to work. Deprecated
unsafe forms receive an actionable error rather than being guessed or silently rewritten.

Existing result consumers retain current fields during one compatibility window. New status,
certainty, measurement-mode, and provenance fields are additive. Ambiguous fields such as
`goodput_per_host` are retained but accompanied by explicit `estimated` or `measured` fields
until callers migrate.

## Verification gates

Each behavior change follows red-green-refactor and is committed independently.

Before GPU use:

1. Focused new tests pass.
2. The complete pytest suite passes.
3. Ruff and Pyright pass for the changed production surface.
4. Shell syntax and CLI regression tests pass.
5. Offline replay of the existing qwen36/qwen38 artifacts preserves mechanically valid
   metrics while changing the six false-completed runtime crashes to incomplete failures.

GPU validation on the authorized Pro5000 host proceeds in increasing scope:

1. Read-only hardware and ownership check.
2. One baseline startup/health/single probe on an isolated port.
3. Four-candidate test covering TP1/TP2/TP4 and one controlled failing candidate.
4. Thirty-two candidates only after the smaller test and cleanup audit pass.

The remote branch is pushed only after local and GPU gates pass and the worktree is clean.

## Delivery decomposition

Implementation is split into independently testable plans and commits:

1. Boundary safety and schemas.
2. Runtime lifecycle and typed failure handling.
3. Search, metrics, and ranking correctness.
4. Reporting, CLI, CI, documentation, and artifact hygiene.

No phase may depend on an untested partial behavior from a later phase. Each phase can be
reverted independently.
