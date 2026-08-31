"""Adaptive concurrency ("goodput boundary") search.

PURE: no docker / ssh / network / wall-clock / randomness. Deterministic given
deterministic callbacks, so it is unit-testable offline with synthetic monotone
pass/fail sequences.

This module ONLY decides which concurrencies to probe and returns the list of
RunResults. Goodput and ranking stay entirely in ranker.py: _best_qualifying
takes max total_throughput over health+SLA-qualifying runs, and because every
probed C (including C*) is recorded here, that max EQUALS throughput(C*). This
module never imports ranker or sla and never recomputes goodput.

Why exponential-expansion + binary-search (not a fixed grid): throughput rises
monotonically with concurrency to a plateau while TTFT/TPOT rise, so the set of
C that pass the SLA is a prefix [start, C*]. A fixed grid only finds a lower
bound on C* (and the bias differs per candidate, so it can flip the ranking).
Expansion breaks past any preset ceiling; bisection pins C* in log2 probes.

Boundary noise is handled symmetrically: every precise concurrency is measured
three fresh times, the SLA verdict is the majority, and the representative is
the field-wise median. Round-1 seeds may choose which coordinates are tried
first, but their verdicts and metrics never enter the authoritative result set.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import ceil, log2
from statistics import median

from runners.metrics import (
    SEARCH_VERDICT_STATUSES,
    ProbeStatus,
    RunResult,
)

Evaluate = Callable[[int], RunResult]    # run ONE bench at C -> parsed metrics
Qualifies = Callable[[RunResult], bool]  # health_check(...) AND passes_sla(...)
OnEvaluateException = Callable[[int, Exception], RunResult]


@dataclass(frozen=True)
class SampleGroup:
    """All fresh statistical samples and their field-wise median result."""

    concurrency: int
    samples: tuple[RunResult, ...]
    representative: RunResult
    qualifies: bool


@dataclass
class SearchOutcome:
    results: list[RunResult]      # one median representative per complete fresh group,
                                  # plus a terminal infrastructure result when present
    c_star: int | None         # largest qualifying C; None when fresh C=1 fails
                               # or when no complete statistical verdict exists
    stop_reason: str              # found_boundary | hit_cap | c1_failed | max_probes
    last_pass: int | None      # == c_star (kept explicit for diagnostics/logging)
    first_fail: int | None     # smallest failing C above last_pass; None if none seen
    num_evals: int                # evaluate() calls made THIS call (budget spent;
                                  #   includes confirm re-probes, excludes seeds)
    newly_probed: list[int]       # distinct C actually evaluated THIS call (excl. seeds),
                                  #   in probe order -> evidence is pulled only for these
    log: list[str] = field(default_factory=list)
    startup_attempts: int = 1
    failures: list[dict] = field(default_factory=list)
    complete: bool = False
    certainty: str = "unknown"
    sample_groups: dict[int, SampleGroup] = field(default_factory=dict)
    incomplete_samples: dict[int, tuple[RunResult, ...]] = field(default_factory=dict)


class _BudgetExhausted(Exception):
    """Internal control-flow signal: max_probes reached. Caught at top level to
    finalize the outcome cleanly (last_pass is always a valid lower bound)."""


class _ProbeIncomplete(Exception):
    """A probe did not produce a valid pass/fail verdict."""

    def __init__(self, result: RunResult) -> None:
        super().__init__(str(result.status))
        self.result = result


class _MonotonicityConflict(Exception):
    """Fresh majority groups contradict the required pass-prefix model."""


def required_sample_budget(
    max_cap: int,
    *,
    start: int = 1,
    factor: int = 2,
    samples_per_concurrency: int = 3,
    seed_hint_endpoints: int = 2,
) -> int:
    """Conservative valid-sample budget for expansion plus exact bisection.

    The seed allowance covers fresh revalidation of up to two hinted endpoints;
    if those hints are false, the ordinary C=1 expansion still fits.  For the
    production domain 1..256 this is 48 samples without seeds and 54 with both
    endpoint hints.
    """

    start = max(1, int(start))
    max_cap = max(start, int(max_cap))
    factor = max(2, int(factor))
    samples_per_concurrency = max(1, int(samples_per_concurrency))
    seed_hint_endpoints = max(0, int(seed_hint_endpoints))
    expansion_points = [start]
    while expansion_points[-1] < max_cap:
        expansion_points.append(min(expansion_points[-1] * factor, max_cap))
    widest_bracket = max(
        (right - left for left, right in zip(expansion_points, expansion_points[1:])),
        default=1,
    )
    bisection_points = ceil(log2(widest_bracket)) if widest_bracket > 1 else 0
    group_budget = len(expansion_points) + bisection_points + seed_hint_endpoints
    return group_budget * samples_per_concurrency


def search_saturation(
    evaluate: Evaluate,
    qualifies: Qualifies,
    *,
    start: int = 1,
    factor: int = 2,
    max_cap: int = 256,
    max_probes: int | None = None,
    refine: bool = True,
    confirm: int = 3,
    seeds: list[RunResult] | None = None,  # round-1 coordinates used only as hints
    on_evaluate_exception: OnEvaluateException | None = None,
) -> SearchOutcome:
    start = max(1, int(start))
    factor = max(2, int(factor))
    max_cap = max(start, int(max_cap))
    confirm = max(1, int(confirm))
    max_probes = (
        required_sample_budget(
            max_cap,
            start=start,
            factor=factor,
            samples_per_concurrency=confirm,
            seed_hint_endpoints=2 if seeds else 0,
        )
        if max_probes is None
        else int(max_probes)
    )

    cache: dict[int, RunResult] = {}       # C -> fresh median representative
    verdict: dict[int, bool] = {}          # C -> fresh majority verdict
    sample_groups: dict[int, SampleGroup] = {}
    incomplete_samples: dict[int, tuple[RunResult, ...]] = {}
    order: list[int] = []  # record order of distinct C (seeds asc, then probe order)
    newly: list[int] = []  # distinct C evaluated this call (excludes seeds)
    log: list[str] = []
    state = {"evals": 0}
    infrastructure_failure: RunResult | None = None

    def _search_verdict(result: RunResult) -> bool:
        """Return a statistical verdict or stop on non-statistical evidence."""

        try:
            status = ProbeStatus(result.status)
        except (TypeError, ValueError):
            result.status = ProbeStatus.INVALID_RESULT
            result.failure_reason = result.failure_reason or "unknown probe status"
            raise _ProbeIncomplete(result)
        if status not in SEARCH_VERDICT_STATUSES:
            raise _ProbeIncomplete(result)
        try:
            callback_verdict = bool(qualifies(result))
        except Exception as exc:
            result.status = ProbeStatus.INVALID_RESULT
            result.failure_reason = (
                f"SLA predicate raised {type(exc).__name__}: {exc}"
            )
            raise _ProbeIncomplete(result) from exc
        try:
            status = ProbeStatus(result.status)
        except (TypeError, ValueError):
            result.status = ProbeStatus.INVALID_RESULT
            result.failure_reason = result.failure_reason or "unknown probe status"
            raise _ProbeIncomplete(result)
        if status not in SEARCH_VERDICT_STATUSES:
            raise _ProbeIncomplete(result)
        if status == ProbeStatus.SLA_FAILED:
            return False
        if not callback_verdict:
            result.status = ProbeStatus.SLA_FAILED
            result.failure_reason = result.failure_reason or "SLA predicate failed"
            return False
        return True

    # Seeds are navigation hints only.  Their metrics and verdicts never enter
    # the authoritative Round-2 cache; hinted endpoints are always remeasured.
    seed_hints: list[tuple[int, bool]] = []
    for r in sorted(seeds or [], key=lambda s: int(s.concurrency)):
        c = int(r.concurrency)
        if start <= c <= max_cap:
            try:
                seed_verdict = _search_verdict(r)
            except _ProbeIncomplete:
                continue
            seed_hints.append((c, seed_verdict))

    def _raw_probe(c: int) -> RunResult:
        # Budget is charged only after this returns typed statistical evidence.
        # Executor-owned infrastructure retries and terminal failures are not
        # search samples and must not consume this counter.
        try:
            return evaluate(c)
        except Exception as exc:
            if on_evaluate_exception is None:
                raise
            return on_evaluate_exception(c, exc)

    def _median_representative(
        samples: list[RunResult], *, majority_qualifies: bool
    ) -> RunResult:
        numeric_fields = (
            "num_prompts",
            "completed",
            "success_rate",
            "request_throughput",
            "output_throughput",
            "total_throughput",
            "mean_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "p99_tpot_ms",
            "total_output_tokens",
            "avg_output_tokens",
            "duration",
        )
        medians = {
            name: median(getattr(sample, name) for sample in samples)
            for name in numeric_fields
        }
        for name in ("num_prompts", "completed", "total_output_tokens"):
            medians[name] = int(medians[name])
        failing_reason = next(
            (sample.failure_reason for sample in samples if sample.failure_reason),
            None,
        )
        return replace(
            samples[0],
            **medians,
            status=(ProbeStatus.OK if majority_qualifies else ProbeStatus.SLA_FAILED),
            failure_reason=(None if majority_qualifies else failing_reason),
            raw=dict(samples[0].raw),
        )

    def probe(c: int):
        """Evaluate one complete fresh sample group and memoize its majority."""
        c = max(start, min(int(c), max_cap))
        if c in verdict:
            return c, verdict[c]
        # Never start a group that the remaining statistical budget cannot
        # finish. This keeps every committed C at exactly ``confirm`` samples.
        if max_probes - state["evals"] < confirm:
            raise _BudgetExhausted()
        samples: list[RunResult] = []
        votes: list[bool] = []
        for _ in range(confirm):
            result = _raw_probe(c)
            try:
                sample_verdict = _search_verdict(result)
            except _ProbeIncomplete:
                incomplete_samples[c] = tuple([*samples, result])
                cache[c] = result
                order.append(c)
                newly.append(c)
                raise
            samples.append(result)
            votes.append(sample_verdict)
            state["evals"] += 1
        pass_votes = sum(votes)
        ok = pass_votes > len(votes) / 2
        if pass_votes * 2 == len(votes):
            # Deterministic compatibility for callers that request an even
            # legacy confirmation count. Production Round 2 always requests 3.
            ok = votes[-1]
        representative = _median_representative(samples, majority_qualifies=ok)
        verdict[c] = ok
        cache[c] = representative
        sample_groups[c] = SampleGroup(
            concurrency=c,
            samples=tuple(samples),
            representative=representative,
            qualifies=ok,
        )
        order.append(c)
        newly.append(c)
        if any(
            fail_c < pass_c
            for fail_c, fail_ok in verdict.items()
            if not fail_ok
            for pass_c, pass_ok in verdict.items()
            if pass_ok
        ):
            raise _MonotonicityConflict()
        return c, ok

    def last_pass() -> int | None:
        ps = [c for c, v in verdict.items() if v]
        return max(ps) if ps else None         # optimistic max-pass (monotone prefix)

    def first_fail_above(lp: int | None) -> int | None:
        fs = [c for c, v in verdict.items() if (not v) and (lp is None or c > lp)]
        return min(fs) if fs else None         # ignores a flaky fail BELOW a confirmed pass

    capped = False
    exhausted = False
    monotonicity_conflict = False
    try:
        if infrastructure_failure is not None:
            raise _ProbeIncomplete(infrastructure_failure)
        seed_passes = [c for c, ok in seed_hints if ok]
        hinted_pass = max(seed_passes) if seed_passes else None
        seed_fails = [
            c
            for c, ok in seed_hints
            if not ok and (hinted_pass is None or c > hinted_pass)
        ]
        hinted_fail = min(seed_fails) if seed_fails else None
        for hinted_c in (hinted_pass, hinted_fail):
            if hinted_c is not None:
                probe(hinted_c)
        if last_pass() is None and start not in verdict:
            # A fresh failure at a high hinted endpoint does not prove C=1
            # failed. Establish the lower endpoint before declaring c1_failed.
            probe(start)
        # ---- Phase 1: exponential expansion to first fail / cap / budget --------
        while True:
            lp = last_pass()
            ff = first_fail_above(lp)
            if ff is not None:                 # bracket exists (or smallest C already fails)
                break
            if lp is not None and lp >= max_cap:
                capped = True
                log.append(f"expansion reached cap C={lp} with no fail "
                           f"(measured goodput is a LOWER BOUND)")
                break
            nxt = start if lp is None else min(lp * factor, max_cap)
            probe(nxt)                         # strictly > lp (or == start); may exhaust budget

        # ---- Phase 2: bisection for the LARGEST qualifying C --------------------
        lp = last_pass()
        ff = first_fail_above(lp)
        if refine and lp is not None and ff is not None and (ff - lp) > 1:
            lo, hi = lp, ff                    # invariant: lo qualifies, hi does not
            while (hi - lo) > 1:               # off-by-one safe: ff==lp+1 -> body never runs
                mid = (lo + hi) // 2           # strictly inside (lo, hi)
                _, ok = probe(mid)             # may exhaust budget
                if ok:
                    lo = mid
                else:
                    hi = mid
    except _BudgetExhausted:
        exhausted = True                       # last_pass() below is a valid lower bound
    except _ProbeIncomplete as exc:
        infrastructure_failure = exc.result
    except _MonotonicityConflict:
        monotonicity_conflict = True
        log.append("fresh sample-group majorities violate the pass-prefix model")

    lp = last_pass()
    ff = first_fail_above(lp)
    if infrastructure_failure is not None:
        c_star, stop = None, str(infrastructure_failure.status)
    elif monotonicity_conflict:
        c_star, stop = None, "monotonicity_conflict"
    elif verdict.get(start) is False:
        c_star, stop = None, "c1_failed"       # fresh smallest C fails
    elif exhausted:
        c_star, stop = lp, "max_probes"        # ran out of budget; lp is a lower bound
    elif lp is None:
        # A high seed hint may have been freshly rejected without enough budget
        # left to establish C=1. That is unknown, never an invented C1 boundary.
        c_star, stop = None, "max_probes"
    elif capped:
        c_star, stop = lp, "hit_cap"           # cap passed; true C* >= cap
    else:
        c_star, stop = lp, "found_boundary"    # tight (refine) or coarse (refine=False)

    log.insert(0, f"c_star={c_star} stop={stop} evals={state['evals']} "
                  f"distinct={len(order)} last_pass={lp} first_fail={ff}")
    exact_completion = (
        refine
        and infrastructure_failure is None
        and not monotonicity_conflict
        and stop
        in {
        "found_boundary",
        "c1_failed",
        }
    )
    if exact_completion:
        certainty = "exact"
    elif (
        infrastructure_failure is None
        and not monotonicity_conflict
        and lp is not None
    ):
        certainty = "lower_bound"
    else:
        certainty = "unknown"
    return SearchOutcome(
        results=[cache[c] for c in order],
        c_star=c_star,
        stop_reason=stop,
        last_pass=lp,
        first_fail=ff,
        num_evals=state["evals"],
        newly_probed=list(newly),
        log=log,
        complete=exact_completion,
        certainty=certainty,
        sample_groups=sample_groups,
        incomplete_samples=incomplete_samples,
    )
