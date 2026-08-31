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

Boundary noise is handled symmetrically: near C* the same C can flip pass/fail
run-to-run. A lone FAIL is re-probed before it truncates the search, and a lone
boundary-advancing PASS is re-probed before it inflates C* — leaving passes
unconfirmed while confirming fails would let one lucky probe over-report goodput
and flip the ranking (the exact failure this search exists to prevent).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from runners.metrics import (
    SEARCH_VERDICT_STATUSES,
    ProbeStatus,
    RunResult,
)

Evaluate = Callable[[int], RunResult]    # run ONE bench at C -> parsed metrics
Qualifies = Callable[[RunResult], bool]  # health_check(...) AND passes_sla(...)
OnEvaluateException = Callable[[int, Exception], RunResult]


@dataclass
class SearchOutcome:
    results: list[RunResult]      # every DISTINCT probed C, deduped, in record order
                                  #   (seeds ascending first, then probe order);
                                  #   len(results) == number of distinct C probed.
    c_star: int | None         # largest qualifying C; None iff smallest known C fails
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


class _BudgetExhausted(Exception):
    """Internal control-flow signal: max_probes reached. Caught at top level to
    finalize the outcome cleanly (last_pass is always a valid lower bound)."""


class _ProbeIncomplete(Exception):
    """A probe did not produce a valid pass/fail verdict."""

    def __init__(self, result: RunResult) -> None:
        super().__init__(str(result.status))
        self.result = result


def search_saturation(
    evaluate: Evaluate,
    qualifies: Qualifies,
    *,
    start: int = 1,
    factor: int = 2,
    max_cap: int = 256,
    max_probes: int = 12,
    refine: bool = True,
    confirm: int = 2,        # probes to BELIEVE a boundary-crossing verdict. 1 == naive
                             # (trust the first result; exact naive probe counts). 2 ==
                             # re-probe a boundary-MOVING fail OR a boundary-ADVANCING
                             # pass once; any disagreeing re-probe overturns it. Applied
                             # symmetrically to fails and passes (see module docstring).
    seeds: list[RunResult] | None = None,  # round-1 grid RunResults ingested free
    on_evaluate_exception: OnEvaluateException | None = None,
) -> SearchOutcome:
    start = max(1, int(start))
    factor = max(2, int(factor))
    max_cap = max(start, int(max_cap))
    max_probes = int(max_probes)
    confirm = max(1, int(confirm))

    cache: dict = {}       # C -> representative RunResult (overturned verdict's result if flipped)
    verdict: dict = {}     # C -> bool (qualifies)
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

    # ---- ingest seeds for free (no budget): reuse the round-1 grid as a bracket --
    for r in sorted(seeds or [], key=lambda s: int(s.concurrency)):
        c = int(r.concurrency)
        if start <= c <= max_cap and c not in verdict:
            try:
                seed_verdict = _search_verdict(r)
            except _ProbeIncomplete:
                # Prior-round infrastructure evidence is not authoritative in a
                # fresh search.  Do not cache it: the same C remains probeable.
                continue
            cache[c] = r
            order.append(c)
            verdict[c] = seed_verdict

    def _raw_probe(c: int) -> RunResult:
        # Budget check FIRST, before the try, so a real _BudgetExhausted is never
        # masked by the exception adapter below.  Executor-owned failures become
        # typed incomplete evidence and stop this candidate without inventing an
        # SLA boundary; unexpected programming errors still propagate.
        if state["evals"] >= max_probes:
            raise _BudgetExhausted()
        state["evals"] += 1
        try:
            return evaluate(c)
        except Exception as exc:
            if on_evaluate_exception is None:
                raise
            return on_evaluate_exception(c, exc)

    def probe(c: int):
        """Evaluate C (memoized). Returns (C, qualifies). Confirm-the-boundary:
        the first verdict is re-probed up to ``confirm`` attempts; a single
        disagreeing re-probe overturns it. Applied SYMMETRICALLY — a fresh fail
        is re-probed (a flaky fail must not truncate the search) and a fresh pass
        is re-probed (a flaky pass must not inflate C* / goodput). Raises
        _BudgetExhausted only when there is NO budget for the first attempt (the
        caller stops cleanly)."""
        c = max(start, min(int(c), max_cap))
        if c in verdict:                      # DEDUPE: never evaluate the same C twice
            return c, verdict[c]
        r = _raw_probe(c)                      # first attempt (propagates _BudgetExhausted)
        try:
            ok = _search_verdict(r)
        except _ProbeIncomplete:
            cache[c] = r
            order.append(c)
            newly.append(c)
            raise
        attempts = 1
        # Re-probe until we either exhaust ``confirm`` attempts or see a
        # disagreeing result that overturns the initial verdict. One loop handles
        # both directions: the overturn condition is "re-probe disagrees".
        while attempts < confirm:
            try:
                r2 = _raw_probe(c)
            except _BudgetExhausted:
                break                          # keep the verdict we have; caller stops soon
            attempts += 1
            try:
                ok2 = _search_verdict(r2)
            except _ProbeIncomplete:
                cache[c] = r2
                order.append(c)
                newly.append(c)
                raise
            if ok2 != ok:
                verb = "pass" if ok else "fail"
                log.append(f"C={c}: {verb} overturned by re-probe (attempt {attempts})")
                r, ok = r2, ok2                # store the overturning representative
                break
        verdict[c] = ok
        cache[c] = r
        order.append(c)
        newly.append(c)
        return c, ok

    def last_pass() -> int | None:
        ps = [c for c, v in verdict.items() if v]
        return max(ps) if ps else None         # optimistic max-pass (monotone prefix)

    def first_fail_above(lp: int | None) -> int | None:
        fs = [c for c, v in verdict.items() if (not v) and (lp is None or c > lp)]
        return min(fs) if fs else None         # ignores a flaky fail BELOW a confirmed pass

    capped = False
    exhausted = False
    try:
        if infrastructure_failure is not None:
            raise _ProbeIncomplete(infrastructure_failure)
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

    lp = last_pass()
    ff = first_fail_above(lp)
    if infrastructure_failure is not None:
        c_star, stop = None, str(infrastructure_failure.status)
    elif lp is None:
        # No qualifying C. Distinguish "learned nothing" from "smallest C fails".
        if not verdict:
            c_star, stop = None, "max_probes"  # degenerate: never probed anything
        else:
            c_star, stop = None, "c1_failed"   # smallest known C fails
    elif exhausted:
        c_star, stop = lp, "max_probes"        # ran out of budget; lp is a lower bound
    elif capped:
        c_star, stop = lp, "hit_cap"           # cap passed; true C* >= cap
    else:
        c_star, stop = lp, "found_boundary"    # tight (refine) or coarse (refine=False)

    log.insert(0, f"c_star={c_star} stop={stop} evals={state['evals']} "
                  f"distinct={len(order)} last_pass={lp} first_fail={ff}")
    exact_completion = refine and infrastructure_failure is None and stop in {
        "found_boundary",
        "c1_failed",
    }
    if exact_completion:
        certainty = "exact"
    elif infrastructure_failure is None and lp is not None:
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
    )
