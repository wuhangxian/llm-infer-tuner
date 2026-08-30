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

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from runners.metrics import RunResult  # fixed dataclass; treated as opaque here

Evaluate = Callable[[int], RunResult]    # run ONE bench at C -> parsed metrics
Qualifies = Callable[[RunResult], bool]  # health_check(...) AND passes_sla(...)


@dataclass
class SearchOutcome:
    results: List[RunResult]      # every DISTINCT probed C, deduped, in record order
                                  #   (seeds ascending first, then probe order);
                                  #   len(results) == number of distinct C probed.
    c_star: Optional[int]         # largest qualifying C; None iff smallest known C fails
    stop_reason: str              # found_boundary | hit_cap | c1_failed | max_probes
    last_pass: Optional[int]      # == c_star (kept explicit for diagnostics/logging)
    first_fail: Optional[int]     # smallest failing C above last_pass; None if none seen
    num_evals: int                # evaluate() calls made THIS call (budget spent;
                                  #   includes confirm re-probes, excludes seeds)
    newly_probed: List[int]       # distinct C actually evaluated THIS call (excl. seeds),
                                  #   in probe order -> evidence is pulled only for these
    log: List[str] = field(default_factory=list)
    startup_attempts: int = 1
    failures: list[dict] = field(default_factory=list)


class _BudgetExhausted(Exception):
    """Internal control-flow signal: max_probes reached. Caught at top level to
    finalize the outcome cleanly (last_pass is always a valid lower bound)."""


def _crash_result(concurrency: int, reason: str) -> RunResult:
    """Overload collapse / evaluate() raising -> a NON-qualifying RunResult, so the
    point counts as a boundary 'fail' instead of aborting the whole search. The
    crashed point is still recorded in results (ranker ignores non-qualifying).

    Field list is kept in lockstep with runners.metrics.RunResult.
    """
    return RunResult(
        candidate_id="", concurrency=concurrency, num_prompts=0, completed=0,
        success_rate=0.0, request_throughput=0.0, output_throughput=0.0,
        total_throughput=0.0, mean_ttft_ms=float("inf"), p99_ttft_ms=float("inf"),
        mean_tpot_ms=float("inf"), p99_tpot_ms=float("inf"), total_output_tokens=0,
        avg_output_tokens=0.0, duration=0.0, status="health_check_failed",
        failure_reason=reason, raw={},
    )


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
    seeds: Optional[List[RunResult]] = None,  # round-1 grid RunResults ingested free
) -> SearchOutcome:
    start = max(1, int(start))
    factor = max(2, int(factor))
    max_cap = max(start, int(max_cap))
    max_probes = int(max_probes)
    confirm = max(1, int(confirm))

    cache: dict = {}       # C -> representative RunResult (overturned verdict's result if flipped)
    verdict: dict = {}     # C -> bool (qualifies)
    order: List[int] = []  # record order of distinct C (seeds asc, then probe order)
    newly: List[int] = []  # distinct C evaluated this call (excludes seeds)
    log: List[str] = []
    state = {"evals": 0}

    # ---- ingest seeds for free (no budget): reuse the round-1 grid as a bracket --
    for r in sorted(seeds or [], key=lambda s: int(s.concurrency)):
        c = int(r.concurrency)
        if start <= c <= max_cap and c not in verdict:
            verdict[c] = bool(qualifies(r))
            cache[c] = r
            order.append(c)

    def _raw_probe(c: int) -> RunResult:
        # Budget check FIRST, before the try, so a real _BudgetExhausted is never
        # masked by the except-Exception below; evaluate()'s own crashes (overload
        # collapse) become a non-qualifying result rather than aborting the search.
        if state["evals"] >= max_probes:
            raise _BudgetExhausted()
        state["evals"] += 1
        try:
            return evaluate(c)
        except Exception as exc:  # collapse != abort
            return _crash_result(c, f"evaluate_raised:{type(exc).__name__}")

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
        ok = bool(qualifies(r))
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
            ok2 = bool(qualifies(r2))
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

    def last_pass() -> Optional[int]:
        ps = [c for c, v in verdict.items() if v]
        return max(ps) if ps else None         # optimistic max-pass (monotone prefix)

    def first_fail_above(lp: Optional[int]) -> Optional[int]:
        fs = [c for c, v in verdict.items() if (not v) and (lp is None or c > lp)]
        return min(fs) if fs else None         # ignores a flaky fail BELOW a confirmed pass

    capped = False
    exhausted = False
    try:
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

    lp = last_pass()
    ff = first_fail_above(lp)
    if lp is None:
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
    return SearchOutcome(
        results=[cache[c] for c in order],
        c_star=c_star,
        stop_reason=stop,
        last_pass=lp,
        first_fail=ff,
        num_evals=state["evals"],
        newly_probed=list(newly),
        log=log,
    )
