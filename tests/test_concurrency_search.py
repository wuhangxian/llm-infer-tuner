"""Offline tests for runners.concurrency_search (no ssh/docker/network).

The search is exercised with synthetic ``evaluate``/``qualifies`` callbacks so
the monotone-boundary behaviour, noise handling, dedupe, budget/cap limits, and
seeding can all be asserted deterministically. Throughput is made to rise with C
so goodput (max total_throughput over qualifying probes, as ranker computes it)
lands on the C* run.
"""

from __future__ import annotations

from collections import Counter

from runners.concurrency_search import search_saturation
from runners.metrics import RunResult
from runners.ranker import candidate_goodput
from schemas.job_spec import SLA


# --- synthetic harness --------------------------------------------------------


def _mk(concurrency: int, *, ok: bool, throughput: float | None = None) -> RunResult:
    """A RunResult that qualifies (ok=True) or not, with throughput ~ concurrency."""
    tput = throughput if throughput is not None else float(concurrency * 100)
    if ok:
        return RunResult(
            candidate_id="c", concurrency=concurrency, num_prompts=concurrency * 4,
            completed=concurrency * 4, success_rate=1.0,
            request_throughput=1.0, output_throughput=tput, total_throughput=tput,
            mean_ttft_ms=100.0, p99_ttft_ms=200.0, mean_tpot_ms=10.0, p99_tpot_ms=20.0,
            total_output_tokens=concurrency * 4 * 1000, avg_output_tokens=1000.0,
            duration=10.0, status="ok",
        )
    # non-qualifying: blow past the TTFT gate (still a normal, non-crash result)
    return RunResult(
        candidate_id="c", concurrency=concurrency, num_prompts=concurrency * 4,
        completed=concurrency * 4, success_rate=1.0,
        request_throughput=1.0, output_throughput=tput, total_throughput=tput,
        mean_ttft_ms=9e9, p99_ttft_ms=9e9, mean_tpot_ms=10.0, p99_tpot_ms=20.0,
        total_output_tokens=concurrency * 4 * 1000, avg_output_tokens=1000.0,
        duration=10.0, status="ok",
    )


def _sla() -> SLA:
    return SLA(max_avg_ttft_ms=2000.0, max_avg_tpot_ms=80.0, min_success_rate=0.99)


def _qualifies(sla: SLA):
    def q(r: RunResult) -> bool:
        return (
            r.mean_ttft_ms <= sla.max_avg_ttft_ms
            and r.mean_tpot_ms <= sla.max_avg_tpot_ms
            and r.success_rate >= sla.min_success_rate
            and r.avg_output_tokens >= 900  # health gate (output_len 1000)
        )
    return q


def _threshold_evaluate(cstar: int, counter: Counter | None = None):
    """evaluate: qualifies iff C <= cstar. Records call counts if a Counter given."""
    def evaluate(c: int) -> RunResult:
        if counter is not None:
            counter[c] += 1
        return _mk(c, ok=(c <= cstar))
    return evaluate


# --- clean monotone boundary --------------------------------------------------


def test_clean_boundary_refine_finds_cstar_and_dedupes() -> None:
    counter: Counter = Counter()
    sla = _sla()
    oc = search_saturation(
        _threshold_evaluate(10, counter), _qualifies(sla),
        start=1, factor=2, max_cap=256, max_probes=20, refine=True, confirm=1,
    )
    assert oc.c_star == 10
    assert oc.stop_reason == "found_boundary"
    assert oc.first_fail == 11
    # every distinct C probed exactly once (no C evaluated twice)
    assert all(v == 1 for v in counter.values())
    assert len(oc.results) == len(counter)
    # goodput via the unchanged ranker == throughput at C*=10
    assert candidate_goodput(oc.results, sla, output_len=1000) == 1000.0


def test_off_by_one_no_bisect() -> None:
    sla = _sla()
    oc = search_saturation(
        _threshold_evaluate(1), _qualifies(sla),
        start=1, factor=2, refine=True, confirm=1,
    )
    assert oc.c_star == 1
    assert oc.first_fail == 2
    assert oc.stop_reason == "found_boundary"
    assert oc.num_evals == 2  # C=1 pass, C=2 fail; nothing to bisect


# --- degenerate: smallest C fails ---------------------------------------------


def test_c1_failed_naive() -> None:
    sla = _sla()
    oc = search_saturation(
        _threshold_evaluate(0), _qualifies(sla), start=1, refine=True, confirm=1,
    )
    assert oc.c_star is None
    assert oc.stop_reason == "c1_failed"
    assert oc.first_fail == 1
    assert oc.num_evals == 1
    assert len(oc.results) == 1  # the failing run is still recorded


def test_c1_failed_confirmed_spends_second_eval() -> None:
    sla = _sla()
    oc = search_saturation(
        _threshold_evaluate(0), _qualifies(sla), start=1, refine=True, confirm=2,
    )
    assert oc.c_star is None
    assert oc.stop_reason == "c1_failed"
    assert oc.num_evals == 2  # confirm the start-fail before discarding


# --- noise: the bug the adversarial review caught -----------------------------


def test_flaky_fail_during_expansion_is_overturned() -> None:
    """A single flaky FAIL at C=4 must not truncate the search (true C*=32)."""
    sla = _sla()
    seen: Counter = Counter()

    def evaluate(c: int) -> RunResult:
        seen[c] += 1
        if c == 4 and seen[c] == 1:
            return _mk(c, ok=False)  # first probe of C=4 flukes a fail
        return _mk(c, ok=(c <= 32))

    oc = search_saturation(
        evaluate, _qualifies(sla),
        start=1, factor=2, max_cap=256, max_probes=30, refine=True, confirm=2,
    )
    assert oc.c_star == 32
    assert oc.stop_reason == "found_boundary"


def test_flaky_pass_during_expansion_is_overturned() -> None:
    """SCENARIO A from the adversarial review: a single flaky PASS above the true
    boundary (C*=4) must NOT inflate C*/goodput. With symmetric pass-confirm the
    flaky pass at C=8 is re-probed and overturned -> c_star=4, not 8."""
    sla = _sla()
    seen: Counter = Counter()

    def evaluate(c: int) -> RunResult:
        seen[c] += 1
        if c == 8 and seen[c] == 1:
            # flaky pass, and it even reports a burst throughput that would be a
            # 3x goodput over-report if trusted.
            return _mk(c, ok=True, throughput=9000.0)
        return _mk(c, ok=(c <= 4))

    oc = search_saturation(
        evaluate, _qualifies(sla),
        start=1, factor=2, max_cap=256, max_probes=30, refine=True, confirm=2,
    )
    assert oc.c_star == 4, f"flaky pass inflated C* to {oc.c_star}"
    # goodput must be throughput(4)=400, NOT the flaky 9000 burst at C=8
    assert candidate_goodput(oc.results, sla, output_len=1000) == 400.0


def test_flaky_pass_during_bisection_is_overturned() -> None:
    """SCENARIO B: flaky PASS at a bisection midpoint (true C*=4)."""
    sla = _sla()
    seen: Counter = Counter()

    def evaluate(c: int) -> RunResult:
        seen[c] += 1
        if c == 6 and seen[c] == 1:
            return _mk(c, ok=True, throughput=8000.0)  # flaky pass mid-bisect
        return _mk(c, ok=(c <= 4))

    oc = search_saturation(
        evaluate, _qualifies(sla),
        start=1, factor=2, max_cap=256, max_probes=30, refine=True, confirm=2,
    )
    assert oc.c_star == 4
    assert candidate_goodput(oc.results, sla, output_len=1000) == 400.0


# --- cap and budget -----------------------------------------------------------


def test_hit_cap_all_pass() -> None:
    sla = _sla()
    oc = search_saturation(
        _threshold_evaluate(10_000), _qualifies(sla),
        start=1, factor=2, max_cap=16, max_probes=30, refine=True, confirm=2,
    )
    assert oc.c_star == 16
    assert oc.stop_reason == "hit_cap"
    assert oc.first_fail is None


def test_max_probes_mid_expansion_returns_lower_bound() -> None:
    sla = _sla()
    oc = search_saturation(
        _threshold_evaluate(10_000), _qualifies(sla),
        start=1, factor=2, max_cap=256, max_probes=4, refine=True, confirm=1,
    )
    # 1,2,4,8 pass (4 evals); 5th probe (16) hits the budget
    assert oc.c_star == 8
    assert oc.stop_reason == "max_probes"
    assert oc.num_evals == 4


def test_max_probes_degenerate_zero() -> None:
    sla = _sla()
    oc = search_saturation(
        _threshold_evaluate(10), _qualifies(sla), max_probes=0,
    )
    assert oc.c_star is None
    assert oc.stop_reason == "max_probes"
    assert oc.num_evals == 0
    assert oc.results == []


# --- overload collapse (evaluate raises) --------------------------------------


def test_overload_collapse_counts_as_fail_not_abort() -> None:
    sla = _sla()

    def evaluate(c: int) -> RunResult:
        if c > 4:
            raise RuntimeError("server crashed under overload")
        return _mk(c, ok=True)

    oc = search_saturation(
        evaluate, _qualifies(sla),
        start=1, factor=2, max_cap=256, max_probes=30, refine=True, confirm=1,
    )
    assert oc.c_star == 4
    assert oc.stop_reason == "found_boundary"
    # crashed probes recorded but excluded from goodput (throughput 0, unhealthy)
    assert candidate_goodput(oc.results, sla, output_len=1000) == 400.0
    assert any(r.status == "health_check_failed" for r in oc.results)


# --- health gate uniform with SLA ---------------------------------------------


def test_unhealthy_but_fast_probe_treated_as_fail() -> None:
    """A truncated (avg_output_tokens < 0.9*output_len) probe fails the boundary
    even though TTFT/TPOT are within SLA."""
    sla = _sla()

    def evaluate(c: int) -> RunResult:
        r = _mk(c, ok=True)
        if c > 4:
            r.avg_output_tokens = 50.0  # truncated -> health gate fails
        return r

    oc = search_saturation(
        evaluate, _qualifies(sla),
        start=1, factor=2, max_cap=256, max_probes=30, refine=True, confirm=1,
    )
    assert oc.c_star == 4


# --- seeding: reuse round-1 grid ----------------------------------------------


def test_seeded_bracket_skips_expansion() -> None:
    """Round-2 reuses round-1 grid results as the initial bracket; the seeded C
    are served from cache (never re-benched)."""
    sla = _sla()
    counter: Counter = Counter()
    # round-1 grid: 1,2,4,8 pass, 16 fail (true boundary 10)
    seeds = [_mk(c, ok=True) for c in (1, 2, 4, 8)] + [_mk(16, ok=False)]

    oc = search_saturation(
        _threshold_evaluate(10, counter), _qualifies(sla),
        start=1, factor=2, max_cap=256, max_probes=20, refine=True, confirm=1,
        seeds=seeds,
    )
    assert oc.c_star == 10
    assert oc.stop_reason == "found_boundary"
    # none of the seeded C were re-evaluated
    for c in (1, 2, 4, 8, 16):
        assert counter[c] == 0, f"seed C={c} was re-benched"
    # only fresh probes cost budget; they are the bisection midpoints
    assert set(oc.newly_probed) == set(counter.keys())
    assert oc.num_evals == len(oc.newly_probed)


def test_seeded_all_pass_resumes_above_old_ceiling() -> None:
    """High-capacity config: old grid [1..32] all pass; expansion resumes at 64,
    breaking past the 32 ceiling that a fixed grid would have capped at."""
    sla = _sla()
    counter: Counter = Counter()
    seeds = [_mk(c, ok=True) for c in (1, 2, 4, 8, 16, 32)]

    oc = search_saturation(
        _threshold_evaluate(50, counter), _qualifies(sla),
        start=1, factor=2, max_cap=256, max_probes=20, refine=True, confirm=1,
        seeds=seeds,
    )
    assert oc.c_star == 50
    assert oc.stop_reason == "found_boundary"
    # expansion did not re-probe any seed and went above 32
    assert min(oc.newly_probed) >= 33
    for c in (1, 2, 4, 8, 16, 32):
        assert counter[c] == 0
