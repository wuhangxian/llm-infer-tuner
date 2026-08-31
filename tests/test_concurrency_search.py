"""Offline tests for runners.concurrency_search (no ssh/docker/network).

The search is exercised with synthetic ``evaluate``/``qualifies`` callbacks so
the monotone-boundary behaviour, noise handling, dedupe, budget/cap limits, and
seeding can all be asserted deterministically. Throughput is made to rise with C
so goodput (max total_throughput over qualifying probes, as ranker computes it)
lands on the C* run.
"""

from __future__ import annotations

from collections import Counter
from itertools import permutations

import pytest

from runners.concurrency_search import search_saturation
from runners.metrics import ProbeStatus, RunResult
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


def test_c1_failed_uses_complete_three_sample_group() -> None:
    sla = _sla()
    oc = search_saturation(
        _threshold_evaluate(0), _qualifies(sla), start=1, refine=True, confirm=3,
    )
    assert oc.c_star is None
    assert oc.stop_reason == "c1_failed"
    assert oc.num_evals == 3


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
        start=1, factor=2, max_cap=256, max_probes=54, refine=True, confirm=3,
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
        start=1, factor=2, max_cap=256, max_probes=54, refine=True, confirm=3,
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
        start=1, factor=2, max_cap=256, max_probes=54, refine=True, confirm=3,
    )
    assert oc.c_star == 4
    assert candidate_goodput(oc.results, sla, output_len=1000) == 400.0


# --- cap and budget -----------------------------------------------------------


def test_hit_cap_all_pass() -> None:
    sla = _sla()
    oc = search_saturation(
        _threshold_evaluate(10_000), _qualifies(sla),
        start=1, factor=2, max_cap=16, max_probes=30, refine=True, confirm=3,
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


def test_overload_collapse_marks_search_incomplete_not_sla_boundary() -> None:
    sla = _sla()

    def evaluate(c: int) -> RunResult:
        if c > 4:
            return RunResult(
                candidate_id="c",
                concurrency=c,
                num_prompts=c * 4,
                completed=0,
                success_rate=0.0,
                request_throughput=0.0,
                output_throughput=0.0,
                total_throughput=0.0,
                mean_ttft_ms=0.0,
                p99_ttft_ms=0.0,
                mean_tpot_ms=0.0,
                p99_tpot_ms=0.0,
                total_output_tokens=0,
                avg_output_tokens=0.0,
                duration=0.0,
                status="runtime_failed",
                failure_reason="server crashed under overload",
            )
        return _mk(c, ok=True)

    oc = search_saturation(
        evaluate, _qualifies(sla),
        start=1, factor=2, max_cap=256, max_probes=30, refine=True, confirm=1,
    )
    assert oc.complete is False
    assert oc.certainty == "unknown"
    assert oc.c_star is None
    assert oc.first_fail is None
    assert oc.stop_reason == "runtime_failed"
    # Infrastructure failure invalidates any earlier successful points for an
    # official goodput/rank; it is not an SLA fail boundary.
    assert candidate_goodput(oc.results, sla, output_len=1000) == 0.0
    assert any(r.status == "runtime_failed" for r in oc.results)


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


def test_seeded_bracket_endpoints_are_fresh_three_sample_groups() -> None:
    """Round-1 verdicts only choose probe order; neither endpoint is authoritative."""
    sla = _sla()
    counter: Counter = Counter()
    seeds = [
        _mk(4, ok=True, throughput=99_999.0),
        _mk(5, ok=False, throughput=88_888.0),
    ]
    throughputs = {4: [410.0, 390.0, 400.0], 5: [510.0, 490.0, 500.0]}

    def evaluate(c: int) -> RunResult:
        sample_index = counter[c]
        counter[c] += 1
        return _mk(c, ok=(c <= 4), throughput=throughputs[c][sample_index])

    oc = search_saturation(
        evaluate, _qualifies(sla),
        start=1, factor=2, max_cap=256, max_probes=54, refine=True, confirm=3,
        seeds=seeds,
    )
    assert oc.c_star == 4
    assert oc.stop_reason == "found_boundary"
    assert oc.complete is True
    assert oc.certainty == "exact"
    assert counter == Counter({4: 3, 5: 3})
    assert oc.num_evals == 6
    assert oc.newly_probed == [4, 5]
    assert [sample.total_throughput for sample in oc.sample_groups[4].samples] == [
        410.0,
        390.0,
        400.0,
    ]
    assert oc.sample_groups[4].representative.total_throughput == 400.0
    assert [result.total_throughput for result in oc.results] == [400.0, 500.0]
    assert all(result.total_throughput < 10_000 for result in oc.results)
    assert candidate_goodput(oc.results, sla, output_len=1000) == 400.0


def test_seeded_all_pass_revalidates_old_ceiling_then_resumes_above_it() -> None:
    """High-capacity config: old grid [1..32] all pass; expansion resumes at 64,
    breaking past the 32 ceiling that a fixed grid would have capped at."""
    sla = _sla()
    counter: Counter = Counter()
    seeds = [_mk(c, ok=True) for c in (1, 2, 4, 8, 16, 32)]

    oc = search_saturation(
        _threshold_evaluate(50, counter), _qualifies(sla),
        start=1, factor=2, max_cap=256, max_probes=54, refine=True, confirm=3,
        seeds=seeds,
    )
    assert oc.c_star == 50
    assert oc.stop_reason == "found_boundary"
    # The highest seed is a navigation hint, so it is fresh-revalidated once
    # before expansion continues above the old ceiling.
    assert oc.newly_probed[0] == 32
    assert counter[32] == 3
    assert all(counter[c] == 0 for c in (1, 2, 4, 8, 16))


_THREE_SAMPLE_ORDERS = sorted(
    set(permutations((True, True, False)))
    | set(permutations((True, False, False)))
)


@pytest.mark.parametrize("sample_verdicts", _THREE_SAMPLE_ORDERS)
def test_three_sample_group_uses_all_samples_and_majority(
    sample_verdicts: tuple[bool, bool, bool],
) -> None:
    sla = _sla()
    throughputs = (300.0, 100.0, 200.0)
    calls = 0

    def evaluate(c: int) -> RunResult:
        nonlocal calls
        result = _mk(
            c,
            ok=sample_verdicts[calls],
            throughput=throughputs[calls],
        )
        calls += 1
        return result

    oc = search_saturation(
        evaluate,
        _qualifies(sla),
        max_cap=1,
        max_probes=3,
        refine=True,
        confirm=3,
    )

    majority_passes = sum(sample_verdicts) >= 2
    assert calls == 3
    assert oc.num_evals == 3
    assert len(oc.sample_groups[1].samples) == 3
    assert oc.sample_groups[1].qualifies is majority_passes
    assert oc.sample_groups[1].representative.total_throughput == 200.0
    assert oc.sample_groups[1].representative.status == (
        ProbeStatus.OK if majority_passes else ProbeStatus.SLA_FAILED
    )


def test_seeded_high_fail_without_fresh_pass_falls_back_to_c1() -> None:
    sla = _sla()
    counter: Counter = Counter()

    oc = search_saturation(
        _threshold_evaluate(3, counter),
        _qualifies(sla),
        max_cap=16,
        max_probes=54,
        refine=True,
        confirm=3,
        seeds=[_mk(8, ok=False)],
    )

    assert oc.c_star == 3
    assert oc.certainty == "exact"
    assert counter[8] == 3
    assert counter[1] == 3


def test_high_seed_failure_without_budget_for_c1_is_unknown() -> None:
    sla = _sla()
    oc = search_saturation(
        _threshold_evaluate(10),
        _qualifies(sla),
        max_cap=256,
        max_probes=3,
        refine=True,
        confirm=3,
        seeds=[_mk(32, ok=True)],
    )

    assert oc.c_star is None
    assert oc.stop_reason == "max_probes"
    assert oc.complete is False
    assert oc.certainty == "unknown"


def test_fresh_nonmonotone_groups_are_unknown_not_an_exact_boundary() -> None:
    sla = _sla()

    def evaluate(c: int) -> RunResult:
        return _mk(c, ok=(c == 8))

    oc = search_saturation(
        evaluate,
        _qualifies(sla),
        max_cap=16,
        max_probes=54,
        refine=True,
        confirm=3,
        seeds=[_mk(4, ok=True), _mk(8, ok=False)],
    )

    assert set(oc.sample_groups) == {4, 8}
    assert oc.c_star is None
    assert oc.stop_reason == "monotonicity_conflict"
    assert oc.complete is False
    assert oc.certainty == "unknown"


def test_budget_never_starts_or_commits_a_partial_sample_group() -> None:
    sla = _sla()
    calls: Counter = Counter()
    oc = search_saturation(
        _threshold_evaluate(255, calls),
        _qualifies(sla),
        max_cap=256,
        max_probes=47,
        refine=True,
        confirm=3,
    )

    assert oc.stop_reason == "max_probes"
    assert oc.certainty == "lower_bound"
    assert oc.complete is False
    assert oc.num_evals == 45
    assert sum(calls.values()) == 45
    assert all(len(group.samples) == 3 for group in oc.sample_groups.values())


def test_budget_48_finds_worst_case_boundary_255_exactly() -> None:
    sla = _sla()
    calls: Counter = Counter()
    oc = search_saturation(
        _threshold_evaluate(255, calls),
        _qualifies(sla),
        max_cap=256,
        max_probes=48,
        refine=True,
        confirm=3,
    )

    assert oc.c_star == 255
    assert oc.first_fail == 256
    assert oc.num_evals == 48
    assert oc.complete is True
    assert oc.certainty == "exact"


def test_terminal_infrastructure_does_not_commit_partial_group_or_spend_budget() -> None:
    sla = _sla()
    first = _mk(1, ok=True)
    terminal = RunResult(
        **{
            **first.__dict__,
            "status": ProbeStatus.TRANSPORT_FAILED,
            "failure_reason": "ssh lost after one valid sample",
        }
    )
    responses = iter((first, terminal))

    oc = search_saturation(
        lambda _c: next(responses),
        _qualifies(sla),
        max_cap=1,
        max_probes=3,
        refine=True,
        confirm=3,
    )

    assert oc.certainty == "unknown"
    assert oc.complete is False
    assert oc.stop_reason == str(ProbeStatus.TRANSPORT_FAILED)
    assert oc.num_evals == 1
    assert oc.sample_groups == {}
    assert oc.incomplete_samples[1] == (first, terminal)
