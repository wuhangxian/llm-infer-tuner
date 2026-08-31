"""Exhaustive offline properties for precise concurrency search."""

from __future__ import annotations

from collections import Counter

from runners.concurrency_search import required_sample_budget, search_saturation
from runners.metrics import ProbeStatus, RunResult


def _result(concurrency: int, *, passes: bool, source: str) -> RunResult:
    return RunResult(
        candidate_id="candidate",
        concurrency=concurrency,
        num_prompts=concurrency * 4,
        completed=concurrency * 4,
        success_rate=1.0,
        request_throughput=float(concurrency),
        output_throughput=float(concurrency * 100),
        total_throughput=float(concurrency * 100),
        mean_ttft_ms=100.0 if passes else 10_000.0,
        p99_ttft_ms=200.0 if passes else 20_000.0,
        mean_tpot_ms=10.0,
        p99_tpot_ms=20.0,
        total_output_tokens=concurrency * 4 * 1000,
        avg_output_tokens=1000.0,
        duration=10.0,
        status=ProbeStatus.OK,
        raw={"source": source},
    )


def _qualifies(result: RunResult) -> bool:
    return result.mean_ttft_ms <= 2_000.0


def _power_of_two_seeds(boundary: int) -> list[RunResult]:
    # Every prior-round verdict is deliberately wrong. Search may use their
    # coordinates for ordering, never their verdicts or metrics.
    return [
        _result(c, passes=not (c <= boundary), source="seed")
        for c in (1, 2, 4, 8, 16, 32, 64, 128, 256)
    ]


def test_every_boundary_survives_false_seeds_and_alternating_samples() -> None:
    for boundary in range(257):
        calls: Counter[int] = Counter()

        def evaluate(concurrency: int) -> RunResult:
            sample_index = calls[concurrency]
            calls[concurrency] += 1
            true_verdict = concurrency <= boundary
            # One alternating noisy vote in every group; the majority remains
            # the true monotone verdict in all six pass/fail permutations.
            if concurrency % 2:
                verdict = (not true_verdict, true_verdict, true_verdict)[sample_index]
            else:
                verdict = (true_verdict, not true_verdict, true_verdict)[sample_index]
            return _result(concurrency, passes=verdict, source="fresh")

        outcome = search_saturation(
            evaluate,
            _qualifies,
            max_cap=256,
            max_probes=54,
            refine=True,
            confirm=3,
            seeds=_power_of_two_seeds(boundary),
        )

        expected_c_star = None if boundary == 0 else boundary
        assert outcome.c_star == expected_c_star, (boundary, outcome)
        if boundary < 256:
            assert outcome.complete is True
            assert outcome.certainty == "exact"
        else:
            assert outcome.complete is False
            assert outcome.certainty == "lower_bound"
            assert outcome.stop_reason == "hit_cap"
        assert outcome.num_evals == sum(calls.values())
        assert outcome.num_evals <= 54
        assert all(count == 3 for count in calls.values())
        assert outcome.results == [
            outcome.sample_groups[c].representative
            for c in outcome.newly_probed
        ]
        assert all(
            sample.raw["source"] == "fresh"
            for group in outcome.sample_groups.values()
            for sample in group.samples
        )


def test_no_seed_search_needs_at_most_48_valid_samples() -> None:
    maximum = 0
    for boundary in range(257):
        outcome = search_saturation(
            lambda concurrency, boundary=boundary: _result(
                concurrency,
                passes=concurrency <= boundary,
                source="fresh",
            ),
            _qualifies,
            max_cap=256,
            max_probes=48,
            refine=True,
            confirm=3,
        )
        maximum = max(maximum, outcome.num_evals)
        assert outcome.c_star == (None if boundary == 0 else boundary)
        assert outcome.certainty == (
            "lower_bound" if boundary == 256 else "exact"
        )
    assert maximum == 48


def test_dynamic_budget_covers_fallback_after_two_false_seed_endpoints() -> None:
    assert required_sample_budget(256, seed_hint_endpoints=0) == 48
    assert required_sample_budget(256, seed_hint_endpoints=2) == 54


def test_each_individually_wrong_seed_is_only_an_ordering_hint() -> None:
    seed_points = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    for boundary in range(257):
        for flipped_seed in seed_points:
            seeds = [
                _result(
                    concurrency,
                    passes=(concurrency <= boundary) ^ (concurrency == flipped_seed),
                    source="seed",
                )
                for concurrency in seed_points
            ]
            outcome = search_saturation(
                lambda concurrency, boundary=boundary: _result(
                    concurrency,
                    passes=concurrency <= boundary,
                    source="fresh",
                ),
                _qualifies,
                max_cap=256,
                max_probes=54,
                refine=True,
                confirm=3,
                seeds=seeds,
            )
            assert outcome.c_star == (None if boundary == 0 else boundary)
            assert outcome.certainty == (
                "lower_bound" if boundary == 256 else "exact"
            )
