from typing import Any

from planner.candidate_generator import CandidateGenerator
from schemas.search_plan import SearchPlan


def _plan(max_candidates: int = 10) -> SearchPlan:
    return SearchPlan.model_validate(
        {
            "job_id": "job-1",
            "pinned": {"tp_size": 8},
            "search_space": {
                "attention_backend": ["flashinfer", "triton"],
                "chunked_prefill_size": [4096, 8192],
            },
            "axes": {
                "attention_backend": {
                    "values": ["flashinfer", "triton"],
                    "source": "test",
                    "risk": "high",
                },
                "chunked_prefill_size": {
                    "values": [4096, 8192],
                    "source": "test",
                    "risk": "medium",
                },
            },
            "search_policy": {
                "strategy": "baseline_first_bounded_product",
                "max_candidates": max_candidates,
            },
        }
    )


def test_generator_emits_baseline_first_and_respects_budget() -> None:
    candidates = CandidateGenerator().generate(_plan(max_candidates=3))

    assert len(candidates) == 3
    assert [candidate.candidate_id for candidate in candidates] == [
        "sglang-c001",
        "sglang-c002",
        "sglang-c003",
    ]
    assert candidates[0].params == {
        "tp_size": 8,
        "attention_backend": "flashinfer",
        "chunked_prefill_size": 4096,
    }
    assert candidates[0].reasons == ["baseline candidate"]
    assert candidates[0].expected_risk == "high"


def test_generator_applies_candidate_filter() -> None:
    def allow(candidate: dict[str, Any]) -> bool:
        return candidate["chunked_prefill_size"] != 4096

    candidates = CandidateGenerator().generate(_plan(), candidate_filter=allow)

    assert candidates
    assert all(candidate.params["chunked_prefill_size"] == 8192 for candidate in candidates)


def test_generator_uses_parallelism_candidates_as_deployment_units() -> None:
    plan = SearchPlan.model_validate(
        {
            "job_id": "job-1",
            "parallelism_candidates": [
                {
                    "gpu_count": 2,
                    "tp_size": 2,
                    "pp_size": 1,
                    "source": "test",
                    "reason": "TP deployment",
                },
                {
                    "gpu_count": 2,
                    "tp_size": 1,
                    "pp_size": 2,
                    "source": "test",
                    "reason": "PP deployment",
                },
            ],
            "search_space": {"attention_backend": ["flashinfer", "triton"]},
            "search_policy": {
                "strategy": "baseline_first_bounded_product",
                "max_candidates": 10,
            },
        }
    )

    candidates = CandidateGenerator().generate(plan)

    assert [(item.params["tp_size"], item.params["pp_size"]) for item in candidates] == [
        (2, 1),
        (2, 1),
        (1, 2),
        (1, 2),
    ]
