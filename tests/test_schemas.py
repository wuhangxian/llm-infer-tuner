import pytest
from pydantic import ValidationError

from schemas.candidate import Candidate
from schemas.job_spec import JobSpec
from schemas.search_plan import SearchPlan


def test_job_spec_accepts_minimal_valid_job() -> None:
    job = JobSpec.model_validate(
        {
            "job_id": "qwen36_pro5000_random_v1",
            "engine": "sglang",
            "instance_type": "GC50s.192XLARGE2304",
            "model": "qwen36-27b-fp8",
            "workload": "random-32k-1k",
            "benchmark_method": "sglang-bench-serving",
            "sla": {
                "max_avg_ttft_ms": 2000,
                "max_avg_tpot_ms": 80
            },
            "search": {
                "max_candidates": 30,
                "max_runtime_minutes": 180
            }
        }
    )

    assert job.engine == "sglang"
    assert job.search.max_candidates == 30


def test_job_spec_rejects_unsupported_engine_and_unknown_fields() -> None:
    payload = {
        "job_id": "job-1",
        "engine": "vllm",
        "instance_type": "gpu-1",
        "model": "model-1",
        "workload": "workload-1",
        "benchmark_method": "method-1",
        "sla": {"max_avg_ttft_ms": 100, "max_avg_tpot_ms": 20},
        "search": {"max_candidates": 1, "max_runtime_minutes": 1},
        "unexpected": True,
    }

    with pytest.raises(ValidationError):
        JobSpec.model_validate(payload)


def test_search_plan_and_candidate_accept_structured_plan() -> None:
    plan = SearchPlan.model_validate(
        {
            "job_id": "job-1",
            "pinned": {"tp_size": 8},
            "search_space": {"attention_backend": ["flashinfer"]},
            "constraints": [
                {
                    "parameter": "attention_backend",
                    "reason": "validated on target GPU",
                    "source": "references/sglang/parameter_policy.json",
                }
            ],
            "axes": {
                "attention_backend": {
                    "values": ["flashinfer"],
                    "source": "references/sglang/parameter_policy.json",
                    "risk": "low",
                }
            },
            "search_policy": {
                "strategy": "baseline_first_bounded_product",
                "max_candidates": 10,
            },
        }
    )
    candidate = Candidate.model_validate(
        {
            "candidate_id": "sglang-c001",
            "params": {"tp_size": 8, "attention_backend": "flashinfer"},
            "server_command": ["python", "-m", "sglang.launch_server"],
            "benchmark_command": ["python", "-m", "sglang.bench_serving"],
            "reasons": ["baseline candidate"],
            "expected_risk": "low",
        }
    )

    assert plan.search_policy.max_candidates == 10
    assert candidate.params["tp_size"] == 8


def test_schema_models_export_json_schema() -> None:
    assert "properties" in JobSpec.model_json_schema()
    assert "properties" in SearchPlan.model_json_schema()
    assert "properties" in Candidate.model_json_schema()


def test_search_plan_supports_top_level_notes() -> None:
    plan = SearchPlan.model_validate(
        {
            "job_id": "job-1",
            "search_policy": {
                "strategy": "baseline_first_bounded_product",
                "max_candidates": 1,
            },
            "notes": ["TP is a deployment search variable."],
        }
    )

    assert plan.notes == ["TP is a deployment search variable."]
