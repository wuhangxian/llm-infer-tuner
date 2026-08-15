import pytest
from pydantic import ValidationError

from schemas.job_spec import JobSpec


def test_job_spec_accepts_minimal_valid_job() -> None:
    job = JobSpec.model_validate(
        {
            "job_id": "qwen36_pro5000_random_v1",
            "engine": "sglang",
            "instance_type": "GC50s.192XLARGE2304",
            "model": "qwen36-27b-fp8",
            "image": "sglang-v0.5.10",
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


def test_job_spec_exports_json_schema() -> None:
    assert "properties" in JobSpec.model_json_schema()
