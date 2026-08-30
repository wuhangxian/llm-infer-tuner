import pytest
from pydantic import ValidationError

from schemas.job_spec import JobSpec


def test_job_spec_accepts_minimal_valid_job() -> None:
    job = JobSpec.model_validate(
        {
            "job_id": "qwen36-27b-fp8_pro5000_8x72g_qa-chat-3.5k-1k",
            "engine": "sglang",
             "gpu_model": "pro5000",
             "gpu_count": 8,
             "gpu_memory_gb": 72,
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
         "gpu_model": "gpu-1",
         "gpu_count": 1,
         "gpu_memory_gb": 16,
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


def test_job_has_no_required_overall_runtime_limit() -> None:
    payload = {
        "job_id": "long-job",
        "engine": "sglang",
        "gpu_model": "gpu-1",
        "gpu_count": 8,
        "gpu_memory_gb": 80,
        "model": "large-model",
        "image": "sglang",
        "workload": "workload",
        "benchmark_method": "method",
        "sla": {"max_avg_ttft_ms": 1000, "max_avg_tpot_ms": 100},
        "search": {"max_candidates": 100},
    }

    job = JobSpec.model_validate(payload)

    assert job.search.max_runtime_minutes is None
