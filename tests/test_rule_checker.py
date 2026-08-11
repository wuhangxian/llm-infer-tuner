from pathlib import Path

from planner.reference_loader import ReferenceLoader, SGLangReferences
from planner.rule_checker import RuleChecker
from planner.spec_loader import LoadedSpec, SpecLoader
from schemas.job_spec import JobSpec
from schemas.search_plan import SearchPlan

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_context() -> tuple[JobSpec, LoadedSpec, LoadedSpec, LoadedSpec, SGLangReferences]:
    job = JobSpec.model_validate(
        {
            "job_id": "qwen36_pro5000_random_v1",
            "engine": "sglang",
            "instance_type": "GC50s.192XLARGE2304",
            "model": "qwen36-27b-fp8",
            "workload": "random-32k-1k",
            "benchmark_method": "sglang-bench-serving",
            "sla": {"max_avg_ttft_ms": 2000, "max_avg_tpot_ms": 80},
            "search": {"max_candidates": 30, "max_runtime_minutes": 180},
        }
    )
    specs = SpecLoader(PROJECT_ROOT / "specs")
    refs = ReferenceLoader(PROJECT_ROOT / "references" / "sglang").load_sglang()
    return (
        job,
        specs.load_hardware(job.instance_type),
        specs.load_model(job.model),
        specs.load_workload(job.workload),
        refs,
    )


def test_rule_checker_accepts_a_valid_search_plan() -> None:
    job, hardware, model, workload, references = _load_context()
    plan = SearchPlan.model_validate(
        {
            "job_id": job.job_id,
            "pinned": {"tp_size": 8, "context_length": 33792},
            "search_space": {
                "attention_backend": ["flashinfer"],
                "chunked_prefill_size": [4096, 8192],
            },
            "search_policy": {
                "strategy": "baseline_first_bounded_product",
                "max_candidates": 10,
            },
        }
    )

    result = RuleChecker().check(job, hardware, model, workload, references, plan)

    assert result.valid is True
    assert result.errors == []
    assert any(item.code == "target_help_validation_required" for item in result.warnings)


def test_rule_checker_rejects_tp_size_above_gpu_count() -> None:
    job, hardware, model, workload, references = _load_context()
    plan = SearchPlan.model_validate(
        {
            "job_id": job.job_id,
            "pinned": {"tp_size": 16, "context_length": 33792},
            "search_policy": {
                "strategy": "baseline_first_bounded_product",
                "max_candidates": 10,
            },
        }
    )

    result = RuleChecker().check(job, hardware, model, workload, references, plan)

    assert result.valid is False
    assert any(item.code == "tp_size_exceeds_gpu_count" for item in result.errors)


def test_rule_checker_validates_parallelism_card_budget() -> None:
    job, hardware, model, workload, references = _load_context()
    plan = SearchPlan.model_validate(
        {
            "job_id": job.job_id,
            "parallelism_candidates": [
                {
                    "gpu_count": 16,
                    "tp_size": 4,
                    "pp_size": 4,
                    "source": "test",
                    "reason": "too many GPUs",
                }
            ],
            "search_policy": {
                "strategy": "baseline_first_bounded_product",
                "max_candidates": 10,
            },
        }
    )

    result = RuleChecker().check(job, hardware, model, workload, references, plan)

    assert result.valid is False
    assert any(item.code == "parallelism_exceeds_gpu_count" for item in result.errors)


def test_rule_checker_rejects_context_length_below_workload() -> None:
    job, hardware, model, workload, references = _load_context()
    plan = SearchPlan.model_validate(
        {
            "job_id": job.job_id,
            "pinned": {"tp_size": 8, "context_length": 32768},
            "search_policy": {
                "strategy": "baseline_first_bounded_product",
                "max_candidates": 10,
            },
        }
    )

    result = RuleChecker().check(job, hardware, model, workload, references, plan)

    assert result.valid is False
    assert any(item.code == "context_length_too_short" for item in result.errors)


def test_rule_checker_rejects_unknown_or_pinned_search_parameters() -> None:
    job, hardware, model, workload, references = _load_context()
    plan = SearchPlan.model_validate(
        {
            "job_id": job.job_id,
            "pinned": {"tp_size": 8, "context_length": 33792},
            "search_space": {
                "model_path": ["/models/a", "/models/b"],
                "not_a_sglang_parameter": [1],
            },
            "search_policy": {
                "strategy": "baseline_first_bounded_product",
                "max_candidates": 10,
            },
        }
    )

    result = RuleChecker().check(job, hardware, model, workload, references, plan)
    error_codes = {item.code for item in result.errors}

    assert "pinned_parameter_in_search_space" in error_codes
    assert "unknown_parameter" in error_codes
