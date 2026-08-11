import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from planner.claude_code_client import ClaudeCodeClient
from planner.plan_validator import PlanValidationError, PlanValidator
from planner.search_planner import SearchPlanner
from schemas.job_spec import JobSpec


def _job() -> JobSpec:
    return JobSpec.model_validate(
        {
            "job_id": "job-1",
            "engine": "sglang",
            "instance_type": "gpu-1",
            "model": "model-1",
            "workload": "workload-1",
            "benchmark_method": "method-1",
            "sla": {"max_avg_ttft_ms": 100, "max_avg_tpot_ms": 20},
            "search": {"max_candidates": 5, "max_runtime_minutes": 1},
        }
    )


def test_plan_validator_accepts_valid_plan_for_job() -> None:
    plan = PlanValidator().validate(
        {
            "job_id": "job-1",
            "pinned": {"tp_size": 1},
            "search_space": {"attention_backend": ["flashinfer"]},
            "search_policy": {
                "strategy": "baseline_first_bounded_product",
                "max_candidates": 5,
            },
        },
        _job(),
    )

    assert plan.job_id == "job-1"


def test_plan_validator_rejects_job_mismatch_and_budget_overrun() -> None:
    with pytest.raises(PlanValidationError, match="job_id"):
        PlanValidator().validate(
            {
                "job_id": "other-job",
                "search_policy": {
                    "strategy": "baseline_first_bounded_product",
                    "max_candidates": 5,
                },
            },
            _job(),
        )


def test_search_planner_builds_prompt_and_returns_validated_plan(tmp_path: Path) -> None:
    payload = {
        "job_id": "job-1",
        "pinned": {"tp_size": 1},
        "search_space": {"attention_backend": ["flashinfer"]},
        "search_policy": {
            "strategy": "baseline_first_bounded_product",
            "max_candidates": 5,
        },
    }
    captured: dict[str, Any] = {}

    def runner(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    plan = SearchPlanner(ClaudeCodeClient(runner=runner)).generate(_job(), [tmp_path])

    assert plan.job_id == "job-1"
    assert "qwen36_pro5000_random_v1" not in captured["argv"][2]
    assert "Allowed directories" in captured["argv"][2]
    assert str(tmp_path) in captured["argv"][2]
    assert "Every key under `pinned` and `search_space`" in captured["argv"][2]

    with pytest.raises(PlanValidationError, match="max_candidates"):
        PlanValidator().validate(
            {
                "job_id": "job-1",
                "search_policy": {
                    "strategy": "baseline_first_bounded_product",
                    "max_candidates": 6,
                },
            },
            _job(),
        )
