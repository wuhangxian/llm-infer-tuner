import json
from pathlib import Path

from typer.testing import CliRunner

from cli.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_JOB = PROJECT_ROOT / "examples/jobs/qwen36_pro5000_random_v1.json"
FIXTURE_PLAN = PROJECT_ROOT / "tests/fixtures/search_plan_qwen36_pro5000.json"


def test_example_job_completes_offline_plan_and_render_flow(tmp_path: Path) -> None:
    output = tmp_path / "qwen36-plan"
    runner = CliRunner()

    plan_result = runner.invoke(
        app,
        [
            "plan",
            str(EXAMPLE_JOB),
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(output),
            "--search-plan",
            str(FIXTURE_PLAN),
        ],
    )
    render_result = runner.invoke(
        app,
        [
            "render",
            str(output / "search_plan.json"),
            "--job",
            str(EXAMPLE_JOB),
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(output),
            "--model-path",
            "/data1/model/Qwen3.6-27B-FP8/",
            "--dataset-path",
            "ShareGPT_V3_unfiltered_cleaned_split.json",
        ],
    )

    assert plan_result.exit_code == 0, plan_result.stderr
    assert render_result.exit_code == 0, render_result.stderr
    candidates = (output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    rendered = (output / "rendered_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(candidates) == 2
    assert len(rendered) == 2
    assert all(json.loads(row)["server_command"] for row in rendered)
    assert all(json.loads(row)["benchmark_commands"] for row in rendered)


def test_plan_rejects_unknown_backend_parameter_before_rendering(tmp_path: Path) -> None:
    invalid_plan = tmp_path / "invalid_plan.json"
    invalid_plan.write_text(
        json.dumps(
            {
                "job_id": "qwen36_pro5000_random_v1",
                "pinned": {"tp_size": 8, "context_length": 262144},
                "search_space": {"unsupported_backend": ["fast-kernel"]},
                "search_policy": {
                    "strategy": "baseline_first_bounded_product",
                    "max_candidates": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "plan",
            str(EXAMPLE_JOB),
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(tmp_path / "invalid-output"),
            "--search-plan",
            str(invalid_plan),
        ],
    )

    assert result.exit_code == 1
    assert "unknown_parameter" in result.stderr
