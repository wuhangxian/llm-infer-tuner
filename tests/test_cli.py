import json
from pathlib import Path

from typer.testing import CliRunner

from cli.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _write_job(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "job_id": "qwen36_pro5000_random_v1",
                "engine": "sglang",
                "instance_type": "GC50s.192XLARGE2304",
                "model": "qwen36-27b-fp8",
                "workload": "random-32k-1k",
                "benchmark_method": "sglang-bench-serving",
                "sla": {"max_avg_ttft_ms": 2000, "max_avg_tpot_ms": 80},
                "search": {"max_candidates": 4, "max_runtime_minutes": 180},
            }
        ),
        encoding="utf-8",
    )


def _write_plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "job_id": "qwen36_pro5000_random_v1",
                "pinned": {"model_path": "/data1/model/DeepSeek-V4-Flash-FP8/", "tp_size": 8},
                "search_space": {"attention_backend": ["flashinfer"]},
                "search_policy": {
                    "strategy": "baseline_first_bounded_product",
                    "max_candidates": 4,
                },
            }
        ),
        encoding="utf-8",
    )


def test_validate_job_and_plan_write_offline_artifacts(tmp_path: Path) -> None:
    job_path = tmp_path / "job.json"
    plan_path = tmp_path / "fixture_plan.json"
    output_dir = tmp_path / "output"
    _write_job(job_path)
    _write_plan(plan_path)

    validated = runner.invoke(
        app,
        ["validate-job", str(job_path), "--project-root", str(PROJECT_ROOT)],
    )
    planned = runner.invoke(
        app,
        [
            "plan",
            str(job_path),
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(output_dir),
            "--search-plan",
            str(plan_path),
        ],
    )

    assert validated.exit_code == 0, validated.stdout
    assert planned.exit_code == 0, planned.stdout
    assert (output_dir / "job_spec.json").is_file()
    assert (output_dir / "search_plan.json").is_file()
    assert len((output_dir / "candidates.jsonl").read_text().splitlines()) == 1


def test_render_writes_commands_without_executing_them(tmp_path: Path) -> None:
    job_path = tmp_path / "job.json"
    plan_path = tmp_path / "fixture_plan.json"
    output_dir = tmp_path / "output"
    _write_job(job_path)
    _write_plan(plan_path)

    planned = runner.invoke(
        app,
        [
            "plan",
            str(job_path),
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(output_dir),
            "--search-plan",
            str(plan_path),
        ],
    )
    rendered = runner.invoke(
        app,
        [
            "render",
            str(output_dir / "search_plan.json"),
            "--job",
            str(job_path),
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(output_dir),
            "--model-path",
            "/data1/model/DeepSeek-V4-Flash-FP8/",
            "--dataset-path",
            "ShareGPT_V3_unfiltered_cleaned_split.json",
        ],
    )

    assert planned.exit_code == 0, planned.stdout
    assert rendered.exit_code == 0, rendered.stdout
    assert (output_dir / "commands.sh").is_file()
    assert (output_dir / "plan_report.md").is_file()
    assert "sglang.launch_server" in (output_dir / "commands.sh").read_text()
    assert "sglang.bench_serving" in (output_dir / "commands.sh").read_text()


def test_validate_job_returns_nonzero_for_invalid_json(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"engine": "vllm"}', encoding="utf-8")

    result = runner.invoke(
        app,
        ["validate-job", str(invalid), "--project-root", str(PROJECT_ROOT)],
    )

    assert result.exit_code == 1
    assert "Invalid JobSpec" in result.stderr


def test_render_rejects_parameter_missing_from_help_snapshot(tmp_path: Path) -> None:
    job_path = tmp_path / "job.json"
    plan_path = tmp_path / "fixture_plan.json"
    output_dir = tmp_path / "output"
    _write_job(job_path)
    _write_plan(plan_path)

    planned = runner.invoke(
        app,
        [
            "plan",
            str(job_path),
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(output_dir),
            "--search-plan",
            str(plan_path),
        ],
    )
    server_help = tmp_path / "server_help.txt"
    server_help.write_text(
        "  --model-path TEXT\n  --tp-size INTEGER\n  --context-length INTEGER\n",
        encoding="utf-8",
    )
    rendered = runner.invoke(
        app,
        [
            "render",
            str(output_dir / "search_plan.json"),
            "--job",
            str(job_path),
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(tmp_path / "rendered"),
            "--model-path",
            "/data1/model/DeepSeek-V4-Flash-FP8/",
            "--dataset-path",
            "ShareGPT_V3_unfiltered_cleaned_split.json",
            "--server-help",
            str(server_help),
        ],
    )

    assert planned.exit_code == 0, planned.stdout
    assert rendered.exit_code == 1
    assert "--attention-backend" in rendered.stderr
