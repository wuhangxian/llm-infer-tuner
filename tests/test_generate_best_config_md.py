from __future__ import annotations

import json

from scripts.generate_best_config_md import main


def _write_fixture(tmp_path, *, qualified: bool = True):
    project_root = tmp_path / "project"
    output_dir = project_root / "outputs" / "demo-job"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True)

    goodput = 120.0 if qualified else 0.0
    best_concurrency = 8 if qualified else None
    ranking = [
        {
            "candidate_id": "c001",
            "tp_size": 2,
            "instances_per_host": 4.0,
            "goodput_raw": 30.0 if qualified else 0.0,
            "goodput_per_host": goodput,
            "best_concurrency": best_concurrency,
            "mean_ttft_ms": 100.0 if qualified else 0.0,
            "mean_tpot_ms": 20.0 if qualified else 0.0,
            "success_rate": 1.0 if qualified else 0.0,
            "avg_output_tokens": 1024.0 if qualified else 0.0,
            "rank": 1,
        },
        {
            "candidate_id": "c002",
            "tp_size": 4,
            "instances_per_host": 2.0,
            "goodput_raw": 25.0,
            "goodput_per_host": 100.0 if qualified else 0.0,
            "best_concurrency": 8 if qualified else None,
            "mean_ttft_ms": 120.0 if qualified else 0.0,
            "mean_tpot_ms": 22.0 if qualified else 0.0,
            "success_rate": 1.0 if qualified else 0.0,
            "avg_output_tokens": 1024.0 if qualified else 0.0,
            "rank": 2,
        },
    ]
    (results_dir / "ranking.json").write_text(json.dumps(ranking), encoding="utf-8")
    rows = []
    for row in ranking:
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "status": "completed" if qualified else "failed",
                "effective_params": {
                    "tp_size": row["tp_size"],
                    "attention_backend": "fa" if row["candidate_id"] == "c001" else "triton",
                    **({"chunked_prefill_size": 8192} if row["candidate_id"] == "c002" else {}),
                },
                "requested_command": "python -m sglang.launch_server --tp-size 2",
                "failure_reason": None if qualified else "没有找到满足 SLA 的并发点",
                "concurrency_points": (
                    []
                    if qualified
                    else [
                        {
                            "concurrency": 1,
                            "mean_ttft_ms": 2200.0,
                            "mean_tpot_ms": 55.0,
                            "success_rate": 1.0,
                            "avg_output_tokens": 1024.0,
                            "status": "ok",
                        }
                    ]
                ),
            }
        )
    (results_dir / "candidate_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (results_dir / "task_status.json").write_text(
        json.dumps({"job_id": "demo-job", "ranking_status": "FINAL"}), encoding="utf-8"
    )
    (output_dir / "configs.jsonl").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "id": "c001",
                        "cmd": "python -m sglang.launch_server --tp-size 2",
                        "reasons": ["TP2 适合当前显卡拓扑", "flash attention 实测吞吐更高"],
                    },
                    {"id": "c002", "reasons": ["对照候选"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    jobs_dir = project_root / "input" / "jobs"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "demo-job.json").write_text(
        json.dumps(
            {
                "job_id": "demo-job",
                "model": "M_demo",
                "gpu_model": "G_demo",
                "gpu_count": 8,
                "gpu_memory_gb": 72,
                "image": "I_demo",
                "workload": "W_demo",
                "sla": {
                    "max_avg_ttft_ms": 2000,
                    "max_avg_tpot_ms": 80,
                    "min_success_rate": 0.99,
                },
            }
        ),
        encoding="utf-8",
    )
    catalogs_dir = project_root / "catalogs"
    catalogs_dir.mkdir()
    (catalogs_dir / "workloads.yaml").write_text(
        "workloads:\n  W_demo:\n    output_tokens: 1024\n", encoding="utf-8"
    )
    return project_root, output_dir, results_dir


def test_generates_report_from_job_output_directory(tmp_path) -> None:
    project_root, output_dir, _ = _write_fixture(tmp_path)

    assert main([str(output_dir), "--project-root", str(project_root)]) == 0

    report = (output_dir / "best_config.md").read_text(encoding="utf-8")
    assert "推荐候选 **`c001`**" in report
    assert "flash attention 实测吞吐更高" in report
    assert "goodput / host 高 `20.00`" in report
    assert "平均 TTFT" in report


def test_report_explains_unspecified_parameters_instead_of_null(tmp_path) -> None:
    project_root, output_dir, _ = _write_fixture(tmp_path)

    assert main([str(output_dir), "--project-root", str(project_root)]) == 0

    report = (output_dir / "best_config.md").read_text(encoding="utf-8")
    assert "最佳=未显式指定（使用 SGLang/镜像默认值）" in report
    assert "最佳=null" not in report
    assert "实际测试采用 SGLang/镜像默认行为" in report


def test_generates_report_in_results_directory_when_results_is_passed(tmp_path) -> None:
    project_root, _, results_dir = _write_fixture(tmp_path)

    assert main([str(results_dir), "--project-root", str(project_root)]) == 0
    assert (results_dir / "best_config.md").is_file()


def test_reports_when_no_candidate_meets_sla(tmp_path) -> None:
    project_root, output_dir, _ = _write_fixture(tmp_path, qualified=False)

    assert main([str(output_dir), "--project-root", str(project_root)]) == 0

    report = (output_dir / "best_config.md").read_text(encoding="utf-8")
    assert "没有找到同时满足 SLA 和输出健康检查的候选配置" in report
    assert "不能推荐可上线参数" in report
    assert "## 本次 SLA" in report
    assert "## 未通过候选的实测点" in report
