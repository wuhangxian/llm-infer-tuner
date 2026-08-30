from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from runners.bench_runner import (
    build_benchmark_command_template,
    rewrite_bench_command,
)
from runners.executor import ExecutorConfig, run_executor
from schemas.job_spec import JobSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _job(**overrides: object) -> JobSpec:
    data: dict[str, object] = {
        "job_id": "deterministic-bench",
        "engine": "sglang",
        "gpu_model": "G24_pro5000",
        "gpu_count": 8,
        "gpu_memory_gb": 72,
        "model": "M02_qwen36-35b-a3b-fp8",
        "image": "I03_sglang-v0.5.16",
        "workload": "W04_input-4k-output-1k",
        "benchmark_method": "sglang-bench-serving",
        "sla": {
            "max_avg_ttft_ms": 2000,
            "max_avg_tpot_ms": 80,
            "min_success_rate": 0.99,
        },
        "search": {"max_candidates": 32},
    }
    data.update(overrides)
    return JobSpec.model_validate(data)


def _flag(parts: list[str], name: str) -> str:
    index = parts.index(name)
    return parts[index + 1]


def test_w04_builds_rewritable_bench_template_without_ai() -> None:
    template = build_benchmark_command_template(_job(), project_root=PROJECT_ROOT)
    parts = shlex.split(template)

    assert parts[:3] == ["python", "-m", "sglang.bench_serving"]
    assert _flag(parts, "--backend") == "sglang"
    assert _flag(parts, "--dataset-name") == "random-ids"
    assert _flag(parts, "--random-range-ratio") == "1.0"
    assert _flag(parts, "--host") == "${BENCHMARK_HOST}"
    assert _flag(parts, "--port") == "${BENCHMARK_PORT}"
    assert _flag(parts, "--model") == "${MODEL_PATH}"
    assert _flag(parts, "--random-input-len") == "4096"
    assert _flag(parts, "--random-output-len") == "1024"
    assert _flag(parts, "--max-concurrency") == "1"
    assert _flag(parts, "--num-prompts") == "4"
    assert _flag(parts, "--output-file") == "result_${JOB_ID}_${TIMESTAMP}.jsonl"
    assert "--context-length" not in parts

    rewritten, num_prompts = rewrite_bench_command(
        template, concurrency=8, multiplier=4
    )
    rewritten_parts = shlex.split(rewritten)
    assert _flag(rewritten_parts, "--max-concurrency") == "8"
    assert _flag(rewritten_parts, "--num-prompts") == "32"
    assert num_prompts == 32


class _NeverRemote:
    def run(self, command: str, *, timeout=None):
        raise AssertionError(f"remote machine was touched before local validation: {command}")


def test_invalid_benchmark_method_fails_before_remote_preflight(tmp_path: Path) -> None:
    job = _job(benchmark_method="missing-method")
    job_path = tmp_path / "job.json"
    job_path.write_text(job.model_dump_json(), encoding="utf-8")
    configs_path = tmp_path / "configs.jsonl"
    configs_path.write_text(
        json.dumps(
            {
                "id": "c001",
                "params": {},
                "cmd": "python -m sglang.launch_server --model-path ${MODEL_PATH}",
                "reasons": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = ExecutorConfig(
        job_path=job_path,
        configs_path=configs_path,
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/models/qwen",
        model_container_path="/models/qwen",
        project_root=PROJECT_ROOT,
        max_candidates=1,
    )

    with pytest.raises(ValueError, match="unknown benchmark_method: missing-method"):
        run_executor(config, remote=_NeverRemote())
