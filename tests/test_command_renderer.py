import json
from pathlib import Path

from planner.command_renderer import BenchmarkMethod, CommandRenderContext, CommandRenderer
from planner.spec_loader import SpecLoader
from schemas.candidate import Candidate

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_renderer_builds_server_and_all_benchmark_commands() -> None:
    specs = SpecLoader(PROJECT_ROOT / "specs")
    workload = specs.load_workload("random-32k-1k")
    method_payload = json.loads(
        (PROJECT_ROOT / "references/benchmark_methods/sglang_bench_serving.json").read_text()
    )
    method = BenchmarkMethod.model_validate(method_payload)
    candidate = Candidate(
        candidate_id="sglang-c001",
        params={
            "model_path": "/data1/model/DeepSeek-V4-Flash-FP8/",
            "tp_size": 8,
            "context_length": 33792,
            "attention_backend": "flashinfer",
        },
    )

    rendered = CommandRenderer().render(
        candidate,
        workload,
        method,
        CommandRenderContext(
            model_path="/data1/model/DeepSeek-V4-Flash-FP8/",
            dataset_path="ShareGPT_V3_unfiltered_cleaned_split.json",
            output_file="result_job_20260807.jsonl",
        ),
    )

    assert rendered.server_command[:4] == [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
    ]
    assert "--tp-size" in rendered.server_command
    assert "8" in rendered.server_command
    assert len(rendered.benchmark_commands) == 6
    assert rendered.benchmark_command == rendered.benchmark_commands[0]
    assert "--max-concurrency" in rendered.benchmark_commands[-1]
    assert "32" in rendered.benchmark_commands[-1]
    assert "--num-prompts" in rendered.benchmark_commands[-1]
    assert "128" in rendered.benchmark_commands[-1]
