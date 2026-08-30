"""Generate sglang.bench_serving commands via the client skill and run them in-container."""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from planner.claude_code_client import ClaudeCodeClient
from runners.container import Container
from runners.remote import CommandResult
from schemas.job_spec import JobSpec

CLIENT_SKILL_DIR = ".claude/skills/sglang-client-config-gen"

BENCH_SCHEMA: dict = {
    "type": "object",
    "required": ["benchmark_commands"],
    "properties": {
        "benchmark_commands": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["concurrency", "num_prompts", "command", "reason"],
                "properties": {
                    "concurrency": {"type": "integer"},
                    "num_prompts": {"type": "integer"},
                    "command": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}


@dataclass
class BenchCommand:
    """One benchmark command for a single concurrency level, still holding placeholders."""

    concurrency: int
    num_prompts: int
    command: str  # holds ${BENCHMARK_HOST}/${BENCHMARK_PORT}/${MODEL_PATH}/${JOB_ID}/${TIMESTAMP}
    reason: str = ""


def _build_prompt(job: JobSpec) -> str:
    job_json = json.dumps(job.model_dump(mode="json"), ensure_ascii=False, indent=2)
    return (
        "# 生成 SGLang 压测命令(客户端)\n\n"
        f"按 `{CLIENT_SKILL_DIR}/SKILL.md` 的读序与输出契约,为下面这个 JobSpec 生成"
        "每个并发档一条 `python -m sglang.bench_serving` 命令。\n\n"
        "读:① 该 SKILL.md → ② 同目录 knowledge.md → ③ catalogs/workloads.yaml"
        "(按 JobSpec.workload 取卡)→ ④ references/benchmark_methods/"
        "<JobSpec.benchmark_method>.json。\n\n"
        "host/port/model/输出文件用占位符 ${BENCHMARK_HOST} / ${BENCHMARK_PORT} / "
        "${MODEL_PATH} / ${JOB_ID} / ${TIMESTAMP};绝不写 --context-length;"
        "--backend 恒 sglang。\n\n"
        "## JobSpec\n```json\n"
        f"{job_json}\n"
        "```\n\n"
        "严格按 schema 只返回 JSON 对象 {\"benchmark_commands\": [...]}。"
    )


def generate_benchmark_commands(
    job: JobSpec,
    *,
    project_root: str | Path,
    client: ClaudeCodeClient,
    allow_dangerous_permissions: bool = True,
) -> list[BenchCommand]:
    """Ask the client skill for one bench command per concurrency level."""
    root = Path(project_root)
    add_dirs: Sequence[str | Path] = [root, root / CLIENT_SKILL_DIR]
    payload = client.run(
        prompt=_build_prompt(job),
        json_schema=BENCH_SCHEMA,
        add_dirs=add_dirs,
        allow_dangerous_permissions=allow_dangerous_permissions,
    )
    commands: list[BenchCommand] = []
    for item in payload.get("benchmark_commands", []):
        commands.append(
            BenchCommand(
                concurrency=int(item["concurrency"]),
                num_prompts=int(item["num_prompts"]),
                command=str(item["command"]),
                reason=str(item.get("reason", "")),
            )
        )
    return commands


def substitute_placeholders(
    command: str,
    *,
    host: str,
    port: int,
    model_path: str,
    job_id: str,
    timestamp: str,
    dataset_path: str = "",
) -> str:
    """Replace the runtime placeholders left in a generated bench command."""
    replacements = {
        "${BENCHMARK_HOST}": host,
        "${BENCHMARK_PORT}": str(port),
        "${MODEL_PATH}": model_path,
        "${JOB_ID}": job_id,
        "${TIMESTAMP}": timestamp,
        "${DATASET_PATH}": dataset_path,
    }
    for placeholder, value in replacements.items():
        command = command.replace(placeholder, value)
    return command


def _set_flag(parts: list[str], flag: str, value: str) -> list[str]:
    """Set ``flag value`` in a shlex-split arg list; replace in place or append.

    Handles both ``--flag value`` and ``--flag=value`` spellings and preserves the
    position of an existing flag so the rest of the command is untouched.
    """
    for index, part in enumerate(parts):
        if part == flag and index + 1 < len(parts):
            parts[index + 1] = value
            return parts
        if part.startswith(f"{flag}="):
            parts[index] = f"{flag}={value}"
            return parts
    parts.extend([flag, value])
    return parts


def rewrite_bench_command(
    template: str, *, concurrency: int, multiplier: int
) -> tuple[str, int]:
    """Rewrite ONE bench command template for a given concurrency.

    The fairness rule (see client knowledge §3): every candidate must be pressed
    with the *same* workload, and per-concurrency num_prompts must scale as a
    constant ratio of C. So we take a single template (generated once per job by
    the client skill) and deterministically rewrite ONLY ``--max-concurrency``
    (-> concurrency) and ``--num-prompts`` (-> concurrency * multiplier), leaving
    every other flag — model, input/output len, dataset, range-ratio, seed —
    byte-for-byte identical across all candidates and all concurrency probes.

    Pure and idempotent. Returns (command, num_prompts).
    """
    num_prompts = concurrency * multiplier
    parts = shlex.split(template)
    _set_flag(parts, "--max-concurrency", str(concurrency))
    _set_flag(parts, "--num-prompts", str(num_prompts))
    return shlex.join(parts), num_prompts


def run_benchmark(
    container: Container, command: str, *, timeout: int | None = 0
) -> CommandResult:
    """Execute a fully-substituted bench command inside the container."""
    return container.exec(command, timeout=timeout)
