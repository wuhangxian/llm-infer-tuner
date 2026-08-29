"""Parse sglang.bench_serving v0.5.10 JSONL output into structured RunResult records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    candidate_id: str
    concurrency: int
    num_prompts: int
    completed: int
    success_rate: float
    request_throughput: float
    output_throughput: float
    total_throughput: float
    mean_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    p99_tpot_ms: float
    total_output_tokens: int
    avg_output_tokens: float
    duration: float
    tp_size: int = 1
    # 本条 total_throughput 是几个并发实例求和得来的。round1 单实例粗筛 = 1;
    # round2 整机满载 = floor(gpu_count/tp_size) 个副本的实测求和。ranker 用它
    # 把外推乘数除回去,使满载实测不被二次外推(详见 ranker._best_qualifying)。
    instances: int = 1
    status: str = "ok"
    failure_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _empty_result(
    candidate_id: str, concurrency: int, num_prompts: int, tp_size: int = 1
) -> RunResult:
    return RunResult(
        candidate_id=candidate_id,
        tp_size=tp_size,
        concurrency=concurrency,
        num_prompts=num_prompts,
        completed=0,
        success_rate=0.0,
        request_throughput=0.0,
        output_throughput=0.0,
        total_throughput=0.0,
        mean_ttft_ms=0.0,
        p99_ttft_ms=0.0,
        mean_tpot_ms=0.0,
        p99_tpot_ms=0.0,
        total_output_tokens=0,
        avg_output_tokens=0.0,
        duration=0.0,
        status="bad_args",
        failure_reason="empty or unparseable bench output",
    )


def _select_record(text: str, concurrency: int) -> dict[str, Any] | None:
    """Return the record whose max_concurrency matches, else the last valid record."""
    last_valid: dict[str, Any] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        last_valid = record
        if _as_int(record.get("max_concurrency")) == concurrency:
            return record
    return last_valid


def parse_bench_text(
    text: str, *, candidate_id: str, concurrency: int, num_prompts: int,
    tp_size: int = 1,
) -> RunResult:
    record = _select_record(text or "", concurrency)
    if record is None:
        return _empty_result(candidate_id, concurrency, num_prompts, tp_size)

    completed = _as_int(record.get("completed"))
    total_output_tokens = _as_int(record.get("total_output_tokens"))
    success_rate = completed / num_prompts if num_prompts > 0 else 0.0
    avg_output_tokens = total_output_tokens / completed if completed > 0 else 0.0

    return RunResult(
        candidate_id=candidate_id,
        tp_size=tp_size,
        concurrency=concurrency,
        num_prompts=num_prompts,
        completed=completed,
        success_rate=success_rate,
        request_throughput=_as_float(record.get("request_throughput")),
        output_throughput=_as_float(record.get("output_throughput")),
        total_throughput=_as_float(record.get("total_throughput")),
        mean_ttft_ms=_as_float(record.get("mean_ttft_ms")),
        p99_ttft_ms=_as_float(record.get("p99_ttft_ms")),
        mean_tpot_ms=_as_float(record.get("mean_tpot_ms")),
        p99_tpot_ms=_as_float(record.get("p99_tpot_ms")),
        total_output_tokens=total_output_tokens,
        avg_output_tokens=avg_output_tokens,
        duration=_as_float(record.get("duration")),
        status="ok",
        failure_reason=None,
        raw=record,
    )


def parse_bench_file(
    path: str | Path, *, candidate_id: str, concurrency: int, num_prompts: int,
    tp_size: int = 1,
) -> RunResult:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return _empty_result(candidate_id, concurrency, num_prompts, tp_size)
    return parse_bench_text(
        text, candidate_id=candidate_id, concurrency=concurrency,
        num_prompts=num_prompts, tp_size=tp_size
    )
