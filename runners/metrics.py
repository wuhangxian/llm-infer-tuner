"""Parse sglang.bench_serving v0.5.10 JSONL output into structured RunResult records."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ProbeStatus(StrEnum):
    """Typed outcome of one benchmark probe.

    Only :attr:`OK` and :attr:`SLA_FAILED` are statistical search verdicts.  The
    remaining values describe failures to obtain trustworthy SLA evidence.
    """

    OK = "ok"
    SLA_FAILED = "sla_failed"
    STARTUP_FAILED = "startup_failed"
    RUNTIME_FAILED = "runtime_failed"
    BENCHMARK_FAILED = "benchmark_failed"
    TRANSPORT_FAILED = "transport_failed"
    INVALID_RESULT = "invalid_result"
    INTERRUPTED = "interrupted"


SEARCH_VERDICT_STATUSES = frozenset({ProbeStatus.OK, ProbeStatus.SLA_FAILED})
SGLANG_0516_TRITON_EAGLE_KNOWN_ISSUE = (
    "sglang-0.5.16-triton-eagle-custom-mask-shape"
)

_REQUIRED_BENCH_FIELDS = (
    "max_concurrency",
    "completed",
    "total_output_tokens",
    "request_throughput",
    "output_throughput",
    "total_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "duration",
)
_INTEGER_BENCH_FIELDS = ("max_concurrency", "completed", "total_output_tokens")
_NUMERIC_BENCH_FIELDS = (
    "request_throughput",
    "output_throughput",
    "total_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "duration",
)


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
    # round2 整机满载 = topology 实际可放置副本的实测求和。
    instances: int = 1
    # True means total_throughput already is the measured aggregate for this
    # host/topology and must never be extrapolated to floor(gpu_count/tp_size).
    full_host_measured: bool = False
    status: ProbeStatus | str = ProbeStatus.OK
    failure_reason: str | None = None
    known_issue: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        aliases = {
            "bad_args": ProbeStatus.INVALID_RESULT,
            "health_check_failed": ProbeStatus.RUNTIME_FAILED,
        }
        if isinstance(self.status, str):
            self.status = aliases.get(self.status, self.status)
        try:
            self.status = ProbeStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown probe status: {self.status!r}") from exc


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError):
        return 0.0


def failed_probe_result(
    candidate_id: str,
    *,
    status: ProbeStatus,
    reason: str,
    concurrency: int,
    num_prompts: int,
    tp_size: int = 1,
    raw: dict[str, Any] | None = None,
    known_issue: str | None = None,
) -> RunResult:
    """Build a typed failure without losing the requested probe coordinates."""

    if status in SEARCH_VERDICT_STATUSES:
        raise ValueError(f"failure result requires a non-verdict status, got {status}")
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
        status=status,
        failure_reason=reason,
        known_issue=known_issue,
        raw=raw or {},
    )


def _select_record(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return the sole JSON object, rejecting noise, truncation, and stale appends."""

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
        except json.JSONDecodeError as exc:
            return None, f"malformed JSON on line {line_number}: {exc.msg}"
        except RecursionError:
            return None, f"invalid JSON on line {line_number}: nesting is too deep"
        except ValueError as exc:
            return None, f"invalid JSON on line {line_number}: {exc}"
        if not isinstance(record, dict):
            return None, f"line {line_number} is not a JSON object"
        records.append(record)
    if len(records) != 1:
        return None, f"expected exactly one benchmark record, found {len(records)}"
    return records[0], None


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON key {key!r}")
        record[key] = value
    return record


def _validate_required_metrics(record: dict[str, Any]) -> str | None:
    missing = [field_name for field_name in _REQUIRED_BENCH_FIELDS if field_name not in record]
    if missing:
        return "missing required benchmark fields: " + ", ".join(missing)
    for field_name in _INTEGER_BENCH_FIELDS:
        value = record[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return f"{field_name} must be a non-negative integer, got {value!r}"
    for field_name in _NUMERIC_BENCH_FIELDS:
        value = record[field_name]
        if not _is_finite_nonnegative_number(value):
            return f"{field_name} must be finite and non-negative, got {value!r}"
    return None


def _is_finite_nonnegative_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric >= 0


def classify_known_runtime_issue(
    *,
    engine_version: str,
    attention_backend: str | None,
    speculative_algorithm: str | None,
    traceback_text: str,
) -> str | None:
    """Recognize only the confirmed SGLang v0.5.16 Triton+EAGLE defect."""

    # Runtime package metadata is expected to return a release string.  Allow a
    # PEP-440 local build suffix (e.g. ``0.5.16+cu129``), but reject rc/dev/post
    # variants and any surrounding prose: the known-issue label is evidence,
    # not a guess derived from a loosely matching image tag.
    if not re.fullmatch(
        r"v?0\.5\.16(?:\+[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*)?",
        engine_version.strip(),
    ):
        return None
    traceback_lower = traceback_text.lower()
    if (attention_backend or "").lower() != "triton":
        return None
    if (speculative_algorithm or "").upper() != "EAGLE":
        return None
    frame = re.search(
        r"(?im)^[ \t]*File[ \t]+[\"'][^\"'\r\n]*triton_backend\.py[\"']"
        r",[ \t]*line[ \t]+\d+,[ \t]*in[ \t]+_update_target_verify_buffers[ \t]*$",
        traceback_text,
    )
    if frame is None:
        return None
    # Require the tensor-shape diagnostic after that exact frame and before a
    # subsequent traceback/frame begins.  This prevents unrelated log snippets
    # from being concatenated into a false known-issue match.
    frame_tail = traceback_lower[frame.end():]
    boundary = re.search(r"(?im)^[ \t]*(?:traceback\b|file[ \t]+[\"'])", frame_tail)
    trace_segment = frame_tail[: boundary.start()] if boundary else frame_tail
    trace_segment = trace_segment[:4096]
    exception_line: str | None = None
    diagnostic_prefix: list[str] = []
    for line in trace_segment.splitlines():
        if re.match(
            r"^[ \t]*(?:[A-Za-z_][\w.]*Error|Exception):",
            line,
            flags=re.IGNORECASE,
        ):
            exception_line = line.lower()
            break
        diagnostic_prefix.append(line.lower())
    if exception_line is not None:
        diagnostic_prefix.append(exception_line)
    if "custom_mask" not in "\n".join(diagnostic_prefix):
        return None
    if exception_line is None or not exception_line.lstrip().startswith("runtimeerror:"):
        return None
    if not all(
        fragment in exception_line for fragment in ("expanded size", "existing size")
    ):
        return None
    return SGLANG_0516_TRITON_EAGLE_KNOWN_ISSUE


def parse_bench_text(
    text: str, *, candidate_id: str, concurrency: int, num_prompts: int,
    output_len: int, tp_size: int = 1,
) -> RunResult:
    for name, value in (
        ("concurrency", concurrency),
        ("num_prompts", num_prompts),
        ("output_len", output_len),
        ("tp_size", tp_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    record, parse_error = _select_record(text or "")
    if record is None:
        return failed_probe_result(
            candidate_id,
            status=ProbeStatus.INVALID_RESULT,
            reason=parse_error or "empty or unparseable bench output",
            concurrency=concurrency,
            num_prompts=num_prompts,
            tp_size=tp_size,
        )

    metric_error = _validate_required_metrics(record)
    if metric_error is not None:
        return failed_probe_result(
            candidate_id,
            status=ProbeStatus.INVALID_RESULT,
            reason=metric_error,
            concurrency=concurrency,
            num_prompts=num_prompts,
            tp_size=tp_size,
            raw=record,
        )

    actual_concurrency = record.get("max_concurrency")
    if (
        isinstance(actual_concurrency, bool)
        or not isinstance(actual_concurrency, int)
        or actual_concurrency != concurrency
    ):
        return failed_probe_result(
            candidate_id,
            status=ProbeStatus.INVALID_RESULT,
            reason=(
                "max_concurrency does not exactly match requested concurrency: "
                f"expected {concurrency}, got {actual_concurrency!r}"
            ),
            concurrency=concurrency,
            num_prompts=num_prompts,
            tp_size=tp_size,
            raw=record,
        )

    if record["completed"] != num_prompts:
        return failed_probe_result(
            candidate_id,
            status=ProbeStatus.INVALID_RESULT,
            reason=(
                "completed does not exactly match requested num_prompts: "
                f"expected {num_prompts}, got {record['completed']!r}"
            ),
            concurrency=concurrency,
            num_prompts=num_prompts,
            tp_size=tp_size,
            raw=record,
        )

    completed = record["completed"]
    total_output_tokens = record["total_output_tokens"]
    success_rate = completed / num_prompts if num_prompts > 0 else 0.0
    # Compare integer totals before dividing.  Besides avoiding float rounding,
    # this keeps an untrusted 1000-digit token count from raising OverflowError.
    scaled_tokens = 10 * total_output_tokens
    lower_scaled = 9 * output_len * completed
    upper_scaled = 11 * output_len * completed
    if not lower_scaled <= scaled_tokens <= upper_scaled:
        return failed_probe_result(
            candidate_id,
            status=ProbeStatus.INVALID_RESULT,
            reason=(
                "average output tokens outside expected range: "
                f"expected 90%..110% of {output_len} per request, "
                f"got total_output_tokens={total_output_tokens!r}"
            ),
            concurrency=concurrency,
            num_prompts=num_prompts,
            tp_size=tp_size,
            raw=record,
        )
    try:
        avg_output_tokens = total_output_tokens / completed
    except OverflowError:
        return failed_probe_result(
            candidate_id,
            status=ProbeStatus.INVALID_RESULT,
            reason="average output tokens is not representable as a finite number",
            concurrency=concurrency,
            num_prompts=num_prompts,
            tp_size=tp_size,
            raw=record,
        )

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
        status=ProbeStatus.OK,
        failure_reason=None,
        raw=record,
    )


def parse_bench_file(
    path: str | Path, *, candidate_id: str, concurrency: int, num_prompts: int,
    output_len: int, tp_size: int = 1,
) -> RunResult:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return failed_probe_result(
            candidate_id,
            status=ProbeStatus.INVALID_RESULT,
            reason=f"cannot read benchmark output: {path}",
            concurrency=concurrency,
            num_prompts=num_prompts,
            tp_size=tp_size,
        )
    return parse_bench_text(
        text, candidate_id=candidate_id, concurrency=concurrency,
        num_prompts=num_prompts, tp_size=tp_size, output_len=output_len
    )
