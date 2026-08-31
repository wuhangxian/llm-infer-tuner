"""Build and persist complete, one-row-per-candidate benchmark reports."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from runners.metrics import RunResult

REPORT_SCHEMA_VERSION = 1


def annotate_baseline_threshold(
    ranking: list[dict[str, Any]], *, threshold_pct: float
) -> list[dict[str, Any]]:
    """Annotate every row relative to baseline; never remove a candidate."""
    rows = deepcopy(ranking)
    baseline = next(
        (
            _finite_float(row.get("goodput_per_host"))
            for row in rows
            if row["candidate_id"] == "baseline"
        ),
        None,
    )
    threshold = _finite_product(
        baseline,
        _finite_sum(1.0, _finite_quotient(_finite_float(threshold_pct), 100.0)),
    )
    for rank, row in enumerate(rows, 1):
        value = _finite_float(row.get("goodput_per_host"))
        delta = _finite_difference(value, baseline)
        relative = _finite_difference(_finite_quotient(value, baseline), 1.0)
        delta_pct = _finite_product(relative, 100.0)
        row["rank"] = rank
        row["baseline_goodput_per_host"] = baseline
        row["threshold_goodput_per_host"] = threshold
        row["goodput_delta"] = delta
        row["goodput_delta_pct"] = round(delta_pct, 6) if delta_pct is not None else None
        row["beats_baseline_threshold"] = (
            value >= threshold
            if value is not None
            and threshold is not None
            and row["candidate_id"] != "baseline"
            else False
        )
    return rows


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _finite_binary(
    left: float | None,
    right: float | None,
    operation: str,
) -> float | None:
    if left is None or right is None:
        return None
    try:
        if operation == "add":
            result = left + right
        elif operation == "subtract":
            result = left - right
        elif operation == "multiply":
            result = left * right
        elif operation == "divide":
            if right == 0:
                return None
            result = left / right
        else:  # pragma: no cover - all callers use a literal operation above
            raise ValueError(f"unknown finite operation: {operation}")
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _finite_sum(left: float | None, right: float | None) -> float | None:
    return _finite_binary(left, right, "add")


def _finite_difference(left: float | None, right: float | None) -> float | None:
    return _finite_binary(left, right, "subtract")


def _finite_product(left: float | None, right: float | None) -> float | None:
    return _finite_binary(left, right, "multiply")


def _finite_quotient(left: float | None, right: float | None) -> float | None:
    return _finite_binary(left, right, "divide")


def _effective_params(candidate: dict[str, Any]) -> dict[str, Any]:
    params = deepcopy(candidate.get("params", {}))
    requested_params = deepcopy(candidate.get("requested_params", params))
    params.pop("disable-radix-cache", None)
    params.pop("disable_radix_cache", None)
    params["disable_radix_cache"] = True
    for name in (
        "mamba_radix_cache_strategy",
        "mamba_scheduler_strategy",
        "mamba-radix-cache-strategy",
        "mamba-scheduler-strategy",
    ):
        params.pop(name, None)
    requested_mamba = requested_params.get("mamba_radix_cache_strategy")
    if requested_mamba is None:
        requested_mamba = requested_params.get("mamba_scheduler_strategy")
    if requested_mamba is None:
        requested_mamba = requested_params.get("mamba-radix-cache-strategy")
    if requested_mamba is None:
        requested_mamba = requested_params.get("mamba-scheduler-strategy")
    if requested_mamba is not None:
        params["mamba_cache_strategy"] = "inactive(radix_off)"
    return params


def _point(result: RunResult, round_number: int, output_len: int) -> dict[str, Any]:
    row = asdict(result)
    row.pop("raw", None)
    row["round"] = round_number
    valid_counts = (
        type(result.completed) is int
        and result.completed > 0
        and type(result.total_output_tokens) is int
        and result.total_output_tokens >= 0
        and type(output_len) is int
        and output_len > 0
    )
    row["output_healthy"] = valid_counts and (
        9 * output_len * result.completed
        <= 10 * result.total_output_tokens
        <= 11 * output_len * result.completed
    )
    return row


def build_candidate_rows(
    candidates: list[dict[str, Any]],
    candidate_summaries: dict[str, dict[str, Any]],
    round_results: dict[str, dict[int, list[RunResult]]],
    ranking: list[dict[str, Any]],
    *,
    output_len: int,
) -> list[dict[str, Any]]:
    """Return input-ordered candidate records, including failures and every measured point."""
    rank_by_id = {row["candidate_id"]: row for row in ranking}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("id", "unknown"))
        summary = candidate_summaries.get(candidate_id, {})
        round2 = summary.get("round2")
        completed = bool(round2 and round2.get("complete") is True)
        failures = list(summary.get("failures", []))
        last_failure = failures[-1] if failures else {}
        points = [
            _point(result, round_number, output_len)
            for round_number in (1, 2)
            for result in round_results.get(candidate_id, {}).get(round_number, [])
        ]
        rows.append(
            {
                "report_schema_version": REPORT_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "status": "completed" if completed else "incomplete",
                "requested_params": deepcopy(
                    candidate.get("requested_params", candidate.get("params", {}))
                ),
                "effective_params": _effective_params(candidate),
                "requested_command": candidate.get("requested_cmd", candidate.get("cmd")),
                "round1": summary.get("round1"),
                "round2": round2,
                "round1_batch": summary.get("round1_batch"),
                "round2_batch": summary.get("round2_batch"),
                "attempts": summary.get("attempts", 1),
                "failures": failures,
                "failed_at": last_failure.get("failed_at"),
                "failed_round": last_failure.get("round"),
                "failed_batch": last_failure.get("batch"),
                "failed_concurrency": last_failure.get("concurrency"),
                "failed_num_prompts": last_failure.get("num_prompts"),
                "failed_tp_size": last_failure.get("tp_size"),
                "failure_status": last_failure.get("status"),
                "known_issue": last_failure.get("known_issue"),
                "failure_reason": last_failure.get("reason"),
                "concurrency_points": points,
                **rank_by_id.get(candidate_id, {}),
            }
        )
    return rows


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_reports(
    results_dir: Path,
    *,
    ranking: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    task_status: dict[str, Any],
) -> None:
    # Serialize the complete report set before replacing any file.  If one
    # payload contains NaN/Infinity (or another JSON error), all previous files
    # remain untouched instead of leaving a mixed-generation report directory.
    ranking_text = json.dumps(
        ranking, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    candidates_text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in candidate_rows
    )
    task_status_text = json.dumps(
        task_status, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    _atomic_write(results_dir / "ranking.json", ranking_text)
    _atomic_write(results_dir / "candidate_results.jsonl", candidates_text)
    _atomic_write(results_dir / "task_status.json", task_status_text)


def render_candidate_preview(rows: list[dict[str, Any]]) -> list[str]:
    """Render exactly one compact terminal line per candidate."""
    rendered: list[str] = []
    for row in rows:
        requested_params = json.dumps(
            row["requested_params"], ensure_ascii=False, separators=(",", ":")
        )
        effective_params = json.dumps(
            row["effective_params"], ensure_ascii=False, separators=(",", ":")
        )
        points = ",".join(
            f"r{p['round']}/C{p['concurrency']}:status={p['status']},"
            f"tput={p['total_throughput']:.1f},"
            f"ttft={p['mean_ttft_ms']:.1f}ms,tpot={p['mean_tpot_ms']:.1f}ms,"
            f"succ={p['success_rate']:.3f},out_ok={'yes' if p['output_healthy'] else 'no'}"
            for p in row["concurrency_points"]
        ) or "none"
        rendered.append(
            f"{row['candidate_id']} status={row['status']} rank={row.get('rank', '-')} "
            f"requested_params={requested_params} effective_params={effective_params} "
            f"batches=r1:{row.get('round1_batch')},r2:{row.get('round2_batch')} "
            f"attempts={row['attempts']} best_C={row.get('best_concurrency')} "
            f"goodput_host={row.get('goodput_per_host', 0):.1f} "
            f"beats_baseline_threshold={'yes' if row.get('beats_baseline_threshold') else 'no'} "
            f"points=[{points}] failed_at={row.get('failed_at')} "
            f"failure_status={row.get('failure_status')} "
            f"failed_tp={row.get('failed_tp_size')} "
            f"failed_C={row.get('failed_concurrency')} "
            f"failed_N={row.get('failed_num_prompts')} "
            f"known_issue={row.get('known_issue')} "
            f"failure={row.get('failure_reason')}"
        )
    return rendered
