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
    baseline_row = next(
        (row for row in rows if row["candidate_id"] == "baseline"), None
    )
    strict_intervals = bool(
        any(
            key in row
            for row in rows
            for key in (
                "goodput_per_host_min",
                "goodput_per_host_median",
                "goodput_per_host_max",
            )
        )
    )

    def strict_interval(row: dict[str, Any] | None) -> tuple[float, float, float] | None:
        if row is None:
            return None
        interval = (
            _finite_float(row.get("goodput_per_host_min")),
            _finite_float(row.get("goodput_per_host_median")),
            _finite_float(row.get("goodput_per_host_max")),
        )
        if any(value is None for value in interval):
            return None
        lower, center, upper = interval
        assert lower is not None and center is not None and upper is not None
        if lower < 0 or not lower <= center <= upper:
            return None
        return lower, center, upper

    baseline_interval = strict_interval(baseline_row) if strict_intervals else None
    baseline = (
        baseline_interval[1]
        if baseline_interval is not None
        else (
            _finite_float(baseline_row.get("goodput_per_host"))
            if baseline_row is not None and not strict_intervals
            else None
        )
    )
    baseline_mode = baseline_row.get("measurement_mode") if baseline_row else None
    modes_match = bool(
        not strict_intervals
        or (
            baseline_mode in {"full_host", "estimated"}
            and all(row.get("measurement_mode") == baseline_mode for row in rows)
        )
    )
    baseline_valid = bool(
        baseline is not None
        and baseline > 0
        and (
            not strict_intervals
            or (
                baseline_row is not None
                and baseline_row.get("ranking_eligible") is True
                and baseline_interval is not None
                and modes_match
            )
        )
    )
    threshold = (
        _finite_product(
            baseline,
            _finite_sum(
                1.0,
                _finite_quotient(_finite_float(threshold_pct), 100.0),
            ),
        )
        if baseline_valid
        else None
    )
    for rank, row in enumerate(rows, 1):
        interval = strict_interval(row) if strict_intervals else None
        value = (
            interval[1]
            if interval is not None
            else (
                _finite_float(row.get("goodput_per_host"))
                if not strict_intervals
                else None
            )
        )
        interval_min = interval[0] if interval is not None else value
        interval_max = interval[2] if interval is not None else value
        delta = _finite_difference(value, baseline)
        relative = _finite_difference(_finite_quotient(value, baseline), 1.0)
        delta_pct = _finite_product(relative, 100.0)
        row_valid = bool(
            interval_min is not None
            and interval_max is not None
            and interval_min <= interval_max
            and (
                not strict_intervals
                or (
                    row.get("ranking_eligible") is True
                    and interval is not None
                    and modes_match
                )
            )
        )
        if (
            row["candidate_id"] == "baseline"
            or not baseline_valid
            or not row_valid
            or threshold is None
        ):
            threshold_status = "unknown"
        elif interval_min is not None and interval_min >= threshold:
            threshold_status = "yes"
        elif interval_max is not None and interval_max < threshold:
            threshold_status = "no"
        else:
            threshold_status = "unknown"
        row["rank"] = rank
        row["baseline_goodput_per_host"] = baseline
        row["threshold_goodput_per_host"] = threshold
        row["goodput_delta"] = delta
        row["goodput_delta_pct"] = round(delta_pct, 6) if delta_pct is not None else None
        row["baseline_threshold_status"] = threshold_status
        row["beats_baseline_threshold"] = threshold_status == "yes"
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
        measurement_mode = summary.get("measurement_mode")
        strict_measurement = measurement_mode in {"full_host", "estimated"}
        completed = bool(
            round2
            and round2.get("complete") is True
            and round2.get("certainty") == "exact"
            and (
                not strict_measurement
                or summary.get("measurement_valid") is True
            )
        )
        ranking_eligible = bool(summary.get("ranking_eligible", False))
        eligibility_reason = summary.get("ranking_eligibility_reason")
        if not ranking_eligible and not eligibility_reason:
            eligibility_reason = "fresh round-2 measurement unavailable"
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
                "measurement_mode": measurement_mode,
                "ranking_eligible": ranking_eligible,
                "ranking_eligibility_reason": eligibility_reason,
                "rank_group": None,
                "baseline_threshold_status": "unknown",
                "beats_baseline_threshold": False,
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
