"""Build and persist complete, one-row-per-candidate benchmark reports."""

from __future__ import annotations

import json
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
        (float(row["goodput_per_host"]) for row in rows if row["candidate_id"] == "baseline"),
        None,
    )
    threshold = baseline * (1 + threshold_pct / 100) if baseline is not None else None
    for rank, row in enumerate(rows, 1):
        value = float(row.get("goodput_per_host", 0))
        row["rank"] = rank
        row["baseline_goodput_per_host"] = baseline
        row["threshold_goodput_per_host"] = threshold
        row["goodput_delta"] = value - baseline if baseline is not None else None
        row["goodput_delta_pct"] = (
            round((value / baseline - 1) * 100, 6) if baseline not in (None, 0) else None
        )
        row["beats_baseline_threshold"] = (
            value >= threshold
            if threshold is not None and row["candidate_id"] != "baseline"
            else False
        )
    return rows


def _effective_params(candidate: dict[str, Any]) -> dict[str, Any]:
    params = deepcopy(candidate.get("params", {}))
    params.pop("disable-radix-cache", None)
    params.pop("disable_radix_cache", None)
    params["disable_radix_cache"] = True
    requested_mamba = params.pop("mamba_radix_cache_strategy", None)
    requested_mamba = params.pop("mamba_scheduler_strategy", requested_mamba)
    requested_mamba = params.pop("mamba-radix-cache-strategy", requested_mamba)
    requested_mamba = params.pop("mamba-scheduler-strategy", requested_mamba)
    if requested_mamba is not None:
        params["mamba_cache_strategy"] = "inactive(radix_off)"
    return params


def _point(result: RunResult, round_number: int, output_len: int) -> dict[str, Any]:
    row = asdict(result)
    row.pop("raw", None)
    row["round"] = round_number
    row["output_healthy"] = (
        result.completed > 0 and result.avg_output_tokens >= output_len * 0.9
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
        completed = bool(round2 and round2.get("stop_reason") != "health_check_failed")
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
                "status": "completed" if completed else "failed",
                "requested_params": deepcopy(candidate.get("params", {})),
                "effective_params": _effective_params(candidate),
                "requested_command": candidate.get("cmd"),
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
    _atomic_write(
        results_dir / "ranking.json",
        json.dumps(ranking, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        results_dir / "candidate_results.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidate_rows),
    )
    _atomic_write(
        results_dir / "task_status.json",
        json.dumps(task_status, ensure_ascii=False, indent=2) + "\n",
    )


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
            f"r{p['round']}/C{p['concurrency']}:tput={p['total_throughput']:.1f},"
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
            f"failure={row.get('failure_reason')}"
        )
    return rendered
