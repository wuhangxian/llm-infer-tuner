"""Build and persist complete, one-row-per-candidate benchmark reports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from runners.metrics import SEARCH_VERDICT_STATUSES, ProbeStatus, RunResult

REPORT_SCHEMA_VERSION = 2


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
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


_REPORT_FILES = (
    "ranking.json",
    "candidate_results.jsonl",
    "probe_results.jsonl",
    "task_status.json",
    "provenance.json",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def _stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _sanitize_optional_json(value: Any, *, path: str = "raw") -> tuple[Any, list[str]]:
    reasons: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        return None, [f"{path}: non-finite numeric value replaced with null"]
    if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > 4096:
        return None, [f"{path}: unrepresentable integer replaced with null"]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            safe, nested = _sanitize_optional_json(item, path=f"{path}.{key}")
            sanitized[str(key)] = safe
            reasons.extend(nested)
        return sanitized, reasons
    if isinstance(value, (list, tuple)):
        sanitized_items: list[Any] = []
        for index, item in enumerate(value):
            safe, nested = _sanitize_optional_json(item, path=f"{path}[{index}]")
            sanitized_items.append(safe)
            reasons.extend(nested)
        return sanitized_items, reasons
    return value, reasons


def _sanitize_unrepresentable_integers(
    value: Any, *, path: str = "normalized"
) -> tuple[Any, list[str]]:
    """Replace only integers too large for Python's JSON digit limit.

    Required floating-point metrics remain strict; this narrow normalization
    prevents malformed arbitrary-precision integers from escaping as a writer
    crash before the hierarchy validator can downgrade the report.
    """

    if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > 4096:
        return None, [f"{path}: unrepresentable integer replaced with null"]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        reasons: list[str] = []
        for key, item in value.items():
            safe, nested = _sanitize_unrepresentable_integers(
                item, path=f"{path}.{key}"
            )
            sanitized[str(key)] = safe
            reasons.extend(nested)
        return sanitized, reasons
    if isinstance(value, (list, tuple)):
        sanitized_items: list[Any] = []
        reasons = []
        for index, item in enumerate(value):
            safe, nested = _sanitize_unrepresentable_integers(
                item, path=f"{path}[{index}]"
            )
            sanitized_items.append(safe)
            reasons.extend(nested)
        return sanitized_items, reasons
    # Sets, bytes, Paths, and other application objects cannot be represented
    # by strict JSON.  Keep the report writable and make the invalidity
    # explicit in ``invariant_errors`` rather than leaking a TypeError from the
    # encoder.  (Unknown candidate scalar extensions are intentionally not
    # passed through this helper; see ``write_reports``.)
    if value is not None and not isinstance(value, (str, bool, int, float)):
        return None, [f"{path}: unsupported JSON value replaced with null"]
    return value, []


def _sanitize_report_tree(
    value: Any, *, path: str
) -> tuple[Any, list[str]]:
    """Make schema-bearing report data JSON-safe without hiding invalidity.

    Required evidence is validated below and must remain distinguishable from
    a valid measurement.  Replacing a non-finite number (or an integer beyond
    Python's JSON digit limit) with ``None`` lets the validator record a
    provisional/invariant error instead of crashing while serializing the
    report.  Callers add the returned reasons to ``invariant_errors``.
    """

    if isinstance(value, float) and not math.isfinite(value):
        return None, [f"{path}: non-finite numeric value replaced with null"]
    if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > 4096:
        return None, [f"{path}: unrepresentable integer replaced with null"]
    if isinstance(value, dict):
        sanitized: dict[Any, Any] = {}
        reasons: list[str] = []
        for key, item in value.items():
            safe, nested = _sanitize_report_tree(item, path=f"{path}.{key}")
            if isinstance(key, str):
                safe_key: Any = key
            else:
                try:
                    safe_key = str(key)
                except Exception:
                    safe_key = "<invalid-key>"
                reasons.append(f"{path}: non-string object key replaced with text")
            if safe_key in sanitized:
                reasons.append(f"{path}: duplicate key after JSON key normalization")
            sanitized[safe_key] = safe
            reasons.extend(nested)
        return sanitized, reasons
    if isinstance(value, (list, tuple)):
        sanitized_items: list[Any] = []
        reasons = []
        for index, item in enumerate(value):
            safe, nested = _sanitize_report_tree(item, path=f"{path}[{index}]")
            sanitized_items.append(safe)
            reasons.extend(nested)
        return sanitized_items, reasons
    # Sets, bytes, Paths, and other application objects cannot be represented
    # by strict JSON. Keep the report writable and make the invalidity
    # explicit in ``invariant_errors`` rather than leaking a TypeError from the
    # encoder.
    if value is not None and not isinstance(value, (str, bool, int, float)):
        return None, [f"{path}: unsupported JSON value replaced with null"]
    return value, []


# Candidate summaries contain a few free-form compatibility fields (for
# example an arbitrary legacy ``metric`` key).  Sanitize only fields that are
# part of the report schema so an unknown malformed extension still retains
# the historical strict-JSON rejection behavior.
_CANDIDATE_NUMERIC_FIELDS = frozenset(
    {
        "rank",
        "rank_group",
        "best_concurrency",
        "actual_instances",
        "sample_count",
        "instances_per_host",
        "goodput_raw",
        "goodput_per_host",
        "goodput_per_host_min",
        "goodput_per_host_median",
        "goodput_per_host_max",
        "baseline_goodput_per_host",
        "threshold_goodput_per_host",
        "baseline_threshold_pct",
        "goodput_delta",
        "goodput_delta_pct",
        "attempts",
        "recovery_count",
        "failed_round",
        "failed_concurrency",
        "failed_num_prompts",
        "failed_tp_size",
    }
)
_CANDIDATE_KNOWN_SUBTREES = frozenset(
    {
        "requested_params",
        "effective_params",
        "round1",
        "round2",
        "sample_groups",
        "incomplete_groups",
        "failures",
        "concurrency_points",
    }
)
_CANDIDATE_INTEGER_FIELDS = frozenset(
    {
        "rank",
        "rank_group",
        "best_concurrency",
        "actual_instances",
        "sample_count",
        "attempts",
        "recovery_count",
        "failed_round",
        "failed_concurrency",
        "failed_num_prompts",
        "failed_tp_size",
    }
)
_CANDIDATE_FLOAT_FIELDS = frozenset(_CANDIDATE_NUMERIC_FIELDS - _CANDIDATE_INTEGER_FIELDS)


_PROBE_REQUIRED_FIELDS = frozenset(
    {
        "probe_id",
        "candidate_id",
        "record_type",
        "round",
        "batch",
        "concurrency",
        "repeat",
        "recovery",
        "measurement_mode",
        "started_at",
        "ended_at",
        "failed_at",
        "status",
        "failure_reason",
        "known_issue",
        "raw",
        "normalized",
        "instances",
        "output_healthy",
        "server_health",
        "artifacts",
        "statistical_vote",
    }
)


def _is_finite_nonnegative_metric(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(converted) and converted >= 0


def _finite_nonnegative_float(value: Any) -> float | None:
    if not _is_finite_nonnegative_metric(value):
        return None
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError):
        return None


def _probe_hierarchy_errors(
    candidate_rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    # This validator is also used at the report boundary, where callers may
    # hand us decoded but untrusted JSON values.  Keep malformed top-level
    # rows in the error stream, but never let a scalar/non-hashable value abort
    # validation before a provisional report can be written.
    if not isinstance(candidate_rows, list):
        errors.append("candidate_rows must be a list")
        candidate_rows = []
    else:
        valid_candidates: list[dict[str, Any]] = []
        for candidate in candidate_rows:
            if isinstance(candidate, dict):
                valid_candidates.append(candidate)
            else:
                errors.append("candidate row must be an object")
        candidate_rows = valid_candidates
    if not isinstance(probe_rows, list):
        errors.append("probe_rows must be a list")
        probe_rows = []
    else:
        valid_probes: list[dict[str, Any]] = []
        for probe in probe_rows:
            if isinstance(probe, dict):
                valid_probes.append(probe)
            else:
                errors.append("probe row must be an object")
        probe_rows = valid_probes

    probes: dict[str, dict[str, Any]] = {}
    for row in probe_rows:
        probe_id = row.get("probe_id")
        if isinstance(probe_id, str) and probe_id:
            probes[probe_id] = row
        else:
            errors.append(f"probe {probe_id} has invalid probe_id")
    physical_parent_count: dict[str, int] = {}
    aggregate_group_parent_count: dict[str, int] = {}
    for row in probe_rows:
        probe_id = row.get("probe_id")
        missing = sorted(_PROBE_REQUIRED_FIELDS - set(row))
        if missing:
            errors.append(f"probe {probe_id} missing fields: {','.join(missing)}")
            continue
        record_type = row.get("record_type")
        if not isinstance(record_type, str) or record_type not in {
            "physical_probe",
            "aggregate_sample",
            "infrastructure_attempt",
        }:
            errors.append(f"probe {probe_id} has invalid record_type")
        round_number = row.get("round")
        if type(round_number) is not int or round_number not in {1, 2}:
            errors.append(f"probe {probe_id} has invalid round")
        measurement_mode = row.get("measurement_mode")
        if not isinstance(measurement_mode, str) or measurement_mode not in {
            "full_host",
            "estimated",
        }:
            errors.append(f"probe {probe_id} has invalid measurement_mode")
        if not isinstance(row.get("batch"), str) or not row["batch"]:
            errors.append(f"probe {probe_id} has invalid batch")
        started_at = _parse_timestamp(row.get("started_at"))
        ended_at = _parse_timestamp(row.get("ended_at"))
        failed_at = (
            _parse_timestamp(row.get("failed_at"))
            if row.get("failed_at") is not None
            else None
        )
        if started_at is None or ended_at is None or ended_at < started_at:
            errors.append(f"probe {probe_id} has invalid timestamps")
        else:
            started_offset = started_at.utcoffset()
            ended_offset = ended_at.utcoffset()
            if (
                started_offset is None
                or ended_offset is None
                or started_offset.total_seconds() != 0
                or ended_offset.total_seconds() != 0
            ):
                errors.append(f"probe {probe_id} timestamps must be UTC")
        if row.get("failed_at") is not None and failed_at is None:
            errors.append(f"probe {probe_id} has invalid failed_at")
        elif (
            failed_at is not None
            and started_at is not None
            and ended_at is not None
            and not started_at <= failed_at <= ended_at
        ):
            errors.append(f"probe {probe_id} failed_at is outside attempt timestamps")
        for text_field in ("failure_reason", "known_issue"):
            text_value = row.get(text_field)
            if text_value is not None and not isinstance(text_value, str):
                errors.append(f"probe {probe_id} {text_field} must be text or null")
        minimum_concurrency = 0 if record_type == "infrastructure_attempt" else 1
        if (
            type(row.get("concurrency")) is not int
            or row["concurrency"] < minimum_concurrency
        ):
            errors.append(f"probe {probe_id} has invalid concurrency")
        if type(row.get("repeat")) is not int or row["repeat"] < -1:
            errors.append(f"probe {probe_id} has invalid repeat")
        if type(row.get("recovery")) is not int or row["recovery"] < 0:
            errors.append(f"probe {probe_id} has invalid recovery")
        if type(row.get("instances")) is not int or row["instances"] <= 0:
            errors.append(f"probe {probe_id} has invalid instances")
        try:
            probe_status = ProbeStatus(row.get("status"))
        except (TypeError, ValueError):
            errors.append(f"probe {probe_id} has invalid status")
            probe_status = None
        for field_name in ("raw", "normalized", "server_health"):
            if not isinstance(row.get(field_name), dict):
                errors.append(f"probe {probe_id} has non-object {field_name}")
        if row.get("output_healthy") is not None and type(
            row.get("output_healthy")
        ) is not bool:
            errors.append(f"probe {probe_id} has invalid output_healthy")
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"probe {probe_id} has invalid artifacts")
        else:
            for artifact in artifacts:
                if not isinstance(artifact, dict) or not isinstance(
                    artifact.get("path"), str
                ):
                    errors.append(f"probe {probe_id} has malformed artifact")
                    continue
                artifact_path = Path(artifact["path"])
                if not artifact["path"]:
                    errors.append(f"probe {probe_id} has empty artifact path")
                if artifact_path.is_absolute() or ".." in artifact_path.parts:
                    errors.append(f"probe {probe_id} has unsafe artifact path")
                if not _valid_sha256(artifact.get("sha256")) and not (
                    artifact.get("sha256") is None
                    and isinstance(artifact.get("unavailable_reason"), str)
                    and artifact.get("unavailable_reason")
                ):
                    errors.append(
                        f"probe {probe_id} artifact requires sha256 or unavailable_reason"
                    )
        if probe_status in SEARCH_VERDICT_STATUSES:
            if row.get("failed_at") is not None:
                errors.append(f"verdict probe {probe_id} cannot have failed_at")
            if probe_status == ProbeStatus.OK and row.get("failure_reason") is not None:
                errors.append(f"ok probe {probe_id} cannot have failure_reason")
            if probe_status == ProbeStatus.SLA_FAILED and (
                not isinstance(row.get("failure_reason"), str)
                or not row.get("failure_reason")
            ):
                errors.append(f"sla_failed probe {probe_id} lacks failure_reason")
            if row.get("output_healthy") is not True:
                errors.append(f"verdict probe {probe_id} output is not healthy")
            normalized = row.get("normalized")
            if not isinstance(normalized, dict) or any(
                not _is_finite_nonnegative_metric(normalized.get(name))
                for name in (
                    "total_throughput",
                    "mean_ttft_ms",
                    "mean_tpot_ms",
                    "success_rate",
                )
            ):
                errors.append(f"probe {probe_id} has invalid normalized metrics")
            elif normalized["success_rate"] > 1:
                errors.append(f"probe {probe_id} success_rate exceeds one")
            health = row.get("server_health")
            if not isinstance(health, dict) or health.get("before") != "healthy" or health.get(
                "after"
            ) != "healthy":
                errors.append(f"verdict probe {probe_id} server health is not healthy")
        elif probe_status is not None and (
            failed_at is None
            or not isinstance(row.get("failure_reason"), str)
            or not row.get("failure_reason")
        ):
            errors.append(f"failed probe {probe_id} lacks failure time/reason")
        if row.get("record_type") == "physical_probe":
            if row.get("statistical_vote") is not False:
                errors.append(f"physical probe {probe_id} cannot be a statistical vote")
            if row.get("instances") != 1:
                errors.append(f"physical probe {probe_id} must have one instance")
            if type(row.get("replica_index")) is not int or row["replica_index"] < 0:
                errors.append(f"physical probe {probe_id} has invalid replica_index")
            if type(row.get("port")) is not int or not 1 <= row["port"] <= 65535:
                errors.append(f"physical probe {probe_id} has invalid port")
        elif row.get("record_type") == "infrastructure_attempt":
            if row.get("statistical_vote") is not False:
                errors.append(f"infrastructure attempt {probe_id} cannot vote")
        elif row.get("record_type") == "aggregate_sample":
            replica_ids = row.get("replica_probe_ids")
            if (
                not isinstance(replica_ids, list)
                or not replica_ids
                or any(not isinstance(value, str) for value in replica_ids)
                or len(replica_ids) != len(set(replica_ids))
            ):
                errors.append(f"aggregate probe {probe_id} has invalid replica_probe_ids")
                continue
            linked_optional = [probes.get(replica_id) for replica_id in replica_ids]
            if any(probe is None for probe in linked_optional):
                errors.append(f"aggregate probe {probe_id} references missing replica probe")
                continue
            linked = [probe for probe in linked_optional if probe is not None]
            if any(probe.get("record_type") != "physical_probe" for probe in linked):
                errors.append(f"aggregate probe {probe_id} links a non-physical probe")
            replica_indexes = [probe.get("replica_index") for probe in linked]
            if any(type(index) is not int for index in replica_indexes) or len(
                replica_indexes
            ) != len(set(index for index in replica_indexes if type(index) is int)):
                errors.append(f"aggregate probe {probe_id} has duplicate replica indexes")
            else:
                dense_indexes: list[int] = [
                    index for index in replica_indexes if type(index) is int
                ]
                if sorted(dense_indexes) != list(range(len(dense_indexes))):
                    # Replica indices are local to one logical aggregate
                    # sample, not arbitrary GPU IDs.  Requiring a dense
                    # zero-based set catches dropped/duplicated physical
                    # children while keeping NUMA/global placement details in
                    # the separate topology provenance fields.
                    errors.append(
                        f"aggregate probe {probe_id} has non-contiguous replica indexes"
                    )
            for replica_id in replica_ids:
                physical_parent_count[replica_id] = (
                    physical_parent_count.get(replica_id, 0) + 1
                )
            coordinates = (
                row.get("candidate_id"),
                row.get("round"),
                row.get("batch"),
                row.get("concurrency"),
                row.get("repeat"),
                row.get("recovery"),
                row.get("measurement_mode"),
            )
            if any(
                (
                    probe.get("candidate_id"),
                    probe.get("round"),
                    probe.get("batch"),
                    probe.get("concurrency"),
                    probe.get("repeat"),
                    probe.get("recovery"),
                    probe.get("measurement_mode"),
                )
                != coordinates
                for probe in linked
            ):
                errors.append(f"aggregate probe {probe_id} replica coordinates differ")
            if row.get("instances") != len(replica_ids):
                errors.append(f"aggregate probe {probe_id} instance count differs")
            child_statuses: list[ProbeStatus | None] = []
            for probe in linked:
                try:
                    child_statuses.append(ProbeStatus(probe.get("status")))
                except (TypeError, ValueError):
                    child_statuses.append(None)
            # An aggregate ``ok`` verdict certifies the whole host sample,
            # so every physical replica must itself have produced an ``ok``
            # verdict.  In particular, do not let a physical SLA failure be
            # hidden behind an apparently healthy aggregate row.  Aggregate
            # ``sla_failed`` may still be a host-level verdict even when each
            # replica is individually healthy (for example, combined
            # throughput/latency can miss the SLA), so the implication is
            # intentionally one-way.
            if probe_status == ProbeStatus.OK and any(
                child_status != ProbeStatus.OK for child_status in child_statuses
            ):
                errors.append(
                    f"aggregate probe {probe_id} status ok conflicts with physical child status"
                )
            if probe_status in SEARCH_VERDICT_STATUSES and any(
                child_status not in SEARCH_VERDICT_STATUSES
                for child_status in child_statuses
            ):
                errors.append(f"aggregate probe {probe_id} hides failed physical probe")
            if probe_status in SEARCH_VERDICT_STATUSES and all(
                isinstance(probe.get("normalized"), dict)
                and all(
                    _is_finite_nonnegative_metric(probe["normalized"].get(name))
                    for name in (
                        "total_throughput",
                        "success_rate",
                        "mean_ttft_ms",
                        "mean_tpot_ms",
                    )
                )
                for probe in linked
            ):
                aggregate_normalized = row.get("normalized", {})
                if isinstance(aggregate_normalized, dict):
                    expected_throughput = sum(
                        probe["normalized"].get("total_throughput", math.nan)
                        for probe in linked
                    )
                    expected_success = min(
                        probe["normalized"].get("success_rate", math.nan)
                        for probe in linked
                    )
                    expected_ttft = max(
                        probe["normalized"].get("mean_ttft_ms", math.nan)
                        for probe in linked
                    )
                    expected_tpot = max(
                        probe["normalized"].get("mean_tpot_ms", math.nan)
                        for probe in linked
                    )
                    for name, expected_value in (
                        ("total_throughput", expected_throughput),
                        ("success_rate", expected_success),
                        ("mean_ttft_ms", expected_ttft),
                        ("mean_tpot_ms", expected_tpot),
                    ):
                        if aggregate_normalized.get(name) != expected_value:
                            errors.append(
                                f"aggregate probe {probe_id} normalized {name} differs"
                            )
            repeat = row.get("repeat")
            expected_vote = (
                type(repeat) is int
                and repeat >= 0
                and probe_status in SEARCH_VERDICT_STATUSES
            )
            if row.get("statistical_vote") is not expected_vote:
                errors.append(f"aggregate probe {probe_id} has invalid statistical_vote")

    aggregate_recoveries: dict[tuple[Any, ...], list[int]] = {}
    physical_identities: set[tuple[str, int, str, int, int, int, str, int]] = set()
    for row in probe_rows:
        candidate_id = row.get("candidate_id")
        round_number = row.get("round")
        batch = row.get("batch")
        concurrency = row.get("concurrency")
        repeat = row.get("repeat")
        recovery = row.get("recovery")
        measurement_mode = row.get("measurement_mode")
        if not (
            isinstance(candidate_id, str)
            and type(round_number) is int
            and isinstance(batch, str)
            and type(concurrency) is int
            and type(repeat) is int
            and type(recovery) is int
            and isinstance(measurement_mode, str)
        ):
            continue
        record_type = row.get("record_type")
        if record_type in ("aggregate_sample", "infrastructure_attempt"):
            replica_identity: tuple[Any, ...] = ()
            if record_type == "infrastructure_attempt":
                replica_index = row.get("replica_index")
                port = row.get("port")
                replica_identity = (
                    (replica_index, port)
                    if type(replica_index) is int and type(port) is int
                    else ("unknown",)
                )
            logical_key = (
                candidate_id,
                record_type,
                round_number,
                batch,
                concurrency,
                repeat,
                measurement_mode,
                replica_identity,
            )
            aggregate_recoveries.setdefault(logical_key, []).append(recovery)
        elif record_type == "physical_probe":
            logical_key = (
                candidate_id,
                round_number,
                batch,
                concurrency,
                repeat,
                measurement_mode,
            )
            replica_index = row.get("replica_index")
            if type(replica_index) is not int:
                continue
            identity = (*logical_key[:-1], recovery, logical_key[-1], replica_index)
            if identity in physical_identities:
                errors.append(
                    f"probe {row.get('probe_id')} duplicates a physical recovery identity"
                )
            physical_identities.add(identity)
    for logical_key, recoveries in aggregate_recoveries.items():
        unique_recoveries = sorted(set(recoveries))
        if len(unique_recoveries) != len(recoveries):
            errors.append(
                "duplicate aggregate recovery for "
                f"{logical_key[0]} round {logical_key[2]} C={logical_key[4]} "
                f"repeat={logical_key[5]}"
            )
        # Recovery ordinals are small attempt counters.  Bound the value
        # before constructing ``range`` so an untrusted huge integer cannot
        # trigger an OverflowError/DoS in the report writer; it is recorded as
        # an ordinary invariant violation and the generation becomes
        # provisional instead.
        if unique_recoveries and unique_recoveries[-1] > 10_000:
            errors.append(
                "recovery ordinal is unreasonably large for "
                f"{logical_key[0]} round {logical_key[2]} C={logical_key[4]} "
                f"repeat={logical_key[5]}"
            )
        elif unique_recoveries and unique_recoveries != list(
            range(unique_recoveries[-1] + 1)
        ):
            errors.append(
                "non-contiguous recovery chain for "
                f"{logical_key[0]} round {logical_key[2]} C={logical_key[4]} "
                f"repeat={logical_key[5]}"
            )

    for candidate in candidate_rows:
        candidate_id = candidate.get("candidate_id")
        complete_groups = candidate.get("sample_groups", [])
        incomplete_groups = candidate.get("incomplete_groups", [])
        if not isinstance(complete_groups, list) or not isinstance(incomplete_groups, list):
            errors.append(f"candidate {candidate_id} has invalid sample group lists")
            continue
        referenced_aggregates: set[str] = set()
        group_ids: set[str] = set()
        logical_group_ids: set[tuple[Any, ...]] = set()
        for group, complete in [
            *((group, True) for group in complete_groups),
            *((group, False) for group in incomplete_groups),
        ]:
            if not isinstance(group, dict):
                errors.append(f"candidate {candidate_id} has invalid sample group")
                continue
            group_id = group.get("group_id")
            if not isinstance(group_id, str) or not group_id or group_id in group_ids:
                errors.append(f"candidate {candidate_id} has duplicate or invalid group_id")
            else:
                group_ids.add(group_id)
            aggregate_ids = group.get("aggregate_probe_ids")
            round_number = group.get("round")
            logical_group_id = (
                round_number,
                group.get("batch"),
                group.get("concurrency"),
            )
            if type(round_number) is not int or round_number not in {1, 2}:
                errors.append(f"candidate {candidate_id} sample group has invalid round")
            group_concurrency = group.get("concurrency")
            if type(group_concurrency) is not int or group_concurrency <= 0:
                errors.append(
                    f"candidate {candidate_id} sample group has invalid concurrency"
                )
            if (
                complete
                and type(round_number) is int
                and type(group.get("concurrency")) is int
                and (isinstance(group.get("batch"), str) or group.get("batch") is None)
            ):
                if logical_group_id in logical_group_ids:
                    errors.append(
                        f"candidate {candidate_id} has duplicate logical sample group"
                    )
                else:
                    logical_group_ids.add(logical_group_id)
            expected_count = 1 if round_number == 1 else 3
            if (
                not isinstance(aggregate_ids, list)
                or any(not isinstance(value, str) for value in aggregate_ids)
                or len(aggregate_ids) != len(set(aggregate_ids))
                or (complete and len(aggregate_ids) != expected_count)
                or (not complete and len(aggregate_ids) >= expected_count)
            ):
                errors.append(f"candidate {candidate_id} has invalid aggregate sample group")
                continue
            if complete and not isinstance(group.get("representative"), dict):
                errors.append(f"candidate {candidate_id} sample group has no representative")
            aggregate_rows_optional = [
                probes.get(probe_id) for probe_id in aggregate_ids
            ]
            if any(row is None for row in aggregate_rows_optional):
                errors.append(f"candidate {candidate_id} sample group references missing probe")
                continue
            aggregate_rows = [
                row for row in aggregate_rows_optional if row is not None
            ]
            if any(row.get("record_type") != "aggregate_sample" for row in aggregate_rows):
                errors.append(f"candidate {candidate_id} sample group includes physical probe")
            if any(
                row.get("candidate_id") != candidate_id
                or row.get("round") != round_number
                or row.get("concurrency") != group.get("concurrency")
                or (
                    group.get("batch") is not None
                    and row.get("batch") != group.get("batch")
                )
                or (
                    group.get("measurement_mode") is not None
                    and row.get("measurement_mode") != group.get("measurement_mode")
                )
                for row in aggregate_rows
            ):
                errors.append(f"candidate {candidate_id} sample group coordinates differ")
            if complete:
                repeats = [row.get("repeat") for row in aggregate_rows]
                if any(type(repeat) is not int for repeat in repeats) or set(
                    repeat for repeat in repeats if type(repeat) is int
                ) != set(range(expected_count)):
                    errors.append(f"candidate {candidate_id} sample group repeats differ")
            if complete:
                aggregate_statuses: list[ProbeStatus | None] = []
                for aggregate_row in aggregate_rows:
                    try:
                        aggregate_statuses.append(
                            ProbeStatus(aggregate_row.get("status"))
                        )
                    except (TypeError, ValueError):
                        aggregate_statuses.append(None)
                if any(
                    status not in SEARCH_VERDICT_STATUSES
                    for status in aggregate_statuses
                ) or any(
                    aggregate_row.get("statistical_vote") is not True
                    for aggregate_row in aggregate_rows
                ):
                    errors.append(
                        f"candidate {candidate_id} sample group contains non-verdict vote"
                    )
                pass_votes = sum(
                    status == ProbeStatus.OK for status in aggregate_statuses
                )
                expected_qualifies = pass_votes > expected_count / 2
                if group.get("qualifies") is not expected_qualifies:
                    errors.append(
                        f"candidate {candidate_id} sample group qualifies differs"
                    )
                representative = group.get("representative")
                expected_status = (
                    ProbeStatus.OK if expected_qualifies else ProbeStatus.SLA_FAILED
                )
                if not isinstance(representative, dict) or representative.get(
                    "status"
                ) != expected_status.value:
                    errors.append(
                        f"candidate {candidate_id} sample group representative status differs"
                    )
                elif all(
                    isinstance(aggregate_row.get("normalized"), dict)
                    and all(
                        _is_finite_nonnegative_metric(
                            aggregate_row["normalized"].get(name)
                        )
                        for name in (
                            "total_throughput",
                            "mean_ttft_ms",
                            "mean_tpot_ms",
                            "success_rate",
                        )
                    )
                    for aggregate_row in aggregate_rows
                ):
                    for name in (
                        "total_throughput",
                        "mean_ttft_ms",
                        "mean_tpot_ms",
                        "success_rate",
                    ):
                        expected_median = median(
                            aggregate_row["normalized"][name]
                            for aggregate_row in aggregate_rows
                        )
                        if representative.get(name) != expected_median:
                            errors.append(
                                f"candidate {candidate_id} representative {name} differs"
                            )
            for aggregate_id in aggregate_ids:
                aggregate_group_parent_count[aggregate_id] = (
                    aggregate_group_parent_count.get(aggregate_id, 0) + 1
                )
            referenced_aggregates.update(aggregate_ids)
        ungrouped_votes = [
            str(row.get("probe_id"))
            for row in probe_rows
            if row.get("candidate_id") == candidate_id
            and row.get("record_type") == "aggregate_sample"
            and row.get("statistical_vote") is True
            and row.get("probe_id") not in referenced_aggregates
        ]
        if ungrouped_votes:
            errors.append(
                f"candidate {candidate_id} has ungrouped aggregate votes "
                f"{','.join(ungrouped_votes)}"
            )
    for row in probe_rows:
        probe_id = str(row.get("probe_id"))
        if row.get("record_type") == "physical_probe" and physical_parent_count.get(
            probe_id, 0
        ) != 1:
            errors.append(f"physical probe {probe_id} must have exactly one aggregate parent")
        if (
            row.get("record_type") == "aggregate_sample"
            and row.get("statistical_vote") is True
            and aggregate_group_parent_count.get(probe_id, 0) != 1
        ):
            errors.append(f"aggregate probe {probe_id} must have exactly one sample group")
        try:
            row_status = ProbeStatus(row.get("status"))
        except (TypeError, ValueError):
            row_status = None
        record_type = row.get("record_type")
        if isinstance(record_type, str) and record_type in {
            "aggregate_sample",
            "infrastructure_attempt",
        } and row_status not in SEARCH_VERDICT_STATUSES:
            row_recovery = row.get("recovery")
            if type(row_recovery) is not int:
                continue
            recovered = any(
                candidate.get("candidate_id") == row.get("candidate_id")
                and candidate.get("record_type") == record_type
                and candidate.get("round") == row.get("round")
                and candidate.get("batch") == row.get("batch")
                and candidate.get("concurrency") == row.get("concurrency")
                and candidate.get("repeat") == row.get("repeat")
                and candidate.get("measurement_mode") == row.get("measurement_mode")
                and type(candidate.get("recovery")) is int
                and candidate["recovery"] > row_recovery
                and (
                    candidate.get("status") in {status.value for status in SEARCH_VERDICT_STATUSES}
                )
                and (
                    row.get("record_type") == "infrastructure_attempt"
                    or aggregate_group_parent_count.get(
                        str(candidate.get("probe_id")), 0
                    )
                    == 1
                )
                for candidate in probe_rows
                if isinstance(candidate.get("status"), str)
            )
            if not recovered:
                errors.append(f"terminal aggregate probe {probe_id} has no recovered verdict")
    return errors


def _valid_sha256(value: Any, *, prefix: bool = False) -> bool:
    pattern = r"sha256:[0-9a-f]{64}" if prefix else r"[0-9a-f]{64}"
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _provenance_errors(provenance: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(provenance, dict):
        return ["provenance must be an object"]
    git = provenance.get("git")
    if (
        not isinstance(git, dict)
        or not isinstance(git.get("sha"), str)
        or re.fullmatch(r"[0-9a-f]{40}", git["sha"]) is None
        or type(git.get("dirty")) is not bool
    ):
        errors.append("provenance git sha/dirty is missing or invalid")
    inputs = provenance.get("inputs")
    if not isinstance(inputs, dict) or not _valid_sha256(
        inputs.get("job_sha256")
    ) or not _valid_sha256(inputs.get("config_sha256")):
        errors.append("provenance canonical job/config hashes are missing or invalid")
    image = provenance.get("image")
    if (
        not isinstance(image, dict)
        or not isinstance(image.get("reference"), str)
        or not image.get("reference")
    ):
        errors.append("provenance image reference is missing")
    elif image.get("digest") is None:
        if not isinstance(image.get("unavailable_reason"), str) or not image.get(
            "unavailable_reason"
        ):
            errors.append("provenance image digest requires unavailable_reason")
    elif not _valid_sha256(image.get("digest"), prefix=True):
        errors.append("provenance image digest is invalid")
    actual_gpu = provenance.get("actual_gpu")
    if not isinstance(actual_gpu, dict):
        errors.append("provenance actual GPU facts are missing")
    else:
        gpu_fields = ("count", "model", "memory_bytes", "topology")
        if actual_gpu.get("count") is not None and type(actual_gpu.get("count")) is not int:
            errors.append("provenance actual GPU count is invalid")
        if actual_gpu.get("memory_bytes") is not None and type(
            actual_gpu.get("memory_bytes")
        ) is not int:
            errors.append("provenance actual GPU memory is invalid")
        if actual_gpu.get("model") is not None and not isinstance(
            actual_gpu.get("model"), str
        ):
            errors.append("provenance actual GPU model is invalid")
        if actual_gpu.get("topology") is not None and not isinstance(
            actual_gpu.get("topology"), (dict, list)
        ):
            errors.append("provenance actual GPU topology is invalid")
        if any(actual_gpu.get(name) is None for name in gpu_fields):
            if not isinstance(actual_gpu.get("unavailable_reason"), str) or not actual_gpu.get(
                "unavailable_reason"
            ):
                errors.append("provenance unavailable GPU facts require unavailable_reason")
        else:
            if type(actual_gpu.get("count")) is not int or actual_gpu["count"] <= 0:
                errors.append("provenance actual GPU count is invalid")
            if not isinstance(actual_gpu.get("model"), str) or not actual_gpu["model"]:
                errors.append("provenance actual GPU model is invalid")
            if type(actual_gpu.get("memory_bytes")) is not int or actual_gpu[
                "memory_bytes"
            ] <= 0:
                errors.append("provenance actual GPU memory is invalid")
    engine = provenance.get("engine")
    if not isinstance(engine, dict):
        errors.append("provenance engine metadata is missing")
    elif engine.get("version") is None:
        if not isinstance(engine.get("unavailable_reason"), str) or not engine.get(
            "unavailable_reason"
        ):
            errors.append("provenance engine version requires unavailable_reason")
    elif not isinstance(engine.get("version"), str) or not engine.get("version"):
        errors.append("provenance engine version is invalid")
    started = _parse_timestamp(provenance.get("run_started_at"))
    ended = _parse_timestamp(provenance.get("run_ended_at"))
    if started is None or ended is None or ended < started:
        errors.append("provenance run timestamps are missing or invalid")
    else:
        started_offset = started.utcoffset()
        ended_offset = ended.utcoffset()
        if (
            started_offset is None
            or ended_offset is None
            or started_offset.total_seconds() != 0
            or ended_offset.total_seconds() != 0
        ):
            errors.append("provenance run timestamps are missing or invalid")
    return errors


def _authoritative_ranking_errors(
    candidate: dict[str, Any],
    ranking_row: dict[str, Any] | None,
    probe_by_id: dict[str, dict[str, Any]],
    expected_rank_group: int | None = None,
    baseline_row: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Cross-check ranking fields against complete fresh Round-2 sample groups."""

    candidate_id = candidate.get("candidate_id")
    errors: list[str] = []
    sample_groups = candidate.get("sample_groups")
    if not isinstance(sample_groups, list):
        return False, [f"candidate {candidate_id} ranking has no sample groups"]
    qualifying_groups = [
        group
        for group in sample_groups
        if isinstance(group, dict)
        and group.get("round") == 2
        and group.get("qualifies") is True
    ]
    authoritative_eligible = bool(qualifying_groups)
    if candidate.get("ranking_eligible") is not authoritative_eligible:
        errors.append(
            f"candidate {candidate_id} ranking eligibility differs from Round-2 evidence"
        )
    if not authoritative_eligible:
        if ranking_row is not None:
            errors.append(f"candidate {candidate_id} is ineligible but ranked")
        if candidate.get("rank") is not None:
            errors.append(f"candidate {candidate_id} ineligible rank must be null")
        return False, errors
    if ranking_row is None:
        errors.append(f"candidate {candidate_id} is eligible but missing ranking row")
        return True, errors

    group_samples: list[tuple[dict[str, Any], list[float]]] = []
    for group in qualifying_groups:
        aggregate_ids = group.get("aggregate_probe_ids")
        if not (
            isinstance(aggregate_ids, list)
            and len(aggregate_ids) == 3
            and all(isinstance(probe_id, str) for probe_id in aggregate_ids)
        ):
            errors.append(f"candidate {candidate_id} ranking group is malformed")
            continue
        aggregate_rows = [probe_by_id.get(probe_id) for probe_id in aggregate_ids]
        if any(not isinstance(probe, dict) for probe in aggregate_rows):
            errors.append(f"candidate {candidate_id} ranking group probe is missing")
            continue
        throughputs: list[float] = []
        for aggregate in aggregate_rows:
            normalized = aggregate.get("normalized") if aggregate is not None else None
            throughput = (
                normalized.get("total_throughput")
                if isinstance(normalized, dict)
                else None
            )
            converted_throughput = _finite_nonnegative_float(throughput)
            if converted_throughput is None:
                break
            throughputs.append(converted_throughput)
        if len(throughputs) != 3:
            errors.append(f"candidate {candidate_id} ranking group metrics are invalid")
            continue
        group_samples.append((group, throughputs))
    if not group_samples:
        return True, errors

    best_group, sample_totals = max(
        group_samples,
        key=lambda item: (
            median(item[1]),
            -item[0]["concurrency"]
            if type(item[0].get("concurrency")) is int
            else float("-inf"),
        ),
    )
    best_concurrency = best_group.get("concurrency")
    candidate_mode = candidate.get("measurement_mode")
    actual_instances = candidate.get("actual_instances")
    instances_per_host = _finite_nonnegative_float(
        ranking_row.get("instances_per_host")
    )
    if instances_per_host is None or instances_per_host <= 0:
        errors.append(f"candidate {candidate_id} ranking instances_per_host is invalid")
        return True, errors
    if (
        candidate_mode == "full_host"
        and type(actual_instances) is int
        and actual_instances > 0
    ):
        raw_samples = [value / actual_instances for value in sample_totals]
        per_host_samples = sample_totals
        expected_instances_per_host = float(actual_instances)
    elif candidate_mode == "estimated":
        raw_samples = sample_totals
        per_host_samples = [value * instances_per_host for value in sample_totals]
        expected_instances_per_host = instances_per_host
    else:
        errors.append(f"candidate {candidate_id} ranking measurement mode is invalid")
        return True, errors
    expected_fields: dict[str, Any] = {
        "best_concurrency": best_concurrency,
        "measurement_mode": candidate_mode,
        "actual_instances": actual_instances,
        "sample_count": 3,
        "instances_per_host": expected_instances_per_host,
        "goodput_raw": median(raw_samples),
        "goodput_per_host_min": min(per_host_samples),
        "goodput_per_host_median": median(per_host_samples),
        "goodput_per_host_max": max(per_host_samples),
        "goodput_per_host": median(per_host_samples),
    }
    for field_name, expected_value in expected_fields.items():
        value = ranking_row.get(field_name)
        expected_type_ok = (
            type(value) is int
            if field_name
            in {"best_concurrency", "actual_instances", "sample_count"}
            else isinstance(value, str)
            if field_name == "measurement_mode"
            else _finite_nonnegative_float(value) is not None
        )
        if not expected_type_ok or value != expected_value:
            errors.append(
                f"candidate {candidate_id} ranking {field_name} differs from evidence"
            )
    representative = best_group.get("representative")
    if isinstance(representative, dict):
        for field_name in (
            "request_throughput",
            "output_throughput",
            "total_throughput",
            "mean_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "p99_tpot_ms",
            "success_rate",
            "avg_output_tokens",
        ):
            if field_name not in ranking_row or ranking_row[field_name] is None:
                continue
            expected_value = representative.get(field_name)
            if expected_value is None or ranking_row[field_name] != expected_value:
                errors.append(
                    f"candidate {candidate_id} ranking {field_name} differs from evidence"
                )
    if type(ranking_row.get("rank_group")) is not int or (
        expected_rank_group is not None and ranking_row.get("rank_group") != expected_rank_group
    ):
        errors.append(f"candidate {candidate_id} ranking rank_group differs from evidence")
    threshold_status = ranking_row.get("baseline_threshold_status")
    if threshold_status not in ("yes", "no", "unknown"):
        errors.append(f"candidate {candidate_id} ranking threshold status is invalid")
    elif threshold_status != "unknown":
        baseline_median = (
            _finite_nonnegative_float(baseline_row.get("goodput_per_host_median"))
            if baseline_row is not None
            else None
        )
        baseline_min = (
            _finite_nonnegative_float(baseline_row.get("goodput_per_host_min"))
            if baseline_row is not None
            else None
        )
        baseline_max = (
            _finite_nonnegative_float(baseline_row.get("goodput_per_host_max"))
            if baseline_row is not None
            else None
        )
        threshold_pct = _finite_nonnegative_float(
            ranking_row.get("baseline_threshold_pct")
        )
        candidate_min = _finite_nonnegative_float(
            ranking_row.get("goodput_per_host_min")
        )
        candidate_max = _finite_nonnegative_float(
            ranking_row.get("goodput_per_host_max")
        )
        baseline_valid = (
            baseline_row is not None
            and baseline_median is not None
            and baseline_median > 0
            and baseline_min is not None
            and baseline_max is not None
            and baseline_min <= baseline_median <= baseline_max
            and baseline_row.get("measurement_mode") == ranking_row.get("measurement_mode")
            and baseline_row.get("actual_instances") == ranking_row.get("actual_instances")
            and threshold_pct is not None
            and candidate_min is not None
            and candidate_max is not None
        )
        if (
            not baseline_valid
            or baseline_median is None
            or threshold_pct is None
            or candidate_min is None
            or candidate_max is None
        ):
            errors.append(
                f"candidate {candidate_id} ranking threshold lacks authoritative baseline"
            )
        else:
            threshold = baseline_median * (1.0 + threshold_pct / 100.0)
            expected_threshold_status = (
                "yes"
                if candidate_min >= threshold
                else "no"
                if candidate_max < threshold
                else "unknown"
            )
            if threshold_status != expected_threshold_status:
                errors.append(
                    f"candidate {candidate_id} ranking threshold differs from evidence"
                )
    if type(ranking_row.get("beats_baseline_threshold")) is not bool or ranking_row.get(
        "beats_baseline_threshold"
    ) is not (threshold_status == "yes"):
        errors.append(f"candidate {candidate_id} ranking threshold fields differ")
    return True, errors


def _expected_rank_groups(ranking: list[dict[str, Any]]) -> dict[str, int]:
    """Derive interval-overlap components in the same deterministic order as ranker."""

    ordered: list[dict[str, Any]] = []
    for row in ranking:
        if not isinstance(row, dict):
            continue
        candidate_id = row.get("candidate_id")
        minimum = _finite_nonnegative_float(row.get("goodput_per_host_min"))
        median_value = _finite_nonnegative_float(row.get("goodput_per_host_median"))
        maximum = _finite_nonnegative_float(row.get("goodput_per_host_max"))
        if (
            not isinstance(candidate_id, str)
            or minimum is None
            or median_value is None
            or maximum is None
            or not minimum <= median_value <= maximum
        ):
            continue
        ordered.append(row)
    ordered.sort(
        key=lambda row: (
            -float(row["goodput_per_host_median"]),
            str(row.get("candidate_id")),
        )
    )
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(ordered)):
        left_min = _finite_nonnegative_float(ordered[left].get("goodput_per_host_min"))
        left_max = _finite_nonnegative_float(ordered[left].get("goodput_per_host_max"))
        if left_min is None or left_max is None:
            continue
        for right in range(left + 1, len(ordered)):
            right_min = _finite_nonnegative_float(
                ordered[right].get("goodput_per_host_min")
            )
            right_max = _finite_nonnegative_float(
                ordered[right].get("goodput_per_host_max")
            )
            if (
                right_min is not None
                and right_max is not None
                and left_min <= right_max
                and right_min <= left_max
            ):
                left_root, right_root = find(left), find(right)
                if left_root != right_root:
                    parents[right_root] = left_root
    component_by_root: dict[int, int] = {}
    groups = {}
    for index, row in enumerate(ordered):
        root = find(index)
        component_by_root.setdefault(root, len(component_by_root) + 1)
        candidate_id = row.get("candidate_id")
        if isinstance(candidate_id, str):
            groups[candidate_id] = component_by_root[root]
    return groups


def _expected_rank_order(ranking: list[dict[str, Any]]) -> list[str]:
    valid_rows = [
        row
        for row in ranking
        if isinstance(row, dict)
        if isinstance(row.get("candidate_id"), str)
        and _finite_nonnegative_float(row.get("goodput_per_host_median")) is not None
    ]
    valid_rows.sort(
        key=lambda row: (
            -float(row["goodput_per_host_median"]),
            str(row["candidate_id"]),
        )
    )
    return [row["candidate_id"] for row in valid_rows]


def _canonicalize_report_set(
    *,
    ranking: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    task_status: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Return a deterministic, fail-closed report set with exact foreign keys."""
    # Treat every argument as untrusted boundary data.  The normal executor
    # supplies lists/dicts, but a partially written or hand-edited report must
    # be downgraded rather than crashing the writer/loader with ``.get`` or
    # iteration errors.
    if isinstance(task_status, dict):
        try:
            status = deepcopy(task_status)
        except Exception:
            status = {}
    else:
        status = {}
    input_shape_errors: list[str] = []
    if not isinstance(task_status, dict):
        input_shape_errors.append("task_status must be an object")
    if isinstance(ranking, list):
        ranking_items: list[Any] = ranking
    else:
        ranking_items = []
        input_shape_errors.append("ranking must be a list")
    if isinstance(candidate_rows, list):
        candidate_items: list[Any] = candidate_rows
    else:
        candidate_items = []
        input_shape_errors.append("candidate_rows must be a list")
    if isinstance(probe_rows, list):
        probe_items: list[Any] = probe_rows
    else:
        probe_items = []
        input_shape_errors.append("probe_rows must be a list")
    final_requested = (
        status.get("ranking_status") == "FINAL"
        or status.get("task_status") == "COMPLETED"
    )
    configured_errors = status.get("invariant_errors", [])
    if isinstance(configured_errors, list):
        errors = [str(error) for error in configured_errors]
    else:
        errors = ["invariant_errors must be a list"]
    errors.extend(input_shape_errors)
    if type(status.get("interrupted")) is not bool:
        errors.append("interrupted must be a boolean")
    task_state = status.get("task_status")
    ranking_state = status.get("ranking_status")
    if not (
        isinstance(task_state, str)
        and isinstance(ranking_state, str)
        and (task_state, ranking_state)
        in {
        ("COMPLETED", "FINAL"),
        ("INCOMPLETE", "PROVISIONAL"),
        ("INTERRUPTED", "PROVISIONAL"),
        }
    ):
        errors.append("task_status/ranking_status pair is invalid")
    configured_expected = status.get("expected_candidate_ids")
    if configured_expected is None:
        raw_expected = [
            row.get("candidate_id")
            for row in candidate_items
            if isinstance(row, dict)
            and isinstance(row.get("candidate_id"), str)
            and row.get("candidate_id")
        ]
        if final_requested:
            errors.append("FINAL requires explicit expected_candidate_ids")
    elif isinstance(configured_expected, list):
        raw_expected = configured_expected
    else:
        raw_expected = []
        errors.append("expected_candidate_ids must be a list")
    valid_expected = [
        candidate_id
        for candidate_id in raw_expected
        if isinstance(candidate_id, str) and candidate_id
    ]
    if len(valid_expected) != len(raw_expected):
        errors.append("expected_candidate_ids contains an invalid identifier")
    expected = _stable_unique(valid_expected)
    if len(expected) != len(valid_expected):
        errors.append("expected_candidate_ids contains duplicates")
    if not expected:
        errors.append("expected_candidate_ids must be non-empty")
    expected_set = set(expected)

    candidate_by_id: dict[str, dict[str, Any]] = {}
    observed: list[str] = []
    for row in candidate_items:
        if not isinstance(row, dict):
            errors.append("candidate row must be an object")
            continue
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            errors.append("candidate row has no valid candidate_id")
            continue
        if candidate_id in candidate_by_id:
            errors.append(f"duplicate candidate row: {candidate_id}")
            continue
        candidate_by_id[candidate_id] = deepcopy(row)
        observed.append(candidate_id)
    extra_candidates = [
        candidate_id for candidate_id in observed if candidate_id not in expected_set
    ]
    if extra_candidates:
        errors.append(f"unexpected candidate rows: {','.join(extra_candidates)}")
    canonical_candidates: list[dict[str, Any]] = []
    for candidate_id in expected:
        row = candidate_by_id.get(candidate_id)
        if row is None:
            errors.append(f"missing candidate row: {candidate_id}")
            row = {
                "candidate_id": candidate_id,
                "status": "incomplete",
                "completion_state": "missing",
                "ranking_eligible": False,
                "ranking_eligibility_reason": "candidate result missing",
                "probe_ids": [],
            }
        canonical_candidates.append(row)

    canonical_probes: list[dict[str, Any]] = []
    probe_by_id: dict[str, dict[str, Any]] = {}
    for row in probe_items:
        if not isinstance(row, dict):
            errors.append("probe row must be an object")
            continue
        probe_id = row.get("probe_id")
        candidate_id = row.get("candidate_id")
        if not isinstance(probe_id, str) or not probe_id:
            errors.append("probe row has no valid probe_id")
            continue
        if probe_id in probe_by_id:
            errors.append(f"duplicate probe row: {probe_id}")
            continue
        if not isinstance(candidate_id, str) or candidate_id not in expected_set:
            errors.append(f"probe {probe_id} references foreign candidate {candidate_id}")
            continue
        copied = deepcopy(row)
        probe_by_id[probe_id] = copied
        canonical_probes.append(copied)
    for row in canonical_candidates:
        candidate_id = row["candidate_id"]
        # Validate scalar summary fields even when probe_ids/sample_groups are
        # absent.  Otherwise an arbitrary object can survive in a provisional
        # placeholder (or exploit Python's permissive equality rules) and
        # later become an apparently valid FINAL row.
        text_or_none_fields = (
            "measurement_mode",
            "ranking_eligibility_reason",
            "requested_command",
            "round1_batch",
            "round2_batch",
            "failed_at",
            "failed_batch",
            "known_issue",
            "failure_reason",
        )
        for field_name in text_or_none_fields:
            value = row.get(field_name)
            if field_name in row and value is not None and not isinstance(value, str):
                errors.append(f"candidate {candidate_id} {field_name} must be text or null")
        candidate_status = row.get("status")
        if "status" in row and (
            not isinstance(candidate_status, str)
            or candidate_status not in {"completed", "incomplete"}
        ):
            errors.append(f"candidate {candidate_id} status is invalid")
        completion_state = row.get("completion_state")
        if "completion_state" in row and (
            not isinstance(completion_state, str)
            or completion_state not in {"completed", "in_progress", "missing"}
        ):
            errors.append(f"candidate {candidate_id} completion_state is invalid")
        final_failure = row.get("final_failure")
        if final_failure is not None and not isinstance(final_failure, dict):
            errors.append(f"candidate {candidate_id} final_failure must be an object or null")
        failure_status = row.get("failure_status")
        if failure_status is not None:
            valid_failure_statuses = {status.value for status in ProbeStatus}
            if not isinstance(failure_status, str) or failure_status not in valid_failure_statuses:
                errors.append(f"candidate {candidate_id} failure_status is invalid")
        for timestamp_field in ("failed_at",):
            timestamp_value = row.get(timestamp_field)
            if timestamp_value is not None:
                parsed_timestamp = _parse_timestamp(timestamp_value)
                timestamp_offset = (
                    parsed_timestamp.utcoffset() if parsed_timestamp is not None else None
                )
                if (
                    parsed_timestamp is None
                    or timestamp_offset is None
                    or timestamp_offset.total_seconds() != 0
                ):
                    errors.append(
                        f"candidate {candidate_id} {timestamp_field} must be UTC text or null"
                    )
        linked = row.get("probe_ids")
        if linked is None:
            orphan_ids = [
                probe_id
                for probe_id, probe in probe_by_id.items()
                if probe.get("candidate_id") == candidate_id
            ]
            if orphan_ids:
                errors.append(
                    f"candidate {candidate_id} does not link probes "
                    f"{','.join(orphan_ids)}"
                )
            continue
        if (
            not isinstance(linked, list)
            or any(not isinstance(probe_id, str) or not probe_id for probe_id in linked)
            or len(linked) != len(set(linked))
        ):
            errors.append(f"candidate {candidate_id} has invalid or duplicate probe_ids")
            continue
        for probe_id in linked:
            probe = probe_by_id.get(probe_id)
            if probe is None or probe.get("candidate_id") != candidate_id:
                errors.append(f"candidate {candidate_id} references missing probe {probe_id}")
        orphan_ids = [
            probe_id
            for probe_id, probe in probe_by_id.items()
            if probe.get("candidate_id") == candidate_id and probe_id not in linked
        ]
        if orphan_ids:
            errors.append(
                f"candidate {candidate_id} does not link probes {','.join(orphan_ids)}"
            )

        # Candidate summaries are persisted evidence too: optional legacy
        # fields must not exploit Python's ``True == 1`` coercion or carry a
        # non-finite value into a supposedly final generation.
        for field_name in _CANDIDATE_INTEGER_FIELDS:
            value = row.get(field_name)
            if field_name in row and value is not None and type(value) is not int:
                errors.append(f"candidate {candidate_id} {field_name} must be an integer")
        for field_name in _CANDIDATE_FLOAT_FIELDS:
            value = row.get(field_name)
            if field_name in row and value is not None and _finite_float(value) is None:
                errors.append(f"candidate {candidate_id} {field_name} must be finite")
        for field_name in ("ranking_eligible", "measurement_valid", "beats_baseline_threshold"):
            value = row.get(field_name)
            if field_name in row and value is not None and type(value) is not bool:
                errors.append(f"candidate {candidate_id} {field_name} must be a boolean")
        for round_name in ("round1", "round2"):
            diagnostics = row.get(round_name)
            if not isinstance(diagnostics, dict):
                continue
            for field_name in ("num_evals", "c_star", "last_pass", "first_fail"):
                value = diagnostics.get(field_name)
                if field_name in diagnostics and value is not None and type(value) is not int:
                    errors.append(
                        f"candidate {candidate_id} {round_name} {field_name} must be an integer"
                    )
            for field_name in ("complete",):
                value = diagnostics.get(field_name)
                if field_name in diagnostics and type(value) is not bool:
                    errors.append(
                        f"candidate {candidate_id} {round_name} {field_name} must be a boolean"
                    )
            for field_name in ("stop_reason", "certainty"):
                value = diagnostics.get(field_name)
                if field_name in diagnostics and value is not None and not isinstance(value, str):
                    errors.append(
                        f"candidate {candidate_id} {round_name} {field_name} must be a string"
                    )

    canonical_ranking: list[dict[str, Any]] = []
    ranked_ids: set[str] = set()
    for row in ranking_items:
        if not isinstance(row, dict):
            errors.append("ranking row must be an object")
            continue
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in expected_set:
            errors.append(f"ranking references foreign candidate {candidate_id}")
            continue
        if candidate_id in ranked_ids:
            errors.append(f"duplicate ranking row: {candidate_id}")
            continue
        ranked_ids.add(candidate_id)
        if row.get("ranking_eligible") is not True:
            errors.append(f"ranking includes ineligible candidate {candidate_id}")
            continue
        for field_name in (
            "rank",
            "rank_group",
            "best_concurrency",
            "actual_instances",
            "sample_count",
            "attempts",
            "recovery_count",
        ):
            if field_name in row and row.get(field_name) is not None and type(
                row.get(field_name)
            ) is not int:
                errors.append(f"candidate {candidate_id} ranking {field_name} must be an integer")
        for field_name in (
            "goodput_raw",
            "goodput_per_host",
            "goodput_per_host_min",
            "goodput_per_host_median",
            "goodput_per_host_max",
            "baseline_goodput_per_host",
            "threshold_goodput_per_host",
            "baseline_threshold_pct",
            "goodput_delta",
            "goodput_delta_pct",
        ):
            if field_name in row and row.get(field_name) is not None and _finite_float(
                row.get(field_name)
            ) is None:
                errors.append(f"candidate {candidate_id} ranking {field_name} must be finite")
        canonical_ranking.append(deepcopy(row))
    expected_rank_groups = _expected_rank_groups(canonical_ranking)
    baseline_ranking_row = next(
        (
            row
            for row in canonical_ranking
            if row.get("candidate_id") == "baseline"
        ),
        None,
    )
    expected_rank_order = _expected_rank_order(canonical_ranking)
    actual_rank_order = [
        row.get("candidate_id")
        for row in canonical_ranking
        if isinstance(row.get("candidate_id"), str)
    ]
    if expected_rank_order and actual_rank_order != expected_rank_order:
        errors.append("ranking rows are not ordered by authoritative Round-2 scores")
    expected_rank_numbers = {
        candidate_id: index + 1
        for index, candidate_id in enumerate(expected_rank_order)
    }

    if final_requested:
        for row in canonical_candidates:
            candidate_id = row["candidate_id"]
            if row.get("status") != "completed":
                errors.append(f"candidate {candidate_id} is not completed")
            if row.get("completion_state") != "completed":
                errors.append(f"candidate {candidate_id} completion_state is unresolved")
            if row.get("final_failure") is not None:
                errors.append(f"candidate {candidate_id} has an unresolved terminal failure")
            candidate_mode = row.get("measurement_mode")
            if not isinstance(candidate_mode, str) or candidate_mode not in {
                "full_host",
                "estimated",
            }:
                errors.append(f"candidate {candidate_id} measurement mode is invalid")
            elif row.get("measurement_valid") is not True:
                errors.append(f"candidate {candidate_id} measurement is invalid")
            if not isinstance(row.get("probe_ids"), list) or not row.get("probe_ids"):
                errors.append(f"candidate {candidate_id} has no linked probe evidence")
            round1 = row.get("round1")
            round2 = row.get("round2")
            round1_stop = round1.get("stop_reason") if isinstance(round1, dict) else None
            if not (
                isinstance(round1_stop, str)
                and round1_stop
                in {"found_boundary", "c1_failed", "hit_cap", "max_probes"}
            ):
                errors.append(f"candidate {candidate_id} has incomplete round 1")
            if (
                not isinstance(round2, dict)
                or round2.get("complete") is not True
                or round2.get("certainty") != "exact"
            ):
                errors.append(f"candidate {candidate_id} has incomplete round 2")
            complete_groups = row.get("sample_groups")
            incomplete_groups = row.get("incomplete_groups")
            if not isinstance(complete_groups, list) or not complete_groups:
                errors.append(f"candidate {candidate_id} has no complete sample groups")
            elif isinstance(incomplete_groups, list) and incomplete_groups:
                errors.append(f"candidate {candidate_id} has partial sample groups")
            elif isinstance(complete_groups, list):
                for round_number, diagnostics in ((1, round1), (2, round2)):
                    round_groups = [
                        group
                        for group in complete_groups
                        if isinstance(group, dict) and group.get("round") == round_number
                    ]
                    if not round_groups:
                        errors.append(
                            f"candidate {candidate_id} round {round_number} has no sample group"
                        )
                        continue
                    sample_count = sum(
                        len(group.get("aggregate_probe_ids", []))
                        for group in round_groups
                        if isinstance(group.get("aggregate_probe_ids"), list)
                    )
                    num_evals = (
                        diagnostics.get("num_evals")
                        if isinstance(diagnostics, dict)
                        else None
                    )
                    if (
                        type(num_evals) is not int
                        or num_evals < 0
                        or num_evals != sample_count
                    ):
                        errors.append(
                            f"candidate {candidate_id} round {round_number} sample count differs"
                        )
                    if isinstance(diagnostics, dict):
                        for field_name in ("c_star", "last_pass", "first_fail"):
                            value = diagnostics.get(field_name)
                            if value is not None and type(value) is not int:
                                errors.append(
                                    f"candidate {candidate_id} round {round_number} "
                                    f"{field_name} must be an integer"
                                )
                        certainty = diagnostics.get("certainty")
                        if isinstance(certainty, str) and certainty not in {
                            "exact",
                            "lower_bound",
                            "unknown",
                        }:
                            errors.append(
                                f"candidate {candidate_id} round {round_number} "
                                "certainty is invalid"
                            )
                        if "complete" in diagnostics and type(
                            diagnostics.get("complete")
                        ) is not bool:
                            errors.append(
                                f"candidate {candidate_id} round {round_number} "
                                "complete must be a boolean"
                            )
                        if "certainty" in diagnostics and not isinstance(
                            diagnostics.get("certainty"), str
                        ):
                            errors.append(
                                f"candidate {candidate_id} round {round_number} "
                                "certainty must be a string"
                            )
                        if round_number == 1:
                            group_concurrencies = {
                                group.get("concurrency")
                                for group in round_groups
                                if isinstance(group, dict)
                                and type(group.get("concurrency")) is int
                            }
                            diagnostic_values = {
                                field_name: diagnostics.get(field_name)
                                for field_name in (
                                    "c_star",
                                    "last_pass",
                                    "first_fail",
                                )
                            }
                            for field_name, value in diagnostic_values.items():
                                if (
                                    type(value) is int
                                    and value not in group_concurrencies
                                ):
                                    errors.append(
                                        f"candidate {candidate_id} round 1 "
                                        f"{field_name} differs from sample groups"
                                    )
                            c_star = diagnostic_values["c_star"]
                            last_pass = diagnostic_values["last_pass"]
                            first_fail = diagnostic_values["first_fail"]
                            if (
                                type(c_star) is int
                                and type(last_pass) is int
                                and c_star != last_pass
                            ):
                                errors.append(
                                    f"candidate {candidate_id} round 1 "
                                    "c_star differs from last_pass"
                                )
                            if (
                                type(last_pass) is int
                                and type(first_fail) is int
                                and first_fail <= last_pass
                            ):
                                errors.append(
                                    f"candidate {candidate_id} round 1 "
                                    "first_fail is not above last_pass"
                                )
                    tested = diagnostics.get("newly_probed") if isinstance(
                        diagnostics, dict
                    ) else None
                    if not isinstance(tested, list) or any(
                        type(concurrency) is not int for concurrency in tested
                    ):
                        errors.append(
                            f"candidate {candidate_id} round {round_number} tested C differs"
                        )
                    else:
                        group_concurrencies = [
                            group.get("concurrency") for group in round_groups
                        ]
                        if any(
                            type(concurrency) is not int or concurrency <= 0
                            for concurrency in group_concurrencies
                        ) or len(tested) != len(set(tested)) or set(tested) != set(
                            group_concurrencies
                        ):
                            errors.append(
                                f"candidate {candidate_id} round {round_number} tested C differs"
                            )
                actual_instances = row.get("actual_instances")
                if type(actual_instances) is not int or actual_instances <= 0:
                    errors.append(f"candidate {candidate_id} actual_instances is invalid")
                else:
                    for probe in canonical_probes:
                        if (
                            probe.get("candidate_id") == candidate_id
                            and probe.get("round") == 2
                            and probe.get("record_type") == "aggregate_sample"
                            and (
                                probe.get("measurement_mode")
                                != row.get("measurement_mode")
                                or probe.get("instances") != actual_instances
                            )
                        ):
                            errors.append(
                                f"candidate {candidate_id} measurement mode/instances differ"
                            )
                            break
            if isinstance(round2, dict):
                round2_groups = [
                    group
                    for group in complete_groups
                    if isinstance(group, dict) and group.get("round") == 2
                ] if isinstance(complete_groups, list) else []
                group_by_c = {
                    group.get("concurrency"): group
                    for group in round2_groups
                    if type(group.get("concurrency")) is int
                }
                if round2.get("stop_reason") == "found_boundary":
                    last_pass = round2.get("last_pass")
                    first_fail = round2.get("first_fail")
                    if (
                        type(last_pass) is not int
                        or type(first_fail) is not int
                        or first_fail != last_pass + 1
                        or round2.get("c_star") != last_pass
                        or not isinstance(group_by_c.get(last_pass), dict)
                        or group_by_c[last_pass].get("qualifies") is not True
                        or not isinstance(group_by_c.get(first_fail), dict)
                        or group_by_c[first_fail].get("qualifies") is not False
                    ):
                        errors.append(f"candidate {candidate_id} exact boundary is unproven")
                elif round2.get("stop_reason") == "c1_failed":
                    if (
                        round2.get("c_star") is not None
                        or not isinstance(group_by_c.get(1), dict)
                        or group_by_c[1].get("qualifies") is not False
                    ):
                        errors.append(f"candidate {candidate_id} C1 failure is unproven")
                else:
                    errors.append(f"candidate {candidate_id} round 2 stop is not exact")

            linked_values = row.get("probe_ids")
            linked_probe_ids = (
                set(linked_values)
                if isinstance(linked_values, list)
                and all(isinstance(value, str) for value in linked_values)
                else set()
            )
            failures = row.get("failures", [])
            if not isinstance(failures, list):
                errors.append(f"candidate {candidate_id} failures must be a list")
                failures = []
            failure_probe_ids: list[str] = []
            for failure in failures:
                if not isinstance(failure, dict):
                    errors.append(f"candidate {candidate_id} has malformed failure evidence")
                    continue
                failure_probe_id = failure.get("probe_id")
                if not isinstance(failure_probe_id, str) or not failure_probe_id:
                    errors.append(
                        f"candidate {candidate_id} failure has an invalid probe_id"
                    )
                    continue
                if failure_probe_id not in linked_probe_ids:
                    errors.append(f"candidate {candidate_id} failure has no linked probe")
                elif failure.get("resolved") is not True:
                    errors.append(f"candidate {candidate_id} has unresolved terminal failure")
                else:
                    failure_probe_ids.append(failure_probe_id)
            if len(failure_probe_ids) != len(set(failure_probe_ids)):
                errors.append(f"candidate {candidate_id} has duplicate failure summary rows")
            failed_attempt_ids: set[str] = set()
            for probe in canonical_probes:
                if (
                    probe.get("candidate_id") != candidate_id
                    or probe.get("record_type")
                    not in ("aggregate_sample", "infrastructure_attempt")
                ):
                    continue
                try:
                    attempt_status = ProbeStatus(probe.get("status"))
                except (TypeError, ValueError):
                    continue
                if attempt_status not in SEARCH_VERDICT_STATUSES and isinstance(
                    probe.get("probe_id"), str
                ):
                    failed_attempt_ids.add(probe["probe_id"])
            if set(failure_probe_ids) != failed_attempt_ids:
                errors.append(
                    f"candidate {candidate_id} failure summary does not conserve attempts"
                )
            recovery_count = row.get("recovery_count")
            if type(recovery_count) is not int or recovery_count != len(
                failed_attempt_ids
            ):
                errors.append(f"candidate {candidate_id} recovery_count differs")

            ranked = next(
                (
                    ranking_row
                    for ranking_row in canonical_ranking
                    if ranking_row.get("candidate_id") == candidate_id
                ),
                None,
            )
            eligible, ranking_errors = _authoritative_ranking_errors(
                row,
                ranked,
                probe_by_id,
                expected_rank_group=expected_rank_groups.get(candidate_id),
                baseline_row=baseline_ranking_row,
            )
            errors.extend(ranking_errors)
            if ranked is not None:
                for field_name in (
                    "best_concurrency",
                    "measurement_mode",
                    "actual_instances",
                    "sample_count",
                    "goodput_raw",
                    "goodput_per_host_min",
                    "goodput_per_host_median",
                    "goodput_per_host_max",
                    "goodput_per_host",
                    "baseline_threshold_status",
                    "beats_baseline_threshold",
                ):
                    candidate_value = row.get(field_name)
                    if field_name in row and (
                        (
                            field_name
                            in {"best_concurrency", "actual_instances", "sample_count"}
                            and type(candidate_value) is not int
                        )
                        or (
                            field_name == "measurement_mode"
                            and not isinstance(candidate_value, str)
                        )
                        or (
                            field_name == "baseline_threshold_status"
                            and not isinstance(candidate_value, str)
                        )
                        or (
                            field_name == "beats_baseline_threshold"
                            and type(candidate_value) is not bool
                        )
                        or (
                            field_name
                            not in {
                                "best_concurrency",
                                "actual_instances",
                                "sample_count",
                                "measurement_mode",
                                "baseline_threshold_status",
                                "beats_baseline_threshold",
                            }
                            and _finite_nonnegative_float(candidate_value) is None
                        )
                        or candidate_value != ranked.get(field_name)
                    ):
                        errors.append(
                            f"candidate {candidate_id} {field_name} differs from ranking"
                        )
                if (
                    "rank_group" in row
                    and (
                        type(row.get("rank_group")) is not int
                        or row.get("rank_group") != expected_rank_groups.get(candidate_id)
                    )
                ):
                    errors.append(f"candidate {candidate_id} rank_group differs from evidence")
                if ranked.get("rank") != expected_rank_numbers.get(candidate_id):
                    errors.append(f"candidate {candidate_id} ranking rank differs from evidence")
            candidate_rank = row.get("rank")
            if eligible:
                if (
                    ranked is None
                    or type(candidate_rank) is not int
                    or candidate_rank <= 0
                    or candidate_rank != ranked.get("rank")
                    or row.get("rank_group") != ranked.get("rank_group")
                ):
                    errors.append(f"candidate {candidate_id} ranking rank fields differ")

        ranking_ranks = [row.get("rank") for row in canonical_ranking]
        if any(type(rank) is not int for rank in ranking_ranks) or ranking_ranks != list(
            range(1, len(canonical_ranking) + 1)
        ):
            errors.append("ranking ranks must be unique and consecutive")

    cleanup_failures = status.get("cleanup_failures", [])
    if not isinstance(cleanup_failures, list) or cleanup_failures:
        errors.append("cleanup is incomplete or unproven")

    errors.extend(_probe_hierarchy_errors(canonical_candidates, canonical_probes))
    errors.extend(_provenance_errors(provenance))

    status["expected_candidate_ids"] = expected
    status["expected_candidate_count"] = len(expected)
    status["observed_candidate_ids"] = [
        candidate_id for candidate_id in expected if candidate_id in candidate_by_id
    ]
    status["observed_candidate_count"] = len(status["observed_candidate_ids"])
    status["completed_candidate_count"] = sum(
        row.get("status") == "completed" for row in canonical_candidates
    )
    status["incomplete_candidate_count"] = (
        len(expected) - status["completed_candidate_count"]
    )
    status["total_candidates"] = len(expected)
    status["completed_candidates"] = status["completed_candidate_count"]
    status["failed_candidates"] = status["incomplete_candidate_count"]
    if status.get("interrupted") is True:
        errors.append("run was interrupted")
    status["invariant_errors"] = _stable_unique(errors)
    if errors:
        if status.get("interrupted") is True or status.get("task_status") == "INTERRUPTED":
            status["task_status"] = "INTERRUPTED"
        else:
            status["task_status"] = "INCOMPLETE"
        status["ranking_status"] = "PROVISIONAL"
    return canonical_ranking, canonical_candidates, canonical_probes, status


def load_report_generation(results_dir: Path) -> dict[str, Any]:
    """Load and verify the immutable generation selected by the manifest."""
    manifest = _strict_json_loads(
        (results_dir / "report_manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(manifest, dict):
        raise ValueError("report_manifest.json must contain an object")
    if (
        type(manifest.get("report_schema_version")) is not int
        or manifest["report_schema_version"] != REPORT_SCHEMA_VERSION
    ):
        raise ValueError("report manifest schema version is invalid")
    manifest_run_id = manifest.get("run_id")
    if (
        not isinstance(manifest_run_id, str)
        or len(manifest_run_id) > 200
        or re.fullmatch(r"[A-Za-z0-9_.:-]+", manifest_run_id) is None
        or ".." in manifest_run_id
    ):
        raise ValueError("report manifest run_id is invalid")
    generation_id = manifest.get("generation_id")
    snapshot_id = manifest.get("snapshot_id")
    if (
        not isinstance(generation_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", generation_id) is None
        or snapshot_id != generation_id
    ):
        raise ValueError("report manifest has no generation_id")
    generation_root = (results_dir / ".report_generations").resolve()
    generation_dir = (generation_root / generation_id).resolve()
    if generation_dir.parent != generation_root:
        raise ValueError("report manifest generation_id escapes generation root")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(_REPORT_FILES):
        raise ValueError("report manifest file set is incomplete")

    texts: dict[str, str] = {}
    for name in _REPORT_FILES:
        text = (generation_dir / name).read_text(encoding="utf-8")
        metadata = files[name]
        if (
            not isinstance(metadata, dict)
            or not _valid_sha256(metadata.get("sha256"))
            or type(metadata.get("size_bytes")) is not int
            or metadata["size_bytes"] < 0
            or type(metadata.get("row_count")) is not int
            or metadata["row_count"] < 0
            or metadata.get("sha256") != _sha256_text(text)
            or metadata.get("size_bytes") != len(text.encode("utf-8"))
        ):
            raise ValueError(f"report generation hash mismatch: {name}")
        texts[name] = text

    ranking = _strict_json_loads(texts["ranking.json"])
    candidates = [
        _strict_json_loads(line)
        for line in texts["candidate_results.jsonl"].splitlines()
    ]
    probes = [
        _strict_json_loads(line) for line in texts["probe_results.jsonl"].splitlines()
    ]
    task_status = _strict_json_loads(texts["task_status.json"])
    provenance = _strict_json_loads(texts["provenance.json"])
    if not isinstance(ranking, list):
        raise ValueError("ranking.json must contain a list")
    if not isinstance(task_status, dict) or not isinstance(provenance, dict):
        raise ValueError("task status and provenance must contain objects")
    row_counts = {
        "ranking.json": len(ranking),
        "candidate_results.jsonl": len(candidates),
        "probe_results.jsonl": len(probes),
        "task_status.json": 1,
        "provenance.json": 1,
    }
    for name, row_count in row_counts.items():
        if files[name].get("row_count") != row_count:
            raise ValueError(f"report generation row count mismatch: {name}")
    run_id = manifest_run_id
    schema_version = manifest.get("report_schema_version")
    for payload in [*ranking, *candidates, *probes, task_status, provenance]:
        if not isinstance(payload, dict):
            raise ValueError("report payload must contain objects")
        if payload.get("run_id") != run_id:
            raise ValueError("report payload run_id does not match manifest")
        if payload.get("report_schema_version") != schema_version:
            raise ValueError("report payload schema does not match manifest")
    # Hashes prove that the five files belong to one immutable generation, but
    # they do not prove that a producer did not commit a self-consistent yet
    # semantically false FINAL payload.  Re-run the canonical evidence checks
    # when the stored status claims finality and reject the generation rather
    # than handing callers an unverifiable result.  Provisional generations
    # remain loadable so interrupted/partial evidence can be audited.
    if task_status.get("task_status") == "COMPLETED" or task_status.get(
        "ranking_status"
    ) == "FINAL":
        _, _, _, validated_status = _canonicalize_report_set(
            ranking=ranking,
            candidate_rows=candidates,
            probe_rows=probes,
            task_status=task_status,
            provenance=provenance,
        )
        if (
            validated_status.get("task_status") != "COMPLETED"
            or validated_status.get("ranking_status") != "FINAL"
            or validated_status.get("invariant_errors")
        ):
            raise ValueError("loaded FINAL report fails canonical invariants")
    return {
        "manifest": manifest,
        "ranking": ranking,
        "candidate_rows": candidates,
        "probe_rows": probes,
        "task_status": task_status,
        "provenance": provenance,
    }


def write_reports(
    results_dir: Path,
    *,
    ranking: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    task_status: dict[str, Any],
    probe_rows: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    report_run_id = uuid.uuid4().hex if run_id is None else run_id
    if (
        not isinstance(report_run_id, str)
        or len(report_run_id) > 200
        or re.fullmatch(r"[A-Za-z0-9_.:-]+", report_run_id) is None
        or ".." in report_run_id
    ):
        raise ValueError("run_id must be a safe non-empty identifier")

    sanitization_errors: list[str] = []

    def sanitize_whole(value: Any, *, path: str) -> Any:
        safe, reasons = _sanitize_report_tree(value, path=path)
        sanitization_errors.extend(reasons)
        return safe

    ranking_input = ranking if isinstance(ranking, list) else []
    if not isinstance(ranking, list):
        sanitization_errors.append("ranking must be a list")
    sanitized_ranking: list[dict[str, Any]] = []
    for index, row in enumerate(ranking_input):
        safe = sanitize_whole(row, path=f"ranking[{index}]")
        if isinstance(safe, dict):
            sanitized_ranking.append(safe)
        else:
            sanitization_errors.append(f"ranking[{index}] must be an object")

    if probe_rows is None:
        probe_input: list[Any] = []
    elif isinstance(probe_rows, list):
        probe_input = probe_rows
    else:
        probe_input = []
        sanitization_errors.append("probe_rows must be a list")
    sanitized_probes: list[dict[str, Any]] = []
    for index, probe in enumerate(probe_input):
        copied_probe = deepcopy(probe)
        if isinstance(copied_probe, dict) and "raw" in copied_probe:
            copied_probe["raw"], sanitization = _sanitize_optional_json(
                copied_probe["raw"], path=f"probe[{index}].raw"
            )
            if sanitization:
                copied_probe["raw_sanitization"] = sanitization
        safe = sanitize_whole(copied_probe, path=f"probe[{index}]")
        if isinstance(safe, dict):
            sanitized_probes.append(safe)
        else:
            sanitization_errors.append(f"probe[{index}] must be an object")

    candidate_input = candidate_rows if isinstance(candidate_rows, list) else []
    if not isinstance(candidate_rows, list):
        sanitization_errors.append("candidate_rows must be a list")
    sanitized_candidates: list[dict[str, Any]] = []
    # v2 callers hand us a typed task-status envelope.  In that mode every
    # candidate field is untrusted and must be made JSON-safe before the
    # canonical validator runs.  Keep the legacy (status-only) compatibility
    # path strict for unknown extension fields: older callers historically
    # relied on a non-finite extension raising before any files were replaced.
    structured_report = isinstance(task_status, dict) and (
        "task_status" in task_status or "ranking_status" in task_status
    )
    for index, candidate in enumerate(candidate_input):
        if not isinstance(candidate, dict):
            sanitization_errors.append(f"candidate[{index}] must be an object")
            continue
        copied_candidate = deepcopy(candidate)
        if structured_report:
            safe_candidate = sanitize_whole(
                copied_candidate, path=f"candidate[{index}]"
            )
            if isinstance(safe_candidate, dict):
                copied_candidate = safe_candidate
            else:  # pragma: no cover - whole-tree sanitizer preserves objects
                sanitization_errors.append(f"candidate[{index}] must be an object")
                continue
        else:
            for field_name in _CANDIDATE_NUMERIC_FIELDS:
                if field_name in copied_candidate:
                    copied_candidate[field_name] = sanitize_whole(
                        copied_candidate[field_name],
                        path=f"candidate[{index}].{field_name}",
                    )
            for field_name in _CANDIDATE_KNOWN_SUBTREES:
                if field_name in copied_candidate:
                    copied_candidate[field_name] = sanitize_whole(
                        copied_candidate[field_name],
                        path=f"candidate[{index}].{field_name}",
                    )
        sanitized_candidates.append(copied_candidate)

    sanitized_status = sanitize_whole(deepcopy(task_status), path="task_status")
    sanitized_provenance = sanitize_whole(
        deepcopy(provenance or {}), path="provenance"
    )
    if not isinstance(sanitized_status, dict):
        sanitized_status = {}
        sanitization_errors.append("task_status must be an object")
    configured_sanitization = sanitized_status.get("invariant_errors")
    if not isinstance(configured_sanitization, list):
        configured_sanitization = []
        if "invariant_errors" in sanitized_status:
            configured_sanitization.append("invariant_errors must be a list")
    sanitized_status["invariant_errors"] = [
        *configured_sanitization,
        *sanitization_errors,
    ]

    (
        canonical_ranking,
        canonical_candidates,
        canonical_probes,
        canonical_status,
    ) = _canonicalize_report_set(
        ranking=sanitized_ranking,
        candidate_rows=sanitized_candidates,
        probe_rows=sanitized_probes,
        task_status=sanitized_status,
        provenance=sanitized_provenance
        if isinstance(sanitized_provenance, dict)
        else {},
    )

    def stamped(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
        return {
            **deepcopy(payload),
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "run_id": report_run_id,
        }

    ranking_payload = [stamped(row, label="ranking") for row in canonical_ranking]
    candidate_payload = [
        stamped(row, label=f"candidate[{index}]")
        for index, row in enumerate(canonical_candidates)
    ]
    probe_payload = [
        stamped(row, label=f"probe[{index}]")
        for index, row in enumerate(canonical_probes)
    ]
    provenance_payload = stamped(
        sanitized_provenance if isinstance(sanitized_provenance, dict) else {},
        label="provenance",
    )
    task_status_payload = stamped(canonical_status, label="task_status")
    # Serialize the complete report set before replacing any file.  If one
    # payload contains NaN/Infinity (or another JSON error), all previous files
    # remain untouched instead of leaving a mixed-generation report directory.
    ranking_text = json.dumps(
        ranking_payload, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    candidates_text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in candidate_payload
    )
    probes_text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in probe_payload
    )
    task_status_text = json.dumps(
        task_status_payload, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    provenance_text = json.dumps(
        provenance_payload, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    texts = {
        "ranking.json": ranking_text,
        "candidate_results.jsonl": candidates_text,
        "probe_results.jsonl": probes_text,
        "task_status.json": task_status_text,
        "provenance.json": provenance_text,
    }
    generation_id = uuid.uuid4().hex
    generation_root = results_dir / ".report_generations"
    generation_root.mkdir(parents=True, exist_ok=True)
    staging_dir = generation_root / f".{generation_id}.tmp"
    generation_dir = generation_root / generation_id
    staging_dir.mkdir()
    try:
        for name in _REPORT_FILES:
            _atomic_write(staging_dir / name, texts[name])
        manifest = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "run_id": report_run_id,
            "generation_id": generation_id,
            "snapshot_id": generation_id,
            "files": {
                name: {
                    "sha256": _sha256_text(texts[name]),
                    "size_bytes": len(texts[name].encode("utf-8")),
                    "row_count": (
                        len(ranking_payload)
                        if name == "ranking.json"
                        else len(candidate_payload)
                        if name == "candidate_results.jsonl"
                        else len(probe_payload)
                        if name == "probe_results.jsonl"
                        else 1
                    ),
                }
                for name in _REPORT_FILES
            },
        }
        manifest_text = json.dumps(
            manifest, ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n"
        _atomic_write(staging_dir / "report_manifest.json", manifest_text)
        staging_dir.replace(generation_dir)
        _fsync_directory(generation_root)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    # Root files are a compatibility view. Publish a deliberately provisional
    # loose status before changing any view file, then atomically commit the
    # manifest. A best-effort final loose status after the commit can only
    # create a false negative; it can never expose FINAL before its generation.
    loose_status_payload = deepcopy(task_status_payload)
    if loose_status_payload.get("ranking_status") == "FINAL":
        loose_status_payload["task_status"] = "INCOMPLETE"
        loose_status_payload["ranking_status"] = "PROVISIONAL"
        loose_status_payload["publication_state"] = "pending_manifest"
    loose_status_text = json.dumps(
        loose_status_payload, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    _atomic_write(results_dir / "task_status.json", loose_status_text)
    for name in _REPORT_FILES:
        if name == "task_status.json":
            continue
        _atomic_write(results_dir / name, texts[name])
    _atomic_write(results_dir / "report_manifest.json", manifest_text)
    try:
        _atomic_write(results_dir / "task_status.json", texts["task_status.json"])
    except OSError:
        pass
    return load_report_generation(results_dir)


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
