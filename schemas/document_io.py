"""Strict JSON document loaders shared by CLIs and the executor."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from schemas.candidate_spec import CandidateSet
from schemas.job_spec import JobSpec, SearchBudget


def strict_json_load(text: str, *, source: str) -> Any:
    """Decode JSON while rejecting duplicate keys and non-finite numbers."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{source}: non-finite JSON value {value!r} is not allowed")

    def reject_nonfinite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{source}: non-finite JSON value {value!r} is not allowed")
        return parsed

    return json.loads(
        text,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
        parse_float=reject_nonfinite_float,
    )


def load_job(path: Path) -> JobSpec:
    """Load and validate one strict JobSpec document."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{path}: cannot read job: {exc}") from exc
    try:
        payload = strict_json_load(text, source=str(path))
        return JobSpec.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"{path}: invalid JobSpec: {exc}") from exc


def _candidate_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{path}: cannot read candidates: {exc}") from exc
    source = str(path)
    try:
        document = strict_json_load(text, source=source)
    except json.JSONDecodeError:
        document = None

    if document is not None:
        if isinstance(document, list):
            candidates = document
        elif isinstance(document, dict):
            if "id" in document:
                candidates = [document]
            else:
                unsupported = set(document) - {"candidates", "_meta"}
                if unsupported or "candidates" not in document:
                    raise ValueError(
                        f"{source}: top-level JSON object must contain only candidates "
                        "(and optional _meta)"
                    )
                candidates = document["candidates"]
        else:
            raise ValueError(
                f"{source}: top-level JSON must be an object or array of candidates"
            )
        if not isinstance(candidates, list):
            raise ValueError(f"{source}: candidates must be a JSON array")
        return candidates

    candidates = []
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"{source}: candidate file is empty")
    for row_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{source}: row {row_number} is empty")
        try:
            row = strict_json_load(line, source=f"{source}: row {row_number}")
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{source}: row {row_number} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{source}: row {row_number} must be a candidate object")
        candidates.append(row)
    return candidates


def load_candidate_set(path: Path, *, search: SearchBudget) -> CandidateSet:
    """Load one complete candidate set and enforce its job search contract."""
    try:
        return CandidateSet.from_candidates(_candidate_rows(path), search=search)
    except (ValidationError, ValueError) as exc:
        raise ValueError(f"{path}: candidate validation failed: {exc}") from exc


def load_candidates(path: Path, *, search: SearchBudget) -> list[dict[str, Any]]:
    """Return the validated executor payload while retaining legacy dict callers."""
    candidate_set = load_candidate_set(path, search=search)
    return [candidate.model_dump() for candidate in candidate_set.candidates]


__all__ = ["load_candidate_set", "load_candidates", "load_job", "strict_json_load"]
