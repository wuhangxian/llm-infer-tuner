"""Regression tests for strict, fail-closed candidate file loading."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from runners.executor import _build_cmd_from_params, _load_candidates
from schemas.job_spec import SearchBudget

FIXTURES = Path(__file__).parent / "fixtures" / "candidates"


def _candidate(candidate_id: str, **params: object) -> dict:
    tuning_flags = " ".join(
        f"--{key.replace('_', '-')} {value}"
        for key, value in params.items()
        if key not in {"is_baseline", "disable_radix_cache", "disable-radix-cache"}
    )
    return {
        "id": candidate_id,
        "params": params,
        "cmd": (
            "python -m sglang.launch_server --model-path ${MODEL_PATH} "
            f"{tuning_flags}"
        ).rstrip(),
        "reasons": [],
    }


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _search(max_candidates: int = 2, *, baseline: bool = False) -> SearchBudget:
    return SearchBudget.model_validate(
        {"max_candidates": max_candidates, **({"baseline": {}} if baseline else {})}
    )


def test_load_candidates_accepts_json_object_array_and_jsonl(tmp_path: Path) -> None:
    candidates = [_candidate("c001", tp_size=1), _candidate("c002", tp_size=2)]
    json_path = _write(tmp_path / "configs.json", json.dumps({"candidates": candidates}))
    list_path = _write(tmp_path / "configs-list.json", json.dumps(candidates))
    jsonl_path = _write(
        tmp_path / "configs.jsonl", "\n".join(json.dumps(row) for row in candidates) + "\n"
    )

    for path in (json_path, list_path, jsonl_path):
        assert [row["id"] for row in _load_candidates(path, search=_search())] == [
            "c001",
            "c002",
        ]


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ('{"candidates": [{"id": "c001"}]}', "candidate"),
        ('{"candidates": []}', "at least 1"),
        ('{"not_candidates": []}', "top-level"),
        ('{"id":"c001"}\nnot-json\n', "row 2"),
        ('{"id":"c001"}\n["not a candidate"]\n', "row 2"),
        (json.dumps(_candidate("c001")) + "\n", "expected 2"),
    ],
)
def test_load_candidates_rejects_every_malformed_or_count_mismatched_input(
    tmp_path: Path, content: str, match: str
) -> None:
    path = _write(tmp_path / "configs.jsonl", content)

    with pytest.raises(ValueError, match=match):
        _load_candidates(path, search=_search())


def test_load_candidates_preserves_path_and_candidate_validation_context(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "configs.jsonl",
        json.dumps(_candidate("bad id")) + "\n" + json.dumps(_candidate("c002")),
    )

    with pytest.raises(ValueError, match=r"configs\.jsonl.*candidate"):
        _load_candidates(path, search=_search())


def test_params_are_shell_quoted_when_rendered_to_a_legacy_command() -> None:
    command = _build_cmd_from_params({"served_model_name": "safe; not-a-command"})

    parts = shlex.split(command)
    assert parts[parts.index("--served-model-name") + 1] == "safe; not-a-command"


def test_load_candidates_keeps_mamba_audit_only_and_returns_safe_executor_payload() -> None:
    rows = _load_candidates(
        FIXTURES / "mamba_baseline_compat.json", search=_search(baseline=True)
    )

    assert [row["id"] for row in rows] == ["baseline", "c001", "c002"]
    for row in rows:
        assert "mamba_radix_cache_strategy" in row["requested_params"]
        assert "mamba_radix_cache_strategy" not in row["params"]
        assert "--mamba-radix-cache-strategy" in row["requested_cmd"]
        assert "--mamba-radix-cache-strategy" not in row["cmd"]
        assert "--model-path" not in row["cmd"]
        assert "--host" not in row["cmd"]
        assert "--port" not in row["cmd"]
        assert row["cmd"].split().count("--disable-radix-cache") == 1


def test_loader_rejects_overflowing_exponent_anywhere_in_the_document(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "configs.json",
        '{"_meta":{"overflow":1e400},"candidates":[]}',
    )

    with pytest.raises(ValueError, match="non-finite"):
        _load_candidates(path, search=_search())
