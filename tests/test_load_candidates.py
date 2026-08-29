"""Unit tests for _load_candidates baseline-aware capping (issue #1 regression).

约定(gen_configs.sh:169):baseline 候选(params.is_baseline==true)不占 max_candidates
名额,总候选数 = max_candidates + 1。旧代码 candidates[:max_candidates] 按裸长度截断,
baseline 在场时会把最后一条真候选(如 tp=4 的 c002)静默丢弃。这些用例钉死修复后的行为。
"""

from __future__ import annotations

import json
from pathlib import Path

from runners.executor import _load_candidates


def _write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "configs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _write_json(tmp_path: Path, candidates: list[dict]) -> Path:
    path = tmp_path / "configs.json"
    path.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
    return path


def _baseline(cid: str = "baseline") -> dict:
    return {"id": cid, "params": {"is_baseline": True, "tp_size": 2}}


def _cand(cid: str, tp: int = 1) -> dict:
    return {"id": cid, "params": {"tp_size": tp}}


def test_jsonl_baseline_exempt_keeps_last_candidate(tmp_path):
    """场景1(核心回归):JSONL,leading baseline + max_candidates 条真候选,
    max_candidates=2 时应返回 3 条,tp=4 的 c002 不能被丢。"""
    rows = [_baseline(), _cand("c001"), _cand("c002", tp=4)]
    path = _write_jsonl(tmp_path, rows)

    out = _load_candidates(path, max_candidates=2)

    assert [c["id"] for c in out] == ["baseline", "c001", "c002"]
    assert len(out) == 3  # max_candidates + 1


def test_jsonl_no_baseline_anchor_caps_exactly(tmp_path):
    """场景2:JSONL,无 is_baseline 标记的锚点 + 超过 max_candidates 条,
    锚点计入名额,恰好返回 max_candidates 条,不多读(gen_configs.sh:172 场景)。"""
    rows = [_cand("anchor"), _cand("c001"), _cand("c002")]
    path = _write_jsonl(tmp_path, rows)

    out = _load_candidates(path, max_candidates=2)

    assert [c["id"] for c in out] == ["anchor", "c001"]
    assert len(out) == 2


def test_json_baseline_exempt_on_return_path(tmp_path):
    """场景3:JSON {"candidates":[baseline, ...max_candidates 条]},
    return 路径同样对 baseline 豁免名额。"""
    candidates = [_baseline(), _cand("c001"), _cand("c002", tp=4)]
    path = _write_json(tmp_path, candidates)

    out = _load_candidates(path, max_candidates=2)

    assert [c["id"] for c in out] == ["baseline", "c001", "c002"]


def test_nonpositive_max_returns_all_uncapped(tmp_path):
    """场景4:max_candidates <= 0 表示不限,原样返回。"""
    candidates = [_baseline(), _cand("c001"), _cand("c002"), _cand("c003")]
    path = _write_json(tmp_path, candidates)

    out = _load_candidates(path, max_candidates=0)

    assert [c["id"] for c in out] == ["baseline", "c001", "c002", "c003"]


def test_multiple_baselines_all_kept(tmp_path):
    """加固:即便出现多条 baseline,也全部保留、均不占名额。"""
    rows = [_baseline("b0"), _baseline("b1"), _cand("c001"), _cand("c002")]
    path = _write_jsonl(tmp_path, rows)

    out = _load_candidates(path, max_candidates=1)

    assert [c["id"] for c in out] == ["b0", "b1", "c001"]
