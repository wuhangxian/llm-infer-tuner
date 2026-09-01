"""Unit tests for scripts/sync_hf_models.py bug fixes.

Covers three fixes:
  Bug 1 — _clean(): strip stray backtick/quote/backslash from scraped flag values
           (root cause of a trailing-backtick reasoning-parser polluting models.yaml).
  Bug 2 — auto-card provenance: source/needs_review markers + source_url rename +
           divider tag written by save_models_yaml().
  Bug 3 — _normalize_model_id(): reject garbage IDs (reserved namespaces, asset
           paths, local/env refs) so scan_cookbook_models() stops minting junk cards.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO_ROOT / "scripts" / "sync_hf_models.py"
    spec = importlib.util.spec_from_file_location("sync_hf_models", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shm = _load_module()


# ── Bug 1: _clean ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("qwen3`", "qwen3"),                # trailing backtick — the actual bug
    ("`qwen3`", "qwen3"),               # inline-code wrapped
    ('"qwen3"', "qwen3"),               # double-quoted
    ("'qwen3'", "qwen3"),               # single-quoted
    ("  qwen3  ", "qwen3"),             # whitespace
    ("qwen3_coder\\", "qwen3_coder"),   # trailing backslash
    ("lfm2`", "lfm2"),                  # from the M04 card in the repo
    ("auto`**", "auto"),                # backtick + markdown bold tail
    ("EAGLE3`:", "EAGLE3"),             # backtick + colon tail
    ("dots`)", "dots"),                 # backtick + paren tail
    ("qwen3", "qwen3"),                 # clean value untouched
])
def test_clean_strips_wrappers(raw, expected):
    assert shm._clean(raw) == expected


def test_clean_passes_non_strings_through():
    assert shm._clean(True) is True
    assert shm._clean(64) == 64
    assert shm._clean(None) is None


# ── Bug 3: _normalize_model_id ──────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
    ("`Qwen/Qwen3-8B`", "Qwen/Qwen3-8B"),          # cleaned then accepted
    ('"deepseek-ai/DeepSeek-V3"', "deepseek-ai/DeepSeek-V3"),
    ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
])
def test_normalize_accepts_real_ids(raw, expected):
    assert shm._normalize_model_id(raw) == expected


@pytest.mark.parametrize("raw", [
    "docs/cookbook",              # reserved namespace
    "datasets/squad",             # reserved namespace
    "blog/some-post",             # reserved namespace
    "settings/tokens",            # reserved namespace
    "/data/models/local",         # local absolute path
    "$MODEL_PATH/foo",            # env-var ref
    "~/models/foo",               # home path
    "https://huggingface.co/x",   # url
    "single-segment",             # no slash
    "a/b/c",                      # too many segments
    "org/asset.css",              # asset extension
    "org/bundle.min.js",          # asset extension
    "org/readme.mdx",             # asset extension
    "",                           # empty
])
def test_normalize_rejects_garbage(raw):
    assert shm._normalize_model_id(raw) is None


def test_normalize_rejects_non_string():
    assert shm._normalize_model_id(None) is None
    assert shm._normalize_model_id(123) is None


# ── Bug 2: auto-card provenance ─────────────────────────────────────────────

def test_generate_card_has_provenance_markers():
    info = {
        "is_moe": False,
        "quant_method": "fp8",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
    }
    card = shm.generate_new_model_card("Qwen/Qwen3-8B", info, sglang_repo="")
    assert card["source"] == "auto"
    assert card["needs_review"] is True
    # HF URL moved off the colliding "source" key onto source_url.
    assert card["source_url"] == "https://huggingface.co/Qwen/Qwen3-8B"


def test_save_models_yaml_tags_auto_cards(tmp_path, monkeypatch):
    out = tmp_path / "models.yaml"
    monkeypatch.setattr(shm, "MODELS_YAML", out)
    data = {
        "version": 1,
        "models": {
            "M01_reviewed": {
                "hf_model_id": "Qwen/Reviewed-8B", "arch": "dense",
                "default_precision": "fp8",
            },
            "M02_auto": {
                "hf_model_id": "Qwen/Auto-8B", "arch": "dense",
                "default_precision": "fp8", "source": "auto", "needs_review": True,
            },
        },
    }
    shm.save_models_yaml(data)
    text = out.read_text(encoding="utf-8")
    # Auto card divider carries the tag; reviewed one does not.
    auto_divider = next(
        line for line in text.splitlines() if "M02_auto:" in line
    )
    reviewed_divider = next(
        line for line in text.splitlines() if "M01_reviewed:" in line
    )
    assert "[AUTO needs_review]" in auto_divider
    assert "[AUTO needs_review]" not in reviewed_divider


# ── Bug 1 integration: extracted flags are clean ────────────────────────────

def test_extract_default_flags_cleans_backticks(tmp_path):
    mdx = tmp_path / "model.mdx"
    mdx.write_text(
        "Launch with:\n"
        "--reasoning-parser qwen3`\n"
        "--tool-call-parser `lfm2`\n"
        "--trust-remote-code\n",
        encoding="utf-8",
    )
    flags = shm.extract_default_flags_from_cookbook(mdx)
    assert flags["reasoning-parser"] == "qwen3"
    assert flags["tool-call-parser"] == "lfm2"
    assert flags["trust-remote-code"] is True
    # No backtick survives into any value.
    assert not any("`" in v for v in flags.values() if isinstance(v, str))
