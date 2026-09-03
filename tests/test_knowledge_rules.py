"""Offline checks for the modular skill knowledge library."""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import yaml

from scripts.sync_hf_models import extract_mtp_params_from_cookbook
from scripts.sync_sglang_params import extract_speculative_algorithms, resolve_image_key
from scripts.validate_knowledge import DEFAULT_RULES_DIR, validate_rule_files


def test_topic_rule_files_have_valid_schema_and_unique_ids() -> None:
    assert validate_rule_files(DEFAULT_RULES_DIR) == []


def test_each_rule_references_its_topic_and_has_guidance() -> None:
    for path in sorted(DEFAULT_RULES_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["topic"] == path.stem
        assert data["rules"]
        assert all(rule["id"].startswith(f"{path.stem}.") for rule in data["rules"])
        assert all(rule["guidance"].strip() for rule in data["rules"])


def test_a_new_topic_file_needs_no_validator_code_change(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    copytree(DEFAULT_RULES_DIR, rules_dir)
    (rules_dir / "routing.yaml").write_text(
        "schema_version: 1\n"
        "topic: routing\n"
        "description: routing rules\n"
        "rules:\n"
        "  - id: routing.example\n"
        "    title: example\n"
        "    status: experimental\n"
        "    confidence: low\n"
        "    applies_when: {}\n"
        "    guidance: try it\n"
        "    evidence:\n"
        "      kind: judgment\n"
        "      ref: local test\n"
        "      verified_with: test\n",
        encoding="utf-8",
    )
    assert validate_rule_files(rules_dir) == []


def test_cookbook_mtp_parser_removes_code_fence_punctuation(tmp_path: Path) -> None:
    mdx = tmp_path / "model.mdx"
    mdx.write_text(
        "run with --speculative-algorithm EAGLE3`: --speculative-num-steps 3 "
        "--speculative-eagle-topk 1 --speculative-num-draft-tokens 4",
        encoding="utf-8",
    )
    assert extract_mtp_params_from_cookbook(mdx) == {
        "speculative-algorithm": "EAGLE3",
        "speculative-num-steps": 3,
        "speculative-eagle-topk": 1,
        "speculative-num-draft-tokens": 4,
    }


def test_sglang_image_extractor_reads_builtin_speculative_algorithms(tmp_path: Path) -> None:
    repo = tmp_path / "sglang"
    server_args = repo / "python/sglang/srt/server_args.py"
    spec_info = repo / "python/sglang/srt/speculative/spec_info.py"
    server_args.parent.mkdir(parents=True)
    spec_info.parent.mkdir(parents=True)
    server_args.write_text("# server args", encoding="utf-8")
    spec_info.write_text(
        "class SpeculativeAlgorithm(Enum):\n"
        "    DFLASH = auto()\n"
        "    EAGLE3 = auto()\n"
        "    NONE = auto()\n\n"
        "class SpecInputType(IntEnum):\n",
        encoding="utf-8",
    )
    assert extract_speculative_algorithms(server_args) == ["DFLASH", "EAGLE3", "NONE"]


def test_image_sync_reuses_legacy_catalog_key() -> None:
    data = {
        "images": {
            "I03_sglang-v0.5.16": {
                "image_ref": "hai.example/sglang:v0.5.16-cu129",
            }
        }
    }
    assert resolve_image_key(data, "v0.5.16") == "I03_sglang-v0.5.16"
