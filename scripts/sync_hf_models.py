"""Sync model info from HuggingFace API into models.yaml.

For each model in models.yaml (plus an optional watch list), fetch
config.json from HuggingFace and extract:
  - is_moe / num_experts / moe_intermediate_size / intermediate_size
  - weight_block_size (block_size for TP constraint)
  - quantization_config (method / scheme / dtype)
  - mtp layers (supports_mtp)
  - context length (max_position_embeddings)

Then compare with current models.yaml and report differences.
Fields that require manual input (default_flags, mtp_params, weight_gb,
parser names) are NOT auto-updated — only flagged in the diff report.

Usage:
  python scripts/sync_hf_models.py
  python scripts/sync_hf_models.py --watch Qwen/Qwen3.6-27B-FP8 DeepSeek/DeepSeek-V4-Flash-0731

No AI needed. Pure API calls + JSON parsing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = PROJECT_ROOT / "catalogs/models.yaml"


def fetch_config_json(hf_model_id: str) -> dict[str, Any]:
    """Fetch config.json from HuggingFace."""
    url = f"https://huggingface.co/{hf_model_id}/raw/main/config.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llm-infer-tuner-sync"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"  [error] {hf_model_id}: HTTP {exc.code}")
        return {}
    except Exception as exc:
        print(f"  [error] {hf_model_id}: {exc}")
        return {}


def extract_model_info(config: dict) -> dict[str, Any]:
    """Extract relevant fields from HF config.json.

    Handle nested configs (some models put text config under 'text_config').
    """
    # Some VLMs nest the text model config under text_config
    text_config = config.get("text_config", {})
    # Merge: prefer top-level, fall back to text_config
    def get(key, default=None):
        return config.get(key, text_config.get(key, default))

    # MoE detection
    is_moe = get("is_moe", None)
    num_experts = get("num_experts", None)
    moe_intermediate_size = get("moe_intermediate_size", None)
    intermediate_size = get("intermediate_size", None)

    # Some models don't have is_moe but have num_experts
    if is_moe is None and num_experts is not None:
        is_moe = True

    # Quantization
    quant_config = config.get("quantization_config", text_config.get("quantization_config", {}))
    quant_method = quant_config.get("quant_method", "none") if quant_config else "none"
    weight_block_size = quant_config.get("weight_block_size", None) if quant_config else None
    if weight_block_size and isinstance(weight_block_size, list):
        block_size = weight_block_size[0]  # [128, 128] -> 128
    else:
        block_size = None

    # MTP detection
    mtp_config = config.get("mtp", text_config.get("mtp", {}))
    has_mtp = bool(mtp_config) or "mtp" in config or "mtp" in text_config
    # Also check for mtp in modules_to_not_convert
    if not has_mtp:
        modules = quant_config.get("modules_to_not_convert", []) if quant_config else []
        has_mtp = any("mtp" in str(m).lower() for m in modules) if modules else False

    # Context length
    max_pos = get("max_position_embeddings", None)

    # Architecture
    architectures = config.get("architectures", [])
    model_type = config.get("model_type", "")

    return {
        "is_moe": is_moe,
        "num_experts": num_experts,
        "moe_intermediate_size": moe_intermediate_size,
        "intermediate_size": intermediate_size,
        "block_size": block_size,
        "quant_method": quant_method,
        "max_position_embeddings": max_pos,
        "has_mtp": has_mtp,
        "architectures": architectures,
        "model_type": model_type,
    }


def load_models_yaml() -> dict[str, Any]:
    with open(MODELS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compare_and_report(models_data: dict, hf_id: str, hf_info: dict) -> list[str]:
    """Compare HF info with models.yaml entry. Return list of diffs."""
    models = models_data.get("models", {})
    # Find the model key that matches this hf_model_id
    model_key = None
    for key, val in models.items():
        if val.get("hf_model_id") == hf_id:
            model_key = key
            break

    if model_key is None:
        return [f"[NEW MODEL] {hf_id} not in models.yaml — consider adding"]

    entry = models[model_key]
    diffs: list[str] = []

    arch = entry.get("architecture", {})

    # Check is_moe
    yaml_is_moe = arch.get("is_moe", None)
    hf_is_moe = hf_info.get("is_moe")
    if hf_is_moe is not None and yaml_is_moe is not None and hf_is_moe != yaml_is_moe:
        diffs.append(f"is_moe: yaml={yaml_is_moe} vs hf={hf_is_moe}")

    # Check moe_intermediate_size
    yaml_moe_int = arch.get("moe_intermediate_size", None)
    hf_moe_int = hf_info.get("moe_intermediate_size")
    if hf_moe_int is not None and yaml_moe_int is not None and hf_moe_int != yaml_moe_int:
        diffs.append(f"moe_intermediate_size: yaml={yaml_moe_int} vs hf={hf_moe_int}")
    elif hf_moe_int is not None and yaml_moe_int is None:
        diffs.append(f"moe_intermediate_size: yaml=MISSING vs hf={hf_moe_int}")

    # Check block_size
    quant = entry.get("quantization", {})
    yaml_block = quant.get("block_size", None)
    hf_block = hf_info.get("block_size")
    if hf_block is not None and yaml_block is not None and hf_block != yaml_block:
        diffs.append(f"block_size: yaml={yaml_block} vs hf={hf_block}")

    # Check intermediate_size
    yaml_int = arch.get("intermediate_size", None)
    hf_int = hf_info.get("intermediate_size")
    if hf_int is not None and yaml_int is not None and hf_int != yaml_int:
        diffs.append(f"intermediate_size: yaml={yaml_int} vs hf={hf_int}")

    # Check num_experts
    yaml_experts = entry.get("num_experts", None)
    hf_experts = hf_info.get("num_experts")
    if hf_experts is not None and yaml_experts is not None and hf_experts != yaml_experts:
        diffs.append(f"num_experts: yaml={yaml_experts} vs hf={hf_experts}")

    # Check supports_mtp
    caps = entry.get("capabilities", {})
    yaml_mtp = caps.get("supports_mtp", None)
    hf_mtp = hf_info.get("has_mtp")
    if hf_mtp is not None and yaml_mtp is not None and hf_mtp != yaml_mtp:
        diffs.append(f"supports_mtp: yaml={yaml_mtp} vs hf={hf_mtp}")

    return diffs


def save_diff_report(all_diffs: dict[str, list[str]]) -> None:
    """Save diff report for human review."""
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "models_diff.md"
    lines = ["# HuggingFace Model Sync Report\n",
             "Auto-generated by sync_hf_models.py\n",
             "Review diffs and update catalogs/models.yaml manually.\n\n"]
    has_diffs = False
    for hf_id, diffs in all_diffs.items():
        if diffs:
            has_diffs = True
            lines.append(f"## {hf_id}\n")
            for d in diffs:
                lines.append(f"- {d}")
            lines.append("")
    if not has_diffs:
        lines.append("No differences found. All models up to date.\n")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [report] {report_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync model info from HuggingFace")
    parser.add_argument("--watch", nargs="*", default=[],
                        help="Additional HF model IDs to check")
    args = parser.parse_args(argv)

    # Load models.yaml to get existing model IDs
    print("[1/3] Loading models.yaml...")
    models_data = load_models_yaml()
    models = models_data.get("models", {})

    # Collect all HF model IDs to check
    hf_ids: list[str] = []
    for val in models.values():
        hf_id = val.get("hf_model_id")
        if hf_id:
            hf_ids.append(hf_id)
    hf_ids.extend(args.watch)

    print(f"  models in yaml: {len(models)}")
    print(f"  hf_ids to check: {len(hf_ids)}")

    # Fetch and compare
    print("[2/3] Fetching config.json from HuggingFace...")
    all_diffs: dict[str, list[str]] = {}
    for hf_id in hf_ids:
        print(f"  {hf_id}...")
        config = fetch_config_json(hf_id)
        if not config:
            all_diffs[hf_id] = ["[ERROR] Failed to fetch config.json"]
            continue
        hf_info = extract_model_info(config)
        diffs = compare_and_report(models_data, hf_id, hf_info)
        all_diffs[hf_id] = diffs
        if diffs:
            for d in diffs:
                print(f"    [diff] {d}")
        else:
            print("    [ok] up to date")

    # Save report
    print("[3/3] Saving diff report...")
    save_diff_report(all_diffs)

    # Summary
    total_diffs = sum(len(d) for d in all_diffs.values())
    print("\n=== Summary ===")
    print(f"  models checked: {len(hf_ids)}")
    print(f"  total diffs: {total_diffs}")
    print(f"  report: reports/models_diff.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
