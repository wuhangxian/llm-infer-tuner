"""Sync model info from HuggingFace API + SGLang cookbook into models.yaml.

Automatically:
  1. Scans SGLang cookbook (local repo or remote) for model deployment commands
     to discover all models SGLang supports.
  2. For new models not in models.yaml: fetches config.json from HuggingFace,
     extracts architecture/quantization/MoE info, and generates a new model card
     marked [AUTO] for manual completion of default_flags/mtp_params.
  3. For existing models: checks if config.json has changed and reports diffs.

Usage:
  python scripts/sync_hf_models.py
  python scripts/sync_hf_models.py --sglang-repo /data/home/dorianwu/sglang-latest
  python scripts/sync_hf_models.py --watch Qwen/Qwen3-Coder-480B-A35B-Instruct

No AI needed. Pure API calls + JSON parsing + regex.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = PROJECT_ROOT / "catalogs/models.yaml"

# Default SGLang repo path (local clone)
DEFAULT_SGLANG_REPO = "/data/home/dorianwu/sglang-latest"
COOKBOOK_DIRS = [
    "docs/cookbook/autoregressive",
    "docs_new/cookbook/autoregressive",
]


def scan_cookbook_models(sglang_repo: str, since_date: str = "2026-05-01") -> list[str]:
    """Scan SGLang cookbook mdx files added since since_date for model IDs."""
    repo = Path(sglang_repo)
    model_ids: set[str] = set()

    # Use git log to find mdx files first added since since_date
    new_files: list[str] = []
    try:
        result = subprocess.run(
            ["git", "log", "--all", "--since", since_date,
             "--diff-filter=A", "--name-only", "--pretty=",
             "--", "docs_new/cookbook/autoregressive/", "docs/cookbook/autoregressive/"],
            cwd=repo, capture_output=True, text=True, check=True, timeout=30,
        )
        new_files = [f.strip() for f in result.stdout.splitlines()
                     if f.strip().endswith(".mdx") and "intro" not in f]
        print(f"  [git] {len(new_files)} new mdx files since {since_date}")
    except Exception:
        pass

    files_to_scan: list[Path] = []
    if new_files:
        for rel in new_files:
            fp = repo / rel
            if fp.exists() and fp not in files_to_scan:
                files_to_scan.append(fp)
            alt = rel.replace("docs_new/", "docs/") if "docs_new/" in rel else rel.replace("docs/", "docs_new/")
            alt_p = repo / alt
            if alt_p.exists() and alt_p not in files_to_scan:
                files_to_scan.append(alt_p)
    else:
        for cookbook_dir in COOKBOOK_DIRS:
            search_dir = repo / cookbook_dir
            if search_dir.exists():
                files_to_scan.extend(search_dir.rglob("*.mdx"))

    for mdx_file in files_to_scan:
            try:
                text = mdx_file.read_text(encoding="utf-8")
            except OSError:
                continue
            # Find --model-path <model_id> or --model-path=<model_id>
            # Also find --model <model_id>
            patterns = [
                r'--model-path\s+["\']?([\w/-]+/[^"\'\s]+)',
                r'--model-path=([\w/-]+/[^"\'\s]+)',
                r'--model\s+["\']?([\w/-]+/[^"\'\s]+)',
                r'--model=([\w/-]+/[^"\'\s]+)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for m in matches:
                    # Filter out paths that look like file paths (contain /data/ or /tmp/)
                    if not m.startswith("/") and "/" in m and not m.startswith("$"):
                        model_ids.add(m.strip("'\""))

    return sorted(model_ids)


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
    """Extract relevant fields from HF config.json."""
    text_config = config.get("text_config", {})

    def get(key, default=None):
        return config.get(key, text_config.get(key, default))

    is_moe = get("is_moe", None)
    num_experts = get("num_experts", None)
    moe_intermediate_size = get("moe_intermediate_size", None)
    intermediate_size = get("intermediate_size", None)

    if is_moe is None and num_experts is not None:
        is_moe = True

    # Also check for shared_expert in modules_to_not_convert as MoE indicator
    if is_moe is None:
        quant_config = config.get("quantization_config", text_config.get("quantization_config", {}))
        modules = quant_config.get("modules_to_not_convert", []) if quant_config else []
        if modules and any("shared_expert" in str(m) or ".gate" in str(m) for m in modules):
            is_moe = True

    quant_config = config.get("quantization_config", text_config.get("quantization_config", {}))
    quant_method = quant_config.get("quant_method", "none") if quant_config else "none"
    weight_block_size = quant_config.get("weight_block_size", None) if quant_config else None
    if weight_block_size and isinstance(weight_block_size, list):
        block_size = weight_block_size[0]
    else:
        block_size = None

    mtp_config = config.get("mtp", text_config.get("mtp", {}))
    has_mtp = bool(mtp_config) or "mtp" in config or "mtp" in text_config
    if not has_mtp:
        modules = quant_config.get("modules_to_not_convert", []) if quant_config else []
        has_mtp = any("mtp" in str(m).lower() for m in modules) if modules else False

    max_pos = get("max_position_embeddings", None)
    architectures = config.get("architectures", [])
    model_type = config.get("model_type", "")
    num_hidden_layers = get("num_hidden_layers", None)
    hidden_size = get("hidden_size", None)
    vocab_size = get("vocab_size", None)

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
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
    }


def load_models_yaml() -> dict[str, Any]:
    with open(MODELS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_models_yaml(data: dict) -> None:
    data["total"] = len(data.get("models", {}))
    with open(MODELS_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def make_model_key(hf_id: str) -> str:
    """Convert HF model ID to a yaml key (e.g. Qwen/Qwen3.6-27B-FP8 -> qwen36-27b-fp8)."""
    name = hf_id.split("/")[-1]  # Qwen3.6-27B-FP8
    key = name.lower()
    key = re.sub(r"[^a-z0-9]", "-", key)  # qwen3-6-27b-fp8
    key = re.sub(r"-+", "-", key)  # collapse dashes
    key = key.strip("-")
    # Qwen3.6 -> qwen36 pattern
    key = key.replace("3-6", "36")
    return key


def generate_new_model_card(hf_id: str, info: dict) -> dict[str, Any]:
    """Generate a new model card from HF info. Marked [AUTO] for manual completion."""
    is_moe = info.get("is_moe", False)
    arch = "moe" if is_moe else "dense"

    card: dict[str, Any] = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "hf_model_id": hf_id,
        "family": "[AUTO-fill]",
        "parameter_count_b": None,
        "arch": arch,
        "hybrid_mamba": False,  # needs manual check
        "default_precision": info.get("quant_method", "none"),
        "weight_gb": {},
        "_auto_generated": True,
        "_todo": "Fill family, weight_gb, default_flags, mtp_params, hybrid_mamba from cookbook",
        "quantization": {
            "method": info.get("quant_method", "none"),
            "scheme": "[AUTO-detect]",
            "block_size": info.get("block_size"),
        },
        "architecture": {
            "is_moe": is_moe,
            "intermediate_size": info.get("intermediate_size"),
            "num_hidden_layers": info.get("num_hidden_layers"),
            "hidden_size": info.get("hidden_size"),
            "vocab_size": info.get("vocab_size"),
        },
        "capabilities": {
            "supports_mtp": info.get("has_mtp", False),
        },
        "default_flags": {
            "_todo": "Fill from cookbook deployment command (reasoning-parser, tool-call-parser, etc.)",
        },
        "deployment": {
            "model_format": "huggingface",
            "weight_size_gb": None,
            "model_path": None,
        },
        "source": f"https://huggingface.co/{hf_id}",
        "notes": "[AUTO] Auto-discovered from SGLang cookbook. Manual completion required.",
    }

    if is_moe:
        card["architecture"]["num_experts"] = info.get("num_experts")
        card["architecture"]["moe_intermediate_size"] = info.get("moe_intermediate_size")
        card["num_experts"] = info.get("num_experts")

    if info.get("max_position_embeddings"):
        card["context"] = {
            "native_context_length": info["max_position_embeddings"],
        }

    return card


def compare_and_report(models_data: dict, hf_id: str, hf_info: dict) -> list[str]:
    """Compare HF info with models.yaml entry. Return list of diffs."""
    models = models_data.get("models", {})
    model_key = None
    for key, val in models.items():
        if val.get("hf_model_id") == hf_id:
            model_key = key
            break

    if model_key is None:
        return [f"[NEW MODEL] {hf_id} not in models.yaml"]

    entry = models[model_key]
    diffs: list[str] = []
    arch = entry.get("architecture", {})

    yaml_moe_int = arch.get("moe_intermediate_size", None)
    hf_moe_int = hf_info.get("moe_intermediate_size")
    if hf_moe_int is not None and yaml_moe_int is not None and hf_moe_int != yaml_moe_int:
        diffs.append(f"moe_intermediate_size: yaml={yaml_moe_int} vs hf={hf_moe_int}")
    elif hf_moe_int is not None and yaml_moe_int is None:
        diffs.append(f"moe_intermediate_size: yaml=MISSING vs hf={hf_moe_int}")

    quant = entry.get("quantization", {})
    yaml_block = quant.get("block_size", None)
    hf_block = hf_info.get("block_size")
    if hf_block is not None and yaml_block is not None and hf_block != yaml_block:
        diffs.append(f"block_size: yaml={yaml_block} vs hf={hf_block}")

    yaml_int = arch.get("intermediate_size", None)
    hf_int = hf_info.get("intermediate_size")
    if hf_int is not None and yaml_int is not None and hf_int != yaml_int:
        diffs.append(f"intermediate_size: yaml={yaml_int} vs hf={hf_int}")

    return diffs


def save_diff_report(all_diffs: dict[str, list[str]], new_models: list[str]) -> None:
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "models_diff.md"
    lines = ["# HuggingFace Model Sync Report\n",
             "Auto-generated by sync_hf_models.py\n\n"]
    if new_models:
        lines.append(f"## New Models Discovered ({len(new_models)})\n")
        for m in new_models:
            lines.append(f"- {m} — added to models.yaml as [AUTO] card, needs manual completion")
        lines.append("")
    has_diffs = False
    for hf_id, diffs in all_diffs.items():
        if diffs:
            has_diffs = True
            lines.append(f"## {hf_id}\n")
            for d in diffs:
                lines.append(f"- {d}")
            lines.append("")
    if not has_diffs and not new_models:
        lines.append("No differences found. All models up to date.\n")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [report] {report_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync model info from HuggingFace + cookbook")
    parser.add_argument("--sglang-repo", default=DEFAULT_SGLANG_REPO,
                        help="Path to local SGLang repo for cookbook scanning")
    parser.add_argument("--watch", nargs="*", default=[],
                        help="Additional HF model IDs to check")
    parser.add_argument("--since-date", default="2026-05-01",
                        help="Only scan cookbook models added since this date")
    args = parser.parse_args(argv)

    print("=== HuggingFace Model Sync ===")

    # Step 1: Load models.yaml
    print("[1/4] Loading models.yaml...")
    models_data = load_models_yaml()
    models = models_data.get("models", {})
    existing_hf_ids = {v.get("hf_model_id") for v in models.values() if v.get("hf_model_id")}
    print(f"  Existing models: {len(existing_hf_ids)}")

    # Step 2: Scan cookbook for all supported models
    print("[2/4] Scanning SGLang cookbook for models...")
    cookbook_models = scan_cookbook_models(args.sglang_repo, args.since_date)
    print(f"  Found {len(cookbook_models)} models in cookbook (since {args.since_date}):")
    for m in cookbook_models:
        status = "existing" if m in existing_hf_ids else "NEW"
        print(f"    [{status}] {m}")

    # Combine with watch list
    all_hf_ids = list(existing_hf_ids | set(cookbook_models) | set(args.watch))

    # Step 3: Fetch config.json and compare/add
    print(f"[3/4] Fetching config.json for {len(all_hf_ids)} models...")
    all_diffs: dict[str, list[str]] = {}
    new_models_added: list[str] = []
    models_changed = False

    for hf_id in sorted(all_hf_ids):
        print(f"  {hf_id}...")
        config = fetch_config_json(hf_id)
        if not config:
            all_diffs[hf_id] = ["[ERROR] Failed to fetch config.json"]
            continue

        hf_info = extract_model_info(config)

        if hf_id not in existing_hf_ids:
            # New model — generate card
            print(f"    [NEW] Generating model card...")
            card = generate_new_model_card(hf_id, hf_info)
            model_key = make_model_key(hf_id)
            # Avoid key collision
            if model_key in models:
                model_key = model_key + "-auto"
            models[model_key] = card
            new_models_added.append(hf_id)
            models_changed = True
            print(f"    -> Added as {model_key} [AUTO]")
        else:
            # Existing model — check for changes
            diffs = compare_and_report(models_data, hf_id, hf_info)
            all_diffs[hf_id] = diffs
            if diffs:
                for d in diffs:
                    print(f"    [diff] {d}")
                # Stamp the entry with today's date since we found changes
                for key, val in models.items():
                    if val.get("hf_model_id") == hf_id:
                        val["last_updated"] = datetime.now().strftime("%Y-%m-%d")
                        models_changed = True
                        break
            else:
                print("    [ok] up to date")

    if models_changed:
        save_models_yaml(models_data)
        print(f"  -> models.yaml updated ({len(new_models_added)} new models)")

    # Step 4: Save report
    print("[4/4] Saving diff report...")
    save_diff_report(all_diffs, new_models_added)

    # Summary
    print("\n=== Summary ===")
    print(f"  models in cookbook: {len(cookbook_models)}")
    print(f"  models in yaml: {len(existing_hf_ids)}")
    print(f"  new models added: {len(new_models_added)}")
    print(f"  diffs found: {sum(len(d) for d in all_diffs.values())}")
    if new_models_added:
        print(f"  [TODO] New model cards need manual completion:")
        print(f"         family, weight_gb, default_flags, mtp_params, hybrid_mamba")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
