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

from datetime import datetime


def _update_yaml_meta(data: dict, section_key: str) -> None:
    """Update version (+1), updated (timestamp), total (recount) in yaml data."""
    data["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["version"] = data.get("version", 1) + 1
    section = data.get(section_key, {})
    data["total"] = len(section)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = PROJECT_ROOT / "catalogs/models.yaml"

# Default SGLang repo path (local clone)
DEFAULT_SGLANG_REPO = "/data/home/dorianwu/sglang-latest"
COOKBOOK_DIRS = [
    "docs/cookbook/autoregressive",
    "docs_new/cookbook/autoregressive",
]
KNOWN_SPECULATIVE_ALGORITHMS = {
    "DFLASH", "DSPARK", "EAGLE", "EAGLE3", "FROZEN_KV_MTP", "STANDALONE", "NGRAM", "NONE",
}


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
                    if not m.startswith("/") and "/" in m and not m.startswith("$"):
                        model_ids.add(m.strip(chr(39) + chr(34)))

            # Pattern 2: href links
            href_matches = re.findall(r'huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)', text)
            for m in href_matches:
                if not any(x in m.lower() for x in ["/docs", "/datasets", "/blog", "/settings", "/license"]):
                    model_ids.add(m)

            # Pattern 3: model="org/model"
            model_eq = re.findall(r'model=["\']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)["\']', text)
            for m in model_eq:
                model_ids.add(m)

    # Also scan jsx config files
    for jsx_dir in [repo / "docs/src/snippets/configs", repo / "docs_new/src/snippets/configs"]:
        if not jsx_dir.exists():
            continue
        for jsx_file in jsx_dir.rglob("*.jsx"):
            try:
                jsx_text = jsx_file.read_text(encoding="utf-8")
            except OSError:
                continue
            jsx_matches = re.findall(r'["\']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+-[A-Za-z0-9]+)["\']', jsx_text)
            for m in jsx_matches:
                if not any(x in m.lower() for x in [".css", ".js", ".jsx", ".png", ".svg"]):
                    model_ids.add(m)

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



def fetch_model_size_gb(hf_model_id: str) -> int | None:
    """Fetch total safetensors file size from HuggingFace API."""
    url = f"https://huggingface.co/api/models/{hf_model_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llm-infer-tuner-sync"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        siblings = data.get("siblings", [])
        total_bytes = 0
        for sib in siblings:
            rfname = sib.get("rfilename", "")
            if rfname.endswith(".safetensors") or rfname.endswith(".bin"):
                # HF API doesn't return file size directly, estimate from siblings count
                pass
        # HF API siblings don't have size. Try the tree API instead.
        return None
    except Exception:
        return None


def fetch_weight_gb(hf_model_id: str) -> dict[str, int]:
    """Fetch weight file sizes from HuggingFace. Returns {precision: gb}.
    Uses HuggingFace API to get safetensors index and compute total size.
    """
    weights: dict[str, int] = {}
    try:
        # Try fetching model.safetensors.index.json to get total shard sizes
        url = f"https://huggingface.co/{hf_model_id}/raw/main/model.safetensors.index.json"
        req = urllib.request.Request(url, headers={"User-Agent": "llm-infer-tuner-sync"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            index = json.loads(resp.read().decode("utf-8"))
        metadata = index.get("metadata", {})
        total_size = metadata.get("total_size", 0)
        if total_size > 0:
            gb = total_size / (1024 ** 3)
            # Round to nearest GB
            weights["estimated"] = round(gb)
    except Exception:
        pass
    return weights

def extract_model_info(config: dict) -> dict[str, Any]:
    """Extract relevant fields from HF config.json."""
    text_config = config.get("text_config", {})

    def get(key, default=None):
        return config.get(key, text_config.get(key, default))

    is_moe = get("is_moe", None)
    num_experts = get("num_experts", None)
    n_routed_experts = get("n_routed_experts", None)
    moe_intermediate_size = get("moe_intermediate_size", None)
    intermediate_size = get("intermediate_size", None)

    # MoE detection: check is_moe, then num_experts, then n_routed_experts,
    # then moe_intermediate_size, then shared_expert in modules_to_not_convert
    if is_moe is None and (num_experts is not None or n_routed_experts is not None):
        is_moe = True
    if num_experts is None and n_routed_experts is not None:
        num_experts = n_routed_experts
    # Some models have moe_intermediate_size but not is_moe flag
    if is_moe is None and moe_intermediate_size is not None:
        is_moe = True
    # Check shared_expert or .gate in modules_to_not_convert
    if is_moe is None:
        quant_config = config.get("quantization_config", text_config.get("quantization_config", {}))
        modules = quant_config.get("modules_to_not_convert", []) if quant_config else []
        if modules and any("shared_expert" in str(m) or ".gate" in str(m) for m in modules):
            is_moe = True
            if num_experts is None:
                gate_count = sum(1 for m in modules if ".gate" in str(m) and "shared" not in str(m))
                if gate_count > 0:
                    num_experts = gate_count

    # hybrid_mamba / GDN detection: check layer_types for linear_attention,
    # mamba_ssm_dtype, linear_num_value_heads, or gated_deltanet in architecture
    layer_types = get("layer_types", None)
    has_linear_attn = False
    if layer_types and isinstance(layer_types, list):
        has_linear_attn = any("linear" in str(lt).lower() for lt in layer_types)
    has_mamba_ssm = get("mamba_ssm_dtype", None) is not None
    has_linear_heads = get("linear_num_value_heads", None) is not None
    has_gdn = has_linear_attn or has_mamba_ssm or has_linear_heads

    # num_experts_per_tok
    num_experts_per_tok = get("num_experts_per_tok", None)
    if num_experts_per_tok is None and n_routed_experts is not None:
        num_experts_per_tok = get("num_experts_per_tok", None)

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
        "n_routed_experts": n_routed_experts,
        "moe_intermediate_size": moe_intermediate_size,
        "has_gdn": has_gdn,
        "num_experts_per_tok": num_experts_per_tok,
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
    """Save models.yaml: refresh version/updated/total + add comment dividers."""
    data["version"] = data.get("version", 1) + 1
    data["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["total"] = len(data.get("models", {}))

    # Build yaml manually with comment dividers between models
    lines = [
        f"version: {data['version']}",
        f"updated: {data['updated']}",
        f"total: {data['total']}",
        "models:",
    ]
    for key, val in data.get("models", {}).items():
        hf_id = val.get("hf_model_id", key)
        model_name = hf_id.split("/")[-1] if "/" in hf_id else hf_id
        arch = val.get("arch", "unknown")
        prec = val.get("default_precision", "unknown")
        # Comment divider
        lines.append(f"  # ── {key}: {model_name} ({arch}, {prec}) ─" + "─" * 20)
        # Dump this single model entry
        single = {key: val}
        dumped = yaml.dump(single, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)
        for dline in dumped.splitlines():
            if dline:
                lines.append("  " + dline)
        lines.append("")
    with open(MODELS_YAML, "w", encoding="utf-8") as f:
        f.write(chr(10).join(lines) + chr(10))


def make_model_key(hf_id: str, model_index: int) -> str:
    """Convert HF model ID to a yaml key with M{NN}_ prefix.
    e.g. (Qwen/Qwen3.6-27B-FP8, 4) -> M04_qwen36-27b-fp8
    """
    name = hf_id.split("/")[-1]
    key = name.lower()
    key = re.sub(r"[^a-z0-9]", "-", key)
    key = re.sub(r"-+", "-", key)
    key = key.strip("-")
    key = key.replace("3-6", "36")
    return f"M{model_index:02d}_{key}"



def find_cookbook_mdx(sglang_repo: str, hf_id: str) -> Path | None:
    """Find the cookbook mdx file that matches this model.
    Priority: 1) filename match, 2) --model-path exact match, 3) content fuzzy.
    """
    repo = Path(sglang_repo)
    model_name = hf_id.split("/")[-1]
    model_name_clean = re.sub(r"[^a-zA-Z0-9]", "", model_name).lower()

    candidates: list[tuple[int, Path]] = []  # (priority, path)

    for cookbook_dir in COOKBOOK_DIRS:
        search_dir = repo / cookbook_dir
        if not search_dir.exists():
            continue
        for mdx in search_dir.rglob("*.mdx"):
            fname_clean = re.sub(r"[^a-zA-Z0-9]", "", mdx.stem).lower()
            try:
                text = mdx.read_text(encoding="utf-8")
            except OSError:
                continue
            # Priority 1: filename closely matches model name
            if model_name_clean in fname_clean or fname_clean in model_name_clean:
                if len(model_name_clean) > 3:  # avoid short matches
                    candidates.append((1, mdx))
                    continue
            # Priority 2: --model-path with exact HF ID in text
            if f"--model-path {hf_id}" in text or f"--model-path={hf_id}" in text:
                candidates.append((2, mdx))
                continue
            # Priority 3: HF ID appears in text (less precise)
            if hf_id in text:
                candidates.append((3, mdx))
                continue

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    return None


def extract_default_flags_from_cookbook(mdx_path: Path) -> dict[str, Any]:
    """Extract --reasoning-parser, --tool-call-parser, --trust-remote-code from cookbook."""
    try:
        text = mdx_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    flags: dict[str, Any] = {}
    # reasoning-parser
    m = re.search(r'--reasoning-parser\s+(\S+)', text)
    if m:
        flags["reasoning-parser"] = m.group(1).strip('\"')
    # tool-call-parser
    m = re.search(r'--tool-call-parser\s+(\S+)', text)
    if m:
        flags["tool-call-parser"] = m.group(1).strip('\"')
    # trust-remote-code
    if "--trust-remote-code" in text:
        flags["trust-remote-code"] = True
    return flags


def extract_mtp_params_from_cookbook(mdx_path: Path) -> dict[str, Any]:
    """Extract speculative decoding params from cookbook."""
    try:
        text = mdx_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    params: dict[str, Any] = {}
    # Restrict the value to an identifier. Cookbook prose/code fences often
    # leave a trailing backtick or colon, which used to pollute models.yaml
    # with values such as "EAGLE3`:".
    m = re.search(r'''--speculative-algorithm(?:\s+|=)["']?([A-Za-z0-9_-]+)''', text)
    if m:
        algorithm = m.group(1).upper()
        params["speculative-algorithm"] = algorithm
    # speculative-num-steps
    m = re.search(r'--speculative-num-steps\s+(\d+)', text)
    if m:
        params["speculative-num-steps"] = int(m.group(1))
    # speculative-eagle-topk
    m = re.search(r'--speculative-eagle-topk\s+(\d+)', text)
    if m:
        params["speculative-eagle-topk"] = int(m.group(1))
    # speculative-num-draft-tokens
    m = re.search(r'--speculative-num-draft-tokens\s+(\d+)', text)
    if m:
        params["speculative-num-draft-tokens"] = int(m.group(1))
    if params.get("speculative-algorithm") not in KNOWN_SPECULATIVE_ALGORITHMS:
        params.pop("speculative-algorithm", None)
    return params


def extract_weight_gb_from_cookbook(mdx_path: Path, model_name: str) -> dict[str, int]:
    """Extract weight sizes from cookbook. Handles multiple formats."""
    try:
        text = mdx_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    weights: dict[str, int] = {}
    name_lower = model_name.lower()
    # Pattern 1: "27B FP8: ~27GB for weights"
    for prec in ["bf16", "fp8", "nvfp4"]:
        pattern = rf'{re.escape(name_lower)}\s+{prec}[^~]*~(\d+)\s*GB'
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            weights[prec] = int(m.group(1))
    # Pattern 2: "FP8 weights ~28.5GB" or "NVFP4 weights ~16.5GB"
    for prec in ["fp8", "bf16", "nvfp4"]:
        pattern = rf'{prec}\s+weights\s*~?(\d+(?:\.\d+)?)\s*GB'
        m = re.search(pattern, text, re.IGNORECASE)
        if m and prec not in weights:
            weights[prec] = int(float(m.group(1)))
    return weights

def generate_new_model_card(hf_id: str, info: dict, sglang_repo: str = "") -> dict[str, Any]:
    """Generate a complete model card from HF config + cookbook."""
    is_moe = info.get("is_moe", False)
    arch = "moe" if is_moe else "dense"
    model_name = hf_id.split("/")[-1]

    # Try to find cookbook mdx for this model
    cookbook_flags = {}
    cookbook_mtp = {}
    cookbook_weights = {}
    if sglang_repo:
        mdx = find_cookbook_mdx(sglang_repo, hf_id)
        if mdx:
            print(f"    [cookbook] found: {mdx.name}")
            cookbook_flags = extract_default_flags_from_cookbook(mdx)
            cookbook_mtp = extract_mtp_params_from_cookbook(mdx)
            cookbook_weights = extract_weight_gb_from_cookbook(mdx, model_name)
        else:
            print(f"    [cookbook] not found for {hf_id}")

    # Detect quantization scheme
    quant_method = info.get("quant_method", "none")
    block_size = info.get("block_size")
    scheme = "fine-grained" if block_size else "per-tensor"

    # Detect arch subtype from architectures/model_type + has_gdn
    model_type = info.get("model_type", "")
    architectures = info.get("architectures", [])
    arch_str = " ".join(architectures).lower() if architectures else model_type.lower()
    has_gdn = info.get("has_gdn", False)
    if has_gdn or "mamba" in arch_str or "gdn" in arch_str or "deltanet" in arch_str or "linear" in arch_str:
        arch = "moe_hybrid_gdn" if is_moe else "dense_hybrid_gdn"
    elif "deepseek" in arch_str or "dsv" in arch_str or "mla" in arch_str:
        arch = "moe"
    elif is_moe:
        arch = "moe"
    else:
        arch = "dense"

    card: dict[str, Any] = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "hf_model_id": hf_id,
        "family": model_name.split("-")[0].lower(),
        "parameter_count_b": None,  # needs manual fill from cookbook
        "arch": arch,
        "hybrid_mamba": has_gdn or "hybrid" in arch or "gdn" in arch,
        "default_precision": quant_method if quant_method != "none" else "bf16",
    }

    # weight_gb: try cookbook first, then HF API, then estimate from params
    if cookbook_weights:
        card["weight_gb"] = cookbook_weights
    else:
        hf_weights = fetch_weight_gb(hf_id)
        if hf_weights:
            card["weight_gb"] = hf_weights
        else:
            # Estimate from parameter count: fp8=2 bytes/param, bf16=4 bytes/param
            num_layers = info.get("num_hidden_layers", 0) or 0
            hidden = info.get("hidden_size", 0) or 0
            intermediate = info.get("intermediate_size", 0) or 0
            num_exp = info.get("num_experts") or 0
            if num_layers and hidden:
                # Rough: attention + MLP params per layer
                attn_params = 2 * hidden * hidden  # Q + KV (simplified)
                mlp_params = 3 * hidden * intermediate  # gate + up + down
                if num_exp and is_moe:
                    mlp_params = num_exp * mlp_params
                total_params = num_layers * (attn_params + mlp_params) + hidden * (info.get("vocab_size") or 0)
                prec = quant_method if quant_method != "none" else "bf16"
                bytes_per_param = 2 if prec in ("fp8", "fp4", "nvfp4") else 4
                est_gb = total_params * bytes_per_param / (1024**3)
                card["weight_gb"] = {"estimated": round(est_gb)}
            else:
                card["weight_gb"] = {}

    # Modalities
    has_vision = bool(info.get("architectures")) and any("VL" in a or "Vision" in a or "Conditional" in a for a in info.get("architectures", []))
    card["modalities"] = {
        "input": ["text", "image"] if has_vision else ["text"],
        "output": ["text"],
    }

    # Quantization
    card["quantization"] = {
        "method": quant_method,
        "scheme": scheme,
    }
    if block_size:
        card["quantization"]["block_size"] = block_size
    if quant_method == "fp8":
        card["quantization"]["weight_dtype"] = "fp8"
        card["quantization"]["activation_dtype"] = "bf16"

    # NVFP4 support
    if "nvfp4" in model_name.lower():
        card["nvfp4_requires_sm"] = 10  # cookbook: NVFP4 requires B200/B300 (SM100+)
    # Architecture
    arch_data: dict[str, Any] = {
        "is_moe": is_moe,
        "num_hidden_layers": info.get("num_hidden_layers"),
        "hidden_size": info.get("hidden_size"),
        "vocab_size": info.get("vocab_size"),
        "intermediate_size": info.get("intermediate_size"),
    }
    if is_moe:
        arch_data["moe_intermediate_size"] = info.get("moe_intermediate_size")
        arch_data["num_experts"] = info.get("num_experts")
        card["num_experts"] = info.get("num_experts")
        card["num_experts_per_tok"] = info.get("num_experts_per_tok")
        card["num_experts_source"] = "official"  # from HF config.json
    card["architecture"] = arch_data

    # KV/token rough estimate (reference only)
    if is_moe:
        card["kv_gb_per_token"] = 0.00003  # MoE with GDN: most layers linear, small KV
    else:
        card["kv_gb_per_token"] = 0.00012  # dense standard estimate
    card["kv_gb_per_token_source"] = "reference"

    # Context
    if info.get("max_position_embeddings"):
        card["context"] = {
            "native_context_length": info["max_position_embeddings"],
            "maximum_context_length": info["max_position_embeddings"],
            "recommended_initial_context_length": min(131072, info["max_position_embeddings"]),
        }

    # Capabilities
    has_mtp = info.get("has_mtp", False) or bool(cookbook_mtp)
    card["capabilities"] = {
        "supports_reasoning": bool(cookbook_flags.get("reasoning-parser")),
        "supports_chat_template": True,
        "supports_tool_call": bool(cookbook_flags.get("tool-call-parser")),
        "supports_mtp": has_mtp,
    }

    # default_flags from cookbook
    if cookbook_flags:
        card["default_flags"] = cookbook_flags
    else:
        card["default_flags"] = {}
        if has_vision or "qwen" in model_name.lower():
            card["default_flags"]["trust-remote-code"] = True

    # mtp_params from cookbook (kept for backward compatibility). The
    # structured speculative_options field is the preferred extensible form.
    if cookbook_mtp:
        card["mtp_params"] = cookbook_mtp
        algorithm = cookbook_mtp.get("speculative-algorithm")
        if algorithm:
            card["speculative_options"] = [{
                "algorithm": algorithm,
                "params": dict(cookbook_mtp),
            }]

    # Deployment
    weight_key = card.get("default_precision", "fp8")
    if weight_key == "none":
        weight_key = "bf16"
    card["deployment"] = {
        "model_format": "huggingface",
        "weight_size_gb": cookbook_weights.get(weight_key) or card.get("weight_gb", {}).get("estimated"),
        "model_path": None,  # fill after haihub download
    }

    card["source"] = f"https://huggingface.co/{hf_id}"
    card["notes"] = f"Auto-synced from HF config.json + SGLang cookbook."

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
            card = generate_new_model_card(hf_id, hf_info, args.sglang_repo)
            model_index = len(models) + 1
            model_key = make_model_key(hf_id, model_index)
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
            # Update missing fields on existing models
            for mk, mv in models.items():
                if mv.get("hf_model_id") == hf_id:
                    updated = False
                    # Update hybrid_mamba from has_gdn
                    has_gdn = hf_info.get("has_gdn", False)
                    if has_gdn and not mv.get("hybrid_mamba"):
                        mv["hybrid_mamba"] = True
                        is_moe_val = mv.get("architecture",{}).get("is_moe")
                        mv["arch"] = "moe_hybrid_gdn" if is_moe_val else "dense_hybrid_gdn"
                        updated = True
                        print(f"    [updated] hybrid_mamba=True, arch=" + str(mv.get("arch")))
                    mdx = find_cookbook_mdx(args.sglang_repo, hf_id) if args.sglang_repo else None
                    if mdx:
                        # Update default_flags from cookbook if missing.
                        ck_flags_obj = mv.get("default_flags", {})
                        has_real_flags = any(v for v in ck_flags_obj.values() if v is not True and v) or (ck_flags_obj.get("trust-remote-code") == True and len(ck_flags_obj) > 1)
                        if not has_real_flags or len(ck_flags_obj) <= 1:
                            ck_flags = extract_default_flags_from_cookbook(mdx)
                            if ck_flags and len(ck_flags) > len(ck_flags_obj):
                                mv["default_flags"] = ck_flags
                                updated = True
                                print(f"    [updated] default_flags={ck_flags}")

                        # Always inspect MTP metadata. This also repairs old cards
                        # polluted by cookbook punctuation such as "EAGLE3`:".
                        ck_mtp = extract_mtp_params_from_cookbook(mdx)
                        current_mtp = mv.get("mtp_params", {})
                        current_algorithm = current_mtp.get("speculative-algorithm") if isinstance(current_mtp, dict) else None
                        cookbook_algorithm = ck_mtp.get("speculative-algorithm")
                        if ck_mtp and (not current_mtp or current_algorithm not in KNOWN_SPECULATIVE_ALGORITHMS):
                            mv["mtp_params"] = ck_mtp
                            if cookbook_algorithm:
                                mv["speculative_options"] = [{
                                    "algorithm": cookbook_algorithm,
                                    "params": dict(ck_mtp),
                                }]
                            updated = True
                            print(f"    [updated] mtp_params={ck_mtp}")

                        ck_weights = extract_weight_gb_from_cookbook(mdx, hf_id.split("/")[-1])
                        if ck_weights and not any(v for v in mv.get("weight_gb",{}).values() if v):
                            mv["weight_gb"] = ck_weights
                            updated = True
                            print(f"    [updated] weight_gb={ck_weights}")
                    if updated:
                        mv["last_updated"] = datetime.now().strftime("%Y-%m-%d")
                        models_changed = True
                    break
            if diffs:
                for d in diffs:
                    print(f"    [diff] {d}")
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
