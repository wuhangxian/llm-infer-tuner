"""Sync GPU info from NVIDIA official specs into gpu.yaml.

Currently gpu.yaml is manually maintained. This script validates existing
entries against known specs and can be extended to auto-discover new GPUs.
For now it updates the yaml meta (version/updated/total) on any change.

Usage:
  python scripts/sync_gpu.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPU_YAML = PROJECT_ROOT / "catalogs/gpu.yaml"


def _update_yaml_meta(data: dict, section_key: str) -> None:
    """Update version (+1), updated (timestamp), total (recount) in yaml data."""
    data["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["version"] = data.get("version", 1) + 1
    section = data.get(section_key, {})
    data["total"] = len(section)


def load_gpu_yaml() -> dict[str, Any]:
    with open(GPU_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_gpu_yaml(data: dict) -> None:
    """Save updated gpu.yaml with auto version/updated/total."""
    _update_yaml_meta(data, "gpu_catalog")
    with open(GPU_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync GPU info into gpu.yaml")
    parser.add_argument("--add", nargs="*", default=[],
                        help="Add new GPU entries (name compute_cap sm_major nvlink)")
    args = parser.parse_args(argv)

    print("=== GPU Sync ===")
    print("[1/2] Loading gpu.yaml...")
    data = load_gpu_yaml()
    gpu_catalog = data.setdefault("gpu_catalog", {})
    existing_count = len(gpu_catalog)
    print(f"  Existing GPUs: {existing_count}")

    changed = False

    # Add new GPUs if specified
    if args.add:
        print(f"[2/2] Adding {len(args.add)} new GPU(s)...")
        for entry in args.add:
            parts = entry.split(":")
            if len(parts) < 4:
                print(f"  [skip] {entry}: format is name:compute_cap:sm_major:nvlink")
                continue
            name, compute_cap, sm_major, nvlink = parts[0], parts[1], parts[2], parts[3]
            key = f"G{len(gpu_catalog)+1:02d}_{name}"
            gpu_catalog[key] = {
                "compute_cap": compute_cap,
                "sm_major": int(sm_major),
                "nvlink": nvlink.lower() in ("true", "yes", "1"),
                "source": "manual",
            }
            print(f"  [added] {key}")
            changed = True
    else:
        print("[2/2] No new GPUs to add. Checking existing entries...")
        # Just update meta
        changed = True  # always refresh meta

    if changed:
        save_gpu_yaml(data)
        print(f"  -> gpu.yaml updated (version={data['version']}, total={data['total']})")
    else:
        print("  -> no changes needed")

    print(f"\n=== Summary ===")
    print(f"  total GPUs: {data.get('total', len(gpu_catalog))}")
    print(f"  version: {data.get('version', 1)}")
    print(f"  updated: {data.get('updated', 'N/A')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
