"""Sync SGLang server parameters from source code into images.yaml.

This script clones (or updates) the SGLang repo at a given tag, parses
server_args.py with AST to extract:
  - All CLI argument names -> valid_flags
  - attention_backend choices -> attention_backends menu
  - __post_init__ assert/raise -> constraints report

Then compares with the current images.yaml and updates if changed.

Usage:
  python scripts/sync_sglang_params.py [--tag v0.5.16] [--repo-dir /tmp/sglang]
  python scripts/sync_sglang_params.py --tag v0.5.17  # new release

No AI needed. Pure deterministic AST extraction.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_YAML = PROJECT_ROOT / ".claude/skills/sglang-server-config-gen/images.yaml"
SGLANG_REPO = "https://github.com/sgl-project/sglang.git"
SERVER_ARGS_REL = "python/sglang/srt/server_args.py"


def clone_or_update(tag: str, repo_dir: Path) -> Path:
    """Clone SGLang repo at a specific tag, or update if already cloned."""
    server_args = repo_dir / SERVER_ARGS_REL
    if server_args.exists():
        # Already cloned, just checkout the tag
        subprocess.run(["git", "fetch", "--tags", "--depth=1", "origin"],
                       cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "checkout", tag], cwd=repo_dir, check=True, capture_output=True)
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth=1", "--branch", tag,
                        SGLANG_REPO, str(repo_dir)],
                       check=True, capture_output=True)
    return server_args


def extract_valid_flags(server_args_path: Path) -> list[str]:
    """Extract all --flag names from add_argument calls via AST."""
    source = server_args_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    flags: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Look for .add_argument("--foo", ...) calls
            if (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
                if node.args and isinstance(node.args[0], ast.Constant):
                    arg_name = node.args[0].value
                    if isinstance(arg_name, str) and arg_name.startswith("--"):
                        flag = arg_name.lstrip("-").replace("-", "_")
                        flags.append(flag)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return sorted(unique)


def extract_attention_backends(server_args_path: Path) -> list[str]:
    """Extract attention_backend choices from the add_argument call."""
    source = server_args_path.read_text(encoding="utf-8")
    # Find the attention_backend add_argument with choices
    # Pattern: add_argument("--attention-backend", ..., choices=[...])
    pattern = r'add_argument\(\s*["\']--attention-backend["\'].*?choices=\[([^\]]+)\]'
    match = re.search(pattern, source, re.DOTALL)
    if not match:
        return []
    raw = match.group(1)
    # Extract quoted strings
    backends = re.findall(r'["\']([^"\']+)["\']', raw)
    return sorted(backends)


def extract_constraints(server_args_path: Path) -> list[dict[str, str]]:
    """Extract assert/raise statements from __post_init__ for constraint report."""
    source = server_args_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    constraints: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            text = ast.get_source_segment(source, node)
            if text and len(text) < 500:
                constraints.append({"type": "assert", "text": text.strip()})
        elif isinstance(node, ast.Raise):
            text = ast.get_source_segment(source, node)
            if text and len(text) < 500:
                # Only keep raises with ValueError/AssertionError messages
                if "ValueError" in text or "AssertionError" in text or "raise" in text:
                    constraints.append({"type": "raise", "text": text.strip()})
    return constraints


def load_images_yaml() -> dict[str, Any]:
    """Load current images.yaml."""
    with open(IMAGES_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def update_image_entry(data: dict, image_key: str, valid_flags: list[str],
                        attention_backends: list[str]) -> bool:
    """Update a specific image entry in images.yaml. Returns True if changed."""
    images = data.get("images", {})
    if image_key not in images:
        print(f"  [skip] {image_key} not in images.yaml")
        return False
    entry = images[image_key]
    changed = False

    # Update valid_flags
    current_flags = set(entry.get("valid_flags", []))
    new_flags = set(valid_flags)
    if current_flags != new_flags:
        added = new_flags - current_flags
        removed = current_flags - new_flags
        if added:
            print(f"  [flags] +{len(added)} new: {sorted(added)[:10]}...")
        if removed:
            print(f"  [flags] -{len(removed)} removed: {sorted(removed)[:10]}...")
        entry["valid_flags"] = sorted(new_flags)
        changed = True

    # Update attention_backends
    current_backends = set(entry.get("attention_backends", []))
    new_backends = set(attention_backends)
    if current_backends != new_backends:
        added = new_backends - current_backends
        removed = current_backends - new_backends
        if added:
            print(f"  [backends] +{sorted(added)}")
        if removed:
            print(f"  [backends] -{sorted(removed)}")
        entry["attention_backends"] = sorted(new_backends)
        changed = True

    return changed


def save_images_yaml(data: dict) -> None:
    """Save updated images.yaml."""
    with open(IMAGES_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def save_constraints_report(constraints: list[dict], tag: str) -> None:
    """Save constraints report for human review."""
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"constraints_{tag}.md"
    lines = [f"# SGLang {tag} Constraints Report\n",
             f"Auto-extracted from server_args.py ({len(constraints)} assert/raise)\n",
             "Review and update knowledge.md section 5 if new constraints found.\n\n"]
    for i, c in enumerate(constraints, 1):
        lines.append(f"## {i}. [{c['type']}]\n```
{c['text']}
```\n")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [report] {report_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync SGLang params from source code")
    parser.add_argument("--tag", default="v0.5.16", help="SGLang git tag")
    parser.add_argument("--repo-dir", default="/tmp/sglang-sync",
                        help="Local dir for SGLang repo clone")
    args = parser.parse_args(argv)

    tag = args.tag
    image_key = f"sglang-{tag.lstrip('v')}"
    print(f"=== Syncing SGLang {tag} -> images.yaml[{image_key}] ===")

    # Step 1: Clone/update repo
    print(f"[1/4] Cloning SGLang {tag}...")
    repo_dir = Path(args.repo_dir)
    server_args_path = clone_or_update(tag, repo_dir)
    print(f"  -> {server_args_path}")

    # Step 2: Extract from source
    print("[2/4] Extracting parameters via AST...")
    valid_flags = extract_valid_flags(server_args_path)
    attention_backends = extract_attention_backends(server_args_path)
    constraints = extract_constraints(server_args_path)
    print(f"  valid_flags: {len(valid_flags)}")
    print(f"  attention_backends: {len(attention_backends)} -> {attention_backends}")
    print(f"  constraints: {len(constraints)} assert/raise")

    # Step 3: Compare and update images.yaml
    print(f"[3/4] Updating {IMAGES_YAML.relative_to(PROJECT_ROOT)}...")
    data = load_images_yaml()
    changed = update_image_entry(data, image_key, valid_flags, attention_backends)
    if changed:
        save_images_yaml(data)
        print("  -> images.yaml updated")
    else:
        print("  -> no changes needed")

    # Step 4: Save constraints report
    print("[4/4] Saving constraints report...")
    save_constraints_report(constraints, tag)

    # Summary
    print("\n=== Summary ===")
    print(f"  tag: {tag}")
    print(f"  image_key: {image_key}")
    print(f"  valid_flags: {len(valid_flags)}")
    print(f"  attention_backends: {attention_backends}")
    print(f"  constraints: {len(constraints)}")
    print(f"  images_yaml_changed: {changed}")
    print(f"  constraints_report: reports/constraints_{tag}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
