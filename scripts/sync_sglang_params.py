"""Sync SGLang server parameters from source code into catalogs/sglang-images.yaml.

Automatically scans all SGLang git tags, discovers new versions not yet in
sglang-images.yaml, and updates existing entries when parameters change.

For each tag:
  - Clones (or updates) the SGLang repo at that tag
  - Parses server_args.py/spec_info.py to extract valid_flags, attention_backends,
    and built-in speculative algorithm names
  - Extracts assert/raise constraints for a human-review report
  - Updates sglang-images.yaml if changed, or creates a new entry for new versions

Usage:
  python scripts/sync_sglang_params.py
  python scripts/sync_sglang_params.py --tags v0.5.10 v0.5.16  # specific tags only

No AI needed. Pure deterministic AST extraction.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from datetime import datetime
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
IMAGES_YAML = PROJECT_ROOT / "catalogs" / "sglang-images.yaml"
SGLANG_REPO = "https://github.com/sgl-project/sglang.git"
SERVER_ARGS_REL = "python/sglang/srt/server_args.py"
SPEC_INFO_REL = "python/sglang/srt/speculative/spec_info.py"


def list_all_tags() -> list[str]:
    """List all SGLang git tags (v0.5.x release tags only)."""
    result = subprocess.run(
        ["git", "ls-remote", "--tags", SGLANG_REPO],
        capture_output=True, text=True, check=True, timeout=60,
    )
    tags: list[str] = []
    for line in result.stdout.splitlines():
        # Format: <sha>	refs/tags/v0.5.10
        if "refs/tags/" in line:
            tag = line.split("refs/tags/")[-1].strip()
            # Only keep release tags like v0.5.x (skip gateway-*, nightly, etc.)
            if re.match(r"^v\d+\.\d+\.\d+$", tag):
                tags.append(tag)
    return sorted(tags, key=lambda t: [int(x) for x in t.lstrip("v").split(".")])


def get_existing_image_keys() -> list[str]:
    """Get image keys already in sglang-images.yaml."""
    with open(IMAGES_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data.get("images", {}).keys())


def resolve_image_key(data: dict[str, Any], tag: str) -> str:
    """Reuse an existing catalog key for a tag, including legacy Ixx keys."""
    version = tag.lstrip("v")
    images = data.get("images", {})
    for key, entry in images.items():
        if str(entry.get("sglang_version", "")) == version:
            return key
        image_ref = str(entry.get("image_ref", ""))
        if re.search(rf":v?{re.escape(version)}(?:[-:]|$)", image_ref):
            return key
    return f"sglang-{version}"


def clone_or_update(tag: str, repo_dir: Path) -> Path:
    """Clone SGLang repo at a specific tag."""
    server_args = repo_dir / SERVER_ARGS_REL
    if server_args.exists():
        subprocess.run(["git", "fetch", "--tags", "--depth=1", "origin"],
                       cwd=repo_dir, check=True, capture_output=True, timeout=60)
        subprocess.run(["git", "checkout", tag], cwd=repo_dir, check=True, capture_output=True)
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth=1", "--branch", tag,
                        SGLANG_REPO, str(repo_dir)],
                       check=True, capture_output=True, timeout=120)
    return server_args


def extract_valid_flags(server_args_path: Path) -> list[str]:
    """Extract all valid flag names from server_args.py.

    Handles two syntaxes across SGLang versions:
    1. Legacy:  parser.add_argument("--flag-name", ...)
    2. Modern:  flag_name: A[Optional[str], Arg(...)]  (AnnAssign + A[] subscript)
    """
    source = server_args_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    flags: list[str] = []

    # 1) Legacy: add_argument("--flag-name", ...)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant):
                    arg_name = node.args[0].value
                    if isinstance(arg_name, str) and arg_name.startswith("--"):
                        flag = arg_name.lstrip("-").replace("-", "_")
                        flags.append(flag)

    # 2) Modern: flag_name: A[Optional[str], Arg(...)]
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            ann = node.annotation
            if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name) and ann.value.id == "A":
                target_id = getattr(node.target, "id", "")
                if target_id and not target_id.startswith("_"):
                    flags.append(target_id)

    seen: set[str] = set()
    unique: list[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return sorted(unique)


def extract_attention_backends(server_args_path: Path) -> list[str]:
    """Extract attention_backend choices from server_args.py.

    Handles three syntaxes across SGLang versions:
    1. Legacy:  parser.add_argument("--attention-backend", choices=[...])
    2. Modern:  attention_backend: A[Optional[str], Arg(choices=VAR_NAME, ...)]
    3. Inline:  choices=["triton", ...] inside Arg()
    """
    source = server_args_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    def _resolve_list_var(var_name: str) -> list[str]:
        """Resolve a module-level list variable, including .extend() calls."""
        backends: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        if isinstance(node.value, ast.List):
                            backends.update(
                                el.value for el in node.value.elts
                                if isinstance(el, ast.Constant) and isinstance(el.value, str)
                            )
        # Collect .extend([...]) additions on the variable
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if (isinstance(f, ast.Attribute) and f.attr == "extend"
                        and isinstance(f.value, ast.Name) and f.value.id == var_name):
                    if node.args and isinstance(node.args[0], ast.List):
                        backends.update(
                            el.value for el in node.args[0].elts
                            if isinstance(el, ast.Constant) and isinstance(el.value, str)
                        )
        return sorted(backends)

    def _extract_from_arg_call(call_node: ast.Call) -> list[str] | None:
        """Extract choices from an Arg() or add_argument() call."""
        for kw in call_node.keywords:
            if kw.arg == "choices":
                if isinstance(kw.value, ast.List):
                    return sorted({
                        el.value for el in kw.value.elts
                        if isinstance(el, ast.Constant) and isinstance(el.value, str)
                    })
                if isinstance(kw.value, ast.Name):
                    return _resolve_list_var(kw.value.id)
        return None

    # 1) Legacy: add_argument("--attention-backend", ..., choices=...)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant):
                    if node.args[0].value == "--attention-backend":
                        result = _extract_from_arg_call(node)
                        if result is not None:
                            return result

    # 2) Modern: attention_backend: A[Optional[str], Arg(choices=VAR_NAME, ...)]
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            target_id = getattr(node.target, "id", "")
            if target_id == "attention_backend":
                for child in ast.walk(node.annotation):
                    if isinstance(child, ast.Call):
                        f = child.func
                        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
                        if name in ("Arg", "Argument"):
                            result = _extract_from_arg_call(child)
                            if result is not None:
                                return result

    # 3) Fallback: regex on choices=VAR_NAME anywhere in source
    var_match = re.search(r'choices=([A-Z_][A-Z0-9_]*)', source)
    if var_match:
        result = _resolve_list_var(var_match.group(1))
        if result:
            return result

    return []


def extract_speculative_algorithms(server_args_path: Path) -> list[str]:
    """Extract built-in speculative algorithm names from spec_info.py.

    The image catalog records names recognized by the engine. Model-specific
    draft weights and parameters remain in models.yaml and are intersected at
    generation time.
    """
    repo_root = server_args_path.parents[3]
    spec_info_path = repo_root / SPEC_INFO_REL
    if not spec_info_path.exists():
        return []
    source = spec_info_path.read_text(encoding="utf-8")
    match = re.search(
        r"class\s+SpeculativeAlgorithm\b(?P<body>.*?)(?=^class\s|\Z)",
        source,
        flags=re.DOTALL | re.MULTILINE,
    )
    if not match:
        return []
    names = re.findall(
        r"^\s{4}([A-Z][A-Z0-9_]*)\s*=\s*auto\(\)",
        match.group("body"),
        re.MULTILINE,
    )
    return list(dict.fromkeys(names))



def extract_constraints(server_args_path: Path) -> list[dict[str, str]]:
    """Extract assert/raise statements for constraint report."""
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
            if text and len(text) < 500 and ("ValueError" in text or "AssertionError" in text):
                constraints.append({"type": "raise", "text": text.strip()})
    return constraints


def load_images_yaml() -> dict[str, Any]:
    with open(IMAGES_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_images_yaml(data: dict) -> None:
    data["total"] = len(data.get("images", {}))
    with open(IMAGES_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def save_constraints_report(constraints: list[dict], tag: str) -> None:
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"constraints_{tag}.md"
    lines = [f"# SGLang {tag} Constraints Report\n",
             f"Auto-extracted from server_args.py ({len(constraints)} assert/raise)\n",
             "Review and update .claude/skills/sglang-server-config-gen/references/rules/\n"
             "attention.yaml or speculative.yaml if new constraints are found.\n\n"]
    for i, c in enumerate(constraints, 1):
        lines.append("## " + str(i) + ". [" + c["type"] + "]\n```\n" + c["text"] + "\n```\n")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def process_tag(tag: str, repo_dir: Path, data: dict) -> bool:
    """Process a single tag. Returns True if sglang-images.yaml was changed."""
    image_key = resolve_image_key(data, tag)
    is_new = image_key not in data.get("images", {})

    if is_new:
        print(f"  [NEW] {image_key} — cloning and extracting...")
    else:
        print(f"  [EXISTING] {image_key} — checking for changes...")

    try:
        server_args_path = clone_or_update(tag, repo_dir)
    except subprocess.CalledProcessError as exc:
        print(f"  [error] clone failed: {exc}")
        return False

    valid_flags = extract_valid_flags(server_args_path)
    attention_backends = extract_attention_backends(server_args_path)
    speculative_algorithms = extract_speculative_algorithms(server_args_path)
    constraints = extract_constraints(server_args_path)

    print(f"    valid_flags: {len(valid_flags)}")
    print(f"    attention_backends: {len(attention_backends)} -> {attention_backends}")
    print(f"    speculative_algorithms: {len(speculative_algorithms)} -> {speculative_algorithms}")
    print(f"    constraints: {len(constraints)}")

    save_constraints_report(constraints, tag)

    images = data.setdefault("images", {})
    changed = False

    if is_new:
        # Create new entry with defaults
        images[image_key] = {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "image_ref": f"lmsysorg/sglang:{tag}",
            "sglang_version": tag.lstrip("v"),
            "cuda_version": None,  # needs manual fill or Dockerfile inspection
            "digest": None,
            "attention_backends": attention_backends,
            "speculative_algorithms": speculative_algorithms,
            "startup_floor": {
                "verified": None,
                "source": None,
            },
            "flag_aliases": {},
            "valid_flags": valid_flags,
            "_auto_generated": True,  # mark for manual review
        }
        print(f"  -> Created new entry {image_key}")
        print(f"  -> [TODO] Fill cuda_version, digest, startup_floor manually")
        changed = True
    else:
        entry = images[image_key]
        current_flags = set(entry.get("valid_flags", []))
        new_flags = set(valid_flags)
        if current_flags != new_flags:
            added = new_flags - current_flags
            removed = current_flags - new_flags
            if added:
                print(f"  [flags] +{len(added)} new: {sorted(added)[:10]}")
            if removed:
                print(f"  [flags] -{len(removed)} removed: {sorted(removed)[:10]}")
            entry["valid_flags"] = sorted(new_flags)
            changed = True

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

        current_algorithms = set(entry.get("speculative_algorithms", []))
        new_algorithms = set(speculative_algorithms)
        if new_algorithms and current_algorithms != new_algorithms:
            added = new_algorithms - current_algorithms
            removed = current_algorithms - new_algorithms
            if added:
                print(f"  [speculative] +{sorted(added)}")
            if removed:
                print(f"  [speculative] -{sorted(removed)}")
            entry["speculative_algorithms"] = sorted(new_algorithms)
            changed = True

        if changed:
            entry["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync SGLang params from source code")
    parser.add_argument("--tags", nargs="*", default=[],
                        help="Specific tags to check (default: auto-scan all)")
    parser.add_argument("--repo-dir", default="/tmp/sglang-sync",
                        help="Local dir for SGLang repo clone")
    args = parser.parse_args(argv)

    print("=== SGLang Parameter Sync ===")

    # Step 1: Determine which tags to process
    if args.tags:
        tags = args.tags
        print(f"[1/3] Using specified tags: {tags}")
    else:
        print("[1/3] Scanning all SGLang tags...")
        all_tags = list_all_tags()
        # Filter: only tags from 2026-05-01 onwards (v0.5.13+)
        tags = [t for t in all_tags if t >= "v0.5.13"]
        print(f"  Found {len(all_tags)} total tags, {len(tags)} since 2026-05-01: {tags}")

    existing_keys = get_existing_image_keys()
    new_tags = [t for t in tags if f"sglang-{t.lstrip('v')}" not in existing_keys]
    existing_tags = [t for t in tags if f"sglang-{t.lstrip('v')}" in existing_keys]
    print(f"  New versions to add: {new_tags or 'none'}")
    print(f"  Existing versions to check: {existing_tags}")

    # Step 2: Load images.yaml
    print("[2/3] Loading sglang-images.yaml...")
    data = load_images_yaml()

    # Step 3: Process each tag
    print("[3/3] Processing tags...")
    any_changed = False
    for tag in tags:
        print(f"\n  --- {tag} ---")
        changed = process_tag(tag, Path(args.repo_dir), data)
        if changed:
            any_changed = True

    if any_changed:
        save_images_yaml(data)
        print(f"\n  -> sglang-images.yaml updated")
    else:
        print(f"\n  -> no changes needed")

    # Summary
    print("\n=== Summary ===")
    print(f"  tags scanned: {len(tags)}")
    print(f"  new entries: {len(new_tags)}")
    print(f"  existing checked: {len(existing_tags)}")
    print(f"  images_yaml_changed: {any_changed}")
    if new_tags:
        print(f"  [TODO] New entries need manual fill: cuda_version, digest, startup_floor")
    print(f"  constraints reports: reports/constraints_*.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
