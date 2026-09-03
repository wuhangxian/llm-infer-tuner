#!/usr/bin/env python3
"""Validate the modular rule library used by the SGLang config skill.

This checks the knowledge files themselves (YAML shape, duplicate IDs and
provenance). It deliberately does not validate or reject generated candidate
commands; those remain exploratory and are decided by the remote executor.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RULES_DIR = PROJECT_ROOT / ".claude/skills/sglang-server-config-gen/references/rules"
REQUIRED_RULE_FILES = (
    "attention.yaml",
    "parallelism.yaml",
    "memory.yaml",
    "speculative.yaml",
    "scheduling.yaml",
    "fairness.yaml",
)
ALLOWED_STATUS = {"active", "experimental", "deprecated"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_EVIDENCE = {"official", "source", "measured", "judgment"}
ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$")


def _non_empty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def discover_rule_files(rules_dir: Path = DEFAULT_RULES_DIR) -> list[Path]:
    """Return required topic files plus any future *.yaml topics."""
    required = [rules_dir / filename for filename in REQUIRED_RULE_FILES]
    extras = sorted(
        path for path in rules_dir.glob("*.yaml") if path.name not in REQUIRED_RULE_FILES
    )
    return required + extras


def validate_rule_files(rules_dir: Path = DEFAULT_RULES_DIR) -> list[str]:
    """Return human-readable validation errors for all topic rule files."""
    errors: list[str] = []
    seen_ids: dict[str, Path] = {}

    for path in discover_rule_files(rules_dir):
        if not path.is_file():
            errors.append(f"{path}: missing required topic file")
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            errors.append(f"{path}: invalid YAML: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"{path}: top level must be a mapping")
            continue
        if data.get("schema_version") != 1:
            errors.append(f"{path}: schema_version must be 1")
        expected_topic = path.stem
        if data.get("topic") != expected_topic:
            errors.append(f"{path}: topic must be {expected_topic!r}")
        if not _non_empty(data.get("description")):
            errors.append(f"{path}: description must be non-empty")

        rules = data.get("rules")
        if not isinstance(rules, list) or not rules:
            errors.append(f"{path}: rules must be a non-empty list")
            continue
        for index, rule in enumerate(rules, 1):
            prefix = f"{path}:{index}"
            if not isinstance(rule, dict):
                errors.append(f"{prefix}: rule must be a mapping")
                continue
            required_fields = (
                "id", "title", "status", "confidence",
                "applies_when", "guidance", "evidence",
            )
            for field in required_fields:
                if field not in rule:
                    errors.append(f"{prefix}: missing {field}")
            rule_id = rule.get("id")
            if not _non_empty(rule_id) or not ID_RE.fullmatch(rule_id):
                errors.append(
                    f"{prefix}: id must be lowercase dotted/slashed words, got {rule_id!r}"
                )
            elif rule_id in seen_ids:
                errors.append(
                    f"{prefix}: duplicate id {rule_id!r}; first seen in {seen_ids[rule_id]}"
                )
            else:
                seen_ids[rule_id] = path
            if not _non_empty(rule.get("title")):
                errors.append(f"{prefix}: title must be non-empty")
            if rule.get("status") not in ALLOWED_STATUS:
                errors.append(f"{prefix}: status must be one of {sorted(ALLOWED_STATUS)}")
            if rule.get("confidence") not in ALLOWED_CONFIDENCE:
                errors.append(f"{prefix}: confidence must be one of {sorted(ALLOWED_CONFIDENCE)}")
            if not isinstance(rule.get("applies_when"), dict):
                errors.append(f"{prefix}: applies_when must be a mapping")
            if not _non_empty(rule.get("guidance")):
                errors.append(f"{prefix}: guidance must be non-empty")
            evidence = rule.get("evidence")
            if not isinstance(evidence, dict):
                errors.append(f"{prefix}: evidence must be a mapping")
            else:
                if evidence.get("kind") not in ALLOWED_EVIDENCE:
                    errors.append(
                        f"{prefix}: evidence.kind must be one of {sorted(ALLOWED_EVIDENCE)}"
                    )
                if not _non_empty(evidence.get("ref")):
                    errors.append(f"{prefix}: evidence.ref must be non-empty")
                if not _non_empty(evidence.get("verified_with")):
                    errors.append(f"{prefix}: evidence.verified_with must be non-empty")
            tags = rule.get("tags")
            if tags is not None and (
                not isinstance(tags, list) or not all(_non_empty(tag) for tag in tags)
            ):
                errors.append(f"{prefix}: tags must be a list of non-empty strings")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate modular SGLang knowledge rules")
    parser.add_argument("--rules-dir", type=Path, default=DEFAULT_RULES_DIR)
    args = parser.parse_args(argv)
    errors = validate_rule_files(args.rules_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"Knowledge validation failed: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(f"Knowledge validation passed: {len(discover_rule_files(args.rules_dir))} topic files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
