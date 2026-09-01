#!/usr/bin/env python3
"""Fail when tracked repository files contain plaintext credentials.

The scanner is intentionally conservative about what it reports: empty values
and explicit environment/placeholder references are allowed, while values in
credential-shaped fields, private-key blocks, and secret-looking shell
assignments are rejected.  It prints paths and line numbers only; matched
values are never echoed to stdout/stderr.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SecretFinding:
    path: Path
    line: int
    kind: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: possible plaintext {self.kind}"


_FIELD_PATTERN = re.compile(
    r"(?i)(?:\"|')(?P<field>ssh_password|password|passwd|passphrase|"
    r"access[_-]?token|api[_-]?key|secret(?:[_-]?key)?|private[_-]?key)"
    r"(?:\"|')\s*:\s*(?:\"(?P<double>[^\"\r\n]*)\"|'(?P<single>[^'\r\n]*)')"
)
_ASSIGNMENT_PATTERN = re.compile(
    # Keep this case-sensitive: ordinary lower-case shell variables such as
    # ``has_plain_password`` are bookkeeping, not credential stores.
    r"(?m)^\s*(?:export\s+)?(?P<field>[A-Z][A-Z0-9_]*(?:PASSWORD|PASSWD|"
    r"TOKEN|SECRET|API_KEY))\s*=\s*(?P<value>[^\s#]+)"
)
_PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_URL_CREDENTIAL_PATTERN = re.compile(
    r"(?i)https?://[^/\s:@]+:[^/@\s]+@"
)

_PLACEHOLDER_PREFIXES = ("$", "${", "<", "{{")
_PLACEHOLDERS = frozenset(
    {
        "REDACTED",
        "CHANGE_ME",
        "CHANGEME",
        "TEST-ONLY-PASSWORD",
        "TEST_PASSWORD",
        "SENTINEL",
        "DUMMY",
        "FAKE",
        "YOUR_PASSWORD",
        "YOUR_TOKEN",
        "EXAMPLE",
    }
)


def _is_non_plaintext_reference(value: str) -> bool:
    stripped = value.strip().strip("'\"").strip()
    if not stripped:
        return True
    upper = stripped.upper()
    if upper in _PLACEHOLDERS or any(
        marker in upper for marker in ("SENTINEL", "TEST-", "DUMMY", "FAKE")
    ):
        return True
    return stripped.startswith(_PLACEHOLDER_PREFIXES)


def scan_text(text: str, path: Path) -> list[SecretFinding]:
    """Return redacted findings in one decoded text file."""

    findings: list[SecretFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in _FIELD_PATTERN.finditer(line):
            value = match.group("double")
            if value is None:
                value = match.group("single") or ""
            if not _is_non_plaintext_reference(value):
                findings.append(
                    SecretFinding(path, line_number, f"credential field {match.group('field')}")
                )
        # Shell assignments are the only non-JSON form with a stable secret
        # contract.  Python variables such as ``sample_token`` are ordinary
        # runtime identifiers, not checked-in credentials.
        if path.suffix in {".sh", ".env"} or path.name == ".env":
            for match in _ASSIGNMENT_PATTERN.finditer(line):
                if not _is_non_plaintext_reference(match.group("value")):
                    findings.append(
                        SecretFinding(
                            path,
                            line_number,
                            f"credential assignment {match.group('field')}",
                        )
                    )
        if _PRIVATE_KEY_PATTERN.search(line):
            findings.append(SecretFinding(path, line_number, "private key material"))
        if _URL_CREDENTIAL_PATTERN.search(line):
            findings.append(SecretFinding(path, line_number, "URL credential"))
    return findings


def scan_paths(paths: Iterable[Path]) -> list[SecretFinding]:
    """Scan the supplied files without printing their contents."""

    findings: list[SecretFinding] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Binary assets cannot contain a JSON/shell credential field in a
            # meaningful form; avoid false positives from arbitrary bytes.
            continue
        except OSError:
            findings.append(SecretFinding(path, 0, "unreadable file"))
            continue
        findings.extend(scan_text(text, path))
    return findings


def tracked_paths(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    names = result.stdout.decode("utf-8").split("\0")
    return [repo_root / name for name in names if name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root (defaults to the current directory)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        findings = scan_paths(tracked_paths(root))
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"secret scan could not inspect tracked files: {exc}", file=sys.stderr)
        return 2
    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        print(
            f"secret scan failed: {len(findings)} possible credential(s); "
            "rotate exposed values and use environment references",
            file=sys.stderr,
        )
        return 1
    print("secret scan passed: no plaintext credentials in tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
