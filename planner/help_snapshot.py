"""Load long CLI flags from a target engine's help output."""

from __future__ import annotations

import re
from pathlib import Path


class HelpSnapshotError(ValueError):
    """Raised when a CLI help snapshot cannot be used for validation."""


_LONG_OPTION = re.compile(r"(?<![A-Za-z0-9_])--[A-Za-z0-9][A-Za-z0-9-]*")


def load_help_flags(path: Path) -> set[str]:
    """Return long options found in a text help snapshot.

    The parser intentionally treats the snapshot as an allow-list of flag names;
    it does not infer types or defaults from help formatting.
    """
    path = Path(path)
    if not path.is_file():
        raise HelpSnapshotError(f"Help snapshot does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HelpSnapshotError(f"Could not read help snapshot {path}: {exc}") from exc

    flags = set(_LONG_OPTION.findall(text))
    if not flags:
        raise HelpSnapshotError(f"Help snapshot {path} contains no long options")
    return flags
