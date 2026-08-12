"""Load Claude Code connection settings from a private dotenv-style file."""

from __future__ import annotations

import os
import stat
from pathlib import Path


class ClaudeEnvError(ValueError):
    """Raised when a Claude environment file is invalid or unreadable."""


CLAUDE_ENV_KEYS = {
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_ATTRIBUTION_HEADER",
}


def default_env_file(project_root: Path) -> Path | None:
    """Return the first existing project or user-level Claude env file."""
    candidates = []
    configured = os.environ.get("LLMOPT_CLAUDE_ENV_FILE")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path(project_root) / ".env",
            Path.home() / ".config" / "llmopt-agent" / "claude.env",
        ]
    )
    return next((path for path in candidates if path.is_file()), None)


def load_env_file(path: Path) -> None:
    """Load allowed KEY=VALUE entries without overriding explicit shell exports."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ClaudeEnvError(f"Claude env file does not exist: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ClaudeEnvError(f"Could not read Claude env file {path}: {exc}") from exc

    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ClaudeEnvError(
            f"Claude env file {path} is too permissive ({mode:o}); run chmod 600 {path}"
        )

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ClaudeEnvError(f"Invalid Claude env entry at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in CLAUDE_ENV_KEYS:
            raise ClaudeEnvError(f"Unsupported Claude env key {key!r} at {path}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
