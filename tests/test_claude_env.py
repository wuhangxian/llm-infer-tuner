import os
from pathlib import Path

import pytest

from planner.claude_env import ClaudeEnvError, default_env_file, load_env_file


def test_load_env_file_preserves_explicit_shell_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "claude.env"
    env_file.write_text(
        """export ANTHROPIC_BASE_URL=https://example.test
ANTHROPIC_AUTH_TOKEN='file-token'
""",
        encoding="utf-8",
    )
    env_file.chmod(0o600)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "shell-token")

    load_env_file(env_file)

    assert os.environ["ANTHROPIC_BASE_URL"] == "https://example.test"
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "shell-token"


def test_load_env_file_rejects_permissive_or_unknown_files(tmp_path: Path) -> None:
    env_file = tmp_path / "claude.env"
    env_file.write_text("UNSAFE=value\n", encoding="utf-8")
    env_file.chmod(0o644)

    with pytest.raises(ClaudeEnvError, match="too permissive"):
        load_env_file(env_file)

    env_file.chmod(0o600)
    with pytest.raises(ClaudeEnvError, match="Unsupported Claude env key"):
        load_env_file(env_file)


def test_default_env_file_prefers_configured_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "configured.env"
    env_file.write_text("ANTHROPIC_BASE_URL=https://example.test\n", encoding="utf-8")
    env_file.chmod(0o600)
    monkeypatch.setenv("LLMOPT_CLAUDE_ENV_FILE", str(env_file))

    assert default_env_file(tmp_path) == env_file
