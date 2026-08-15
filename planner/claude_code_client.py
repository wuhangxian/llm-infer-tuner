"""Small, testable wrapper around the non-interactive Claude Code CLI."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

CompletedProcess = subprocess.CompletedProcess[str]
Runner = Callable[..., CompletedProcess]


class ClaudeCodeError(RuntimeError):
    """Raised when Claude Code cannot produce a structured response."""


class ClaudeCodeClient:
    """Invoke Claude Code with explicit directories and a JSON output schema."""

    def __init__(
        self,
        executable: str = "claude",
        timeout_seconds: int = 600,
        runner: Runner = subprocess.run,
    ) -> None:
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.runner = runner

    def run(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        add_dirs: Sequence[str | Path],
        allow_dangerous_permissions: bool = False,
    ) -> dict[str, Any]:
        argv = [
            self.executable,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(json_schema, ensure_ascii=False, separators=(",", ":")),
        ]
        for directory in add_dirs:
            argv.extend(["--add-dir", str(directory)])
        if allow_dangerous_permissions:
            argv.append("--dangerously-skip-permissions")

        try:
            completed = self.runner(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCodeError(
                f"Claude Code timed out after {self.timeout_seconds} seconds"
            ) from exc
        except OSError as exc:
            raise ClaudeCodeError(f"Could not start Claude Code: {exc}") from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise ClaudeCodeError(
                f"Claude Code exited with status {completed.returncode}: {detail}"
            )
        return self._parse_payload(completed.stdout)

    @staticmethod
    def _parse_payload(stdout: str) -> dict[str, Any]:
        try:
            payload: Any = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ClaudeCodeError(f"Claude Code returned invalid JSON: {exc}") from exc

        if isinstance(payload, dict) and isinstance(payload.get("structured_output"), dict):
            payload = payload["structured_output"]
        elif isinstance(payload, dict) and "result" in payload:
            result = payload["result"]
            if isinstance(result, dict):
                payload = result
            elif isinstance(result, str):
                try:
                    payload = json.loads(result)
                except json.JSONDecodeError as exc:
                    raise ClaudeCodeError(
                        f"Claude Code result field is not valid JSON: {exc}"
                    ) from exc

        if not isinstance(payload, dict):
            raise ClaudeCodeError("Claude Code response must be a JSON object")
        return payload
