"""Run commands on a remote host over the system ssh binary, plus local shells."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

CompletedProcess = subprocess.CompletedProcess[str]
Runner = Callable[..., CompletedProcess]

DEFAULT_SSH_OPTIONS: tuple[str, ...] = (
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
)


@dataclass
class CommandResult:
    """Outcome of a single command: exit code plus captured streams."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class RemoteRunner:
    """Execute commands on a remote host via ssh, or locally via a raw argv."""

    def __init__(
        self,
        ssh_target: str,
        *,
        ssh_options: Sequence[str] = DEFAULT_SSH_OPTIONS,
        runner: Runner = subprocess.run,
        default_timeout: int = 600,
    ) -> None:
        self.ssh_target = ssh_target
        self.ssh_options = tuple(ssh_options)
        self.runner = runner
        self.default_timeout = default_timeout

    def build_ssh_argv(self, command: str) -> list[str]:
        return ["ssh", *self.ssh_options, self.ssh_target, command]

    def run(self, command: str, *, timeout: int | None = None) -> CommandResult:
        return self._invoke(self.build_ssh_argv(command), timeout=timeout)

    def run_local(
        self, argv: Sequence[str], *, timeout: int | None = None
    ) -> CommandResult:
        return self._invoke(list(argv), timeout=timeout)

    def _invoke(self, argv: list[str], *, timeout: int | None) -> CommandResult:
        effective_timeout = self.default_timeout if timeout is None else timeout
        try:
            completed = self.runner(
                argv,
                capture_output=True,
                text=True,
                check=False,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr)
            detail = f"command timed out after {effective_timeout}s"
            stderr = f"{stderr}\n{detail}".strip() if stderr else detail
            return CommandResult(returncode=124, stdout=stdout, stderr=stderr)
        except OSError as exc:
            return CommandResult(returncode=127, stdout="", stderr=str(exc))

        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
