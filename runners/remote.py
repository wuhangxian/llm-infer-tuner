"""Run commands on a remote host over the system ssh binary, plus local shells."""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

CompletedProcess = subprocess.CompletedProcess[str]
Runner = Callable[..., CompletedProcess]


class CommandFailureKind(StrEnum):
    """Failures produced by the command transport rather than the remote child."""

    TRANSPORT = "transport"
    TIMEOUT = "timeout"

# Key-based (passwordless) SSH: disable password auth entirely so failures are
# immediate instead of hanging on a prompt.
KEY_SSH_OPTIONS: tuple[str, ...] = (
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
)

# Password-based SSH: allow password auth, still no strict host check.
PASSWORD_SSH_OPTIONS: tuple[str, ...] = (
    "-o",
    "StrictHostKeyChecking=accept-new",
)

# Kept for backward compat with tests/external callers that import it.
DEFAULT_SSH_OPTIONS = KEY_SSH_OPTIONS


@dataclass
class CommandResult:
    """Outcome of a single command: exit code plus captured streams."""

    returncode: int
    stdout: str
    stderr: str
    failure_kind: CommandFailureKind | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class RemoteRunner:
    """Execute commands on a remote host via ssh, or locally via a raw argv.

    If *ssh_password* is provided, ``sshpass -d`` reads it from a protected pipe.
    The secret therefore never appears in process argv. Otherwise key-based SSH
    uses ``BatchMode=yes``.
    """

    def __init__(
        self,
        ssh_target: str,
        *,
        ssh_password: str = "",
        ssh_options: Sequence[str] | None = None,
        runner: Runner = subprocess.run,
        default_timeout: int = 600,
    ) -> None:
        self.ssh_target = ssh_target
        self._ssh_password = ssh_password
        if ssh_options is not None:
            self.ssh_options = tuple(ssh_options)
        elif ssh_password:
            self.ssh_options = PASSWORD_SSH_OPTIONS
        else:
            self.ssh_options = KEY_SSH_OPTIONS
        self.runner = runner
        self.default_timeout = default_timeout

    def build_ssh_argv(
        self, command: str, *, password_fd: int | None = None
    ) -> list[str]:
        if self._ssh_password:
            # ``0`` is only a non-secret structural placeholder for callers that
            # inspect argv. ``run`` always supplies the actual protected pipe FD.
            fd = 0 if password_fd is None else password_fd
            return [
                "sshpass", "-d", str(fd),
                "ssh", *self.ssh_options, self.ssh_target, command,
            ]
        return ["ssh", *self.ssh_options, self.ssh_target, command]

    def run(self, command: str, *, timeout: int | None = None) -> CommandResult:
        if not self._ssh_password:
            return self._invoke(
                self.build_ssh_argv(command), timeout=timeout, via_ssh=True
            )

        read_fd, write_fd = os.pipe()
        writer = threading.Thread(
            target=_write_password,
            args=(write_fd, self._ssh_password.encode("utf-8")),
            name="sshpass-fd-writer",
            daemon=True,
        )
        writer.start()
        try:
            return self._invoke(
                self.build_ssh_argv(command, password_fd=read_fd),
                timeout=timeout,
                pass_fds=(read_fd,),
                via_ssh=True,
            )
        finally:
            os.close(read_fd)
            writer.join()

    def run_local(
        self, argv: Sequence[str], *, timeout: int | None = None
    ) -> CommandResult:
        return self._invoke(list(argv), timeout=timeout, via_ssh=False)

    def _invoke(
        self,
        argv: list[str],
        *,
        timeout: int | None,
        pass_fds: tuple[int, ...] = (),
        via_ssh: bool,
    ) -> CommandResult:
        # timeout=0 is the explicit opt-out used by long-running benchmarks.
        # None retains the ordinary per-command safety default.
        effective_timeout = None if timeout == 0 else (
            self.default_timeout if timeout is None else timeout
        )
        try:
            kwargs = {
                "capture_output": True,
                "text": True,
                "check": False,
                "timeout": effective_timeout,
            }
            if pass_fds:
                kwargs["pass_fds"] = pass_fds
            completed = self.runner(argv, **kwargs)
        except subprocess.TimeoutExpired as exc:
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr)
            detail = f"command timed out after {effective_timeout}s"
            stderr = f"{stderr}\n{detail}".strip() if stderr else detail
            return CommandResult(
                returncode=124,
                stdout=stdout,
                stderr=stderr,
                failure_kind=CommandFailureKind.TIMEOUT,
            )
        except OSError as exc:
            return CommandResult(
                returncode=127,
                stdout="",
                stderr=str(exc),
                failure_kind=CommandFailureKind.TRANSPORT,
            )

        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            failure_kind=(
                CommandFailureKind.TRANSPORT
                if via_ssh and completed.returncode == 255
                else None
            ),
        )


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_password(write_fd: int, encoded: bytes) -> None:
    """Feed sshpass concurrently so a large secret cannot fill the pipe first."""
    try:
        written = 0
        while written < len(encoded):
            written += os.write(write_fd, encoded[written:])
    except BrokenPipeError:
        # The command can fail before sshpass consumes the credential. Closing
        # the reader is sufficient; this must not strand the caller.
        pass
    finally:
        os.close(write_fd)
