"""Persistent docker container lifecycle on a remote host via the ssh runner."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field

from runners.remote import CommandResult, RemoteRunner


@dataclass
class ContainerConfig:
    """Everything needed to `docker run` the sglang image as a long-lived box."""

    image_ref: str
    name: str
    model_host_dir: str  # model directory on the dev machine
    model_container_path: str  # mount point inside the container (used as --model-path)
    outputs_host_dir: str  # results directory on the dev machine
    outputs_container_path: str = "/workspace/outputs"
    gpus: str = "all"  # "all" or "device=0,1" for specific GPUs
    shm_size: str = "32g"
    port: int = 30000
    extra_run_args: Sequence[str] = field(default_factory=tuple)


class Container:
    """Start, exec into, and tear down one persistent container over ssh."""

    def __init__(self, remote: RemoteRunner, config: ContainerConfig) -> None:
        self.remote = remote
        self.config = config

    def start(self, *, timeout: int | None = None) -> CommandResult:
        """Launch the container detached with GPUs, mounts, and port published."""
        cfg = self.config
        argv = [
            "docker",
            "run",
            "-d",
            "--gpus",
            cfg.gpus,
            "--shm-size",
            cfg.shm_size,
            "-v",
            f"{cfg.model_host_dir}:{cfg.model_container_path}",
            "-v",
            f"{cfg.outputs_host_dir}:{cfg.outputs_container_path}",
            "-p",
            f"{cfg.port}:{cfg.port}",
            "--name",
            cfg.name,
            *cfg.extra_run_args,
            cfg.image_ref,
            "sleep",
            "infinity",
        ]
        command = " ".join(shlex.quote(part) for part in argv)
        return self.remote.run(command, timeout=timeout)

    def exec(self, command: str, *, timeout: int | None = None) -> CommandResult:
        """Run a shell command inside the container via `docker exec ... bash -lc`."""
        wrapped = f"docker exec {shlex.quote(self.config.name)} bash -lc {shlex.quote(command)}"
        return self.remote.run(wrapped, timeout=timeout)

    def exec_detached(
        self, command: str, log_container_path: str, *, timeout: int | None = None
    ) -> CommandResult:
        """Start a long-running command in the background inside the container.

        The command is launched with nohup, its stdout/stderr redirected to
        ``log_container_path`` (a path on the container filesystem, i.e. under the
        mounted outputs dir), and returns immediately without blocking on it.
        """
        inner = (
            f"nohup {command} > {shlex.quote(log_container_path)} 2>&1 &"
            " echo $!"
        )
        return self.exec(inner, timeout=timeout)

    def is_running(self, *, timeout: int | None = None) -> bool:
        """True when `docker inspect` reports the container's State.Running is true."""
        command = (
            f"docker inspect -f {shlex.quote('{{.State.Running}}')} "
            f"{shlex.quote(self.config.name)}"
        )
        result = self.remote.run(command, timeout=timeout)
        return result.ok and result.stdout.strip() == "true"

    def stop(self, *, timeout: int | None = None) -> CommandResult:
        command = f"docker stop {shlex.quote(self.config.name)}"
        return self.remote.run(command, timeout=timeout)

    def remove(self, *, force: bool = True, timeout: int | None = None) -> CommandResult:
        flag = "-f " if force else ""
        command = f"docker rm {flag}{shlex.quote(self.config.name)}"
        return self.remote.run(command, timeout=timeout)
