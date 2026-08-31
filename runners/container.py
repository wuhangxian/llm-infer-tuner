"""Persistent docker container lifecycle on a remote host via the ssh runner."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field

from runners.remote import CommandResult, RemoteRunner


def process_group_state_command(pid: int) -> str:
    """Shell probe that treats a process group containing only zombies as absent."""
    if type(pid) is not int or pid <= 1:
        raise ValueError(f"invalid monitored pid: {pid!r}")
    awk = (
        f"($1 == {pid} || $2 == {pid}) && $3 !~ /^Z/ {{ found=1 }} "
        "END { exit(found ? 0 : 1) }"
    )
    return (
        "rows=$(ps -eo pid=,pgid=,stat=) || exit $?; "
        f"if printf '%s\\n' \"$rows\" | awk {shlex.quote(awk)}; then "
        "printf 'RUNNING\\n'; else rc=$?; "
        "if [ \"$rc\" -eq 1 ]; then printf 'MISSING\\n'; else exit \"$rc\"; fi; fi"
    )


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
        # Build --device args from GPU IDs. This server uses CDI (Container
        # Device Interface), where --gpus "device=0,1,2,3" fails with
        # "cannot set both Count and DeviceIDs". Use --device nvidia.com/gpu=N
        # per GPU instead, which works with both CDI and legacy runtimes.
        if cfg.gpus == "all":
            gpu_args = ["--gpus", "all"]
        elif cfg.gpus.startswith("device="):
            ids = cfg.gpus[len("device="):].split(",")
            gpu_args = []
            for gid in ids:
                gpu_args.extend(["--device", f"nvidia.com/gpu={gid.strip()}"])
        else:
            gpu_args = ["--gpus", cfg.gpus]

        argv = [
            "docker",
            "run",
            "-d",
            *gpu_args,
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

    @staticmethod
    def _failed_monitored_start_cleanup(pid_path: str, temporary_pid: str) -> str:
        """Fail closed after a child exists but its PID cannot be committed."""
        return (
            "kill -KILL -- \"-$pid\" 2>/dev/null || true; "
            "kill -KILL \"$pid\" 2>/dev/null || true; "
            "alive=1; "
            "for _proof in $(seq 1 100); do "
            "proof_state=UNKNOWN; "
            "if rows=$(ps -eo pid=,pgid=,stat= 2>/dev/null); then "
            "printf '%s\\n' \"$rows\" | awk -v pid=\"$pid\" "
            "'$3 !~ /^Z/ && ($1 == pid || $2 == pid) { found=1 } "
            "END { exit(found ? 0 : 1) }'; proof_rc=$?; "
            "case \"$proof_rc\" in "
            "0) proof_state=RUNNING ;; 1) proof_state=MISSING ;; esac; fi; "
            "if [ \"$proof_state\" = MISSING ]; then alive=0; break; fi; "
            "kill -KILL -- \"-$pid\" 2>/dev/null || true; "
            "kill -KILL \"$pid\" 2>/dev/null || true; "
            "sleep 0.01; "
            "done; "
            "if [ \"$alive\" -eq 0 ]; then "
            "wait \"$pid\" 2>/dev/null || true; "
            f"rm -f -- {shlex.quote(pid_path)} {shlex.quote(temporary_pid)}; "
            "else "
            "printf 'process cleanup proof failed for pid %s\\n' \"$pid\" >&2; "
            "fi; "
            "exit 125; "
        )

    @classmethod
    def _process_group_publication_barrier(
        cls, pid_path: str, temporary_pid: str
    ) -> str:
        """Bound the fork-to-setsid window before reporting start success."""
        return (
            "ready=0; "
            "for _ in $(seq 1 100); do "
            "pgid=$(ps -o pgid= -p \"$pid\" 2>/dev/null | tr -d ' '); "
            "if [ \"$pgid\" = \"$pid\" ]; then ready=1; break; fi; "
            "if ! kill -0 \"$pid\" 2>/dev/null; then ready=1; break; fi; "
            "sleep 0.01; "
            "done; "
            "if [ \"$ready\" -ne 1 ]; then "
            "printf 'process group publication timed out for pid %s\\n' \"$pid\" >&2; "
            + cls._failed_monitored_start_cleanup(pid_path, temporary_pid)
            + "fi; "
        )

    def start_server_monitored(
        self,
        command: str,
        log_path: str,
        pid_path: str,
        *,
        timeout: int | None = None,
    ) -> CommandResult:
        """Start a server in its own process group and atomically publish PGID."""
        worker = f"exec {command}"
        temporary_pid = f"{pid_path}.tmp.$$"
        inner = (
            f"rm -f -- {shlex.quote(pid_path)}; "
            f"/usr/bin/setsid bash -lc {shlex.quote(worker)} "
            f"> {shlex.quote(log_path)} 2>&1 < /dev/null & pid=$!; "
            f"if ! printf '%s\\n' \"$pid\" > {shlex.quote(temporary_pid)} || "
            f"! mv -f -- {shlex.quote(temporary_pid)} {shlex.quote(pid_path)}; then "
            + self._failed_monitored_start_cleanup(pid_path, temporary_pid)
            + "fi; "
            + self._process_group_publication_barrier(pid_path, temporary_pid)
            + "printf '%s\\n' \"$pid\""
        )
        return self.exec(inner, timeout=timeout)

    def start_monitored(
        self,
        command: str,
        log_path: str,
        status_path: str,
        pid_path: str,
        *,
        timeout: int | None = None,
    ) -> CommandResult:
        """Start a new process group and atomically publish its eventual exit code."""
        temporary_status = f"{status_path}.tmp.$$"
        worker = (
            "set +e; "
            f"{{ {command}; }} > {shlex.quote(log_path)} 2>&1; "
            "rc=$?; "
            f"printf '%s\\n' \"$rc\" > {shlex.quote(temporary_status)} && "
            f"mv -f -- {shlex.quote(temporary_status)} {shlex.quote(status_path)}; "
            'exit "$rc"'
        )
        inner = (
            f"rm -f -- {shlex.quote(status_path)} {shlex.quote(pid_path)}; "
            f"/usr/bin/setsid bash -lc {shlex.quote(worker)} "
            "</dev/null >/dev/null 2>&1 & pid=$!; "
            f"if ! printf '%s\\n' \"$pid\" > "
            f"{shlex.quote(pid_path + '.tmp.$$')} || "
            f"! mv -f -- {shlex.quote(pid_path + '.tmp.$$')} "
            f"{shlex.quote(pid_path)}; then "
            + self._failed_monitored_start_cleanup(
                pid_path, pid_path + ".tmp.$$"
            )
            + "fi; "
            + self._process_group_publication_barrier(
                pid_path, pid_path + ".tmp.$$"
            )
            + "printf '%s\\n' \"$pid\""
        )
        return self.exec(inner, timeout=timeout)

    def monitored_pid(
        self, pid_path: str, *, timeout: int | None = None
    ) -> CommandResult:
        return self.exec(f"cat -- {shlex.quote(pid_path)}", timeout=timeout)

    def monitored_state(
        self, pid: int, status_path: str, *, timeout: int | None = None
    ) -> CommandResult:
        """Return exactly RUNNING, MISSING, or EXIT <integer>."""
        if type(pid) is not int or pid <= 1:
            raise ValueError(f"invalid monitored pid: {pid!r}")
        command = (
            f"if [ -f {shlex.quote(status_path)} ]; then "
            f"printf 'EXIT '; cat -- {shlex.quote(status_path)}; "
            f"elif kill -0 -- {pid} 2>/dev/null; then printf 'RUNNING\\n'; "
            "else printf 'MISSING\\n'; fi"
        )
        return self.exec(command, timeout=timeout)

    def process_group_state(
        self, pid: int, *, timeout: int | None = None
    ) -> CommandResult:
        return self.exec(process_group_state_command(pid), timeout=timeout)

    def process_state(
        self, pid: int, *, timeout: int | None = None
    ) -> CommandResult:
        """Strict liveness for one PID; zombies count as absent."""
        if type(pid) is not int or pid <= 1:
            raise ValueError(f"invalid monitored pid: {pid!r}")
        command = (
            f"state=$(ps -o stat= -p {pid}) || exit $?; "
            "state=${state## }; "
            "if [ -n \"$state\" ] && [ \"${state#Z}\" = \"$state\" ]; then "
            "printf 'RUNNING\\n'; else printf 'MISSING\\n'; fi"
        )
        return self.exec(command, timeout=timeout)

    def file_progress(
        self, paths: Sequence[str], *, timeout: int | None = None
    ) -> CommandResult:
        """Return a JSON map of each exact path to an independent size/mtime token."""
        script = (
            "import json,os,sys; "
            "print(json.dumps({p:(f'{os.stat(p).st_size}:{os.stat(p).st_mtime_ns}' "
            "if os.path.exists(p) else 'missing') for p in sys.argv[1:]}))"
        )
        command = shlex.join(["python", "-c", script, *paths])
        return self.exec(command, timeout=timeout)

    def health(self, port: int, *, timeout: int | None = None) -> CommandResult:
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError(f"invalid health port: {port!r}")
        return self.exec(
            f"curl -sf http://127.0.0.1:{port}/health", timeout=timeout
        )

    def health_many(
        self, ports: Sequence[int], *, timeout: int | None = None
    ) -> CommandResult:
        """Check a replica group in one bounded remote command."""
        normalized = [int(port) for port in ports]
        if not normalized or any(not 1 <= port <= 65535 for port in normalized):
            raise ValueError(f"invalid health ports: {ports!r}")
        checks = " ".join(str(port) for port in normalized)
        command = (
            f"for port in {checks}; do "
            "curl -sf \"http://127.0.0.1:${port}/health\" >/dev/null || "
            "{ rc=$?; printf 'FAILED %s %s\\n' \"$port\" \"$rc\"; exit \"$rc\"; }; "
            "done; printf 'HEALTHY\\n'"
        )
        return self.exec(command, timeout=timeout)

    def signal_process_group(
        self,
        pid: int,
        signal_name: str,
        *,
        timeout: int | None = None,
    ) -> CommandResult:
        if type(pid) is not int or pid <= 1:
            raise ValueError(f"invalid monitored pid: {pid!r}")
        if signal_name not in {"TERM", "KILL"}:
            raise ValueError(f"invalid process signal: {signal_name!r}")
        command = (
            f"if kill -0 -- -{pid} 2>/dev/null; then "
            f"kill -{signal_name} -- -{pid}; "
            f"else kill -{signal_name} -- {pid}; fi"
        )
        return self.exec(command, timeout=timeout)

    def running_state(self, *, timeout: int | None = None) -> CommandResult:
        """Return the raw bounded inspect used for the running postcheck."""
        command = (
            f"docker inspect -f {shlex.quote('{{.State.Running}}')} "
            f"{shlex.quote(self.config.name)}"
        )
        return self.remote.run(command, timeout=timeout)

    def is_running(self, *, timeout: int | None = None) -> bool:
        """True when `docker inspect` reports the container's State.Running is true."""
        result = self.running_state(timeout=timeout)
        return result.ok and result.stdout.strip() == "true"

    def inspect(self, *, timeout: int | None = None) -> CommandResult:
        """Return raw ``docker inspect`` status for strict removal postchecks."""
        command = f"docker inspect {shlex.quote(self.config.name)}"
        return self.remote.run(command, timeout=timeout)

    def stop(self, *, timeout: int | None = None) -> CommandResult:
        command = f"docker stop {shlex.quote(self.config.name)}"
        return self.remote.run(command, timeout=timeout)

    def remove(self, *, force: bool = True, timeout: int | None = None) -> CommandResult:
        flag = "-f " if force else ""
        command = f"docker rm {flag}{shlex.quote(self.config.name)}"
        return self.remote.run(command, timeout=timeout)
