"""Ordered, fail-closed preparation for one executor run.

The public API deliberately separates pure local validation from remote host
inspection.  Callers can therefore reject malformed work before constructing an
SSH runner, and the remote phase can remain read-only until every fact needed to
authorise cleanup has been collected.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any, Protocol

from runners.remote import CommandResult

SSH_PROBE_COMMAND = "printf 'llm-infer-tuner-preflight-ok\\n'"
HOME_QUERY_COMMAND = "printf '%s\\n' \"$HOME\""
GPU_QUERY_COMMAND = (
    "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits"
)
NUMA_QUERY_COMMAND = "nvidia-smi topo -m"
CONTAINERS_QUERY_COMMAND = (
    "ids=$(docker ps -aq) || exit $?; if [ -n \"$ids\" ]; then docker inspect $ids; "
    "else printf '[]\\n'; fi"
)
GPU_PIDS_QUERY_COMMAND = (
    "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits"
)
LISTENERS_QUERY_COMMAND = "ss -H -ltn"


class _Remote(Protocol):
    def run(self, command: str, *, timeout: int | None = None) -> CommandResult: ...


class PreflightError(RuntimeError):
    """A fail-closed local or remote preparation failure."""


@dataclass(frozen=True)
class PreflightRequest:
    """Immutable inputs needed to validate and prepare one remote run."""

    candidates: tuple[dict[str, Any], ...]
    gpu_model: str
    gpu_count: int
    gpu_memory_gb: float
    model_host_dir: str
    image_ref: str
    base_port: int
    container_name: str
    remote_outputs_dir: str = ""
    job_id: str = "preflight"
    allocation_gpu_count: int | None = None
    fill_host: bool = False
    allow_cross_numa: bool = False
    exclusive_host: bool = False

    @property
    def execution_gpu_count(self) -> int:
        return self.gpu_count if self.allocation_gpu_count is None else self.allocation_gpu_count


@dataclass(frozen=True)
class CandidatePlacement:
    """One candidate's resources inside one concurrently executed batch."""

    candidate_id: str
    gpu_ids: tuple[int, ...]
    port: int


@dataclass(frozen=True)
class FillHostPlacement:
    """The concrete replicas used for one fill-host candidate."""

    candidate_id: str
    gpu_slices: tuple[tuple[int, ...], ...]
    ports: tuple[int, ...]


@dataclass(frozen=True)
class PreflightPlan:
    """Fully validated placement plan derived from the observed topology."""

    candidate_ids: tuple[str, ...]
    numa_groups: tuple[tuple[int, ...], ...]
    round1_batches: tuple[tuple[CandidatePlacement, ...], ...]
    round2_batches: tuple[tuple[CandidatePlacement, ...], ...]
    fill_host_placements: tuple[FillHostPlacement, ...]
    required_ports: tuple[int, ...]
    outputs_host_dir: str = ""


@dataclass(frozen=True)
class _GpuFact:
    index: int
    name: str
    memory_mib: float


@dataclass(frozen=True)
class _ContainerFact:
    container_id: str
    name: str
    running: bool
    labels: dict[str, str]
    published_ports: frozenset[int]


@dataclass(frozen=True)
class _HostUsage:
    containers: tuple[_ContainerFact, ...]
    owned_containers: tuple[_ContainerFact, ...]
    gpu_pids: frozenset[int]
    owned_gpu_pids: frozenset[int]
    listening_ports: frozenset[int]


def validate_local_preflight(request: PreflightRequest) -> None:
    """Validate locally knowable execution constraints without an SSH runner."""
    if not request.candidates:
        raise ValueError("candidate set must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", request.container_name):
        raise ValueError(
            "container_name must be a safe anchored Docker name without shell syntax"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", request.job_id):
        raise ValueError("job_id must be a safe non-empty identifier")
    if request.remote_outputs_dir:
        _validate_absolute_remote_path(
            request.remote_outputs_dir,
            label="remote_outputs_dir",
        )
    if any(
        type(value) is not bool
        for value in (
            request.fill_host,
            request.allow_cross_numa,
            request.exclusive_host,
        )
    ):
        raise ValueError("preflight policy values must be booleans")
    if type(request.gpu_count) is not int or request.gpu_count < 1:
        raise ValueError("GPU count must be a positive integer")
    if (
        type(request.gpu_memory_gb) not in (int, float)
        or not math.isfinite(request.gpu_memory_gb)
        or request.gpu_memory_gb <= 0
    ):
        raise ValueError("GPU memory must be a finite positive number")
    if (
        type(request.execution_gpu_count) is not int
        or request.execution_gpu_count < 1
        or request.execution_gpu_count > request.gpu_count
    ):
        raise ValueError("execution GPU count must be in 1..target GPU count")

    candidate_ids = [_candidate_id(candidate) for candidate in request.candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")
    for candidate_id in candidate_ids:
        if len(f"{request.container_name}-{candidate_id}") > 255:
            raise ValueError(
                f"candidate {candidate_id!r}: final Docker container name exceeds 255 characters"
            )
    tp_sizes = [_candidate_tp(candidate) for candidate in request.candidates]
    for candidate, tp_size in zip(request.candidates, tp_sizes, strict=True):
        if tp_size > request.execution_gpu_count:
            raise ValueError(
                f"candidate {candidate.get('id', '?')}: tp_size={tp_size} exceeds "
                f"execution GPU count {request.execution_gpu_count}"
            )

    # The exact simultaneous span depends on the observed NUMA topology and can
    # always shrink by splitting ordinary candidates into more batches.  Before
    # SSH, validate only that the base itself is usable; build_preflight_plan
    # validates every concrete batch and fill-host replica port later, still
    # before any remote mutation.
    _validate_port_span(request.base_port, 1)


def build_preflight_plan(
    request: PreflightRequest,
    *,
    numa_groups: Sequence[Sequence[int]],
) -> PreflightPlan:
    """Build and validate both execution rounds against an observed topology."""
    validate_local_preflight(request)
    observed_topology = _normalise_topology(request.gpu_count, numa_groups)
    execution_ids = set(range(request.execution_gpu_count))
    topology = tuple(
        tuple(gpu_id for gpu_id in group if gpu_id in execution_ids)
        for group in observed_topology
    )
    topology = tuple(group for group in topology if group)
    topology = _normalise_topology(request.execution_gpu_count, topology)
    candidate_ids = tuple(_candidate_id(candidate) for candidate in request.candidates)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")

    round1 = _plan_batches(request, topology)
    fill_host_placements = (
        _build_fill_host_placements(request, topology) if request.fill_host else ()
    )
    if request.fill_host:
        all_gpus = tuple(range(request.execution_gpu_count))
        round2 = tuple(
            (
                CandidatePlacement(
                    candidate_id=_candidate_id(candidate),
                    gpu_ids=all_gpus,
                    port=request.base_port,
                ),
            )
            for candidate in request.candidates
        )
    else:
        round2 = round1
    assigned_ports = {
        placement.port
        for batches in (round1, round2)
        for batch in batches
        for placement in batch
    }
    required_ports = tuple(sorted(assigned_ports))
    plan = PreflightPlan(
        candidate_ids=candidate_ids,
        numa_groups=topology,
        round1_batches=round1,
        round2_batches=round2,
        fill_host_placements=fill_host_placements,
        required_ports=required_ports,
        outputs_host_dir=request.remote_outputs_dir,
    )
    _validate_plan(plan, request)
    return plan


def prepare_remote_host(remote: _Remote, request: PreflightRequest) -> PreflightPlan:
    """Inspect a host read-only, validate it, then perform authorised preparation.

    Every local contract and remote fact is checked before mutation.  Missing
    images are pulled only after ownership validation.  Non-exclusive cleanup is
    limited to containers anchored to ``container_name``; whole-host container,
    GPU-process, and required-port cleanup requires explicit ``exclusive_host``.
    Each mutation is checked and followed by a fresh read-only inventory.
    """
    validate_local_preflight(request)
    probe = _run_readonly(remote, SSH_PROBE_COMMAND, label="SSH probe")
    if probe.stdout.strip() != "llm-infer-tuner-preflight-ok":
        raise PreflightError("SSH probe returned an unexpected response")

    outputs_host_dir = _resolve_outputs_host_dir(remote, request)

    gpu_result = _run_readonly(remote, GPU_QUERY_COMMAND, label="GPU inventory")
    gpus = _parse_gpu_inventory(gpu_result.stdout)
    _validate_gpu_inventory(gpus, request)

    topology_result = _run_readonly(remote, NUMA_QUERY_COMMAND, label="NUMA topology")
    try:
        topology = _parse_numa_topology(topology_result.stdout, request.gpu_count)
        plan = build_preflight_plan(request, numa_groups=topology)
    except ValueError as exc:
        raise PreflightError(str(exc)) from exc

    model_command = f"test -d {shlex.quote(request.model_host_dir)}"
    _run_readonly(remote, model_command, label="model directory")

    image_command = f"docker image inspect {shlex.quote(request.image_ref)}"
    image_present = _inspect_image(remote, image_command)

    usage = _inspect_usage(remote, request)
    _validate_ownership(usage, plan, request)
    output_path = shlex.quote(outputs_host_dir)
    output_command = (
        f"mkdir -p -- {output_path} && test -d {output_path} && test -w {output_path}"
    )
    _run_mutation(
        remote,
        output_command,
        label="remote output directory preparation",
        timeout=60,
    )
    if not image_present:
        pull_command = f"docker pull {shlex.quote(request.image_ref)}"
        _run_mutation(remote, pull_command, label="image pull", timeout=600)
        _run_readonly(remote, image_command, label="post-pull image inspection")
    if request.exclusive_host:
        _cleanup_exclusive_host(remote, usage, plan)
    elif usage.owned_containers:
        _cleanup_owned_containers(remote, usage.owned_containers)

    # Re-read ownership-sensitive facts immediately before handing the plan to
    # the executor.  Image pulls and cleanup can take minutes, so the initial
    # inventory alone cannot safely authorise a later launch.
    after = _inspect_usage(remote, request)
    _validate_cleanup_recheck(after, plan, request)
    return replace(plan, outputs_host_dir=outputs_host_dir)


def _resolve_outputs_host_dir(remote: _Remote, request: PreflightRequest) -> str:
    if request.remote_outputs_dir:
        return request.remote_outputs_dir
    result = _run_readonly(remote, HOME_QUERY_COMMAND, label="remote HOME")
    home = result.stdout.strip()
    try:
        _validate_absolute_remote_path(home, label="remote HOME")
    except ValueError as exc:
        raise PreflightError(str(exc)) from exc
    prefix = home.rstrip("/")
    return f"{prefix}/llm-infer-tuner-outputs/{request.job_id}"


def _validate_absolute_remote_path(path: str, *, label: str) -> None:
    if (
        not path
        or not path.startswith("/")
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
        or ".." in PurePosixPath(path).parts
    ):
        raise ValueError(f"{label} must be an absolute POSIX path without '..'")


def _inspect_image(remote: _Remote, command: str) -> bool:
    result = remote.run(command)
    if result.ok:
        return True
    detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
    if result.returncode == 1 and re.search(
        r"\bno such (?:image|object)\b",
        detail,
        flags=re.IGNORECASE,
    ):
        return False
    raise PreflightError(
        f"image inspection failed with rc={result.returncode}: {detail}"
    )


def _run_readonly(
    remote: _Remote,
    command: str,
    *,
    label: str,
    timeout: int | None = None,
) -> CommandResult:
    result = remote.run(command, timeout=timeout)
    if not result.ok:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise PreflightError(f"{label} failed with rc={result.returncode}: {detail}")
    return result


def _parse_gpu_inventory(stdout: str) -> tuple[_GpuFact, ...]:
    facts: list[_GpuFact] = []
    for row_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            raise PreflightError(f"GPU inventory row {row_number} is malformed")
        try:
            index = int(fields[0])
            memory_mib = float(fields[-1])
        except ValueError as exc:
            raise PreflightError(
                f"GPU inventory row {row_number} has invalid numeric fields"
            ) from exc
        if not math.isfinite(memory_mib) or memory_mib <= 0:
            raise PreflightError(f"GPU inventory row {row_number} has invalid memory")
        name = ",".join(fields[1:-1]).strip()
        if not name:
            raise PreflightError(f"GPU inventory row {row_number} has no name")
        facts.append(_GpuFact(index=index, name=name, memory_mib=memory_mib))
    return tuple(facts)


def _gpu_model_matches(declared: str, actual: str) -> bool:
    """Match complete adjacent model tokens without substring collisions."""
    declared = re.sub(r"(?i)^G\d+_", "", declared.strip())
    declared_tokens = re.findall(r"[a-z0-9]+", declared.lower())
    actual_tokens = re.findall(r"[a-z0-9]+", actual.lower())
    if not declared_tokens or not actual_tokens:
        return False
    declared_compact = "".join(declared_tokens)
    return any(
        "".join(actual_tokens[start:end]) == declared_compact
        for start in range(len(actual_tokens))
        for end in range(start + 1, len(actual_tokens) + 1)
    )


def _validate_gpu_inventory(
    gpus: tuple[_GpuFact, ...], request: PreflightRequest
) -> None:
    if len(gpus) != request.gpu_count:
        raise PreflightError(
            f"actual GPU count {len(gpus)} does not match declared {request.gpu_count}"
        )
    if tuple(gpu.index for gpu in gpus) != tuple(range(request.gpu_count)):
        raise PreflightError("GPU inventory indices are incomplete or reordered")
    for gpu in gpus:
        if not _gpu_model_matches(request.gpu_model, gpu.name):
            raise PreflightError(
                f"actual GPU model {gpu.name!r} does not match {request.gpu_model!r}"
            )
        actual_gib = gpu.memory_mib / 1024
        actual_decimal_gb = gpu.memory_mib * 1_048_576 / 1_000_000_000
        # Historical targets mix binary GiB and vendor-marketed decimal GB.
        # Match the nearer interpretation within 1 GB; strict model-token
        # matching above prevents this compatibility rule crossing SKU names.
        closest_delta = min(
            abs(actual_gib - request.gpu_memory_gb),
            abs(actual_decimal_gb - request.gpu_memory_gb),
        )
        if closest_delta > 1.0:
            raise PreflightError(
                f"actual GPU memory {gpu.memory_mib:.0f}MiB "
                f"({actual_gib:.1f}GiB/{actual_decimal_gb:.1f}GB) does not match "
                f"declared {request.gpu_memory_gb:.1f}GB"
            )


def _parse_numa_topology(stdout: str, gpu_count: int) -> tuple[tuple[int, ...], ...]:
    groups: dict[int, list[int]] = {}
    for line in stdout.splitlines():
        if not line.startswith("GPU"):
            continue
        fields = line.split()
        try:
            gpu_id = int(fields[0].removeprefix("GPU"))
        except (IndexError, ValueError):
            continue
        numa_id: int | None = None
        for field in reversed(fields[1:]):
            try:
                numa_id = int(field)
                break
            except ValueError:
                continue
        if numa_id is not None:
            groups.setdefault(numa_id, []).append(gpu_id)
    return _normalise_topology(gpu_count, tuple(groups.values()))


def _parse_container_inventory(stdout: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise PreflightError("container inventory is not valid JSON") from exc
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise PreflightError("container inventory must be a JSON array of objects")
    return payload


def _container_facts(rows: list[dict[str, Any]]) -> tuple[_ContainerFact, ...]:
    facts: list[_ContainerFact] = []
    for row_number, row in enumerate(rows, start=1):
        container_id = row.get("Id")
        raw_name = row.get("Name")
        if (
            not isinstance(container_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", container_id)
            or not isinstance(raw_name, str)
            or not raw_name
        ):
            raise PreflightError(f"container inventory row {row_number} has invalid identity")
        config = row.get("Config") or {}
        state = row.get("State") or {}
        network = row.get("NetworkSettings") or {}
        if not isinstance(config, dict) or not isinstance(state, dict) or not isinstance(network, dict):
            raise PreflightError(f"container inventory row {row_number} is malformed")
        raw_labels = config.get("Labels") or {}
        if not isinstance(raw_labels, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in raw_labels.items()
        ):
            raise PreflightError(f"container inventory row {row_number} has invalid labels")
        raw_ports = network.get("Ports") or {}
        if not isinstance(raw_ports, dict):
            raise PreflightError(f"container inventory row {row_number} has invalid ports")
        published_ports: set[int] = set()
        for bindings in raw_ports.values():
            if bindings is None:
                continue
            if not isinstance(bindings, list):
                raise PreflightError(
                    f"container inventory row {row_number} has invalid port bindings"
                )
            for binding in bindings:
                if not isinstance(binding, dict):
                    raise PreflightError(
                        f"container inventory row {row_number} has invalid port binding"
                    )
                host_port = binding.get("HostPort")
                if not isinstance(host_port, str) or not host_port.isdigit():
                    raise PreflightError(
                        f"container inventory row {row_number} has invalid host port"
                    )
                port = int(host_port)
                if not 1 <= port <= 65535:
                    raise PreflightError(
                        f"container inventory row {row_number} has out-of-range host port"
                    )
                published_ports.add(port)
        running = state.get("Running")
        if type(running) is not bool:
            raise PreflightError(f"container inventory row {row_number} has invalid state")
        facts.append(
            _ContainerFact(
                container_id=container_id,
                name=raw_name.removeprefix("/"),
                running=running,
                labels=dict(raw_labels),
                published_ports=frozenset(published_ports),
            )
        )
    return tuple(facts)


def _is_owned_container(
    container: _ContainerFact,
    owner: str,
    legacy_names: frozenset[str],
) -> bool:
    label_name = "llm-infer-tuner.owner"
    if label_name in container.labels:
        return container.labels[label_name] == owner
    return container.name in legacy_names


def _inspect_usage(remote: _Remote, request: PreflightRequest) -> _HostUsage:
    containers_result = _run_readonly(
        remote, CONTAINERS_QUERY_COMMAND, label="container inventory"
    )
    containers = _container_facts(_parse_container_inventory(containers_result.stdout))
    legacy_names = frozenset(
        {
            request.container_name,
            *(
                f"{request.container_name}-{_candidate_id(candidate)}"
                for candidate in request.candidates
            ),
        }
    )
    owned = tuple(
        container
        for container in containers
        if _is_owned_container(container, request.container_name, legacy_names)
    )
    owned_pids: set[int] = set()
    for container in owned:
        if not container.running:
            continue
        command = f"docker top {shlex.quote(container.container_id)} -eo pid"
        top = _run_readonly(remote, command, label=f"container {container.name} process inventory")
        lines = top.stdout.splitlines()
        if not lines or lines[0].strip().upper() != "PID":
            raise PreflightError(f"container {container.name} process inventory is malformed")
        owned_pids.update(
            _parse_pid_lines("\n".join(lines[1:]), label="container process inventory")
        )

    gpu_result = _run_readonly(
        remote, GPU_PIDS_QUERY_COMMAND, label="GPU process inventory"
    )
    gpu_pids = _parse_pid_lines(gpu_result.stdout, label="GPU process inventory")
    listener_result = _run_readonly(
        remote, LISTENERS_QUERY_COMMAND, label="listener inventory"
    )
    listeners = _parse_listener_ports(listener_result.stdout)
    return _HostUsage(
        containers=containers,
        owned_containers=owned,
        gpu_pids=frozenset(gpu_pids),
        owned_gpu_pids=frozenset(gpu_pids & owned_pids),
        listening_ports=frozenset(listeners),
    )


def _validate_ownership(
    usage: _HostUsage,
    plan: PreflightPlan,
    request: PreflightRequest,
) -> None:
    if request.exclusive_host:
        return
    unrelated_gpu_pids = sorted(usage.gpu_pids - usage.owned_gpu_pids)
    if unrelated_gpu_pids:
        raise PreflightError(
            f"unrelated GPU processes occupy the non-exclusive host: {unrelated_gpu_pids}"
        )
    owned_ports = {
        port
        for container in usage.owned_containers
        if container.running
        for port in container.published_ports
    }
    blocked_ports = sorted(
        (set(plan.required_ports) & set(usage.listening_ports)) - owned_ports
    )
    if blocked_ports:
        raise PreflightError(f"required port is already occupied: {blocked_ports}")


def _run_mutation(
    remote: _Remote,
    command: str,
    *,
    label: str,
    timeout: int | None = None,
) -> CommandResult:
    result = remote.run(command, timeout=timeout)
    if not result.ok:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise PreflightError(f"{label} failed with rc={result.returncode}: {detail}")
    return result


def _cleanup_owned_containers(
    remote: _Remote,
    containers: tuple[_ContainerFact, ...],
) -> None:
    running_ids = [container.container_id for container in containers if container.running]
    if running_ids:
        command = "docker stop --time 30 " + " ".join(map(shlex.quote, running_ids))
        _run_mutation(remote, command, label="owned container stop", timeout=180)
    all_ids = [container.container_id for container in containers]
    if all_ids:
        command = "docker rm -f " + " ".join(map(shlex.quote, all_ids))
        _run_mutation(remote, command, label="owned container removal", timeout=180)


def _cleanup_exclusive_host(
    remote: _Remote,
    usage: _HostUsage,
    plan: PreflightPlan,
) -> None:
    _cleanup_owned_containers(remote, usage.containers)

    gpu_result = _run_readonly(
        remote, GPU_PIDS_QUERY_COMMAND, label="post-container GPU process inventory"
    )
    remaining_gpu_pids = sorted(
        _parse_pid_lines(gpu_result.stdout, label="GPU process inventory")
    )
    if remaining_gpu_pids:
        command = "kill -9 " + " ".join(str(pid) for pid in remaining_gpu_pids)
        _run_mutation(remote, command, label="exclusive GPU process cleanup", timeout=60)

    listener_result = _run_readonly(
        remote, LISTENERS_QUERY_COMMAND, label="post-container listener inventory"
    )
    occupied_ports = _parse_listener_ports(listener_result.stdout)
    ports_to_clear = sorted(set(plan.required_ports) & occupied_ports)
    if ports_to_clear:
        command = "fuser -k " + " ".join(f"{port}/tcp" for port in ports_to_clear)
        _run_mutation(remote, command, label="exclusive port cleanup", timeout=60)


def _validate_cleanup_recheck(
    usage: _HostUsage,
    plan: PreflightPlan,
    request: PreflightRequest,
) -> None:
    if request.exclusive_host and usage.containers:
        names = sorted(container.name for container in usage.containers)
        raise PreflightError(f"containers remain after exclusive cleanup: {names}")
    if usage.owned_containers:
        names = sorted(container.name for container in usage.owned_containers)
        raise PreflightError(f"owned containers remain after cleanup: {names}")
    if usage.gpu_pids:
        raise PreflightError(
            f"GPU processes remain after cleanup: {sorted(usage.gpu_pids)}"
        )
    blocked_ports = sorted(set(plan.required_ports) & set(usage.listening_ports))
    if blocked_ports:
        raise PreflightError(f"required ports remain occupied after cleanup: {blocked_ports}")
    _validate_ownership(usage, plan, request)


def _parse_pid_lines(stdout: str, *, label: str) -> set[int]:
    pids: set[int] = set()
    for line in stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        if not value.isdigit() or int(value) < 1:
            raise PreflightError(f"{label} contains invalid PID {value!r}")
        pids.add(int(value))
    return pids


def _parse_listener_ports(stdout: str) -> set[int]:
    ports: set[int] = set()
    for row_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4:
            raise PreflightError(f"listener inventory row {row_number} is malformed")
        match = re.search(r":(\d+)$", fields[3])
        if match is None:
            raise PreflightError(f"listener inventory row {row_number} has no local port")
        ports.add(int(match.group(1)))
    return ports


def _candidate_tp(candidate: dict[str, Any]) -> int:
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    params = candidate.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"candidate {candidate.get('id', '?')}: params must be an object")
    tp_size = params.get("tp_size", 1)
    if type(tp_size) is not int or tp_size < 1:
        raise ValueError(
            f"candidate {candidate.get('id', '?')}: tp_size must be a positive integer"
        )
    return tp_size


def _candidate_id(candidate: dict[str, Any]) -> str:
    candidate_id = candidate.get("id")
    if not isinstance(candidate_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", candidate_id
    ):
        raise ValueError("candidate id must be a safe non-empty identifier")
    return candidate_id


def _normalise_topology(
    gpu_count: int,
    numa_groups: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(numa_groups, Sequence) or not numa_groups:
        raise ValueError("NUMA topology is missing")
    groups: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for raw_group in numa_groups:
        if not isinstance(raw_group, Sequence) or not raw_group:
            raise ValueError("NUMA topology contains an empty or invalid group")
        group: list[int] = []
        for gpu_id in raw_group:
            if type(gpu_id) is not int or not 0 <= gpu_id < gpu_count:
                raise ValueError("NUMA topology contains an invalid GPU ID")
            if gpu_id in seen:
                raise ValueError(f"NUMA topology contains duplicate GPU ID {gpu_id}")
            seen.add(gpu_id)
            group.append(gpu_id)
        groups.append(tuple(group))
    expected = set(range(gpu_count))
    if seen != expected:
        raise ValueError(
            f"NUMA topology is incomplete; missing GPU IDs: {sorted(expected - seen)}"
        )
    return tuple(groups)


def _allocate_batch(
    candidates: Iterable[dict[str, Any]],
    *,
    request: PreflightRequest,
    topology: tuple[tuple[int, ...], ...],
) -> tuple[CandidatePlacement, ...]:
    free_by_group = [list(group) for group in topology]
    placements: list[CandidatePlacement] = []
    for candidate in candidates:
        tp_size = _candidate_tp(candidate)
        selected: list[int] | None = None
        for free_ids in free_by_group:
            if len(free_ids) >= tp_size:
                selected = free_ids[:tp_size]
                del free_ids[:tp_size]
                break
        if selected is None:
            free_ids = [gpu_id for group in free_by_group for gpu_id in group]
            if len(free_ids) < tp_size:
                raise ValueError(
                    f"candidate {_candidate_id(candidate)}: tp_size={tp_size} cannot fit"
                )
            if not request.allow_cross_numa:
                raise ValueError(
                    f"candidate {_candidate_id(candidate)}: tp_size={tp_size} requires cross-NUMA"
                )
            selected = free_ids[:tp_size]
            selected_set = set(selected)
            for group in free_by_group:
                group[:] = [gpu_id for gpu_id in group if gpu_id not in selected_set]
        placements.append(
            CandidatePlacement(
                candidate_id=_candidate_id(candidate),
                gpu_ids=tuple(selected),
                port=request.base_port + len(placements),
            )
        )
    _validate_port_span(request.base_port, max(len(placements), 1))
    return tuple(placements)


def _plan_batches(
    request: PreflightRequest,
    topology: tuple[tuple[int, ...], ...],
) -> tuple[tuple[CandidatePlacement, ...], ...]:
    batches: list[tuple[CandidatePlacement, ...]] = []
    current: list[dict[str, Any]] = []
    for candidate in request.candidates:
        _allocate_batch((candidate,), request=request, topology=topology)
        trial = [*current, candidate]
        try:
            _allocate_batch(trial, request=request, topology=topology)
        except ValueError:
            if not current:
                raise
            batches.append(_allocate_batch(current, request=request, topology=topology))
            current = [candidate]
        else:
            current = trial
    if current:
        batches.append(_allocate_batch(current, request=request, topology=topology))
    return tuple(batches)


def _build_fill_host_placements(
    request: PreflightRequest,
    topology: tuple[tuple[int, ...], ...],
) -> tuple[FillHostPlacement, ...]:
    placements: list[FillHostPlacement] = []
    for candidate in request.candidates:
        tp_size = _candidate_tp(candidate)
        slices: list[tuple[int, ...]] = []
        leftovers: list[int] = []
        for group in topology:
            local_width = (len(group) // tp_size) * tp_size
            slices.extend(
                tuple(group[offset:offset + tp_size])
                for offset in range(0, local_width, tp_size)
            )
            leftovers.extend(group[local_width:])
        if request.allow_cross_numa:
            cross_width = (len(leftovers) // tp_size) * tp_size
            slices.extend(
                tuple(leftovers[offset:offset + tp_size])
                for offset in range(0, cross_width, tp_size)
            )
        if not slices:
            raise ValueError(
                f"candidate {_candidate_id(candidate)}: fill-host cannot place tp_size={tp_size}"
            )
        ports = tuple(request.base_port + index for index in range(len(slices)))
        _validate_port_span(request.base_port, len(ports))
        placements.append(
            FillHostPlacement(
                candidate_id=_candidate_id(candidate),
                gpu_slices=tuple(slices),
                ports=ports,
            )
        )
    return tuple(placements)


def _validate_plan(plan: PreflightPlan, request: PreflightRequest) -> None:
    expected = list(plan.candidate_ids)
    for round_name, batches in (
        ("round1", plan.round1_batches),
        ("round2", plan.round2_batches),
    ):
        observed: list[str] = []
        for batch in batches:
            batch_gpus: list[int] = []
            batch_ports: list[int] = []
            for placement in batch:
                candidate = request.candidates[expected.index(placement.candidate_id)]
                expected_gpu_width = (
                    request.execution_gpu_count
                    if request.fill_host and round_name == "round2"
                    else _candidate_tp(candidate)
                )
                if len(placement.gpu_ids) != expected_gpu_width:
                    raise ValueError(f"{round_name}: candidate TP allocation is incomplete")
                if any(
                    not 0 <= gpu_id < request.execution_gpu_count
                    for gpu_id in placement.gpu_ids
                ):
                    raise ValueError(f"{round_name}: GPU allocation is out of range")
                batch_gpus.extend(placement.gpu_ids)
                batch_ports.append(placement.port)
                observed.append(placement.candidate_id)
            if len(batch_gpus) != len(set(batch_gpus)):
                raise ValueError(f"{round_name}: duplicate GPU allocation in batch")
            if len(batch_ports) != len(set(batch_ports)):
                raise ValueError(f"{round_name}: duplicate port allocation in batch")
        if observed != expected:
            raise ValueError(f"{round_name}: candidate plan is incomplete or reordered")
    if request.fill_host:
        if [placement.candidate_id for placement in plan.fill_host_placements] != expected:
            raise ValueError("fill-host candidate plan is incomplete or reordered")
        for placement in plan.fill_host_placements:
            candidate = request.candidates[expected.index(placement.candidate_id)]
            tp_size = _candidate_tp(candidate)
            flattened = [gpu for replica in placement.gpu_slices for gpu in replica]
            if any(len(replica) != tp_size for replica in placement.gpu_slices):
                raise ValueError("fill-host replica TP allocation is incomplete")
            if len(flattened) != len(set(flattened)):
                raise ValueError("fill-host GPU allocation contains duplicates")
            if len(placement.ports) != len(set(placement.ports)):
                raise ValueError("fill-host port allocation contains duplicates")
    assigned_ports = {
        placement.port
        for batches in (plan.round1_batches, plan.round2_batches)
        for batch in batches
        for placement in batch
    }
    if assigned_ports != set(plan.required_ports):
        raise ValueError("required_ports must exactly match assigned ports")


def _validate_port_span(base_port: int, span: int) -> tuple[int, int]:
    if type(base_port) is not int or not 1 <= base_port <= 65535:
        raise ValueError("base port must be an integer in 1..65535")
    if type(span) is not int or span < 1:
        raise ValueError("port span must be a positive integer")
    end_port = base_port + span - 1
    if end_port > 65535:
        raise ValueError(
            f"port span {base_port}-{end_port} exceeds 65535 (span={span})"
        )
    return base_port, end_port


__all__ = [
    "CONTAINERS_QUERY_COMMAND",
    "CandidatePlacement",
    "FillHostPlacement",
    "GPU_PIDS_QUERY_COMMAND",
    "GPU_QUERY_COMMAND",
    "HOME_QUERY_COMMAND",
    "LISTENERS_QUERY_COMMAND",
    "NUMA_QUERY_COMMAND",
    "PreflightError",
    "PreflightPlan",
    "PreflightRequest",
    "SSH_PROBE_COMMAND",
    "build_preflight_plan",
    "prepare_remote_host",
    "validate_local_preflight",
]
