"""Fail-closed executor preflight tests (all offline)."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping

import pytest

from runners import executor as executor_module
from runners.preflight import (
    CONTAINERS_QUERY_COMMAND,
    GPU_PIDS_QUERY_COMMAND,
    GPU_QUERY_COMMAND,
    HOME_QUERY_COMMAND,
    LISTENERS_QUERY_COMMAND,
    NUMA_QUERY_COMMAND,
    SSH_PROBE_COMMAND,
    PreflightError,
    PreflightRequest,
    _gpu_model_matches,
    build_preflight_plan,
    prepare_remote_host,
    validate_local_preflight,
)
from runners.remote import CommandResult

_Response = CommandResult | list[CommandResult]


class _RecordingRemote:
    """Strict fake: every command must be explicitly supported by the test."""

    def __init__(
        self,
        responses: Mapping[str, _Response],
    ) -> None:
        self.responses = dict(responses)
        self.commands: list[str] = []

    def run(self, command: str, *, timeout=None) -> CommandResult:
        del timeout
        self.commands.append(command)
        response = self.responses.get(
            command,
            CommandResult(returncode=127, stdout="", stderr="unexpected fake command"),
        )
        if isinstance(response, list):
            if not response:
                return CommandResult(
                    returncode=127,
                    stdout="",
                    stderr="fake response sequence exhausted",
                )
            return response.pop(0)
        return response


def _ok(stdout: str = "") -> CommandResult:
    return CommandResult(returncode=0, stdout=stdout, stderr="")


def _gpu_rows(count: int = 8) -> str:
    return "".join(f"{index}, NVIDIA PRO 5000, 72832\n" for index in range(count))


def _numa_rows() -> str:
    return "".join(
        f"GPU{index} X SYS {0 if index < 4 else 1}\n" for index in range(8)
    )


def _free_host_responses(request: PreflightRequest) -> dict[str, _Response]:
    output_dir = request.remote_outputs_dir or (
        f"/home/test/llm-infer-tuner-outputs/{request.job_id}"
    )
    output_path = shlex.quote(output_dir)
    output_command = (
        f"mkdir -p -- {output_path} && test -d {output_path} && test -w {output_path}"
    )
    return {
        SSH_PROBE_COMMAND: _ok("llm-infer-tuner-preflight-ok\n"),
        HOME_QUERY_COMMAND: _ok("/home/test\n"),
        GPU_QUERY_COMMAND: _ok(_gpu_rows(request.gpu_count)),
        NUMA_QUERY_COMMAND: _ok(_numa_rows()),
        f"test -d {request.model_host_dir}": _ok(),
        f"docker image inspect {request.image_ref}": _ok("sha256:cached\n"),
        CONTAINERS_QUERY_COMMAND: _ok("[]\n"),
        GPU_PIDS_QUERY_COMMAND: _ok(),
        LISTENERS_QUERY_COMMAND: _ok(),
        output_command: _ok(),
    }


def _request(
    *,
    candidates: tuple[dict, ...] = ({"id": "c001", "params": {"tp_size": 1}},),
    base_port: int = 30000,
    exclusive_host: bool = False,
    container_name: str = "llm-infer-tuner-job",
) -> PreflightRequest:
    return PreflightRequest(
        candidates=candidates,
        gpu_model="pro5000",
        gpu_count=8,
        gpu_memory_gb=72.0,
        model_host_dir="/models/qwen",
        image_ref="registry.example/sglang:test",
        base_port=base_port,
        container_name=container_name,
        exclusive_host=exclusive_host,
    )


def _owned_container(*, running: bool = True) -> str:
    return json.dumps(
        [
            {
                "Id": "owned123",
                "Name": "/llm-infer-tuner-job-c001",
                "Config": {"Labels": {}},
                "State": {"Running": running, "Pid": 100 if running else 0},
                "NetworkSettings": {
                    "Ports": {"30000/tcp": [{"HostPort": "30000"}]}
                },
            }
        ]
    )


def _unrelated_container() -> str:
    return json.dumps(
        [
            {
                "Id": "other123",
                "Name": "/unrelated-service",
                "Config": {"Labels": {}},
                "State": {"Running": True, "Pid": 200},
                "NetworkSettings": {
                    "Ports": {"30000/tcp": [{"HostPort": "30000"}]}
                },
            }
        ]
    )


def _assert_no_cleanup(commands: list[str]) -> None:
    assert not any(
        command.startswith(("docker stop", "docker rm", "kill -9", "fuser -k"))
        for command in commands
    )


def test_empty_candidate_set_fails_during_local_preflight() -> None:
    with pytest.raises(ValueError, match="candidate"):
        validate_local_preflight(_request(candidates=()))


@pytest.mark.parametrize(
    "candidates",
    [
        ({"params": {"tp_size": 1}},),
        ({"id": "", "params": {"tp_size": 1}},),
        ({"id": 7, "params": {"tp_size": 1}},),
        (
            {"id": "same", "params": {"tp_size": 1}},
            {"id": "same", "params": {"tp_size": 1}},
        ),
    ],
)
def test_candidate_identity_fails_during_local_preflight(candidates: tuple[dict, ...]) -> None:
    with pytest.raises(ValueError, match="candidate id|candidate IDs"):
        validate_local_preflight(_request(candidates=candidates))


@pytest.mark.parametrize("memory_gb", [0, -1, True, float("nan"), float("inf"), "72"])
def test_invalid_declared_gpu_memory_fails_during_local_preflight(
    memory_gb: object,
) -> None:
    request = _request()
    request = PreflightRequest(**{**request.__dict__, "gpu_memory_gb": memory_gb})

    with pytest.raises(ValueError, match="GPU memory"):
        validate_local_preflight(request)


@pytest.mark.parametrize("base_port", [0, -1, 65536])
def test_invalid_ports_fail_during_local_preflight(
    base_port: int,
) -> None:
    candidates = (
        {"id": "c001", "params": {"tp_size": 1}},
        {"id": "c002", "params": {"tp_size": 1}},
    )

    with pytest.raises(ValueError, match="port"):
        validate_local_preflight(_request(candidates=candidates, base_port=base_port))


@pytest.mark.parametrize("tp_size", [0, -1, True, 1.5, "2", 9])
def test_invalid_or_impossible_tp_fails_during_local_preflight(tp_size: object) -> None:
    candidates = ({"id": "c001", "params": {"tp_size": tp_size}},)

    with pytest.raises(ValueError, match="tp_size"):
        validate_local_preflight(_request(candidates=candidates))


@pytest.mark.parametrize(
    "container_name",
    ["", "-starts-with-option", "job name", "job;docker-rm", "job/subpath"],
)
def test_unsafe_cleanup_owner_anchor_fails_during_local_preflight(
    container_name: str,
) -> None:
    with pytest.raises(ValueError, match="container_name"):
        validate_local_preflight(_request(container_name=container_name))


def test_preflight_plan_is_complete_disjoint_and_uses_exact_ports() -> None:
    candidates = (
        {"id": "c001", "params": {"tp_size": 3}},
        {"id": "c002", "params": {"tp_size": 3}},
        {"id": "c003", "params": {"tp_size": 2}},
    )
    request = _request(candidates=candidates)

    plan = build_preflight_plan(
        request,
        numa_groups=((0, 1, 2, 3), (4, 5, 6, 7)),
    )

    assert plan.candidate_ids == ("c001", "c002", "c003")
    assert [
        [(placement.candidate_id, placement.gpu_ids, placement.port) for placement in batch]
        for batch in plan.round1_batches
    ] == [
        [("c001", (0, 1, 2), 30000), ("c002", (4, 5, 6), 30001)],
        [("c003", (0, 1), 30000)],
    ]
    assert plan.round2_batches == plan.round1_batches
    assert plan.required_ports == (30000, 30001)


def test_port_span_waits_for_numa_plan_instead_of_flat_overestimating() -> None:
    request = PreflightRequest(
        candidates=tuple(
            {"id": f"c{index:03d}", "params": {"tp_size": 2}}
            for index in range(1, 4)
        ),
        gpu_model="pro5000",
        gpu_count=6,
        gpu_memory_gb=72.0,
        model_host_dir="/models/qwen",
        image_ref="registry.example/sglang:test",
        base_port=65534,
        container_name="llm-infer-tuner-job",
    )

    validate_local_preflight(request)
    plan = build_preflight_plan(request, numa_groups=((0, 1, 2), (3, 4, 5)))

    assert [len(batch) for batch in plan.round1_batches] == [2, 1]
    assert plan.required_ports == (65534, 65535)


def test_fill_host_plan_uses_only_real_numa_replicas_for_required_ports() -> None:
    request = PreflightRequest(
        candidates=({"id": "c001", "params": {"tp_size": 2}},),
        gpu_model="pro5000",
        gpu_count=6,
        gpu_memory_gb=72.0,
        model_host_dir="/models/qwen",
        image_ref="registry.example/sglang:test",
        base_port=65534,
        container_name="llm-infer-tuner-job",
        fill_host=True,
    )

    plan = build_preflight_plan(
        request,
        numa_groups=((0, 1, 2), (3, 4, 5)),
    )

    assert len(plan.fill_host_placements) == 1
    placement = plan.fill_host_placements[0]
    assert placement.candidate_id == "c001"
    assert placement.gpu_slices == ((0, 1), (3, 4))
    assert placement.ports == (65534, 65535)
    assert plan.required_ports == (65534,)


def test_fill_host_exact_replica_ports_reject_overflow_after_topology() -> None:
    request = PreflightRequest(
        candidates=({"id": "c001", "params": {"tp_size": 2}},),
        gpu_model="pro5000",
        gpu_count=6,
        gpu_memory_gb=72.0,
        model_host_dir="/models/qwen",
        image_ref="registry.example/sglang:test",
        base_port=65535,
        container_name="llm-infer-tuner-job",
        fill_host=True,
    )

    validate_local_preflight(request)
    with pytest.raises(ValueError, match="port span"):
        build_preflight_plan(request, numa_groups=((0, 1, 2), (3, 4, 5)))


def test_fill_host_internal_replica_port_is_not_treated_as_published_host_port() -> None:
    request = PreflightRequest(
        candidates=({"id": "c001", "params": {"tp_size": 2}},),
        gpu_model="pro5000",
        gpu_count=6,
        gpu_memory_gb=72.0,
        model_host_dir="/models/qwen",
        image_ref="registry.example/sglang:test",
        base_port=30000,
        container_name="llm-infer-tuner-job",
        fill_host=True,
    )
    responses = _free_host_responses(request)
    responses[GPU_QUERY_COMMAND] = _ok(_gpu_rows(6))
    responses[NUMA_QUERY_COMMAND] = _ok(
        "".join(
            f"GPU{index} X SYS {0 if index < 3 else 1}\n" for index in range(6)
        )
    )
    responses[LISTENERS_QUERY_COMMAND] = _ok(
        "LISTEN 0 4096 0.0.0.0:30001 0.0.0.0:*\n"
    )
    remote = _RecordingRemote(responses)

    plan = prepare_remote_host(remote, request)

    assert plan.required_ports == (30000,)
    assert plan.fill_host_placements[0].ports == (30000, 30001)


def test_fill_host_round2_runs_each_candidate_in_a_singleton_whole_host_batch() -> None:
    request = _request(
        candidates=(
            {"id": "c001", "params": {"tp_size": 2}},
            {"id": "c002", "params": {"tp_size": 4}},
        )
    )
    request = PreflightRequest(**{**request.__dict__, "fill_host": True})

    plan = build_preflight_plan(
        request,
        numa_groups=((0, 1, 2, 3), (4, 5, 6, 7)),
    )

    assert [
        [(placement.candidate_id, placement.gpu_ids) for placement in batch]
        for batch in plan.round2_batches
    ] == [
        [("c001", tuple(range(8)))],
        [("c002", tuple(range(8)))],
    ]


def test_remote_preflight_success_is_read_only_when_host_is_free() -> None:
    request = _request()
    model_command = "test -d /models/qwen"
    image_command = "docker image inspect registry.example/sglang:test"
    remote = _RecordingRemote(
        {
            SSH_PROBE_COMMAND: _ok("llm-infer-tuner-preflight-ok\n"),
            HOME_QUERY_COMMAND: _ok("/home/test\n"),
            GPU_QUERY_COMMAND: _ok(_gpu_rows()),
            NUMA_QUERY_COMMAND: _ok(_numa_rows()),
            model_command: _ok(),
            image_command: _ok("sha256:cached\n"),
            CONTAINERS_QUERY_COMMAND: _ok("[]\n"),
            GPU_PIDS_QUERY_COMMAND: _ok(),
            LISTENERS_QUERY_COMMAND: _ok(),
            (
                "mkdir -p -- /home/test/llm-infer-tuner-outputs/preflight && "
                "test -d /home/test/llm-infer-tuner-outputs/preflight && "
                "test -w /home/test/llm-infer-tuner-outputs/preflight"
            ): _ok(),
        }
    )

    plan = prepare_remote_host(remote, request)

    assert plan.candidate_ids == ("c001",)
    assert plan.required_ports == (30000,)
    assert plan.outputs_host_dir == "/home/test/llm-infer-tuner-outputs/preflight"
    assert remote.commands == [
        SSH_PROBE_COMMAND,
        HOME_QUERY_COMMAND,
        GPU_QUERY_COMMAND,
        NUMA_QUERY_COMMAND,
        model_command,
        image_command,
        CONTAINERS_QUERY_COMMAND,
        GPU_PIDS_QUERY_COMMAND,
        LISTENERS_QUERY_COMMAND,
        (
            "mkdir -p -- /home/test/llm-infer-tuner-outputs/preflight && "
            "test -d /home/test/llm-infer-tuner-outputs/preflight && "
            "test -w /home/test/llm-infer-tuner-outputs/preflight"
        ),
        CONTAINERS_QUERY_COMMAND,
        GPU_PIDS_QUERY_COMMAND,
        LISTENERS_QUERY_COMMAND,
    ]
    assert not any(
        marker in command
        for command in remote.commands
        for marker in ("docker stop", "docker rm", "kill -9", "fuser -k", "docker pull")
    )


def test_observed_topology_rejection_is_remote_preflight_error_without_mutation() -> None:
    request = _request(
        candidates=({"id": "wide", "params": {"tp_size": 5}},),
    )
    remote = _RecordingRemote(_free_host_responses(request))

    with pytest.raises(PreflightError, match="cross-NUMA"):
        prepare_remote_host(remote, request)

    assert not any(
        marker in command
        for command in remote.commands
        for marker in ("docker stop", "docker rm", "kill -9", "fuser -k", "docker pull")
    )


def test_unowned_required_port_fails_without_remote_mutation() -> None:
    request = _request()
    responses = _free_host_responses(request)
    responses[LISTENERS_QUERY_COMMAND] = _ok(
        "LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"
    )
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="port"):
        prepare_remote_host(remote, request)

    assert not any(
        marker in command
        for command in remote.commands
        for marker in ("docker stop", "docker rm", "kill -9", "fuser -k", "docker pull")
    )


def test_default_ownership_removes_only_job_containers_then_rechecks() -> None:
    request = _request()
    owned = json.dumps(
        [
            {
                "Id": "owned123",
                "Name": "/llm-infer-tuner-job-c001",
                "Config": {"Labels": {}},
                "State": {"Running": True, "Pid": 100},
                "NetworkSettings": {
                    "Ports": {"30000/tcp": [{"HostPort": "30000"}]}
                },
            }
        ]
    )
    responses = _free_host_responses(request)
    responses[CONTAINERS_QUERY_COMMAND] = [_ok(owned), _ok("[]\n")]
    responses[GPU_PIDS_QUERY_COMMAND] = [_ok("111\n"), _ok()]
    responses[LISTENERS_QUERY_COMMAND] = [
        _ok("LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"),
        _ok(),
    ]
    responses["docker top owned123 -eo pid"] = _ok("PID\n111\n")
    responses["docker stop --time 30 owned123"] = _ok("owned123\n")
    responses["docker rm -f owned123"] = _ok("owned123\n")
    remote = _RecordingRemote(responses)

    plan = prepare_remote_host(remote, request)

    assert plan.required_ports == (30000,)
    assert "docker stop --time 30 owned123" in remote.commands
    assert "docker rm -f owned123" in remote.commands
    assert not any("kill -9" in command or "fuser -k" in command for command in remote.commands)
    assert not any(
        command.startswith("docker rm") and "owned123" not in command
        for command in remote.commands
    )


def test_stopped_owned_container_cannot_claim_an_unrelated_listener() -> None:
    request = _request()
    stopped_owned = json.dumps(
        [
            {
                "Id": "owned123",
                "Name": "/llm-infer-tuner-job-c001",
                "Config": {"Labels": {}},
                "State": {"Running": False, "Pid": 0},
                "NetworkSettings": {
                    "Ports": {"30000/tcp": [{"HostPort": "30000"}]}
                },
            }
        ]
    )
    responses = _free_host_responses(request)
    responses[CONTAINERS_QUERY_COMMAND] = _ok(stopped_owned)
    responses[LISTENERS_QUERY_COMMAND] = _ok(
        "LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"
    )
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="port"):
        prepare_remote_host(remote, request)

    assert not any(command.startswith("docker rm") for command in remote.commands)


@pytest.mark.parametrize(
    ("name", "labels"),
    [
        (
            "llm-infer-tuner-job-foreign-c001",
            {"llm-infer-tuner.owner": "llm-infer-tuner-job-foreign"},
        ),
        ("llm-infer-tuner-job-c999", {}),
    ],
)
def test_foreign_label_or_unknown_legacy_name_cannot_be_claimed_by_prefix(
    name: str,
    labels: dict[str, str],
) -> None:
    request = _request()
    foreign = json.dumps(
        [
            {
                "Id": "foreign123",
                "Name": f"/{name}",
                "Config": {"Labels": labels},
                "State": {"Running": True, "Pid": 200},
                "NetworkSettings": {
                    "Ports": {"30000/tcp": [{"HostPort": "30000"}]}
                },
            }
        ]
    )
    responses = _free_host_responses(request)
    responses[CONTAINERS_QUERY_COMMAND] = _ok(foreign)
    responses[LISTENERS_QUERY_COMMAND] = _ok(
        "LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"
    )
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="port"):
        prepare_remote_host(remote, request)

    _assert_no_cleanup(remote.commands)


def test_final_docker_container_name_length_fails_during_local_preflight() -> None:
    request = _request(
        candidates=({"id": "c" * 128, "params": {"tp_size": 1}},),
        container_name="o" * 128,
    )

    with pytest.raises(ValueError, match="Docker container name"):
        validate_local_preflight(request)


def test_missing_image_is_pulled_only_after_all_readonly_checks() -> None:
    request = _request()
    responses = _free_host_responses(request)
    inspect_command = f"docker image inspect {request.image_ref}"
    pull_command = f"docker pull {request.image_ref}"
    responses[inspect_command] = [
        CommandResult(returncode=1, stdout="", stderr="No such image"),
        _ok("sha256:pulled\n"),
    ]
    responses[pull_command] = _ok("pulled\n")
    remote = _RecordingRemote(responses)

    prepare_remote_host(remote, request)

    assert remote.commands.count(inspect_command) == 2
    assert remote.commands.count(pull_command) == 1
    assert remote.commands.index(pull_command) > remote.commands.index(LISTENERS_QUERY_COMMAND)
    assert not any(
        marker in command
        for command in remote.commands
        for marker in ("docker stop", "docker rm", "kill -9", "fuser -k")
    )


def test_final_inventory_catches_foreign_gpu_process_created_during_image_pull() -> None:
    request = _request()
    image_command = f"docker image inspect {request.image_ref}"
    pull_command = f"docker pull {request.image_ref}"
    responses = _free_host_responses(request)
    responses[image_command] = [
        CommandResult(1, "", "No such image"),
        _ok("sha256:pulled\n"),
    ]
    responses[pull_command] = _ok("pulled\n")
    responses[CONTAINERS_QUERY_COMMAND] = [_ok("[]\n"), _ok("[]\n")]
    responses[GPU_PIDS_QUERY_COMMAND] = [_ok(), _ok("999\n")]
    responses[LISTENERS_QUERY_COMMAND] = [_ok(), _ok()]
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="GPU processes remain"):
        prepare_remote_host(remote, request)

    assert remote.commands.count(GPU_PIDS_QUERY_COMMAND) == 2
    _assert_no_cleanup(remote.commands)


def test_final_inventory_rejects_job_named_container_created_during_image_pull() -> None:
    request = _request()
    image_command = f"docker image inspect {request.image_ref}"
    pull_command = f"docker pull {request.image_ref}"
    responses = _free_host_responses(request)
    responses[image_command] = [
        CommandResult(1, "", "No such image"),
        _ok("sha256:pulled\n"),
    ]
    responses[pull_command] = _ok("pulled\n")
    responses[CONTAINERS_QUERY_COMMAND] = [_ok("[]\n"), _ok(_owned_container())]
    responses[GPU_PIDS_QUERY_COMMAND] = [_ok(), _ok("111\n")]
    responses[LISTENERS_QUERY_COMMAND] = [
        _ok(),
        _ok("LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"),
    ]
    responses["docker top owned123 -eo pid"] = _ok("PID\n111\n")
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="owned containers remain"):
        prepare_remote_host(remote, request)

    _assert_no_cleanup(remote.commands)


@pytest.mark.parametrize("memory", ["nan", "inf", "-inf"])
def test_non_finite_gpu_memory_fails_without_remote_mutation(memory: str) -> None:
    request = _request()
    responses = _free_host_responses(request)
    responses[GPU_QUERY_COMMAND] = _ok(
        "".join(
            f"{index}, NVIDIA PRO 5000, {memory if index == 0 else 72832}\n"
            for index in range(request.gpu_count)
        )
    )
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="memory"):
        prepare_remote_host(remote, request)

    _assert_no_cleanup(remote.commands)


@pytest.mark.parametrize(
    ("declared", "actual"),
    [("a100", "NVIDIA A1000"), ("h20", "NVIDIA H200")],
)
def test_gpu_model_matching_rejects_substring_collisions(
    declared: str,
    actual: str,
) -> None:
    request = PreflightRequest(**{**_request().__dict__, "gpu_model": declared})
    responses = _free_host_responses(request)
    responses[GPU_QUERY_COMMAND] = _ok(
        "".join(
            f"{index}, {actual}, 72832\n" for index in range(request.gpu_count)
        )
    )
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="GPU model"):
        prepare_remote_host(remote, request)

    _assert_no_cleanup(remote.commands)


@pytest.mark.parametrize(
    ("declared", "actual"),
    [
        ("G12_l40", "NVIDIA L40"),
        ("G13_l40s", "NVIDIA L40S"),
        ("G24_pro5000", "NVIDIA RTX PRO 5000 Blackwell"),
    ],
)
def test_gpu_model_matching_accepts_complete_adjacent_model_tokens(
    declared: str,
    actual: str,
) -> None:
    assert _gpu_model_matches(declared, actual)


@pytest.mark.parametrize(
    ("declared", "actual"),
    [
        ("G12_l40", "NVIDIA L40S"),
        ("G24_pro5000", "NVIDIA RTX PRO 50000 Blackwell"),
        ("G08_a10", "NVIDIA A100"),
    ],
)
def test_gpu_model_matching_rejects_adjacent_product_names(
    declared: str,
    actual: str,
) -> None:
    assert not _gpu_model_matches(declared, actual)


@pytest.mark.parametrize(
    ("declared_gb", "actual_mib"),
    [(24.0, 23028), (48.0, 46068), (72.0, 72832)],
)
def test_gpu_memory_accepts_binary_or_decimal_vendor_capacity(
    declared_gb: float,
    actual_mib: int,
) -> None:
    request = PreflightRequest(
        **{**_request().__dict__, "gpu_memory_gb": declared_gb}
    )
    responses = _free_host_responses(request)
    responses[GPU_QUERY_COMMAND] = _ok(
        "".join(
            f"{index}, NVIDIA PRO 5000, {actual_mib}\n"
            for index in range(request.gpu_count)
        )
    )

    plan = prepare_remote_host(_RecordingRemote(responses), request)

    assert plan.candidate_ids == ("c001",)


@pytest.mark.parametrize("actual_mib", [81920])
def test_declared_72g_rejects_materially_different_actual_memory(
    actual_mib: int,
) -> None:
    request = _request()
    responses = _free_host_responses(request)
    responses[GPU_QUERY_COMMAND] = _ok(
        "".join(
            f"{index}, NVIDIA PRO 5000, {actual_mib}\n"
            for index in range(request.gpu_count)
        )
    )
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="GPU memory"):
        prepare_remote_host(remote, request)

    _assert_no_cleanup(remote.commands)


@pytest.mark.parametrize(
    ("failed_command", "result", "message"),
    [
        (GPU_QUERY_COMMAND, CommandResult(0, _gpu_rows(7), ""), "GPU count"),
        (NUMA_QUERY_COMMAND, _ok(""), "NUMA topology"),
        (
            "test -d /models/qwen",
            CommandResult(1, "", "missing"),
            "model directory",
        ),
        (
            "docker image inspect registry.example/sglang:test",
            CommandResult(255, "", "transport lost"),
            "image inspection",
        ),
    ],
)
def test_remote_fact_failure_never_mutates_host(
    failed_command: str,
    result: CommandResult,
    message: str,
) -> None:
    request = _request()
    responses = _free_host_responses(request)
    responses[failed_command] = result
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match=message):
        prepare_remote_host(remote, request)

    _assert_no_cleanup(remote.commands)
    assert not any(command.startswith("docker pull") for command in remote.commands)


def test_image_pull_failure_never_starts_owned_cleanup() -> None:
    request = _request()
    image_command = f"docker image inspect {request.image_ref}"
    pull_command = f"docker pull {request.image_ref}"
    responses = _free_host_responses(request)
    responses[image_command] = CommandResult(1, "", "No such image")
    responses[CONTAINERS_QUERY_COMMAND] = _ok(_owned_container())
    responses[GPU_PIDS_QUERY_COMMAND] = _ok("111\n")
    responses[LISTENERS_QUERY_COMMAND] = _ok(
        "LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"
    )
    responses["docker top owned123 -eo pid"] = _ok("PID\n111\n")
    responses[pull_command] = CommandResult(1, "", "registry unavailable")
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="image pull"):
        prepare_remote_host(remote, request)

    assert pull_command in remote.commands
    _assert_no_cleanup(remote.commands)


def test_image_inspect_rc1_permission_error_is_not_treated_as_missing() -> None:
    request = _request()
    image_command = f"docker image inspect {request.image_ref}"
    responses = _free_host_responses(request)
    responses[image_command] = CommandResult(1, "", "permission denied")
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="image inspection"):
        prepare_remote_host(remote, request)

    assert not any(command.startswith("docker pull") for command in remote.commands)
    _assert_no_cleanup(remote.commands)


@pytest.mark.parametrize("home", ["relative/home", "/home/../other"])
def test_invalid_remote_home_is_wrapped_as_preflight_error(home: str) -> None:
    request = _request()
    responses = _free_host_responses(request)
    responses[HOME_QUERY_COMMAND] = _ok(f"{home}\n")
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="remote HOME"):
        prepare_remote_host(remote, request)


def test_remote_output_directory_failure_happens_before_owned_cleanup() -> None:
    request = PreflightRequest(
        **{
            **_request().__dict__,
            "remote_outputs_dir": "/remote/benchmark results",
            "job_id": "job-output-preflight",
        }
    )
    responses = _free_host_responses(request)
    responses[CONTAINERS_QUERY_COMMAND] = _ok(_owned_container())
    responses[GPU_PIDS_QUERY_COMMAND] = _ok("111\n")
    responses[LISTENERS_QUERY_COMMAND] = _ok(
        "LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"
    )
    responses["docker top owned123 -eo pid"] = _ok("PID\n111\n")
    output_command = (
        "mkdir -p -- '/remote/benchmark results' && "
        "test -d '/remote/benchmark results' && test -w '/remote/benchmark results'"
    )
    responses[output_command] = CommandResult(1, "", "permission denied")
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="output directory"):
        prepare_remote_host(remote, request)

    assert output_command in remote.commands
    _assert_no_cleanup(remote.commands)


@pytest.mark.parametrize("failure_stage", ["stop", "remove"])
def test_scoped_cleanup_failure_aborts_later_steps(failure_stage: str) -> None:
    request = _request()
    responses = _free_host_responses(request)
    responses[CONTAINERS_QUERY_COMMAND] = _ok(_owned_container())
    responses[GPU_PIDS_QUERY_COMMAND] = _ok("111\n")
    responses[LISTENERS_QUERY_COMMAND] = _ok(
        "LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"
    )
    responses["docker top owned123 -eo pid"] = _ok("PID\n111\n")
    responses["docker stop --time 30 owned123"] = (
        CommandResult(1, "", "stop failed") if failure_stage == "stop" else _ok()
    )
    responses["docker rm -f owned123"] = CommandResult(1, "", "remove failed")
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="owned container"):
        prepare_remote_host(remote, request)

    if failure_stage == "stop":
        assert "docker rm -f owned123" not in remote.commands
    assert remote.commands.count(CONTAINERS_QUERY_COMMAND) == 1


def test_post_cleanup_recheck_rejects_remaining_owned_container() -> None:
    request = _request()
    owned = _owned_container()
    responses = _free_host_responses(request)
    responses[CONTAINERS_QUERY_COMMAND] = [_ok(owned), _ok(owned)]
    responses[GPU_PIDS_QUERY_COMMAND] = [_ok("111\n"), _ok("111\n")]
    responses[LISTENERS_QUERY_COMMAND] = [
        _ok("LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"),
        _ok("LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"),
    ]
    responses["docker top owned123 -eo pid"] = [
        _ok("PID\n111\n"),
        _ok("PID\n111\n"),
    ]
    responses["docker stop --time 30 owned123"] = _ok()
    responses["docker rm -f owned123"] = _ok()
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="remain"):
        prepare_remote_host(remote, request)

    assert remote.commands.count(CONTAINERS_QUERY_COMMAND) == 2


@pytest.mark.parametrize(
    ("residual", "second_gpu", "second_listener", "message"),
    [
        ("gpu", "999\n", "", "GPU processes remain"),
        (
            "port",
            "",
            "LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n",
            "ports remain",
        ),
    ],
)
def test_post_cleanup_recheck_rejects_gpu_or_port_residual(
    residual: str,
    second_gpu: str,
    second_listener: str,
    message: str,
) -> None:
    del residual
    request = _request()
    responses = _free_host_responses(request)
    responses[CONTAINERS_QUERY_COMMAND] = [_ok(_owned_container()), _ok("[]\n")]
    responses[GPU_PIDS_QUERY_COMMAND] = [_ok("111\n"), _ok(second_gpu)]
    responses[LISTENERS_QUERY_COMMAND] = [
        _ok("LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"),
        _ok(second_listener),
    ]
    responses["docker top owned123 -eo pid"] = _ok("PID\n111\n")
    responses["docker stop --time 30 owned123"] = _ok()
    responses["docker rm -f owned123"] = _ok()
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match=message):
        prepare_remote_host(remote, request)

    assert remote.commands.count(CONTAINERS_QUERY_COMMAND) == 2


def test_container_inventory_shell_propagates_docker_ps_failure() -> None:
    assert "ids=$(docker ps -aq) || exit $?" in CONTAINERS_QUERY_COMMAND


def test_explicit_exclusive_host_cleans_recorded_resources_in_checked_steps() -> None:
    request = _request(exclusive_host=True)
    unrelated = _unrelated_container()
    responses = _free_host_responses(request)
    responses[CONTAINERS_QUERY_COMMAND] = [_ok(unrelated), _ok("[]\n")]
    responses[GPU_PIDS_QUERY_COMMAND] = [_ok("222\n"), _ok("222\n"), _ok()]
    responses[LISTENERS_QUERY_COMMAND] = [
        _ok("LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"),
        _ok("LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"),
        _ok(),
    ]
    responses["docker stop --time 30 other123"] = _ok("other123\n")
    responses["docker rm -f other123"] = _ok("other123\n")
    responses["kill -9 222"] = _ok()
    responses["fuser -k 30000/tcp"] = _ok("30000/tcp: 333\n")
    remote = _RecordingRemote(responses)

    prepare_remote_host(remote, request)

    destructive = [
        command
        for command in remote.commands
        if command.startswith(("docker stop", "docker rm", "kill -9", "fuser -k"))
    ]
    assert destructive == [
        "docker stop --time 30 other123",
        "docker rm -f other123",
        "kill -9 222",
        "fuser -k 30000/tcp",
    ]


@pytest.mark.parametrize(
    ("failure_stage", "expected_destructive"),
    [
        ("stop", ["docker stop --time 30 other123"]),
        (
            "remove",
            ["docker stop --time 30 other123", "docker rm -f other123"],
        ),
        (
            "kill",
            [
                "docker stop --time 30 other123",
                "docker rm -f other123",
                "kill -9 222",
            ],
        ),
        (
            "port",
            [
                "docker stop --time 30 other123",
                "docker rm -f other123",
                "fuser -k 30000/tcp",
            ],
        ),
    ],
)
def test_exclusive_cleanup_failure_aborts_every_later_step(
    failure_stage: str,
    expected_destructive: list[str],
) -> None:
    request = _request(exclusive_host=True)
    responses = _free_host_responses(request)
    responses[CONTAINERS_QUERY_COMMAND] = _ok(_unrelated_container())
    responses[GPU_PIDS_QUERY_COMMAND] = [
        _ok("222\n"),
        _ok("222\n" if failure_stage == "kill" else ""),
    ]
    responses[LISTENERS_QUERY_COMMAND] = [
        _ok("LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"),
        _ok("LISTEN 0 4096 0.0.0.0:30000 0.0.0.0:*\n"),
    ]
    responses["docker stop --time 30 other123"] = (
        CommandResult(1, "", "stop failed") if failure_stage == "stop" else _ok()
    )
    responses["docker rm -f other123"] = (
        CommandResult(1, "", "remove failed") if failure_stage == "remove" else _ok()
    )
    responses["kill -9 222"] = CommandResult(1, "", "kill failed")
    responses["fuser -k 30000/tcp"] = CommandResult(1, "", "fuser failed")
    remote = _RecordingRemote(responses)

    with pytest.raises(PreflightError, match="failed"):
        prepare_remote_host(remote, request)

    destructive = [
        command
        for command in remote.commands
        if command.startswith(("docker stop", "docker rm", "kill -9", "fuser -k"))
    ]
    assert destructive == expected_destructive
    assert remote.commands.count(CONTAINERS_QUERY_COMMAND) == 1


def test_legacy_unscoped_cleanup_entrypoints_are_removed() -> None:
    assert not hasattr(executor_module, "_reclaim_host")
    assert not hasattr(executor_module, "_preflight_checks")
