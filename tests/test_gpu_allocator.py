from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from itertools import product
from typing import Any, cast

import pytest

from runners.executor import (
    _allocate_gpus_and_ports,
    _detect_numa_groups,
    _plan_candidate_batches,
    _plan_fill_host_replica_slices,
)


def _candidates(tp_sizes: Sequence[int | None]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, tp_size in enumerate(tp_sizes, start=1):
        params = {} if tp_size is None else {"tp_size": tp_size}
        rows.append({"id": f"c{index:03d}", "params": params})
    return rows


def _gpu_ids(device_spec: str) -> list[int]:
    assert device_spec.startswith("device=")
    return [int(value) for value in device_spec.removeprefix("device=").split(",")]


def test_three_numa_fragmentation_never_reuses_a_gpu() -> None:
    candidates = _candidates([3, 3, 3, 2, 1])
    before = deepcopy(candidates)
    topology = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))]

    allocations = _allocate_gpus_and_ports(
        candidates,
        gpu_count=12,
        base_port=30000,
        numa_groups=topology,
        allow_cross_numa=True,
    )

    assigned = [_gpu_ids(device_spec) for _, device_spec, _ in allocations]
    flattened = [gpu_id for gpu_ids in assigned for gpu_id in gpu_ids]
    assert [len(gpu_ids) for gpu_ids in assigned] == [3, 3, 3, 2, 1]
    assert len(flattened) == len(set(flattened)) == 12
    assert set(flattened) == set(range(12))
    assert candidates == before


def test_default_policy_refuses_cross_numa_fragmentation() -> None:
    topology = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))]

    with pytest.raises(ValueError, match="cross-NUMA"):
        _allocate_gpus_and_ports(
            _candidates([3, 3, 3, 2, 1]),
            gpu_count=12,
            numa_groups=topology,
        )


@pytest.mark.parametrize(
    "gpu_count,numa_groups",
    [
        (0, [[0]]),
        (-1, [[0]]),
        (True, [[0]]),
        (4, None),
        (4, []),
        (4, [[]]),
        (4, [[0, 1], [1, 2, 3]]),
        (4, [[0, 1], [2, 4]]),
        (4, [[0, 1], [2]]),
        (4, [[0, 1], [2, "3"]]),
    ],
)
def test_invalid_or_incomplete_topology_fails_closed(
    gpu_count: int,
    numa_groups: list[list[int]] | None,
) -> None:
    with pytest.raises(ValueError, match="GPU|NUMA|topology"):
        _allocate_gpus_and_ports(
            _candidates([1]),
            gpu_count=gpu_count,
            numa_groups=numa_groups,
        )


@pytest.mark.parametrize("tp_size", [0, -1, True, 1.5, "2"])
def test_non_positive_or_non_integer_tp_is_rejected(tp_size: Any) -> None:
    with pytest.raises(ValueError, match="tp_size"):
        _allocate_gpus_and_ports(
            _candidates([tp_size]),
            gpu_count=4,
            numa_groups=[[0, 1, 2, 3]],
        )


def test_missing_tp_defaults_to_one() -> None:
    allocations = _allocate_gpus_and_ports(
        _candidates([None]),
        gpu_count=2,
        numa_groups=[[0, 1]],
    )

    assert _gpu_ids(allocations[0][1]) == [0]


@pytest.mark.parametrize(
    "gpu_count,numa_groups,tp_sizes",
    [
        (4, [[0, 1, 2, 3]], [1, 1, 2]),
        (8, [[0, 1, 2, 3], [4, 5, 6, 7]], [2, 2, 4]),
        (12, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [3, 1, 4, 2, 2]),
    ],
)
def test_feasible_allocations_are_disjoint_deterministic_and_numa_local(
    gpu_count: int,
    numa_groups: list[list[int]],
    tp_sizes: list[int],
) -> None:
    candidates = _candidates(tp_sizes)
    before = deepcopy(candidates)

    first = _allocate_gpus_and_ports(
        candidates,
        gpu_count=gpu_count,
        base_port=32000,
        numa_groups=numa_groups,
    )
    second = _allocate_gpus_and_ports(
        candidates,
        gpu_count=gpu_count,
        base_port=32000,
        numa_groups=numa_groups,
    )

    assert first == second
    assigned = [_gpu_ids(device_spec) for _, device_spec, _ in first]
    flattened = [gpu_id for gpu_ids in assigned for gpu_id in gpu_ids]
    assert [len(gpu_ids) for gpu_ids in assigned] == tp_sizes
    assert len(flattened) == len(set(flattened))
    assert all(0 <= gpu_id < gpu_count for gpu_id in flattened)
    assert all(any(set(gpu_ids) <= set(group) for group in numa_groups) for gpu_ids in assigned)
    assert [port for _, _, port in first] == list(range(32000, 32000 + len(first)))
    assert candidates == before


def test_total_capacity_exhaustion_is_explicit() -> None:
    with pytest.raises(ValueError, match="只能分到"):
        _allocate_gpus_and_ports(
            _candidates([4, 1]),
            gpu_count=4,
            numa_groups=[[0, 1, 2, 3]],
        )


@pytest.mark.parametrize("base_port", [0, -1, 65536, 65535])
def test_allocator_rejects_invalid_or_overflowing_port_span(base_port: int) -> None:
    with pytest.raises(ValueError, match="port"):
        _allocate_gpus_and_ports(
            _candidates([1, 1]),
            gpu_count=2,
            base_port=base_port,
            numa_groups=[[0, 1]],
        )


def test_batch_planner_splits_fragmentation_instead_of_crossing_numa() -> None:
    candidates = _candidates([3, 3, 3, 2, 1])
    before = deepcopy(candidates)
    topology = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))]

    batches = _plan_candidate_batches(
        candidates,
        gpu_count=12,
        base_port=30000,
        numa_groups=topology,
    )

    assert [[candidate["id"] for candidate in batch] for batch in batches] == [
        ["c001", "c002", "c003"],
        ["c004", "c005"],
    ]
    for batch in batches:
        allocations = _allocate_gpus_and_ports(
            batch,
            gpu_count=12,
            base_port=30000,
            numa_groups=topology,
        )
        assigned = [_gpu_ids(device_spec) for _, device_spec, _ in allocations]
        assert all(any(set(ids) <= set(group) for group in topology) for ids in assigned)
    assert candidates == before


def test_batch_planner_is_deterministic_and_rejects_unplaceable_candidate() -> None:
    topology = [[0, 1], [2, 3]]
    candidates = _candidates([2, 1, 1, 2])

    first = _plan_candidate_batches(candidates, 4, numa_groups=topology)
    second = _plan_candidate_batches(candidates, 4, numa_groups=topology)
    assert first == second

    with pytest.raises(ValueError, match="cross-NUMA"):
        _plan_candidate_batches(_candidates([3]), 4, numa_groups=topology)


def test_batch_planner_reuses_ports_across_sequential_batches() -> None:
    candidates = _candidates([1, 1, 1])

    batches = _plan_candidate_batches(
        candidates,
        gpu_count=2,
        base_port=65534,
        numa_groups=[[0, 1]],
    )

    assert [[row["id"] for row in batch] for batch in batches] == [
        ["c001", "c002"],
        ["c003"],
    ]
    assert [
        [port for _, _, port in _allocate_gpus_and_ports(
            batch,
            gpu_count=2,
            base_port=65534,
            numa_groups=[[0, 1]],
        )]
        for batch in batches
    ] == [[65534, 65535], [65534]]


def test_fill_host_replica_count_uses_real_numa_placements() -> None:
    topology = [[0, 1, 2], [3, 4, 5]]
    topology_before = deepcopy(topology)

    assert _plan_fill_host_replica_slices(
        list(range(6)),
        tp_size=2,
        numa_groups=topology,
    ) == [[0, 1], [3, 4]]
    assert _plan_fill_host_replica_slices(
        list(range(6)),
        tp_size=2,
        numa_groups=topology,
        allow_cross_numa=True,
    ) == [[0, 1], [3, 4], [2, 5]]
    assert topology == topology_before


def test_fill_host_explicit_cross_numa_allows_one_tp8_replica() -> None:
    topology = [[0, 1, 2, 3], [4, 5, 6, 7]]

    assert _plan_fill_host_replica_slices(
        list(range(8)),
        tp_size=8,
        numa_groups=topology,
    ) == []
    assert _plan_fill_host_replica_slices(
        list(range(8)),
        tp_size=8,
        numa_groups=topology,
        allow_cross_numa=True,
    ) == [list(range(8))]


def test_generated_tp_sequences_preserve_allocator_invariants() -> None:
    for group_count in (1, 2, 3):
        gpu_count = group_count * 4
        topology = [
            list(range(group_index * 4, (group_index + 1) * 4))
            for group_index in range(group_count)
        ]
        topology_before = deepcopy(topology)
        for tp_sizes in product(range(1, 5), repeat=3):
            candidates = _candidates(tp_sizes)
            candidates_before = deepcopy(candidates)
            batches = _plan_candidate_batches(
                candidates,
                gpu_count,
                base_port=40000,
                numa_groups=topology,
            )

            assert [row for batch in batches for row in batch] == candidates
            for batch in batches:
                allocations = _allocate_gpus_and_ports(
                    batch,
                    gpu_count,
                    base_port=40000,
                    numa_groups=topology,
                )
                gpu_sets = [set(_gpu_ids(spec)) for _, spec, _ in allocations]
                flattened = set().union(*gpu_sets)
                assert sum(len(gpu_set) for gpu_set in gpu_sets) == len(flattened)
                assert all(gpu_set <= set(range(gpu_count)) for gpu_set in gpu_sets)
                assert all(
                    any(gpu_set <= set(group) for group in topology)
                    for gpu_set in gpu_sets
                )
                assert [port for _, _, port in allocations] == list(
                    range(40000, 40000 + len(batch))
                )
            assert candidates == candidates_before
            assert topology == topology_before


class _RemoteResult:
    def __init__(self, *, ok: bool, stdout: str = "", stderr: str = "") -> None:
        self.ok = ok
        self.stdout = stdout
        self.stderr = stderr


class _TopologyRemote:
    def __init__(self, result: _RemoteResult) -> None:
        self.result = result

    def run(self, command: str, *, timeout: int) -> _RemoteResult:
        assert command == "nvidia-smi topo -m"
        assert timeout == 30
        return self.result


def test_detect_numa_groups_returns_complete_validated_topology() -> None:
    result = _RemoteResult(
        ok=True,
        stdout=(
            "        GPU0 GPU1 GPU2 GPU3 NUMA Affinity\n"
            "GPU0    X    NV1  SYS  SYS  0\n"
            "GPU1    NV1  X    SYS  SYS  0\n"
            "GPU2    SYS  SYS  X    NV1  1\n"
            "GPU3    SYS  SYS  NV1  X    1\n"
        ),
    )

    assert _detect_numa_groups(cast(Any, _TopologyRemote(result)), 4) == [[0, 1], [2, 3]]


@pytest.mark.parametrize(
    "result",
    [
        _RemoteResult(ok=False, stderr="nvidia-smi failed"),
        _RemoteResult(ok=True, stdout=""),
        _RemoteResult(ok=True, stdout="GPU0 X SYS 0\nGPU1 SYS X 0\n"),
        _RemoteResult(ok=True, stdout="GPU0 X SYS 0\nGPU0 X SYS 0\nGPU1 SYS X 0\n"),
    ],
)
def test_detect_numa_groups_fails_closed_when_topology_is_unavailable_or_invalid(
    result: _RemoteResult,
) -> None:
    with pytest.raises((RuntimeError, ValueError), match="NUMA|topology"):
        _detect_numa_groups(cast(Any, _TopologyRemote(result)), 4)
