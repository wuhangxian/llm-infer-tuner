"""Offline dry-run of the full two-round executor loop (no ssh/docker/network).

Drives ``run_executor`` end to end with a fake RemoteRunner + fake Container.
Asserts the wiring that the unit tests can't reach:

  * every candidate/probe reuses one deterministic benchmark template with only
    --max-concurrency / --num-prompts rewritten (byte-identical workload otherwise);
  * round 1 expands over ALL candidates, round 2 precisely refines ALL candidates
    and REUSES round-1 probes as seeds (seeded C are never re-benched);
  * the final ranking is goodput-descending and matches each candidate's true C*.

The fake "server" is a monotone SLA boundary per candidate: bench at C qualifies
iff C <= cstar[candidate], with throughput rising in C, so ranker goodput lands
on throughput(C*) exactly as in production.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

import pytest

from runners import executor as ex
from runners.executor import ExecutorConfig, run_executor
from runners.preflight import (
    CONTAINERS_QUERY_COMMAND,
    GPU_PIDS_QUERY_COMMAND,
    GPU_QUERY_COMMAND,
    HOME_QUERY_COMMAND,
    LISTENERS_QUERY_COMMAND,
    NUMA_QUERY_COMMAND,
    SSH_PROBE_COMMAND,
    CandidatePlacement,
    PreflightPlan,
)

# --- fakes --------------------------------------------------------------------


class _FakeResult:
    """Mimics remote.CommandResult (returncode / stdout / stderr / .ok)."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class _FakeRemote:
    """Answers only the host-shell commands run_executor issues before docker."""

    def __init__(self, gpu_count: int = 8) -> None:
        self.gpu_count = gpu_count

    def run(self, command: str, *, timeout=None) -> _FakeResult:
        del timeout
        if command == SSH_PROBE_COMMAND:
            return _FakeResult(stdout="llm-infer-tuner-preflight-ok\n")
        if command == GPU_QUERY_COMMAND:
            rows = [
                f"{gpu_id}, NVIDIA PRO 5000, 72832"
                for gpu_id in range(self.gpu_count)
            ]
            return _FakeResult(stdout="\n".join(rows) + "\n")
        if command == NUMA_QUERY_COMMAND:
            boundary = max(1, self.gpu_count // 2)
            rows = [
                f"GPU{gpu_id} X SYS {0 if gpu_id < boundary else 1}"
                for gpu_id in range(self.gpu_count)
            ]
            return _FakeResult(stdout="\n".join(rows) + "\n")
        if command == HOME_QUERY_COMMAND:
            return _FakeResult(stdout="/home/fake\n")
        if command.startswith("test -d"):
            return _FakeResult()
        if command.startswith("docker image inspect"):
            return _FakeResult(stdout="sha256:abcdef\n")
        if command.startswith("docker pull"):
            return _FakeResult(stdout="\n")
        if command == CONTAINERS_QUERY_COMMAND:
            return _FakeResult(stdout="[]\n")
        if command in {GPU_PIDS_QUERY_COMMAND, LISTENERS_QUERY_COMMAND}:
            return _FakeResult()
        if command.startswith("mkdir -p"):
            return _FakeResult()
        return _FakeResult(returncode=127, stderr=f"unexpected fake command: {command}")


class _FakeContainer:
    """A monotone SLA boundary per candidate, exercised through the real search.

    ``exec`` recognizes the handful of shell shapes the executor sends:
      * pgrep sglang.launch_server  -> alive iff a server is "running"
      * curl .../health             -> ready once the current candidate is started
      * cat <result path>           -> the bench jsonl written by the last bench
      * a bench command (contains sglang.bench_serving) -> record the C, "run" it
      * everything else (mkdir, pkill, launch_server nohup) -> succeed
    """

    def __init__(self, cstar_by_candidate: dict[str, int]) -> None:
        self.cstar = cstar_by_candidate
        self.current: str | None = None          # candidate whose server is up
        # Production creates one Container object per candidate.  Most dry-run
        # tests intentionally return this one shared fake from the factory, so
        # model server liveness must be scoped to the worker thread/port rather
        # than one global boolean (otherwise one candidate's cleanup kills a
        # sibling candidate in the fake only).
        self._server_lock = threading.Lock()
        self._alive_ports: set[int] = set()
        self._thread_servers = threading.local()
        self._fallback_alive = False
        self.bench_calls: list[tuple[str, int, int]] = []  # (candidate, C, num_prompts)
        self.warmup_calls: list[tuple[str, int, int]] = []  # 预热压测(丢弃,不计入搜索)
        # 满载测试要证明「同一并发档打到了 N 个不同副本端口」,单独记录 (candidate, C, port),
        # 不动 bench_calls 的三元组形状(现有用例仍按 (cand, conc, num_prompts) 解包)。
        self.bench_ports: list[tuple[str, int, int]] = []
        self.last_bench_jsonl = ""
        self._result_files: dict[str, str] = {}   # container path -> jsonl text
        self.launched_cmds: list[str] = []        # 每次 exec_detached 收到的 server 启动命令
        self.container_exists = False

    def _owned_ports(self) -> set[int]:
        ports = getattr(self._thread_servers, "ports", None)
        if ports is None:
            ports = set()
            self._thread_servers.ports = ports
        return ports

    def _record_launch_port(self, command: str) -> None:
        port = _flag_int(shlex.split(command), "--port")
        if port <= 0:
            return
        ports = self._owned_ports()
        with self._server_lock:
            # ThreadPool workers may be reused by a later candidate.  A worker
            # with no live owned server starts a fresh logical container.
            if not (ports & self._alive_ports):
                ports.clear()
            ports.add(port)

    @property
    def alive(self) -> bool:
        ports = self._owned_ports()
        with self._server_lock:
            if ports:
                return bool(ports & self._alive_ports)
            return self._fallback_alive

    @alive.setter
    def alive(self, value: bool) -> None:
        ports = self._owned_ports()
        with self._server_lock:
            if ports:
                if value:
                    self._alive_ports.update(ports)
                else:
                    self._alive_ports.difference_update(ports)
            else:
                self._fallback_alive = value

    def _forget_owned_servers(self) -> None:
        ports = self._owned_ports()
        with self._server_lock:
            self._alive_ports.difference_update(ports)
        ports.clear()

    # -- lifecycle no-ops the executor calls on the container object ------------
    def start(self, *, timeout=None) -> _FakeResult:
        self.container_exists = True
        return _FakeResult()

    def is_running(self, *, timeout=None) -> bool:
        return True

    def stop(self, *, timeout=None) -> _FakeResult:
        return _FakeResult()

    def remove(self, *, force: bool = True, timeout=None) -> _FakeResult:
        self.container_exists = False
        return _FakeResult()

    def inspect(self, *, timeout=None) -> _FakeResult:
        if self.container_exists:
            return _FakeResult()
        return _FakeResult(returncode=1, stderr="No such object")

    def exec_detached(self, command: str, log_container_path: str, *, timeout=None):
        # launching a candidate's server: infer which candidate from the -v mount
        # path baked into the launch cmd is not available, so track by outputs dir
        self.launched_cmds.append(command)
        self._record_launch_port(command)
        # 复刻真实 nohup 陷阱:exec_detached 会拼成 `nohup {command} ...`,nohup 把
        # 第一个词当程序名。裸 `CUDA_VISIBLE_DEVICES=0 python...` 前缀会让 nohup 去找名为
        # "CUDA_VISIBLE_DEVICES=0" 的可执行文件而失败(线上就是这么崩的)。正确写法是
        # `env CUDA_VISIBLE_DEVICES=0 python...`。这里模拟该判定:裸 KEY=VAL 前缀 → 启动失败。
        first_word = command.strip().split(None, 1)[0] if command.strip() else ""
        if "=" in first_word:
            self.alive = False
            return _FakeResult(returncode=1, stderr=(
                f"nohup: failed to run command '{first_word}': No such file or directory"
            ))
        self.alive = True
        return _FakeResult(stdout="1234\n")

    # -- the workhorse ---------------------------------------------------------
    def exec(self, command: str, *, timeout=None) -> _FakeResult:
        if command == f"pgrep -f {shlex.quote(ex.SERVER_PROCESS_PATTERN)}":
            return _FakeResult(0 if self.alive else 1)
        if command in {
            f"pkill -TERM -f {shlex.quote(ex.SERVER_PROCESS_PATTERN)}",
            f"pkill -KILL -f {shlex.quote(ex.SERVER_PROCESS_PATTERN)}",
        }:
            self._forget_owned_servers()
            self.current = None
            return _FakeResult()
        if "/health" in command:
            match = re.search(r":(\d+)/health\b", command)
            if match is not None:
                with self._server_lock:
                    alive = int(match.group(1)) in self._alive_ports
            else:
                alive = self.alive
            return _FakeResult(0 if alive else 7)
        if command.startswith("mkdir -p"):
            # the candidate output dir tells us which candidate is active
            path = command.split(None, 2)[-1].strip().strip("'\"")
            self.current = Path(path).name
            return _FakeResult()
        if "sglang.bench_serving" in command:
            return self._run_bench(command)
        if command.startswith("cat "):
            path = command.split(None, 1)[1].strip().strip("'\"")
            return _FakeResult(0, stdout=self._result_files.get(path, ""))
        return _FakeResult()

    def _run_bench(self, command: str) -> _FakeResult:
        parts = command.split()
        conc = _flag_int(parts, "--max-concurrency")
        num_prompts = _flag_int(parts, "--num-prompts")
        out_path = _flag_str(parts, "--output-file")
        # 候选身份从 --output-file 路径("/workspace/outputs/<cid>/bench_...jsonl")的
        # 父目录名取,而非共享的 self.current —— 后者在批内并发跑时会被兄弟候选覆盖,
        # 导致压测结果串到错误候选(真实环境每候选独占容器,天然不共享)。
        cand = Path(out_path).parent.name if out_path else (self.current or "unknown")
        # 预热压测(结果丢弃、不是搜索 probe)走单独的 warmup_calls,
        # 不污染 bench_calls —— 现有用例都把 bench_calls 当"真实搜索档序列"。
        if "_warmup" in Path(out_path).name:
            self.warmup_calls.append((cand, conc, num_prompts))
        else:
            self.bench_calls.append((cand, conc, num_prompts))
            self.bench_ports.append((cand, conc, _flag_int(parts, "--port")))

        cstar = self.cstar.get(cand, 0)
        qualifies = conc <= cstar
        completed = num_prompts  # all requests complete -> success_rate 1.0
        # 1000 output tokens/request (health target 1000) so avg_output_tokens=1000
        total_output_tokens = completed * 1000
        record = {
            "max_concurrency": conc,
            "completed": completed,
            "total_output_tokens": total_output_tokens,
            "request_throughput": float(conc),
            "output_throughput": float(conc * 100),
            "total_throughput": float(conc * 100),  # rises with C -> goodput at C*
            "mean_ttft_ms": 100.0 if qualifies else 9e9,  # blow the TTFT gate past C*
            "p99_ttft_ms": 200.0 if qualifies else 9e9,
            "mean_tpot_ms": 10.0,
            "p99_tpot_ms": 20.0,
            "duration": 10.0,
        }
        jsonl = json.dumps(record)
        self.last_bench_jsonl = jsonl
        if out_path:
            self._result_files[out_path] = jsonl
        return _FakeResult(0, stdout=jsonl)


class _StrictLaunchContainer(_FakeContainer):
    """Reject malformed executables so a failed launch cannot look healthy."""

    def __init__(self, cstar_by_candidate: dict[str, int]) -> None:
        super().__init__(cstar_by_candidate)
        self.rejected_launches: list[str] = []

    def exec_detached(self, command: str, log_container_path: str, *, timeout=None):
        if command.split()[:3] != ["python", "-m", "sglang.launch_server"]:
            self.launched_cmds.append(command)
            self._record_launch_port(command)
            self.rejected_launches.append(command)
            self.alive = False
            return _FakeResult(returncode=127, stderr="synthetic invalid executable")
        return super().exec_detached(command, log_container_path, timeout=timeout)


class _FlakyServerContainer(_FakeContainer):
    def __init__(self, cstar_by_candidate: dict[str, int], failures_before_ready: int) -> None:
        super().__init__(cstar_by_candidate)
        self.failures_remaining = failures_before_ready
        self.launch_attempts = 0

    def exec_detached(self, command: str, log_container_path: str, *, timeout=None):
        self.launch_attempts += 1
        self.launched_cmds.append(command)
        self._record_launch_port(command)
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            self.alive = False
            self._result_files[log_container_path] = "RuntimeError: synthetic startup crash"
            return _FakeResult(returncode=1, stderr="synthetic startup crash")
        self.alive = True
        return _FakeResult(stdout="1234\n")


class _FlakyContainerStart(_FakeContainer):
    def __init__(self, cstar_by_candidate: dict[str, int], failures_before_ready: int) -> None:
        super().__init__(cstar_by_candidate)
        self.failures_remaining = failures_before_ready
        self.container_running = False

    def start(self, *, timeout=None) -> _FakeResult:
        self.container_exists = True
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            self.container_running = False
            return _FakeResult(returncode=255, stderr="ssh connection lost")
        self.container_running = True
        return _FakeResult()

    def is_running(self, *, timeout=None) -> bool:
        return self.container_running

    def remove(self, *, force: bool = True, timeout=None) -> _FakeResult:
        self.container_running = False
        self.container_exists = False
        return _FakeResult()


class _BatchStopFailureContainer(_FakeContainer):
    def __init__(self, cstar_by_candidate: dict[str, int]) -> None:
        super().__init__(cstar_by_candidate)
        self.remove_calls = 0

    def stop(self, *, timeout=None) -> _FakeResult:
        return _FakeResult(returncode=1, stderr="synthetic stop failure")

    def remove(self, *, force: bool = True, timeout=None) -> _FakeResult:
        self.remove_calls += 1
        self.container_exists = False
        return _FakeResult()


class _StartupRemoveFailureContainer(_FlakyContainerStart):
    def __init__(self, cstar_by_candidate: dict[str, int]) -> None:
        super().__init__(cstar_by_candidate, failures_before_ready=99)
        self.start_calls = 0
        self.remove_calls = 0

    def start(self, *, timeout=None) -> _FakeResult:
        self.start_calls += 1
        return super().start(timeout=timeout)

    def remove(self, *, force: bool = True, timeout=None) -> _FakeResult:
        self.remove_calls += 1
        return _FakeResult(returncode=1, stderr="synthetic remove failure")


class _ServerCleanupFailureContainer(_FakeContainer):
    def exec(self, command: str, *, timeout=None) -> _FakeResult:
        if command == f"pkill -TERM -f {shlex.quote(ex.SERVER_PROCESS_PATTERN)}":
            return _FakeResult(returncode=2, stderr="synthetic pkill failure")
        return super().exec(command, timeout=timeout)


class _MissingOnFailedStartContainer(_FakeContainer):
    """The first docker run fails before creating a container."""

    def __init__(self, cstar_by_candidate: dict[str, int]) -> None:
        super().__init__(cstar_by_candidate)
        self.start_calls = 0
        self.remove_calls = 0
        self.events: list[str] = []
        self.container_running = False

    def start(self, *, timeout=None) -> _FakeResult:
        self.start_calls += 1
        if self.start_calls == 1:
            self.container_exists = False
            self.container_running = False
            self.events.append("start-failed-before-create")
            return _FakeResult(returncode=125, stderr="synthetic docker run failure")
        self.container_exists = True
        self.container_running = True
        self.events.append("start-created")
        return _FakeResult()

    def is_running(self, *, timeout=None) -> bool:
        return self.container_running

    def inspect(self, *, timeout=None) -> _FakeResult:
        self.events.append("inspect-present" if self.container_exists else "inspect-absent")
        return super().inspect(timeout=timeout)

    def remove(self, *, force: bool = True, timeout=None) -> _FakeResult:
        self.remove_calls += 1
        self.events.append("remove")
        self.container_running = False
        return super().remove(force=force, timeout=timeout)


class _AmbiguousFailedStartContainer(_FakeContainer):
    """A failed docker run followed by an untrustworthy inspect response."""

    def __init__(self, cstar_by_candidate: dict[str, int]) -> None:
        super().__init__(cstar_by_candidate)
        self.start_calls = 0

    def start(self, *, timeout=None) -> _FakeResult:
        self.start_calls += 1
        return _FakeResult(returncode=125, stderr="synthetic docker run failure")

    def is_running(self, *, timeout=None) -> bool:
        return False

    def inspect(self, *, timeout=None) -> _FakeResult:
        return _FakeResult(returncode=1, stderr="permission denied by docker daemon")


class _DelayedTermExitContainer(_FakeContainer):
    """A server needs two post-TERM polls before exiting normally."""

    def __init__(self, cstar_by_candidate: dict[str, int]) -> None:
        super().__init__(cstar_by_candidate)
        self.cleanup_active = False
        self.cleanup_polls = 0
        self.term_calls = 0
        self.kill_calls = 0

    @property
    def _pattern(self) -> str:
        return shlex.quote(ex.SERVER_PROCESS_PATTERN)

    def exec_detached(self, command: str, log_container_path: str, *, timeout=None):
        self.cleanup_active = False
        self.cleanup_polls = 0
        return super().exec_detached(command, log_container_path, timeout=timeout)

    def exec(self, command: str, *, timeout=None) -> _FakeResult:
        if command == f"pkill -TERM -f {self._pattern}":
            self.term_calls += 1
            self.cleanup_active = True
            self.cleanup_polls = 0
            return _FakeResult()
        if command == f"pkill -KILL -f {self._pattern}":
            self.kill_calls += 1
            self.alive = False
            return _FakeResult()
        if command == f"pgrep -f {self._pattern}" and self.cleanup_active:
            self.cleanup_polls += 1
            if self.cleanup_polls >= 2:
                self.alive = False
                return _FakeResult(returncode=1)
            return _FakeResult()
        if command.startswith("pkill "):
            return _FakeResult(returncode=127, stderr=f"unexpected kill command: {command}")
        return super().exec(command, timeout=timeout)


class _RequiresKillServerContainer(_DelayedTermExitContainer):
    """A server ignores TERM and disappears only after KILL."""

    def exec(self, command: str, *, timeout=None) -> _FakeResult:
        if command == f"pgrep -f {self._pattern}" and self.cleanup_active:
            return _FakeResult(0 if self.alive else 1)
        return super().exec(command, timeout=timeout)


def _flag_int(parts: list[str], flag: str) -> int:
    for i, p in enumerate(parts):
        if p == flag and i + 1 < len(parts):
            return int(parts[i + 1])
        if p.startswith(f"{flag}="):
            return int(p.split("=", 1)[1])
    return 0


def _flag_str(parts: list[str], flag: str) -> str:
    for i, p in enumerate(parts):
        if p == flag and i + 1 < len(parts):
            return parts[i + 1]
        if p.startswith(f"{flag}="):
            return p.split("=", 1)[1]
    return ""


# --- fixtures -----------------------------------------------------------------


def _write_job(
    tmp_path: Path,
    *,
    baseline_threshold_pct: float = 0,
    max_candidates: int = 1,
    baseline: bool = False,
    gpu_count: int = 8,
) -> Path:
    job = {
        "job_id": "dryrun-job",
        "engine": "sglang",
         "gpu_model": "pro5000",
         "gpu_count": gpu_count,
         "gpu_memory_gb": 72,
        "model": "qwen36-35b",
        "image": "sglang-test",
        "workload": "W01_input-1k-output-1k",
        "benchmark_method": "sglang-bench-serving",
        "sla": {"max_avg_ttft_ms": 2000.0, "max_avg_tpot_ms": 80.0, "min_success_rate": 0.99},
        "search": {
            "max_candidates": max_candidates,
            "max_runtime_minutes": 120,
            "baseline_threshold_pct": baseline_threshold_pct,
            **({"baseline": {}} if baseline else {}),
        },
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    return path


def _write_configs(tmp_path: Path, candidate_ids: list[str]) -> Path:
    path = tmp_path / "configs.jsonl"
    lines = []
    for index, cid in enumerate(candidate_ids):
        params = {"mem_fraction_static": 0.5 + index / 1000}
        lines.append(
            json.dumps(
                {
                    "id": cid,
                    "params": params,
                    "cmd": (
                        "python -m sglang.launch_server --model-path ${MODEL_PATH} "
                        f"--mem-fraction-static {params['mem_fraction_static']}"
                    ),
                    "reasons": [],
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def workloads_output_len(monkeypatch):
    """Pin output_len=1000 (workloads.yaml lookup) so the health gate is deterministic."""
    monkeypatch.setattr(ex, "_load_output_len", lambda workload: 1000)
    return 1000


# --- the dry run --------------------------------------------------------------


def test_two_round_dryrun_ranks_by_true_goodput(tmp_path, workloads_output_len):
    # Four candidates with distinct true C* -> distinct goodput = C* * 100.
    cstar = {"cand-a": 4, "cand-b": 16, "cand-c": 2, "cand-d": 8}
    job_path = _write_job(tmp_path, max_candidates=len(cstar))
    configs_path = _write_configs(tmp_path, list(cstar))
    results_dir = tmp_path / "results"

    config = ExecutorConfig(
        job_path=job_path,
        configs_path=configs_path,
        results_dir=results_dir,
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=16,
        top_k=2,
        max_cap=256,
    )

    remote = _FakeRemote()
    container = _FakeContainer(cstar)
    # Patch Container so run_executor uses our fake instead of docker-over-ssh.
    import runners.executor as exmod
    original_container = exmod.Container
    exmod.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=remote)
    finally:
        exmod.Container = original_container

    # -- every bench used the deterministic shared template: only C / num_prompts vary, and
    #    num_prompts == C * multiplier (4) every time.
    for cand, conc, num_prompts in container.bench_calls:
        assert num_prompts == conc * 4, (cand, conc, num_prompts)

    server_ports = {
        _flag_int(command.split(), "--port")
        for command in container.launched_cmds
        if "sglang.launch_server" in command
    }
    assert server_ports == {30000, 30001, 30002, 30003}

    # -- ranking is goodput-descending and matches true C* --------------------
    ranking = summary["ranking"]
    ids_in_order = [row["candidate_id"] for row in ranking]
    assert ids_in_order == ["cand-b", "cand-d", "cand-a", "cand-c"]
    by_id = {row["candidate_id"]: row for row in ranking}
    for cid, expected_cstar in cstar.items():
        # gpu_count=8, tp_size=1
        assert by_id[cid]["goodput_per_host"] == expected_cstar * 100.0 * 8
        assert by_id[cid]["best_concurrency"] == expected_cstar

    # -- every candidate enters precise round 2; --top-k is compatibility-only --
    assert "top_k" not in summary
    assert "top_ids" not in summary
    for cid in cstar:
        assert "round2" in summary["candidates"][_index_of(summary, cid)]

    assert summary["task_status"] == "COMPLETED"
    assert summary["ranking_status"] == "FINAL"
    rows = [json.loads(line) for line in (
        results_dir / "candidate_results.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert [row["candidate_id"] for row in rows] == list(cstar)
    assert all(row["status"] == "completed" for row in rows)
    assert all({point["round"] for point in row["concurrency_points"]} == {1, 2}
               for row in rows)

    # -- ranking.json + per-candidate evidence were written -------------------
    assert (results_dir / "ranking.json").exists()
    for cid in cstar:
        assert (results_dir / cid / "run_result.r1.json").exists()


def test_hit_cap_candidate_is_incomplete_and_has_no_final_rank(
    tmp_path, workloads_output_len, monkeypatch
):
    config = ExecutorConfig(
        job_path=_write_job(tmp_path),
        configs_path=_write_configs(tmp_path, ["uncertain"]),
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_cap=4,
    )
    container = _FakeContainer({"uncertain": 1_000})
    monkeypatch.setattr(ex, "Container", lambda _remote, _config: container)

    summary = run_executor(config, remote=_FakeRemote())

    row = summary["candidate_results"][0]
    assert row["status"] == "incomplete"
    assert row["round2"]["complete"] is False
    assert row["round2"]["certainty"] == "lower_bound"
    assert summary["ranking"] == []
    assert "rank" not in row
    assert summary["task_status"] == "INCOMPLETE"
    assert summary["ranking_status"] == "PROVISIONAL"


def test_executor_consumes_preflight_plan_for_both_rounds(
    tmp_path, workloads_output_len, monkeypatch
):
    config = ExecutorConfig(
        job_path=_write_job(tmp_path),
        configs_path=_write_configs(tmp_path, ["solo"]),
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
    )
    placement = CandidatePlacement("solo", (0,), 30000)
    plan = PreflightPlan(
        candidate_ids=("solo",),
        numa_groups=((0, 1, 2, 3), (4, 5, 6, 7)),
        round1_batches=((placement,),),
        round2_batches=((placement,),),
        fill_host_placements=(),
        required_ports=(30000,),
    )
    prepared: list[object] = []

    def fake_prepare(remote, request):
        prepared.append((remote, request))
        return plan

    def stale_path(*args, **kwargs):
        raise AssertionError("executor must consume PreflightPlan, not re-plan")

    monkeypatch.setattr(ex, "prepare_remote_host", fake_prepare, raising=False)
    monkeypatch.setattr(ex, "_preflight_checks", stale_path, raising=False)
    monkeypatch.setattr(ex, "_detect_numa_groups", stale_path)
    monkeypatch.setattr(ex, "_plan_candidate_batches", stale_path)
    monkeypatch.setattr(ex, "_allocate_gpus_and_ports", stale_path)
    container = _FakeContainer({"solo": 2})
    monkeypatch.setattr(ex, "Container", lambda _remote, _config: container)

    summary = run_executor(config, remote=_FakeRemote())

    assert len(prepared) == 1
    assert summary["task_status"] == "COMPLETED"
    assert {
        port for _candidate, _concurrency, port in container.bench_ports
    } == {30000}


def test_baseline_plus_32_candidates_all_enter_round2_and_keep_one_row(
    tmp_path, workloads_output_len
):
    candidate_ids = ["baseline", *(f"c{i:03d}" for i in range(1, 33))]
    params = {
        candidate_id: {
            "tp_size": 1,
            **({"is_baseline": True} if candidate_id == "baseline" else {}),
        }
        for candidate_id in candidate_ids
    }
    config = ExecutorConfig(
        job_path=_write_job(tmp_path, max_candidates=32, baseline=True),
        configs_path=_write_configs_with_params(tmp_path, params),
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=32,
        top_k=5,
    )
    container = _FakeContainer(dict.fromkeys(candidate_ids, 2))
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=_FakeRemote())
    finally:
        ex.Container = original_container

    assert len(summary["candidate_results"]) == 33
    assert [row["candidate_id"] for row in summary["candidate_results"]] == candidate_ids
    assert all("round2" in candidate for candidate in summary["candidates"])
    assert all(row["status"] == "completed" for row in summary["candidate_results"])


def test_round2_reuses_round1_seeds_no_rebench(tmp_path, workloads_output_len):
    """The single top-K candidate's round-1 probes must be reused as round-2 seeds:
    a C benched in round 1 is never benched again in round 2."""
    cstar = {"solo": 10}
    job_path = _write_job(tmp_path)
    configs_path = _write_configs(tmp_path, ["solo"])
    results_dir = tmp_path / "results"

    config = ExecutorConfig(
        job_path=job_path,
        configs_path=configs_path,
        results_dir=results_dir,
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=1,
        top_k=1,
        max_cap=256,
    )

    remote = _FakeRemote()
    container = _FakeContainer(cstar)
    import runners.executor as exmod
    original_container = exmod.Container
    exmod.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=remote)
    finally:
        exmod.Container = original_container

    # Split bench calls into round 1 (coarse expansion) and round 2 (bisection).
    # Round 1 (refine=False) probes the expansion points 1,2,4,8,16 (16 is first
    # fail; true C*=10). Round 2 seeds those and bisects (8,16) -> 12,10,11...
    round1_cs = []
    round2_cs = []
    # round-1 stops at first_fail during expansion; count expansion probes:
    # they are strictly the doubling sequence until the first C that fails.
    seen_first_fail = False
    for _cand, conc, _np in container.bench_calls:
        if not seen_first_fail:
            round1_cs.append(conc)
            if conc > cstar["solo"]:
                seen_first_fail = True
        else:
            round2_cs.append(conc)

    # 每个 server(round-1 + round-2 各一次)都应在正式搜索前预热一次,
    # 且预热用固定小并发 WARMUP_CONCURRENCY(结果丢弃,不进 bench_calls)。
    assert container.warmup_calls, "server ready 后应至少预热一次"
    assert all(c == exmod.WARMUP_CONCURRENCY for _cand, c, _np in container.warmup_calls), (
        f"warmup 应固定用 C={exmod.WARMUP_CONCURRENCY}: {container.warmup_calls}"
    )

    assert round1_cs == [1, 2, 4, 8, 16]  # coarse expansion, no bisection
    # Round 2 must NOT re-bench any C already probed in round 1.
    assert not (set(round2_cs) & set(round1_cs)), (
        f"round-2 re-benched seeded C: {sorted(set(round2_cs) & set(round1_cs))}"
    )
    # And it must find the exact boundary C*=10.
    ranking = summary["ranking"]
    assert ranking[0]["candidate_id"] == "solo"
    assert ranking[0]["best_concurrency"] == 10
    assert ranking[0]["goodput_per_host"] == 1000.0 * 8  # gpu_count=8, tp_size=1


def _write_configs_with_params(tmp_path: Path, params_by_id: dict[str, dict]) -> Path:
    """Like _write_configs but每个候选带自定义 params(满载测试要 tp_size)。"""
    path = tmp_path / "configs.jsonl"
    lines = []
    for index, (cid, supplied_params) in enumerate(params_by_id.items()):
        params = {"mem_fraction_static": 0.5 + index / 1000, **supplied_params}
        tuning_flags = " ".join(
            f"--{key.replace('_', '-')} {value}"
            for key, value in params.items()
            if key not in {"is_baseline", "disable_radix_cache"}
        )
        lines.append(
            json.dumps(
                {
                    "id": cid,
                    "params": params,
                    "cmd": (
                        "python -m sglang.launch_server --model-path ${MODEL_PATH} "
                        f"{tuning_flags}"
                    ).rstrip(),
                    "reasons": [],
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_round2_fill_host_benches_n_replicas_no_double_count(tmp_path, workloads_output_len):
    """round2 满载(fill_host):top-K 候选被复制成 floor(gpu/tp) 个副本、各占端口并发实测求和。

    钉死三件事:
      1. 同一并发档确实打到了 N 个不同副本端口(是真并发实测,不是纸面外推);
      2. 副本数恰为 floor(gpu_count/tp_size)=4,不多不少;
      3. goodput = (ΣN 副本吞吐 / N) × floor(gpu/tp),N 只被计一次 —— 绝不是 ΣN × floor
         的双重计数。tp=2、8 卡下 c*=8:per_host = 3200,而非 3200×4=12800。
    """
    cstar = {"full": 8}
    job_path = _write_job(tmp_path)                       # gpu_count=8
    configs_path = _write_configs_with_params(tmp_path, {"full": {"tp_size": 2}})
    results_dir = tmp_path / "results"

    config = ExecutorConfig(
        job_path=job_path,
        configs_path=configs_path,
        results_dir=results_dir,
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=1,
        top_k=1,
        max_cap=256,
        fill_host=True,          # <-- 只 round2 满载
    )

    remote = _FakeRemote()
    container = _FakeContainer(cstar)
    import runners.executor as exmod
    original_container = exmod.Container
    exmod.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=remote)
    finally:
        exmod.Container = original_container

    # Task 8 will revise the precise-search budget.  With today's fixed 14
    # probes this confirmed fill-host search stops at max_probes before proving
    # the exact boundary, so Task 6 must fail closed instead of publishing the
    # tempting C=8 / 3200 tok/s lower bound as an official rank.
    assert summary["ranking"] == []
    assert summary["task_status"] == "INCOMPLETE"
    assert summary["ranking_status"] == "PROVISIONAL"
    row = summary["candidate_results"][0]
    assert row["status"] == "incomplete"
    assert row["round2"]["stop_reason"] == "max_probes"
    assert row["round2"]["complete"] is False

    # round2 满载:c*=8 这一档必须打满 4 个不同副本端口 30000..30003
    #(round1 单实例也在 30000 压过 8,取并集仍是这 4 个)。
    ports_at_cstar = {p for cand, conc, p in container.bench_ports
                      if cand == "full" and conc == 8}
    assert ports_at_cstar == {30000, 30001, 30002, 30003}, ports_at_cstar
    # 恰好 4 副本,不多起(没有 30004+)。
    assert max(p for _c, _conc, p in container.bench_ports) == 30003

    # 落盘证据:round2 结果的 instances 记为 4(供防双重计数复核)。
    r2 = json.loads((results_dir / "full" / "run_result.r2.json").read_text(encoding="utf-8"))
    passing = [row for row in r2 if row.get("status") == "ok"]
    assert passing, "round2 应有健康的满载结果"
    assert all(row["instances"] == 4 for row in passing)

    # 回归:满载副本的启动命令必须用 `env CUDA_VISIBLE_DEVICES=N` 而非裸前缀,
    # 否则 exec_detached 拼 `nohup {cmd}` 时 nohup 会把 "CUDA_VISIBLE_DEVICES=N" 当程序名而崩
    #(线上真实故障:nohup: failed to run command 'CUDA_VISIBLE_DEVICES=0')。
    gpu_pinned = [c for c in container.launched_cmds if "CUDA_VISIBLE_DEVICES" in c]
    assert gpu_pinned, "满载应有钉卡的副本启动命令"
    for cmd in gpu_pinned:
        assert cmd.strip().startswith("env CUDA_VISIBLE_DEVICES="), (
            f"副本启动命令必须以 `env CUDA_VISIBLE_DEVICES=` 开头(nohup 安全),实际: {cmd[:60]}"
        )
        # 且第一个词不能是裸 KEY=VAL(那正是 nohup 会当成程序名的东西)。
        assert "=" not in cmd.strip().split(None, 1)[0]


def test_round2_fill_host_tp3_uses_only_numa_local_replicas(
    tmp_path, workloads_output_len
):
    """On dual-NUMA 4+4, TP3 has two real local replicas, never flat [3,4,5]."""
    cstar = {"full": 4}
    config = ExecutorConfig(
        job_path=_write_job(tmp_path),
        configs_path=_write_configs_with_params(tmp_path, {"full": {"tp_size": 3}}),
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=1,
        fill_host=True,
    )
    container = _FakeContainer(cstar)
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=_FakeRemote())
    finally:
        ex.Container = original_container

    pinned = {
        token.split("=", 1)[1]
        for command in container.launched_cmds
        for token in command.split()
        if token.startswith("CUDA_VISIBLE_DEVICES=")
    }
    assert pinned == {"0,1,2", "4,5,6"}
    assert summary["ranking"][0]["instances_per_host"] == 2.0
    assert {port for cand, _conc, port in container.bench_ports if cand == "full"} <= {
        30000,
        30001,
    }


def test_explicit_cross_numa_target_allows_tp8_on_dual_numa_host(
    tmp_path, workloads_output_len
):
    config = ExecutorConfig(
        job_path=_write_job(tmp_path),
        configs_path=_write_configs_with_params(tmp_path, {"wide": {"tp_size": 8}}),
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=1,
        allow_cross_numa=True,
    )
    container = _FakeContainer({"wide": 2})
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=_FakeRemote())
    finally:
        ex.Container = original_container

    assert summary["task_status"] == "COMPLETED"
    assert summary["ranking"][0]["candidate_id"] == "wide"
    assert summary["ranking"][0]["tp_size"] == 8


def test_executor_reuses_high_ports_after_topology_aware_batch_split(
    tmp_path, workloads_output_len
):
    candidate_ids = ["c001", "c002", "c003"]
    config = ExecutorConfig(
        job_path=_write_job(tmp_path, max_candidates=3, gpu_count=2),
        configs_path=_write_configs_with_params(
            tmp_path,
            {candidate_id: {"tp_size": 1} for candidate_id in candidate_ids},
        ),
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=3,
        port=65534,
    )
    container = _FakeContainer(dict.fromkeys(candidate_ids, 2))
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=_FakeRemote(gpu_count=2))
    finally:
        ex.Container = original_container

    assert summary["task_status"] == "COMPLETED"
    assert [row["round1_batch"] for row in summary["candidates"]] == [
        "1/2",
        "1/2",
        "2/2",
    ]
    assert {
        port
        for _candidate, _concurrency, port in container.bench_ports
    } == {65534, 65535}


def _index_of(summary: dict, candidate_id: str) -> int:
    for i, row in enumerate(summary["candidates"]):
        if row["candidate_id"] == candidate_id:
            return i
    raise AssertionError(f"{candidate_id} not in summary candidates")


def test_every_launch_forces_exactly_one_disable_radix(tmp_path, workloads_output_len):
    """硬约束:任何候选、任何来源(cmd/params)启动命令都必须钉死关 radix,
    用户没写时补上,用户写了任意值时覆盖成唯一的裸 flag。Mamba 请求值原样留作审计。"""
    path = tmp_path / "configs.jsonl"
    lines = [
        # (a) cmd 整串,带重复 disable-radix —— loader 规范化为一个裸 flag。
        json.dumps({
            "id": "cand-cmd",
            "params": {"tp_size": 1, "attention_backend": "triton"},
            "cmd": "python -m sglang.launch_server --model-path ${MODEL_PATH} "
                   "--tp-size 1 --attention-backend triton "
                   "--disable-radix-cache --disable-radix-cache",
            "reasons": [],
        }),
        # (b) params 字典,没写 disable_radix_cache 字段 —— 回落 SGLang 默认(radix 开)
        json.dumps({
            "id": "cand-params",
            "params": {"tp_size": 1, "attention_backend": "flashinfer"},
            "reasons": [],
        }),
        # (c) 手写了 disable_radix_cache=true —— 不能重复拼成两个 flag
        json.dumps({
            "id": "cand-already",
            "params": {"tp_size": 1, "disable_radix_cache": True},
            "reasons": [],
        }),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    config = ExecutorConfig(
        job_path=_write_job(tmp_path, max_candidates=3),
        configs_path=path,
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=16,
        top_k=1,
        max_cap=256,
    )
    remote = _FakeRemote()
    container = _FakeContainer({"cand-cmd": 2, "cand-params": 2, "cand-already": 2})
    import runners.executor as exmod
    original_container = exmod.Container
    exmod.Container = lambda _remote, _cfg: container
    try:
        run_executor(config, remote=remote)
    finally:
        exmod.Container = original_container

    # 只看真正的 server 启动命令(排除 nohup 包装里的副本 env 前缀差异)
    server_cmds = [c for c in container.launched_cmds if "sglang.launch_server" in c]
    assert server_cmds, "no server launch command was captured"
    for cmd in server_cmds:
        assert cmd.count("--disable-radix-cache") == 1, (
            f"每条启动命令必须带且仅带一个 --disable-radix-cache,实际:{cmd}"
        )
        assert "--disable-radix-cache=" not in cmd, (
            f"必须是裸 flag,不能是 =true/=false 形式,实际:{cmd}"
        )


def test_params_only_candidate_launches_the_structured_command(
    tmp_path, workloads_output_len
):
    configs_path = tmp_path / "configs.jsonl"
    configs_path.write_text(
        json.dumps(
            {
                "id": "params-only",
                "params": {"tp_size": 2, "attention_backend": "flashinfer"},
                "reasons": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = ExecutorConfig(
        job_path=_write_job(tmp_path),
        configs_path=configs_path,
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=1,
    )
    container = _StrictLaunchContainer({"params-only": 2})
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        run_executor(config, remote=_FakeRemote())
    finally:
        ex.Container = original_container

    assert not container.rejected_launches
    command = next(
        item
        for item in container.launched_cmds
        if item.startswith("python -m sglang.launch_server")
    )
    assert command.startswith("python -m sglang.launch_server")
    assert "--tp-size 2" in command
    assert "--attention-backend flashinfer" in command
    assert command.count("--disable-radix-cache") == 1
    assert "--model-path /models/qwen" in command
    assert "--host 0.0.0.0" in command
    assert "--port 30000" in command
    assert "None" not in command.split()


def test_params_only_quoted_disable_text_round_trips_as_one_argument(
    tmp_path, workloads_output_len
):
    served_model_name = "safe --disable-radix-cache marker"
    configs_path = tmp_path / "configs.jsonl"
    configs_path.write_text(
        json.dumps(
            {
                "id": "quoted-disable-text",
                "params": {"served_model_name": served_model_name},
                "reasons": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = ExecutorConfig(
        job_path=_write_job(tmp_path),
        configs_path=configs_path,
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=1,
    )
    container = _StrictLaunchContainer({"quoted-disable-text": 2})
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        run_executor(config, remote=_FakeRemote())
    finally:
        ex.Container = original_container

    assert not container.rejected_launches
    command = next(
        item
        for item in container.launched_cmds
        if item.startswith("python -m sglang.launch_server")
    )
    argv = shlex.split(command)
    assert argv[argv.index("--served-model-name") + 1] == served_model_name
    assert argv.count("--disable-radix-cache") == 1


@pytest.mark.parametrize(
    "port_arg",
    ["--port=30000", "--port 30010", ""],
    ids=["equals", "arbitrary-old-port", "missing"],
)
def test_launch_overrides_any_port_spelling_with_assigned_port(
    tmp_path, workloads_output_len, port_arg
):
    configs_path = tmp_path / "configs.jsonl"
    configs_path.write_text(
        json.dumps(
            {
                "id": "equals-port",
                "params": {"tp_size": 1},
                "cmd": (
                    "python -m sglang.launch_server --model-path ${MODEL_PATH} "
                    f"--tp-size 1 {port_arg}"
                ),
                "reasons": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = ExecutorConfig(
        job_path=_write_job(tmp_path),
        configs_path=configs_path,
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=1,
        port=30005,
    )
    container = _FakeContainer({"equals-port": 2})
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        run_executor(config, remote=_FakeRemote())
    finally:
        ex.Container = original_container

    server_cmds = [
        command
        for command in container.launched_cmds
        if "sglang.launch_server" in command
    ]
    assert server_cmds
    assert all(_flag_int(command.split(), "--port") == 30005 for command in server_cmds)


def test_launch_runtime_argv_removes_all_duplicates_and_preserves_quoted_values() -> None:
    preserved_value = "模型 family's copy --port marker"
    original = [
        "python",
        "-m",
        "sglang.launch_server",
        "--port",
        "30001",
        "-p",
        "30002",
        "--port=30003",
        "-p=30004",
        "--model-path",
        "/old model",
        "--model-path=/another old model",
        "--host",
        "127.0.0.1",
        "--host=localhost",
        "--served-model-name",
        preserved_value,
    ]

    effective = ex._canonicalize_launch_runtime_argv(
        original,
        model_path="/container/模型 family's copy",
        port=30123,
    )

    assert effective[-6:] == [
        "--model-path",
        "/container/模型 family's copy",
        "--host",
        "0.0.0.0",
        "--port",
        "30123",
    ]
    assert effective.count("--port") == 1
    assert not any(token == "-p" or token.startswith("-p=") for token in effective)
    assert sum(token == "--model-path" for token in effective) == 1
    assert sum(token == "--host" for token in effective) == 1
    assert effective[effective.index("--served-model-name") + 1] == preserved_value


def test_launch_runtime_argv_appends_runtime_flags_when_candidate_omits_them() -> None:
    effective = ex._canonicalize_launch_runtime_argv(
        ["python", "-m", "sglang.launch_server", "--tp-size", "2"],
        model_path="/models/example",
        port=30005,
    )

    assert effective[-6:] == [
        "--model-path",
        "/models/example",
        "--host",
        "0.0.0.0",
        "--port",
        "30005",
    ]


def test_threshold_marks_but_never_removes_candidates(tmp_path, workloads_output_len):
    cstar = {"baseline": 10, "fast": 16, "slow": 4}
    job_path = _write_job(
        tmp_path, baseline_threshold_pct=20, max_candidates=2, baseline=True
    )
    configs_path = _write_configs_with_params(
        tmp_path,
        {
            "baseline": {"is_baseline": True, "tp_size": 1},
            "fast": {"tp_size": 1},
            "slow": {"tp_size": 1},
        },
    )
    config = ExecutorConfig(
        job_path=job_path,
        configs_path=configs_path,
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=16,
        top_k=1,
    )
    remote, container = _FakeRemote(), _FakeContainer(cstar)
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=remote)
    finally:
        ex.Container = original_container

    ranking = summary["ranking"]
    assert {row["candidate_id"] for row in ranking} == set(cstar)
    by_id = {row["candidate_id"]: row for row in ranking}
    assert by_id["fast"]["beats_baseline_threshold"] is True
    assert by_id["slow"]["beats_baseline_threshold"] is False
    assert by_id["slow"]["total_throughput"] == 400.0
    assert by_id["slow"]["mean_ttft_ms"] == 100.0
    assert by_id["slow"]["mean_tpot_ms"] == 10.0
    assert by_id["slow"]["success_rate"] == 1.0


def test_server_startup_retries_then_completes(tmp_path, workloads_output_len):
    config = ExecutorConfig(
        job_path=_write_job(tmp_path),
        configs_path=_write_configs(tmp_path, ["flaky"]),
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=1,
        startup_max_attempts=3,
        startup_hard_timeout_s=10,
        startup_stall_timeout_s=5,
    )
    container = _FlakyServerContainer({"flaky": 4}, failures_before_ready=2)
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=_FakeRemote())
    finally:
        ex.Container = original_container

    row = summary["candidate_results"][0]
    assert summary["task_status"] == "COMPLETED"
    assert row["status"] == "completed"
    assert row["attempts"] == 4
    assert len(row["failures"]) == 2
    assert all(failure["failed_at"] for failure in row["failures"])
    assert all(failure["round"] == 1 for failure in row["failures"])


def test_container_or_ssh_startup_failure_recreates_and_retries(
    tmp_path, workloads_output_len
):
    config = ExecutorConfig(
        job_path=_write_job(tmp_path),
        configs_path=_write_configs(tmp_path, ["flaky-container"]),
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=1,
        startup_max_attempts=3,
        startup_hard_timeout_s=10,
        startup_stall_timeout_s=5,
    )
    container = _FlakyContainerStart({"flaky-container": 4}, failures_before_ready=2)
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=_FakeRemote())
    finally:
        ex.Container = original_container

    row = summary["candidate_results"][0]
    assert summary["task_status"] == "COMPLETED"
    assert row["status"] == "completed"
    assert row["attempts"] == 4
    assert len(row["failures"]) == 2
    assert all("ssh connection lost" in failure["reason"] for failure in row["failures"])


def test_exhausted_startup_retries_keeps_failed_candidate_row(
    tmp_path, workloads_output_len
):
    config = ExecutorConfig(
        job_path=_write_job(tmp_path),
        configs_path=_write_configs_with_params(tmp_path, {"broken": {"tp_size": 4}}),
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=1,
        startup_max_attempts=3,
        startup_hard_timeout_s=10,
        startup_stall_timeout_s=5,
    )
    container = _FlakyServerContainer({"broken": 4}, failures_before_ready=99)
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=_FakeRemote())
    finally:
        ex.Container = original_container

    row = summary["candidate_results"][0]
    assert summary["task_status"] == "INCOMPLETE"
    assert summary["ranking_status"] == "PROVISIONAL"
    assert row["candidate_id"] == "broken"
    assert row["status"] == "incomplete"
    assert row["failed_at"]
    assert row["failed_round"] == 2
    assert row["failure_status"] == "startup_failed"
    assert row["failed_tp_size"] == 4
    assert row["failed_concurrency"] == 0
    assert row["failed_num_prompts"] == 0
    assert row["known_issue"] is None
    assert row["failure_reason"]


def _lifecycle_test_config(tmp_path: Path, candidate_id: str) -> ExecutorConfig:
    return ExecutorConfig(
        job_path=_write_job(tmp_path),
        configs_path=_write_configs(tmp_path, [candidate_id]),
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=1,
        startup_max_attempts=3,
    )


def test_batch_stop_failure_still_attempts_remove_and_aborts_final_ranking(
    tmp_path, workloads_output_len
):
    container = _BatchStopFailureContainer({"cleanup-stop": 2})
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        with pytest.raises(RuntimeError, match="cleanup"):
            run_executor(
                _lifecycle_test_config(tmp_path, "cleanup-stop"),
                remote=_FakeRemote(),
            )
    finally:
        ex.Container = original_container

    assert container.remove_calls >= 1
    assert not (tmp_path / "results" / "ranking.json").exists()


def test_startup_remove_failure_stops_retry_and_aborts_final_ranking(
    tmp_path, workloads_output_len
):
    container = _StartupRemoveFailureContainer({"cleanup-startup": 2})
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        with pytest.raises(RuntimeError, match="cleanup"):
            run_executor(
                _lifecycle_test_config(tmp_path, "cleanup-startup"),
                remote=_FakeRemote(),
            )
    finally:
        ex.Container = original_container

    assert container.start_calls == 1
    assert not (tmp_path / "results" / "ranking.json").exists()


def test_server_pkill_failure_aborts_final_ranking(
    tmp_path, workloads_output_len, monkeypatch
):
    monkeypatch.setattr(ex, "SERVER_CLEANUP_TERM_POLL_ATTEMPTS", 2)
    monkeypatch.setattr(ex, "SERVER_CLEANUP_TERM_POLL_INTERVAL_S", 0)
    container = _ServerCleanupFailureContainer({"cleanup-server": 2})
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        with pytest.raises(RuntimeError, match="cleanup"):
            run_executor(
                _lifecycle_test_config(tmp_path, "cleanup-server"),
                remote=_FakeRemote(),
            )
    finally:
        ex.Container = original_container

    assert not (tmp_path / "results" / "ranking.json").exists()


def test_start_failure_before_container_creation_retries_without_remove(
    tmp_path, workloads_output_len
):
    container = _MissingOnFailedStartContainer({"missing-first": 2})
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(
            _lifecycle_test_config(tmp_path, "missing-first"),
            remote=_FakeRemote(),
        )
    finally:
        ex.Container = original_container

    assert summary["task_status"] == "COMPLETED"
    assert container.events[:3] == [
        "start-failed-before-create",
        "inspect-absent",
        "start-created",
    ]
    # There is no remove between the failed run and its safe retry. Later
    # successful round cleanup still removes the containers it created.
    assert container.start_calls == container.remove_calls + 1


def test_ambiguous_inspect_after_failed_start_blocks_retry_and_final_ranking(
    tmp_path, workloads_output_len
):
    container = _AmbiguousFailedStartContainer({"ambiguous": 2})
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        with pytest.raises(RuntimeError, match="permission denied"):
            run_executor(
                _lifecycle_test_config(tmp_path, "ambiguous"),
                remote=_FakeRemote(),
            )
    finally:
        ex.Container = original_container

    assert container.start_calls == 1
    assert not (tmp_path / "results" / "ranking.json").exists()


def test_startup_cleanup_failure_prevents_starting_later_batch_candidates(
    tmp_path, workloads_output_len
):
    first = _StartupRemoveFailureContainer({"first": 2})
    second = _MissingOnFailedStartContainer({"second": 2})
    config = ExecutorConfig(
        job_path=_write_job(tmp_path, max_candidates=2),
        configs_path=_write_configs(tmp_path, ["first", "second"]),
        results_dir=tmp_path / "results",
        ssh_target="fake@host",
        image_ref="sglang-test",
        model_host_dir="/data/models/qwen",
        model_container_path="/models/qwen",
        project_root=Path.cwd(),
        max_candidates=2,
        startup_max_attempts=3,
    )

    def _container_factory(_remote, container_config):
        if container_config.name.endswith("-first"):
            return first
        if container_config.name.endswith("-second"):
            return second
        raise AssertionError(container_config.name)

    original_container = ex.Container
    ex.Container = _container_factory
    try:
        with pytest.raises(RuntimeError, match="cleanup"):
            run_executor(config, remote=_FakeRemote())
    finally:
        ex.Container = original_container

    assert first.start_calls == 1
    assert first.remove_calls >= 2  # initial cleanup plus best-effort batch cleanup
    assert second.start_calls == 0
    assert not (tmp_path / "results" / "ranking.json").exists()


def test_server_process_pattern_avoids_self_match_and_finds_real_target():
    pattern = ex.SERVER_PROCESS_PATTERN
    no_target = subprocess.run(
        ["bash", "-lc", f"pgrep -f {shlex.quote(pattern)}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert no_target.returncode == 1, no_target.stdout

    target = subprocess.Popen(
        ["bash", "-c", "exec -a sglang.launch_server sleep 30"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            matched = subprocess.run(
                ["bash", "-lc", f"pgrep -f {shlex.quote(pattern)}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if str(target.pid) in matched.stdout.split():
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"pattern did not find target pid {target.pid}: {matched!r}")
    finally:
        target.terminate()
        target.wait(timeout=5)


def test_server_cleanup_waits_for_real_delayed_term_exit(monkeypatch):
    name = f"llm-infer-cleanup-{os.getpid()}"
    pattern = f"[l]{name[1:]}"
    script = (
        "import signal,sys,time; "
        "signal.signal(signal.SIGTERM, lambda *_: (time.sleep(0.2), sys.exit(0))); "
        "time.sleep(30)"
    )
    target = subprocess.Popen(
        ["bash", "-c", f"exec -a {shlex.quote(name)} python3 -c {shlex.quote(script)}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    class _LocalProcessContainer:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def exec(self, command: str, *, timeout=None) -> _FakeResult:
            self.commands.append(command)
            result = subprocess.run(
                ["bash", "-lc", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return _FakeResult(result.returncode, result.stdout, result.stderr)

    container = _LocalProcessContainer()
    monkeypatch.setattr(ex, "SERVER_PROCESS_PATTERN", pattern)
    monkeypatch.setattr(ex, "SERVER_CLEANUP_TERM_POLL_ATTEMPTS", 20)
    monkeypatch.setattr(ex, "SERVER_CLEANUP_TERM_POLL_INTERVAL_S", 0.025)
    try:
        for _ in range(100):
            ready = subprocess.run(
                ["bash", "-lc", f"pgrep -f {shlex.quote(pattern)}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if str(target.pid) in ready.stdout.split():
                break
            time.sleep(0.01)
        else:
            pytest.fail("delayed-exit target did not start")

        started = time.monotonic()
        ex._cleanup_servers_checked(container, candidate_id="real-delayed-exit")
        elapsed = time.monotonic() - started

        assert elapsed >= 0.15
        assert not any("pkill -KILL" in command for command in container.commands)
        target.wait(timeout=5)
    finally:
        if target.poll() is None:
            target.kill()
            target.wait(timeout=5)


def test_server_cleanup_allows_bounded_delayed_exit_after_term(
    tmp_path, workloads_output_len, monkeypatch
):
    monkeypatch.setattr(ex, "SERVER_CLEANUP_TERM_POLL_ATTEMPTS", 3)
    monkeypatch.setattr(ex, "SERVER_CLEANUP_TERM_POLL_INTERVAL_S", 0)
    container = _DelayedTermExitContainer({"delayed-exit": 2})
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(
            _lifecycle_test_config(tmp_path, "delayed-exit"),
            remote=_FakeRemote(),
        )
    finally:
        ex.Container = original_container

    assert summary["task_status"] == "COMPLETED"
    assert container.term_calls >= 2
    assert container.kill_calls == 0


def test_server_cleanup_escalates_to_kill_then_proves_absence(
    tmp_path, workloads_output_len, monkeypatch
):
    monkeypatch.setattr(ex, "SERVER_CLEANUP_TERM_POLL_ATTEMPTS", 3)
    monkeypatch.setattr(ex, "SERVER_CLEANUP_TERM_POLL_INTERVAL_S", 0)
    container = _RequiresKillServerContainer({"requires-kill": 2})
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(
            _lifecycle_test_config(tmp_path, "requires-kill"),
            remote=_FakeRemote(),
        )
    finally:
        ex.Container = original_container

    assert summary["task_status"] == "COMPLETED"
    assert container.term_calls >= 2
    assert container.kill_calls >= 2
