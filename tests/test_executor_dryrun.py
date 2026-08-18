"""Offline dry-run of the full two-round executor loop (no ssh/docker/network).

Drives ``run_executor`` end to end with a fake RemoteRunner + fake Container +
fake ClaudeCodeClient. Asserts the wiring that the unit tests can't reach:

  * the client skill (an LLM) is called EXACTLY ONCE per job (fairness), and every
    candidate/probe reuses that one template with only --max-concurrency /
    --num-prompts rewritten (byte-identical workload otherwise);
  * round 1 expands over ALL candidates, round 2 bisects only the top-K and
    REUSES round-1 probes as seeds (top-K seeded C are never re-benched);
  * the final ranking is goodput-descending and matches each candidate's true C*.

The fake "server" is a monotone SLA boundary per candidate: bench at C qualifies
iff C <= cstar[candidate], with throughput rising in C, so ranker goodput lands
on throughput(C*) exactly as in production.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runners import executor as ex
from runners.executor import ExecutorConfig, run_executor


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

    def run(self, command: str, *, timeout=None) -> _FakeResult:
        if command.startswith("echo $HOME"):
            return _FakeResult(stdout="/home/fake\n")
        if command == "echo ok":
            return _FakeResult(stdout="ok\n")
        if command.startswith("test -d"):
            return _FakeResult(stdout="config.json\nsafetensors\n")
        if command.startswith("docker image inspect"):
            return _FakeResult(stdout="sha256:abcdef\n")
        if command.startswith("docker pull"):
            return _FakeResult(stdout="\n")
        return _FakeResult(stdout="")


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
        self.alive = False
        self.bench_calls: list[tuple[str, int, int]] = []  # (candidate, C, num_prompts)
        self.last_bench_jsonl = ""
        self._result_files: dict[str, str] = {}   # container path -> jsonl text

    # -- lifecycle no-ops the executor calls on the container object ------------
    def start(self, *, timeout=None) -> _FakeResult:
        return _FakeResult()

    def is_running(self, *, timeout=None) -> bool:
        return True

    def stop(self, *, timeout=None) -> _FakeResult:
        return _FakeResult()

    def remove(self, *, force: bool = True, timeout=None) -> _FakeResult:
        return _FakeResult()

    def exec_detached(self, command: str, log_container_path: str, *, timeout=None):
        # launching a candidate's server: infer which candidate from the -v mount
        # path baked into the launch cmd is not available, so track by outputs dir
        self.alive = True
        return _FakeResult(stdout="1234\n")

    # -- the workhorse ---------------------------------------------------------
    def exec(self, command: str, *, timeout=None) -> _FakeResult:
        if "pgrep -f sglang.launch_server" in command:
            return _FakeResult(0 if self.alive else 1)
        if "pkill -f sglang.launch_server" in command:
            self.alive = False
            self.current = None
            return _FakeResult()
        if "/health" in command:
            return _FakeResult(0 if self.alive else 7)
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
        cand = self.current or "unknown"
        self.bench_calls.append((cand, conc, num_prompts))

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


class _FakeClient:
    """Returns one bench command per concurrency level; counts how often it runs."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, prompt, json_schema, add_dirs, allow_dangerous_permissions=True):
        self.calls += 1
        base = (
            "python -m sglang.bench_serving --backend sglang "
            "--host ${BENCHMARK_HOST} --port ${BENCHMARK_PORT} "
            "--model ${MODEL_PATH} --dataset-name random "
            "--random-input-len 1000 --random-output-len 1000 "
            "--output-file result_${JOB_ID}_${TIMESTAMP}.jsonl"
        )
        return {
            "benchmark_commands": [
                {
                    "concurrency": c,
                    "num_prompts": c * 4,
                    "command": f"{base} --max-concurrency {c} --num-prompts {c * 4}",
                    "reason": "grid",
                }
                for c in (1, 2, 4, 8, 16, 32)
            ]
        }


# --- fixtures -----------------------------------------------------------------


def _write_job(tmp_path: Path) -> Path:
    job = {
        "job_id": "dryrun-job",
        "engine": "sglang",
         "gpu_model": "pro5000",
         "gpu_count": 8,
         "gpu_memory_gb": 72,
        "model": "qwen36-35b",
        "image": "sglang-test",
        "workload": "chat_1k_1k",
        "benchmark_method": "sglang-bench-serving",
        "sla": {"max_avg_ttft_ms": 2000.0, "max_avg_tpot_ms": 80.0, "min_success_rate": 0.99},
        "search": {"max_candidates": 16, "max_runtime_minutes": 120},
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    return path


def _write_configs(tmp_path: Path, candidate_ids: list[str]) -> Path:
    path = tmp_path / "configs.jsonl"
    lines = []
    for cid in candidate_ids:
        lines.append(
            json.dumps(
                {
                    "id": cid,
                    "params": {},
                    "cmd": f"python -m sglang.launch_server --model-path ${{MODEL_PATH}} # {cid}",
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
    job_path = _write_job(tmp_path)
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
    client = _FakeClient()
    container = _FakeContainer(cstar)
    # Patch Container so run_executor uses our fake instead of docker-over-ssh.
    import runners.executor as exmod
    original_container = exmod.Container
    exmod.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=remote, client=client)
    finally:
        exmod.Container = original_container

    # -- FAIRNESS: the LLM client is called exactly once for the whole job -----
    assert client.calls == 1, f"client skill called {client.calls} times, expected 1"

    # -- every bench used the shared template: only C / num_prompts vary, and
    #    num_prompts == C * multiplier (4) every time.
    for cand, conc, num_prompts in container.bench_calls:
        assert num_prompts == conc * 4, (cand, conc, num_prompts)

    # -- ranking is goodput-descending and matches true C* --------------------
    ranking = summary["ranking"]
    ids_in_order = [row["candidate_id"] for row in ranking]
    assert ids_in_order == ["cand-b", "cand-d", "cand-a", "cand-c"]
    by_id = {row["candidate_id"]: row for row in ranking}
    for cid, expected_cstar in cstar.items():
        assert by_id[cid]["goodput_per_host"] == expected_cstar * 100.0 * 8  # gpu_count=8, tp_size=1
        assert by_id[cid]["best_concurrency"] == expected_cstar

    # -- top-K wiring: only the two best (by round-1 goodput) get round 2 ------
    assert summary["top_k"] == 2
    assert set(summary["top_ids"]) == {"cand-b", "cand-d"}
    for cid in ("cand-b", "cand-d"):
        assert "round2" in summary["candidates"][_index_of(summary, cid)]

    # -- ranking.json + per-candidate evidence were written -------------------
    assert (results_dir / "ranking.json").exists()
    for cid in cstar:
        assert (results_dir / cid / "run_result.r1.json").exists()


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
    client = _FakeClient()
    container = _FakeContainer(cstar)
    import runners.executor as exmod
    original_container = exmod.Container
    exmod.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=remote, client=client)
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


def _index_of(summary: dict, candidate_id: str) -> int:
    for i, row in enumerate(summary["candidates"]):
        if row["candidate_id"] == candidate_id:
            return i
    raise AssertionError(f"{candidate_id} not in summary candidates")
