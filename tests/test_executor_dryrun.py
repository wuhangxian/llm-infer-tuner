"""Offline dry-run of the full two-round executor loop (no ssh/docker/network).

Drives ``run_executor`` end to end with a fake RemoteRunner + fake Container +
fake ClaudeCodeClient. Asserts the wiring that the unit tests can't reach:

  * the client skill (an LLM) is called EXACTLY ONCE per job (fairness), and every
    candidate/probe reuses that one template with only --max-concurrency /
    --num-prompts rewritten (byte-identical workload otherwise);
  * round 1 expands over ALL candidates, round 2 precisely refines ALL candidates
    and REUSES round-1 probes as seeds (seeded C are never re-benched);
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
        self.warmup_calls: list[tuple[str, int, int]] = []  # 预热压测(丢弃,不计入搜索)
        # 满载测试要证明「同一并发档打到了 N 个不同副本端口」,单独记录 (candidate, C, port),
        # 不动 bench_calls 的三元组形状(现有用例仍按 (cand, conc, num_prompts) 解包)。
        self.bench_ports: list[tuple[str, int, int]] = []
        self.last_bench_jsonl = ""
        self._result_files: dict[str, str] = {}   # container path -> jsonl text
        self.launched_cmds: list[str] = []        # 每次 exec_detached 收到的 server 启动命令

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
        self.launched_cmds.append(command)
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


class _FlakyServerContainer(_FakeContainer):
    def __init__(self, cstar_by_candidate: dict[str, int], failures_before_ready: int) -> None:
        super().__init__(cstar_by_candidate)
        self.failures_remaining = failures_before_ready
        self.launch_attempts = 0

    def exec_detached(self, command: str, log_container_path: str, *, timeout=None):
        self.launch_attempts += 1
        self.launched_cmds.append(command)
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
        return _FakeResult()


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


def _write_job(tmp_path: Path, *, baseline_threshold_pct: float = 0) -> Path:
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
        "search": {
            "max_candidates": 16,
            "max_runtime_minutes": 120,
            "baseline_threshold_pct": baseline_threshold_pct,
        },
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
    for cid, params in params_by_id.items():
        lines.append(
            json.dumps(
                {
                    "id": cid,
                    "params": params,
                    "cmd": f"python -m sglang.launch_server --model-path ${{MODEL_PATH}} # {cid}",
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
    client = _FakeClient()
    container = _FakeContainer(cstar)
    import runners.executor as exmod
    original_container = exmod.Container
    exmod.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=remote, client=client)
    finally:
        exmod.Container = original_container

    ranking = summary["ranking"]
    assert ranking[0]["candidate_id"] == "full"
    assert ranking[0]["best_concurrency"] == 8

    # tp=2, gpu=8 -> floor(8/2)=4 副本。per_host = (4×800 / 4) × 4 = 3200。
    # 若双重计数会是 4×800 × 4 = 12800。
    assert ranking[0]["tp_size"] == 2
    assert ranking[0]["instances_per_host"] == 4.0
    assert ranking[0]["goodput_per_host"] == 3200.0

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
        # (a) cmd 整串,带 extra_buffer,没写 disable-radix —— 你手写 config 渲染后的样子
        json.dumps({
            "id": "cand-cmd",
            "cmd": "python -m sglang.launch_server --model-path ${MODEL_PATH} "
                   "--tp-size 1 --mamba-radix-cache-strategy extra_buffer "
                   "--disable-radix-cache=0",
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
        job_path=_write_job(tmp_path),
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
    client = _FakeClient()
    container = _FakeContainer({"cand-cmd": 2, "cand-params": 2, "cand-already": 2})
    import runners.executor as exmod
    original_container = exmod.Container
    exmod.Container = lambda _remote, _cfg: container
    try:
        run_executor(config, remote=remote, client=client)
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


def test_threshold_marks_but_never_removes_candidates(tmp_path, workloads_output_len):
    cstar = {"baseline": 10, "fast": 16, "slow": 4}
    job_path = _write_job(tmp_path, baseline_threshold_pct=20)
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
    remote, client, container = _FakeRemote(), _FakeClient(), _FakeContainer(cstar)
    original_container = ex.Container
    ex.Container = lambda _remote, _cfg: container
    try:
        summary = run_executor(config, remote=remote, client=client)
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
        summary = run_executor(config, remote=_FakeRemote(), client=_FakeClient())
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
        summary = run_executor(config, remote=_FakeRemote(), client=_FakeClient())
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
        configs_path=_write_configs(tmp_path, ["broken"]),
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
        summary = run_executor(config, remote=_FakeRemote(), client=_FakeClient())
    finally:
        ex.Container = original_container

    row = summary["candidate_results"][0]
    assert summary["task_status"] == "INCOMPLETE"
    assert summary["ranking_status"] == "PROVISIONAL"
    assert row["candidate_id"] == "broken"
    assert row["status"] == "failed"
    assert row["failed_at"]
    assert row["failed_round"] == 2
    assert row["failure_reason"]
