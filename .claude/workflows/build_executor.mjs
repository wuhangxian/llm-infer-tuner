export const meta = {
  name: 'build-phase2-executor',
  description: 'Author runners/ phase-2 executor skeleton (SSH+docker, 1 candidate x 1 concurrency) across parallel agents against a shared interface contract',
  phases: [
    { title: 'Foundation', detail: 'leaf modules: remote, metrics, readiness, ranker' },
    { title: 'Integration', detail: 'container (needs remote), bench_runner (needs container+metrics)' },
    { title: 'Orchestrate', detail: 'executor main loop + __main__' },
    { title: 'Verify', detail: 'py_compile + import + minimal pytest + consistency report' },
  ],
}

const ROOT = '/data/home/dorianwu/aaawhx-study/llm-infer-tuner'

// ---------------------------------------------------------------------------
// Shared context handed to EVERY agent so parallel-written modules line up.
// ---------------------------------------------------------------------------
const STYLE = `
## 项目代码风格(必须匹配 planner/ 现有代码)
- 每个文件首行 module docstring(英文,一句话)。
- 紧接 \`from __future__ import annotations\`。
- 类型标注齐全;低注释密度(只在非显然处注释)。
- 依赖注入以便测试:凡是会调 subprocess / 时间 / 网络的,都把 runner/sleep/now 做成构造参数或关键字参数,默认值指向真实实现(参照 planner/claude_code_client.py 的 \`runner: Runner = subprocess.run\`)。
- 不引新第三方依赖(不装 paramiko/fabric/requests)。SSH 走 subprocess 调系统 ssh 二进制;HTTP 探活走容器内 curl。
- dataclass 用 \`from dataclasses import dataclass, field\`。
- 不要在模块顶层执行任何 SSH/docker/网络/真实 IO。这是**离线写代码**,不连服务器。
- 允许在本地跑 \`.venv/bin/python -m py_compile <file>\` 自检语法。

## 项目根
${ROOT}
从项目根用相对 import:\`from planner.claude_code_client import ClaudeCodeClient\`、\`from schemas.job_spec import JobSpec, SLA\`。
`

const CONTRACT = `
## 全体模块统一接口契约(所有 agent 都按这份写,保证 import 对得上)

### runners/remote.py
\`\`\`python
from collections.abc import Callable, Sequence
import subprocess
Runner = Callable[..., "subprocess.CompletedProcess[str]"]

@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    @property
    def ok(self) -> bool: ...   # returncode == 0

class RemoteRunner:
    # ssh_target 形如 "ubuntu@122.51.115.16";免密(key)。
    def __init__(self, ssh_target: str, *, ssh_options: Sequence[str] = ("-o","BatchMode=yes","-o","StrictHostKeyChecking=accept-new"), runner: Runner = subprocess.run, default_timeout: int = 600) -> None: ...
    def build_ssh_argv(self, command: str) -> list[str]: ...   # ["ssh", *ssh_options, ssh_target, command] —— 便于单测
    def run(self, command: str, *, timeout: int | None = None) -> CommandResult: ...   # 经 ssh 在远端跑
    def run_local(self, argv: Sequence[str], *, timeout: int | None = None) -> CommandResult: ...  # 本地跑(如把远端文件 scp 回来)
\`\`\`

### runners/container.py  (import RemoteRunner, CommandResult from runners.remote)
\`\`\`python
@dataclass
class ContainerConfig:
    image_ref: str
    name: str
    model_host_dir: str            # 开发机上模型目录
    model_container_path: str      # 容器内挂载点(= 压测/起服 --model-path 用)
    outputs_host_dir: str          # 开发机上结果目录
    outputs_container_path: str = "/workspace/outputs"
    gpus: str = "all"
    shm_size: str = "32g"
    port: int = 30000
    extra_run_args: Sequence[str] = ()

class Container:
    def __init__(self, remote: RemoteRunner, config: ContainerConfig) -> None: ...
    def start(self) -> CommandResult: ...        # docker run -d --gpus <gpus> --shm-size <shm> -v model:... -v outputs:... -p port:port --name <name> <image> sleep infinity
    def exec(self, command: str, *, timeout: int | None = None) -> CommandResult: ...   # docker exec <name> bash -lc '<command>'  (用 remote.run 拼 docker exec)
    def exec_detached(self, command: str, log_container_path: str) -> CommandResult: ...  # 容器内后台起服:nohup <command> > log 2>&1 & (返回后不阻塞)
    def is_running(self) -> bool: ...            # docker inspect -f {{.State.Running}}
    def stop(self) -> CommandResult: ...
    def remove(self, *, force: bool = True) -> CommandResult: ...
\`\`\`

### runners/readiness.py
\`\`\`python
import time
def wait_until_ready(probe: Callable[[], bool], *, is_alive: Callable[[], bool] | None = None,
                     timeout_s: int = 1800, interval_s: float = 5.0,
                     sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.monotonic) -> bool: ...
    # 轮询 probe();命中 True 立即返回 True。若给了 is_alive 且 is_alive()==False(进程崩)→立即返回 False。超时→False。
def make_health_probe(container, *, host: str = "127.0.0.1", port: int = 30000) -> Callable[[], bool]: ...
    # 返回一个 callable:执行 container.exec(f"curl -sf http://{host}:{port}/health") 并返回 result.ok
\`\`\`

### runners/metrics.py  (解析 sglang.bench_serving v0.5.10 真实输出)
\`\`\`python
from typing import Any
# sglang v0.5.10 bench_serving --output-file 的真实字段(已核对源码):
#   throughput: request_throughput / output_throughput / total_throughput   (goodput 用 total_throughput!)
#   ttft:  mean_ttft_ms / median_ttft_ms / std_ttft_ms / p99_ttft_ms        (SLA 的 avg = mean)
#   tpot:  mean_tpot_ms / median_tpot_ms / std_tpot_ms / p99_tpot_ms
#   counts: completed / total_input_tokens / total_output_tokens / duration
#   注意:输出里没有 num_prompts 字段 —— 由调用方按 concurrency×multiplier 传入。
@dataclass
class RunResult:
    candidate_id: str
    concurrency: int
    num_prompts: int
    completed: int
    success_rate: float          # completed / num_prompts
    request_throughput: float
    output_throughput: float
    total_throughput: float
    mean_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    p99_tpot_ms: float
    total_output_tokens: int
    avg_output_tokens: float     # total_output_tokens / completed (completed>0 否则 0.0)
    duration: float
    status: str = "ok"           # ok | health_check_failed | startup_oom | runtime_oom | bad_args
    failure_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

def parse_bench_text(text: str, *, candidate_id: str, concurrency: int, num_prompts: int) -> RunResult: ...
    # text 是 result jsonl 全文(可能多行,每行一个 JSON dict)。取匹配 max_concurrency==concurrency 的那条;
    # 找不到就取最后一个非空行。空/无法解析 → status="bad_args" 的 RunResult(数值填 0)。
def parse_bench_file(path, *, candidate_id: str, concurrency: int, num_prompts: int) -> RunResult: ...
    # 读本地文件再调 parse_bench_text。
\`\`\`

### runners/ranker.py  (import RunResult from runners.metrics; SLA from schemas.job_spec)
\`\`\`python
def data_health_check(result: RunResult, *, output_len: int) -> tuple[bool, str | None]: ...
    # §5: completed>0;avg_output_tokens >= output_len*0.9 否则 (False,"truncated:...")。通过→(True,None)。
def passes_sla(result: RunResult, sla) -> bool: ...
    # mean_ttft_ms<=sla.max_avg_ttft_ms AND mean_tpot_ms<=sla.max_avg_tpot_ms AND success_rate>=sla.min_success_rate
def candidate_goodput(results: list[RunResult], sla, *, output_len: int) -> float: ...
    # 只在 data_health_check 通过 且 passes_sla 的档里取 max(total_throughput);无合格档→0.0
def rank_candidates(results_by_candidate: dict[str, list[RunResult]], sla, *, output_len: int) -> list[dict]: ...
    # 返回 [{"candidate_id","goodput","best_concurrency"}] 按 goodput 降序;骨架期可只被 executor 调一次。
\`\`\`

### runners/bench_runner.py  (import Container/CommandResult; ClaudeCodeClient; JobSpec)
\`\`\`python
CLIENT_SKILL_DIR = ".claude/skills/sglang-client-config-gen"
BENCH_SCHEMA = {  # 传给 claude 的 --json-schema
  "type":"object","required":["benchmark_commands"],
  "properties":{"benchmark_commands":{"type":"array","items":{"type":"object",
    "required":["concurrency","num_prompts","command","reason"],
    "properties":{"concurrency":{"type":"integer"},"num_prompts":{"type":"integer"},
      "command":{"type":"string"},"reason":{"type":"string"}}}}}}

@dataclass
class BenchCommand:
    concurrency: int
    num_prompts: int
    command: str      # 含占位符 \${BENCHMARK_HOST}/\${BENCHMARK_PORT}/\${MODEL_PATH}/\${JOB_ID}/\${TIMESTAMP}
    reason: str = ""

def generate_benchmark_commands(job: JobSpec, *, project_root, client: ClaudeCodeClient,
                                allow_dangerous_permissions: bool = True) -> list[BenchCommand]: ...
    # 构造 prompt(引用 CLIENT_SKILL_DIR/SKILL.md + knowledge.md,给出 JobSpec JSON),
    # client.run(prompt, json_schema=BENCH_SCHEMA, add_dirs=[project_root, project_root/CLIENT_SKILL_DIR]),
    # 解析 payload["benchmark_commands"] → list[BenchCommand]。参照 planner/search_planner.py 的调法。
def substitute_placeholders(command: str, *, host: str, port: int, model_path: str,
                            job_id: str, timestamp: str, dataset_path: str = "") -> str: ...
    # 替换 \${BENCHMARK_HOST}\${BENCHMARK_PORT}\${MODEL_PATH}\${JOB_ID}\${TIMESTAMP}\${DATASET_PATH}
def run_benchmark(container, command: str, *, timeout: int = 3600) -> "CommandResult": ...
    # container.exec(command)
\`\`\`

### runners/executor.py  (串起全部 + __main__)
\`\`\`python
@dataclass
class ExecutorConfig:
    job_path: Path
    configs_path: Path            # outputs/<job>/configs.jsonl
    results_dir: Path             # outputs/<job>/results
    ssh_target: str               # ubuntu@122.51.115.16
    image_ref: str
    model_host_dir: str
    model_container_path: str
    project_root: Path
    max_candidates: int = 1       # 骨架:1
    concurrencies: list[int] | None = None   # 骨架:[1]
    port: int = 30000
    container_name: str = "llm-infer-tuner-exec"

def run_executor(config: ExecutorConfig, *, remote=None, client=None) -> dict: ...
    # 1) 读 job(JobSpec)+ workload(catalogs/workloads.yaml 取 output_tokens.value 做体检 output_len)
    # 2) 读 configs.jsonl 前 max_candidates 条(每行有 id/params/cmd/reasons)
    # 3) Container(remote 或默认 RemoteRunner(ssh_target)).start()
    # 4) 每候选:把 cmd 里的 \${MODEL_PATH} 换成 model_container_path → exec_detached 后台起服(日志落 outputs)
    #    → wait_until_ready(make_health_probe(container)) → 若失败记 status=health_check_failed 跳过
    #    → generate_benchmark_commands()(client 或默认 ClaudeCodeClient)取 concurrency∈concurrencies 的档
    #    → substitute_placeholders(host=127.0.0.1,port,model_path=model_container_path,job_id,timestamp)
    #    → run_benchmark → 结果 jsonl 在容器内 outputs 挂载 → 本地 results_dir 可直接读(同一挂载)
    #    → parse_bench_file → RunResult → 写 results_dir/<candidate_id>/run_result.json
    #    → 停该候选服务(pkill sglang 或重启容器,骨架期可 container.exec("pkill -f sglang.launch_server || true"))
    # 5) container.stop()+remove();汇总 rank_candidates 写 results_dir/ranking.json;返回 summary dict
def main(argv: list[str] | None = None) -> int: ...   # argparse:--job --configs --results --ssh-target --image-ref --model-host-dir --model-container-path [--max-candidates][--concurrencies]
if __name__ == "__main__": raise SystemExit(main())
\`\`\`
`

async function writeModule(file, responsibilities, phase) {
  const prompt = `你在为 llm-infer-tuner 写第二阶段执行器的一个模块。**只写这一个文件,严格按契约,写完用 Write 工具落盘。**

目标文件:\`${ROOT}/${file}\`

${STYLE}

${CONTRACT}

## 你负责的文件:${file}
${responsibilities}

## 交付
1. 用 Read 看 \`${ROOT}/planner/claude_code_client.py\`(风格样板;若你的模块 import 它或 schemas,也读对应文件确认真实签名)。
2. 把 ${file} 完整写出来(Write 工具,绝对路径 ${ROOT}/${file})。
3. 跑 \`cd ${ROOT} && .venv/bin/python -m py_compile ${file}\` 确认语法通过(不通过就改到通过)。
4. 不要连 SSH/docker/网络。不要改其它文件。
返回:一句话说明你写了什么 + py_compile 是否通过。`
  return agent(prompt, { label: file, phase })
}

phase('Foundation')
await parallel([
  () => writeModule('runners/remote.py',
    '实现 CommandResult + RemoteRunner。run() 用 self.runner 调 ["ssh", *ssh_options, ssh_target, command];run_local() 直接调 argv。build_ssh_argv 单独抽出便于测试。所有 subprocess 调用 capture_output=True,text=True,check=False,带 timeout。', 'Foundation'),
  () => writeModule('runners/metrics.py',
    '实现 RunResult + parse_bench_text + parse_bench_file。严格用契约里列的 sglang v0.5.10 真实字段名(total_throughput/mean_ttft_ms/mean_tpot_ms/p99_*/completed/total_output_tokens/duration)。success_rate=completed/num_prompts(num_prompts>0);avg_output_tokens=total_output_tokens/completed(completed>0 否则 0.0)。多行 jsonl 取 max_concurrency==concurrency 的记录,找不到取最后一行。解析失败→status="bad_args"、数值 0 的 RunResult。raw 存原始 dict。', 'Foundation'),
  () => writeModule('runners/readiness.py',
    '实现 wait_until_ready + make_health_probe。wait_until_ready 用注入的 now()/sleep() 做超时循环(不直接依赖真实时间以便测试):循环内先 is_alive(若提供)判崩溃,再 probe();命中即 True,超时 False。make_health_probe 返回闭包调 container.exec 打 /health。', 'Foundation'),
  () => writeModule('runners/ranker.py',
    '实现 data_health_check + passes_sla + candidate_goodput + rank_candidates。import RunResult from runners.metrics;SLA 从 schemas.job_spec import SLA(先 Read 确认字段:max_avg_ttft_ms/max_avg_tpot_ms/min_success_rate)。goodput 只在体检通过且过 SLA 的档取 max(total_throughput)。rank 按 goodput 降序返回 dict 列表。', 'Foundation'),
])

phase('Integration')
await parallel([
  () => writeModule('runners/container.py',
    '实现 ContainerConfig + Container。import RemoteRunner/CommandResult from runners.remote。start() 拼 docker run -d(--gpus/--shm-size/-v 模型/-v outputs/-p/--name/image/sleep infinity)经 remote.run 跑。exec() 拼 docker exec <name> bash -lc <shlex.quote 后的 command>。exec_detached() 在容器内 nohup ... > log 2>&1 & 后台起服。is_running() 用 docker inspect。stop/remove 对应 docker stop/rm。', 'Integration'),
  () => writeModule('runners/bench_runner.py',
    '实现 BenchCommand + BENCH_SCHEMA + CLIENT_SKILL_DIR + generate_benchmark_commands + substitute_placeholders + run_benchmark。generate 参照 planner/search_planner.py 的 client.run 调法(先 Read 它);add_dirs=[project_root, project_root/CLIENT_SKILL_DIR]。substitute 用 str.replace 逐个换占位符。run_benchmark 调 container.exec。', 'Integration'),
])

phase('Orchestrate')
await writeModule('runners/executor.py',
  '实现 ExecutorConfig + run_executor + main + __main__。import 全部同级模块(runners.remote/container/readiness/bench_runner/metrics/ranker)、schemas.job_spec.JobSpec、planner.claude_code_client.ClaudeCodeClient。run_executor 按契约里的 8 步编排;骨架期 max_candidates 默认 1、concurrencies 默认 [1]。workload 的 output_len 从 catalogs/workloads.yaml 读(可用 yaml,已是项目依赖;不确定就 Read pyproject.toml 确认 pyyaml 在依赖里)。main 用 argparse 暴露全部 ExecutorConfig 字段。写 run_result.json 与 ranking.json 用 json.dump(indent=2)。', 'Orchestrate')

phase('Verify')
const verify = await agent(
  `验证 llm-infer-tuner 第二阶段执行器骨架的一致性与可用性(不连 SSH/docker/网络,纯离线校验)。

工作目录:${ROOT}

做这些:
1. \`cd ${ROOT} && .venv/bin/python -m py_compile runners/*.py\` —— 全部语法通过?
2. \`cd ${ROOT} && .venv/bin/python -c "import runners.executor, runners.container, runners.bench_runner, runners.readiness, runners.metrics, runners.ranker, runners.remote"\` —— 全部 import 通过?(能抓出跨模块签名/名字对不上)
3. 快速读 runners/ 各文件,核对:模块间 import 的类名/函数名/参数与契约一致(RemoteRunner/CommandResult/Container/RunResult/SLA 用法);metrics 用的是 sglang 真实字段名(total_throughput、mean_ttft_ms、mean_tpot_ms,不是 total_token_throughput/avg_*)。
4. 写最小 pytest 到 ${ROOT}/tests/test_runners_skeleton.py,只测纯逻辑(不碰网络):
   - metrics.parse_bench_text 用一段构造的真实字段 jsonl(含 total_throughput/mean_ttft_ms/mean_tpot_ms/p99_*/completed/total_output_tokens/duration/max_concurrency),断言 RunResult 各字段与 success_rate/avg_output_tokens 计算正确。
   - ranker.passes_sla / data_health_check / candidate_goodput 用构造 RunResult 断言(过 SLA、截断作废、goodput 取 max)。
   - remote.RemoteRunner.build_ssh_argv 断言拼出的 argv 结构正确(用假的 runner,不真跑 ssh)。
   然后 \`cd ${ROOT} && .venv/bin/python -m pytest tests/test_runners_skeleton.py -q\` 跑通。
5. 若发现任何模块违反契约/import 失败/字段名错,直接修那个文件到通过。

返回结构化结论:{py_compile: pass/fail, imports: pass/fail, tests: "N passed/M failed", 修了哪些文件, 仍存在的问题列表}。`,
  { label: 'verify+tests', phase: 'Verify', schema: {
    type: 'object',
    required: ['py_compile', 'imports', 'tests', 'issues'],
    properties: {
      py_compile: { type: 'string' },
      imports: { type: 'string' },
      tests: { type: 'string' },
      fixed_files: { type: 'array', items: { type: 'string' } },
      issues: { type: 'array', items: { type: 'string' } },
    },
  } })

return verify
