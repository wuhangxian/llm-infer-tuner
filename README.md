# llm-infer-tuner

大模型推理服务参数寻优工具：给定模型、GPU、SGLang 镜像、输入/输出长度和 SLA，先生成一组可执行的服务端配置，再到真实 GPU 机器上启动服务、压测、搜索可用并发度并排名，最后输出可复现的部署报告。

它把“让 AI 提出候选”和“用确定性程序做实验”拆开：AI 只负责配置推导，执行器不再二次调用 AI，而是按固定规则完成校验、部署、压测和排名。

## 工作流

```text
JobSpec(job.json)
      │
      ├─ gen_configs.sh
      │    └─ tclaude 或 claude 读取 SKILL.md + knowledge.md + catalogs
      │         → outputs/<job_id>/configs.jsonl
      │
      └─ run_executor.sh
           └─ SSH 到目标 GPU 机 → Docker 启动 SGLang → /health 检查
                → 预热 → 多并发压测 → SLA 过滤 → goodput 排名
                     → outputs/<job_id>/results/*
                              │
                              └─ gen_report.sh → best_config.md
```

三个入口分别负责：

| 入口 | 作用 | 需要 AI 登录 | 需要目标 GPU 机 |
| --- | --- | --- | --- |
| `gen_configs.sh` | 根据 JobSpec 生成候选启动配置 | 是 | 否 |
| `run_executor.sh` | 远程启动、压测、搜索并排名 | 否 | 是 |
| `gen_report.sh` | 根据已有结果生成 Markdown 报告 | 否 | 否 |

## 功能总览

### AI 配置规划

- 默认使用腾讯内网 `tclaude`，也支持公开 `claude` CLI。
- AI 按顺序读取调优规范、经验库、GPU/模型/负载/镜像目录，生成结构化候选。
- 候选覆盖并行度、attention 后端、显存比例、KV cache、chunked prefill、调度策略和投机解码等参数。
- `--agent` 切换 `tclaude`/`claude`，`--model` 覆盖模型；公开 `claude` 未指定模型时使用其默认模型。
- 默认流式显示读取和产出事件，并把完整原始返回保存到 `claude-raw-outputs/`。
- AI 调用带单次超时、TERM→KILL 宽限和有限重试；失败或中断不会覆盖已有正式配置。

### 生成后的硬校验

候选生成后，代码会拒绝确定无法启动的配置：

- attention 后端必须同时满足 GPU、CUDA 和目标 SGLang 镜像的支持范围。
- TP/EP、MoE 专家数、块量化切分等整除关系必须合法；dense 模型不会错误生成 EP 轴。
- 启动命令中的参数必须属于目标镜像有效参数白名单，别名会按镜像版本翻译。

### 远程真机执行

`run_executor.sh` 通过 SSH 连接目标机器，并在目标机器上完成：

- SSH、GPU 数量/显存、模型目录、SGLang 镜像和端口的 preflight 校验。
- 清理同名旧容器、占用 GPU 的进程和冲突端口。
- 为每个候选启动独立服务，轮询 `/health`，记录启动失败原因并按配置重试。
- 自动发现 NUMA 分组，让同一个 TP 实例尽量落在同一 NUMA 节点。
- 多候选并行执行，每个候选使用独立 GPU 和端口；也可用 `FILL_HOST=1` 做整机满载第二轮实测。
- 先 warmup 再正式压测，避免预热数据污染正式结果。
- 自适应搜索并发度，按 SLA 过滤延迟、成功率和输出完整性，再按 goodput 排名。
- 压测命令由 `workload` 和 `benchmark_method` 确定性拼装，执行阶段不调用 AI，结果可重复审计。

### 公平性和特殊模型约束

- 所有候选最终都会强制带且只带一个 `--disable-radix-cache`，避免固定测试请求命中前缀缓存，污染候选对比。
- Mamba/GDN 模型只使用与关闭 radix cache 兼容的 `no_buffer`；依赖 radix cache 的 `extra_buffer` 不会生成。
- 投机解码是可搜索轴，但是否生成由模型卡片的 `mtp_params` 和硬约束决定。自带 MTP 权重的 EAGLE/NEXTN 类方案可以直接候选；需要外挂 draft 模型的算法必须同时提供 `speculative_draft_model_path`，否则会被排除。
- 不会把 workload 的输入/输出长度错误写成 `--context-length`；服务上下文长度由模型和 SGLang 自身配置决定。

### 可读报告

`gen_report.sh` 只读取已有的排名、候选和结果文件，不重新压测。报告展示：

- 任务、模型、镜像、GPU、负载和 SLA。
- 最佳候选的生效参数与完整启动命令。
- 最佳并发度、TTFT、TPOT、吞吐、成功率和输出完整性。
- 基线对比、候选差异、失败原因和结果文件位置。

### 知识库自动同步

`update.sh` 会同步 SGLang 参数、attention 后端、模型架构/量化/MoE/MTP 信息和标准 workload，并生成待审核的约束 diff。新增模型、GPU、镜像或经验，优先修改 `catalogs/` 和 `knowledge.md`，不需要改主执行流程。

## 推荐用法：直接使用 Docker 镜像

镜像已包含 Python/uv、Node.js、`tclaude`、公开 `claude`、`jq`、SSH 和 `sshpass`。模型权重、目标 GPU 机和登录态不打进镜像，而是在运行时提供。

镜像地址：

```bash
ghcr.io/wuhangxian/llm-infer-tuner:0903-dorianwu
```

GHCR 包已关联到本项目：[查看镜像包](https://github.com/users/wuhangxian/packages/container/package/llm-infer-tuner)。该包目前为 private，拉取前需要有权限的 GitHub 登录态。

本文件也会被复制到镜像内的 `/app/README.md`，因此在线项目 README 和容器内说明保持一致。

### 1. 拉取镜像和准备目录

```bash
docker login ghcr.io
docker pull ghcr.io/wuhangxian/llm-infer-tuner:0903-dorianwu

export TUNER_IMAGE=ghcr.io/wuhangxian/llm-infer-tuner:0903-dorianwu

mkdir -p input/jobs input/targets input/configs outputs claude-raw-outputs
mkdir -p "$HOME/.tclaude" "$HOME/.claude"
```

宿主机的 `input/`、`outputs/`、`claude-raw-outputs/` 是运行时挂载目录；镜像内已经有代码和知识库，不需要把项目源码复制进容器。

### 2. 首次登录 AI CLI

登录态必须持久化挂载；只在临时容器里登录而不挂载目录，容器删除后登录态也会消失。

腾讯内网账号使用 `tclaude`：

```bash
docker run --rm -it \
  -v "$HOME/.tclaude:/home/runner/.tclaude" \
  "$TUNER_IMAGE" tclaude login
```

如果终端提示浏览器没有自动打开，就复制它打印的 URL 到浏览器完成登录，再回到终端等待完成。

公开 Claude 账号使用 `claude`：

```bash
docker run --rm -it \
  -v "$HOME/.claude:/home/runner/.claude" \
  "$TUNER_IMAGE" claude login
```

可以同时挂载并登录两个 CLI；脚本通过 `--agent tclaude` 或 `--agent claude` 选择实际使用的一个。登录信息只在运行时挂载，不会固化进镜像。

### 3. 第一步：生成候选配置

准备一个 `input/jobs/<job>.json`，格式见后文。然后运行：

```bash
docker run --rm \
  -v "$PWD/input:/app/input:ro" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$PWD/claude-raw-outputs:/app/claude-raw-outputs" \
  -v "$HOME/.tclaude:/home/runner/.tclaude" \
  "$TUNER_IMAGE" \
  ./gen_configs.sh input/jobs/qwen35-27-input8k-output1k-cand16.json
```

默认输出：

```text
outputs/<job_id>/configs.jsonl
claude-raw-outputs/<run-id>.stdout.jsonl
claude-raw-outputs/<run-id>.stderr.log
```

切换公开 Claude 或调整模型：

```bash
docker run --rm \
  -v "$PWD/input:/app/input:ro" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$PWD/claude-raw-outputs:/app/claude-raw-outputs" \
  -v "$HOME/.claude:/home/runner/.claude" \
  "$TUNER_IMAGE" \
  ./gen_configs.sh --agent claude --model claude-opus-4-8 \
  input/jobs/qwen35-27-input8k-output1k-cand16.json
```

常用环境变量：

```bash
# 默认单次 600 秒、超时后再重试 1 次；下面示例改为 900 秒且不重试
docker run --rm -e GEN_TIMEOUT_SECONDS=900 -e GEN_MAX_RETRIES=0 \
  -v "$PWD/input:/app/input:ro" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$PWD/claude-raw-outputs:/app/claude-raw-outputs" \
  -v "$HOME/.tclaude:/home/runner/.tclaude" \
  "$TUNER_IMAGE" ./gen_configs.sh input/jobs/<job>.json

# GEN_STREAM=0 使用一次性 JSON 返回，默认 GEN_STREAM=1
```

### 4. 第二步：远程启动、压测和排名

准备 `input/targets/<target>.json`。目标机器必须有 Docker、NVIDIA 容器运行时、模型权重和 JobSpec 指定的 SGLang 镜像；运行此工具的机器不需要 GPU。

```bash
docker run --rm \
  -v "$PWD/input:/app/input:ro" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$HOME/.ssh:/home/runner/.ssh:ro" \
  "$TUNER_IMAGE" \
  ./run_executor.sh \
  input/jobs/qwen35-27-input8k-output1k-cand16.json \
  input/targets/qwen35-27-input8k-output1k-cand16-100-67-146-43.json
```

默认结果目录：

```text
outputs/<job_id>/results/
├── task_status.json
├── candidate_results.jsonl
├── ranking.json
└── <candidate-id>/              # server、bench、启动尝试日志和原始结果
```

整机满载第二轮实测：

```bash
docker run --rm \
  -e FILL_HOST=1 \
  -e MAX_PARALLEL=8 \
  -v "$PWD/input:/app/input:ro" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$HOME/.ssh:/home/runner/.ssh:ro" \
  "$TUNER_IMAGE" \
  ./run_executor.sh input/jobs/<job>.json input/targets/<target>.json
```

`FILL_HOST=1` 会把候选复制成 `floor(gpu_count / tp_size)` 个实例，在整机上真实并发求和；不设置时只做默认的候选级搜索。

服务启动防护也可以按机器情况调整：

```bash
STARTUP_STALL_TIMEOUT_SECONDS=600 \
STARTUP_HARD_TIMEOUT_SECONDS=1800 \
STARTUP_MAX_ATTEMPTS=3 \
./run_executor.sh input/jobs/<job>.json input/targets/<target>.json
```

### 5. 第三步：生成报告

```bash
docker run --rm \
  -v "$PWD/outputs:/app/outputs:rw" \
  "$TUNER_IMAGE" \
  ./gen_report.sh outputs/<job_id>
```

报告写入 `outputs/<job_id>/best_config.md`。它不会启动服务，也不会重新消耗 GPU 机器上的压测时间。

### Docker 常见问题

- `permission denied`：镜像默认用 UID/GID `1000:1000` 的 `runner` 用户运行。确保宿主机的 `outputs/` 和 `claude-raw-outputs/` 可由 UID 1000 写入，例如 `sudo chown -R 1000:1000 outputs claude-raw-outputs`。
- `docker.sock permission denied`：这是宿主机用户没有 Docker 权限，不是镜像内程序错误；把用户加入 `docker` 组后重新登录，或按机器规范使用 Docker。
- `tclaude` 每次都要求登录：通常是没有挂载 `$HOME/.tclaude`，或者挂载到了错误的容器路径。镜像中的对应路径是 `/home/runner/.tclaude`。
- `gen_configs.sh` 有原始日志但没有正式 `configs.jsonl`：先检查 `outputs/` 写权限；脚本只有在结构化返回完整解析成功后才原子替换正式文件。
- 远程 `/health` 不通过：优先查看 `outputs/<job_id>/results/<candidate-id>/server*.log`，常见原因是模型路径、目标镜像、TP/EP 整除关系、显存比例或模型专属参数不匹配。
- 本地没有 GPU：可以生成配置和报告；真实压测必须能 SSH 到一台有目标 GPU、模型和 SGLang 镜像的机器，本项目没有“本地无 GPU 的假压测”模式。

## 使用源码运行

如果希望修改脚本或知识库，可以直接使用 GitHub 或工蜂上的代码：

```bash
git clone -b main https://github.com/wuhangxian/llm-infer-tuner.git
cd llm-infer-tuner
uv sync
```

生成候选和执行器的命令与镜像内完全相同：

```bash
./gen_configs.sh input/jobs/<job>.json
./run_executor.sh input/jobs/<job>.json input/targets/<target>.json
./gen_report.sh outputs/<job_id>
```

开发依赖包含 pytest、ruff 和 pyright；需要自检时可运行：

```bash
uv run python -m pytest tests/ -q
```

## 两种执行模式

### 模式 A：JobSpec + TargetSpec（推荐）

这是完整的两阶段流程：JobSpec 描述“要测什么”，TargetSpec 描述“去哪台机器测”。候选配置默认放在 `outputs/<job_id>/configs.jsonl`，也可以把第三个参数显式指定为其他路径。

```bash
./run_executor.sh <job.json> <target.json> [configs.jsonl] [results_dir]
```

### 模式 B：单文件手写配置

适合不调用 AI、直接复现一组已知配置。单文件使用 `.json` 后缀，包含 `_meta` 和 `candidates`：

```bash
./run_executor.sh input/configs/<config>.json
```

这种模式仍然经过相同的远程启动、健康检查、压测、SLA 过滤和排名流程。

## 输入文件

### JobSpec：`input/jobs/<job>.json`

```json
{
  "job_id": "qwen35-27b-fp8_pro5000_8x72g_input8k-output1k-cand16",
  "engine": "sglang",
  "gpu_model": "G24_pro5000",
  "gpu_count": 8,
  "gpu_memory_gb": 72,
  "model": "M41_qwen3-5-27b",
  "image": "I03_sglang-v0.5.16",
  "workload": "W10_input-8k-output-1k",
  "benchmark_method": "sglang-bench-serving",
  "sla": {
    "max_avg_ttft_ms": 2000,
    "max_avg_tpot_ms": 80,
    "min_success_rate": 0.99
  },
  "search": {
    "max_candidates": 16,
    "max_runtime_minutes": 180,
    "baseline": {
      "tp_size": 1,
      "attention_backend": "flashinfer",
      "mem_fraction_static": 0.80
    },
    "baseline_threshold_pct": 5
  }
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `gpu_model` | `catalogs/gpu.yaml` 中的 GPU 卡型 ID |
| `gpu_count` / `gpu_memory_gb` | 目标 GPU 数量和单卡显存 |
| `model` | `catalogs/models.yaml` 中的模型 ID |
| `image` | `catalogs/sglang-images.yaml` 中的 SGLang 镜像 ID |
| `workload` | `catalogs/workloads.yaml` 中的输入/输出长度和采样配置 |
| `sla` | 平均 TTFT、平均 TPOT、最低成功率约束 |
| `search.max_candidates` | 非 baseline 候选上限 |
| `search.baseline` | 可选的用户基线；存在时不占 `max_candidates` 名额 |
| `search.baseline_threshold_pct` | 相对 baseline 的 goodput 门槛，用于比较标记 |

填写 `search.baseline` 后，基线会作为第一条候选用于复现和对比；其余候选才消耗 `max_candidates` 名额。无论 AI 是否忘记填写，执行器都会把所有实际测试的候选统一置为 radix off。

### TargetSpec：`input/targets/<target>.json`

```json
{
  "gpu_model": "G24_pro5000",
  "gpu_count": 8,
  "gpu_memory_gb": 72,
  "ssh_target": "ubuntu@100.67.146.43",
  "ssh_password": "",
  "model_host_dir": "/data/autotune/models/Qwen3.5-27B",
  "model_container_path": "/data/autotune/models/Qwen3.5-27B",
  "image_ref": "hai-beijing.tencentcloudcr.com/ai/sglang:v0.5.16-cu129",
  "port": 30000,
  "remote_outputs_dir": ""
}
```

`image_ref` 是目标 GPU 机上运行的 SGLang 服务镜像，不是本项目的 tuner 镜像。优先使用 SSH key；如确实需要密码，可填写 `ssh_password`，但不要把含密码的文件提交到 Git。

### 单文件配置：`input/configs/<config>.json`

```json
{
  "_meta": {
    "job_id": "qwen35-manual-test",
    "gpu_model": "G24_pro5000",
    "gpu_count": 8,
    "gpu_memory_gb": 72,
    "workload": "W10_input-8k-output-1k",
    "benchmark_method": "sglang-bench-serving",
    "sla": {
      "max_avg_ttft_ms": 2000,
      "max_avg_tpot_ms": 80,
      "min_success_rate": 0.99
    },
    "ssh_target": "ubuntu@100.67.146.43",
    "model_host_dir": "/data/autotune/models/Qwen3.5-27B",
    "model_container_path": "/data/autotune/models/Qwen3.5-27B",
    "image_ref": "hai-beijing.tencentcloudcr.com/ai/sglang:v0.5.16-cu129",
    "port": 30000
  },
  "candidates": [
    {
      "id": "c001",
      "params": {
        "tp_size": 4,
        "attention_backend": "flashinfer",
        "mem_fraction_static": 0.88,
        "trust_remote_code": true
      },
      "reasons": ["TP4 + flashinfer + 显存比例 0.88"]
    }
  ]
}
```

`params` 中的下划线会自动转换为 SGLang 的连字符参数，`true` 会变成裸 flag；`--model-path`、`--host`、`--port` 和强制的 `--disable-radix-cache` 由执行器补齐或校正。

## 可搜索的参数轴

具体候选由模型、GPU、镜像和知识库共同决定，不是把所有参数盲目做笛卡尔积。常见参数包括：

- 并行：`tp_size`、`pp_size`、MoE 的 `ep_size`。
- Attention：`attention_backend`、`page_size`。
- 显存和调度：`mem_fraction_static`、`chunked_prefill_size`、`max_running_requests`、`schedule_conservativeness`、`kv_cache_dtype`。
- 投机解码：`speculative_algorithm`、`speculative_num_steps`、`speculative_eagle_topk`、`speculative_num_draft_tokens`、`speculative_draft_model_path`。
- 模型专属：`trust_remote_code`、`reasoning_parser`、`tool_call_parser` 以及模型卡声明的其他 flag。
- 混合 Mamba/GDN：`mamba_radix_cache_strategy` 等字段用于审计，但在本项目公平性口径下最终使用 `no_buffer`/radix off。

## 排名口径和结果文件

执行器只把同时满足以下条件的并发档位用于排名：有完成请求、成功率满足 SLA、平均 TTFT/TPOT 满足 SLA，且平均输出长度没有明显截断。

核心归一化指标是：

```text
goodput_per_host = total_throughput × (gpu_count / tp_size)
```

这样 TP2 的多实例总吞吐可以和 TP8 的单实例吞吐公平比较。每个候选的详细结果会写入：

```text
outputs/<job_id>/
├── configs.jsonl                  # 候选配置
├── best_config.md                 # gen_report.sh 生成
└── results/
    ├── task_status.json           # 完成/失败/中断状态
    ├── candidate_results.jsonl    # 每个候选的汇总和所有并发点
    ├── ranking.json               # 最终或 provisional 排名
    └── <candidate-id>/            # server、bench、启动尝试日志
```

## 目录结构

```text
llm-infer-tuner/
├── gen_configs.sh                 # L1：AI 生成候选
├── run_executor.sh                # L2：远程压测和排名
├── gen_report.sh                  # 从已有结果生成 Markdown 报告
├── Dockerfile                     # 包含 Python/uv、Node、tclaude、claude 的镜像
├── docker/entrypoint.sh           # 镜像入口，默认进入 /app
├── input/
│   ├── jobs/                      # JobSpec
│   ├── targets/                   # TargetSpec
│   └── configs/                   # 单文件手写配置
├── catalogs/
│   ├── gpu.yaml                   # GPU 能力和 NUMA/互联事实
│   ├── models.yaml                # 架构、量化、MoE、parser、MTP 信息
│   ├── workloads.yaml             # 输入/输出长度和压测梯度
│   └── sglang-images.yaml         # 镜像、CUDA、后端和有效 flag
├── .claude/skills/
│   └── sglang-server-config-gen/
│       ├── SKILL.md               # 配置生成流程和输出契约
│       └── knowledge.md            # 调优经验、硬约束和排除项
├── runners/                       # SSH、Docker、压测、搜索、排名和报告编排
├── schemas/                       # JobSpec 等 Pydantic 契约
├── scripts/                       # 目录同步和报告脚本
├── outputs/                       # 本地运行产物
└── claude-raw-outputs/            # AI 原始返回
```

## 自动同步目录和参数知识

```bash
./update.sh
# 等价于
./scripts/run_daily_sync.sh
```

同步内容包括：

1. 从 SGLang tag 的源码中提取 `launch_server` 有效参数和 attention 后端，更新 `catalogs/sglang-images.yaml`。
2. 从 SGLang cookbook 和 HuggingFace 配置同步模型架构、量化、MoE、默认 parser、MTP 参数和权重估算，更新 `catalogs/models.yaml`。
3. 维护标准输入/输出 workload 和 GPU 目录，生成 `reports/` 下的变化摘要供人工审核。

有变化时检查 diff 后再提交和推送；如果需要定时同步，可以用 cron：

```cron
0 9 * * * cd /path/to/llm-infer-tuner && ./update.sh >> logs/sync.log 2>&1
```

## 镜像维护者说明

重新构建并运行本地镜像：

```bash
docker build -t llm-infer-tuner:dev .
docker run --rm -it llm-infer-tuner:dev bash
```

发布镜像时需要保留项目来源标签，否则 GHCR 只会把它识别成独立的用户镜像包：

```dockerfile
LABEL org.opencontainers.image.source="https://github.com/wuhangxian/llm-infer-tuner"
```

镜像不包含模型权重、SSH 私钥或任何 AI 登录 token。发布新镜像后，使用者仍需在运行时挂载自己的登录态和 SSH key。
