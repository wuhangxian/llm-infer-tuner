# llm-infer-tuner

给定模型、GPU、SGLang 镜像、输入/输出长度和 SLA，自动生成候选启动参数，在真实 GPU 机器上压测并找出最优配置，最后生成可复现的部署报告。

核心思路是：**AI 根据可追溯规则决定测试哪些候选，执行器负责真实启动、压测和排名。** 规则用于指导和解释，不会在生成阶段把实验性候选提前拦掉；能否启动和性能好坏以目标机器实测为准。

## 工作流

```text
job.json
   │
   ├─ ./gen_configs.sh
   │    tclaude/claude 读取规则库和 catalogs
   │    → outputs/<job_id>/configs.jsonl
   │
   ├─ ./run_executor.sh
   │    SSH 到 GPU 机 → Docker 启动 SGLang → 预热 → 并发搜索
   │    → SLA 过滤 → goodput 排名
   │    → outputs/<job_id>/results/
   │
   └─ ./gen_report.sh
        → outputs/<job_id>/best_config.md
```

| 命令 | 作用 | AI 登录 | 目标 GPU 机 |
| --- | --- | --- | --- |
| `./gen_configs.sh` | 生成候选配置 | 需要 | 不需要 |
| `./run_executor.sh` | 远程压测和排名 | 不需要 | 需要 |
| `./gen_report.sh` | 从已有结果生成报告 | 不需要 | 不需要 |

## 主要功能

- 支持腾讯内网 `tclaude` 和公开 `claude` CLI，可用 `--agent`、`--model` 切换。
- AI 读取 `SKILL.md`、`knowledge.md`、主题规则 YAML 和 GPU/模型/负载/镜像 catalogs，生成带理由的结构化候选。
- 搜索 TP/EP、attention 后端、显存比例、KV cache、chunked prefill、调度策略和投机解码等参数。
- 把 GPU、CUDA、模型结构和 SGLang 镜像事实写入候选理由；实验性组合也可以交给执行器实测，不在生成阶段硬拦。
- 通过 SSH 在目标机启动独立 Docker 容器，自动检查硬件、模型、镜像、端口和 `/health`。
- 支持启动超时与重试、NUMA 感知 GPU 分配、多候选并行和整机满载实测。
- 先预热，再按两轮策略搜索满足 SLA 的最大并发度。
- 检查成功率、TTFT、TPOT 和输出完整性，避免截断结果获得虚假高吞吐。
- 按 `goodput_per_host` 排名，并保留每个候选的参数、日志、并发点和失败原因。
- 从已有结果生成 `best_config.md`，不会重新启动服务或重复压测。
- 自动同步 SGLang 参数、attention 后端、模型架构、量化、MoE、MTP 和 workload 知识。

### 公平性约束

- 所有候选最终都会强制带且只带一个 `--disable-radix-cache`，避免固定请求命中前缀缓存后污染对比。
- Mamba/GDN 模型只使用与 radix off 兼容的 `no_buffer`，不会生成依赖 radix cache 的 `extra_buffer`。
- 投机解码由“模型能力/参数”与“镜像内置算法”取交集决定：模型卡优先读取 `speculative_options`，兼容旧的 `mtp_params`；镜像卡读取 `speculative_algorithms`。`NONE` 始终作为对照，EAGLE 不是唯一方案；需要外挂 draft 模型的算法必须提供 `speculative_draft_model_path`。
- 不会根据 workload 人为写紧 `--context-length`，服务上下文长度由模型和 SGLang 配置决定。

## 最简单的 Docker 用法

推荐方式不是每执行一步都重新 `docker run`，而是：**容器创建一次，以后进入容器像普通目录一样直接运行脚本。**

镜像地址：

```text
ghcr.io/wuhangxian/llm-infer-tuner:0903-dorianwu
```

镜像包含 Python/uv、Node.js、`tclaude`、公开 `claude`、Git、nano、jq、SSH 和 sshpass。镜像内的 `/app/README.md` 就是本说明。

### 1. 第一次只做一次：拉镜像并创建容器

```bash
docker login ghcr.io
docker pull ghcr.io/wuhangxian/llm-infer-tuner:0903-dorianwu

mkdir -p ~/llm-tuner-work/{input/jobs,input/targets,input/configs,outputs,claude-raw-outputs}
mkdir -p "$HOME/.tclaude" "$HOME/.claude"
cd ~/llm-tuner-work

docker run -dit \
  --name llm-infer-tuner \
  --restart unless-stopped \
  -v "$PWD/input:/app/input" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$PWD/claude-raw-outputs:/app/claude-raw-outputs" \
  -v "$HOME/.tclaude:/home/runner/.tclaude" \
  -v "$HOME/.claude:/home/runner/.claude" \
  -v "$HOME/.ssh:/home/runner/.ssh:ro" \
  ghcr.io/wuhangxian/llm-infer-tuner:0903-dorianwu \
  bash
```

这条命令虽然长，但只执行一次。它把输入、输出、AI 登录态和 SSH key 固定挂载好；以后不需要再写这些 `-v` 参数。

如果宿主机之前用 root 创建过输出目录，先修复一次权限：

```bash
sudo chown -R 1000:1000 ~/llm-tuner-work/outputs ~/llm-tuner-work/claude-raw-outputs
```

### 2. 进入容器

```bash
docker exec -it llm-infer-tuner bash
```

进入后默认就在 `/app`。现在可以像普通项目一样工作：

```bash
nano input/jobs/my-job.json
nano input/targets/my-target.json
```

### 3. 首次登录一次

腾讯内网账号：

```bash
tclaude login
```

公开 Claude 账号：

```bash
claude login
```

终端如果没有自动打开浏览器，复制它打印的 URL 到浏览器完成登录。登录态已经挂载到宿主机，停止或重启容器后仍然保留。

### 4. 以后每一步都是一行

生成候选：

```bash
./gen_configs.sh input/jobs/my-job.json
```

使用公开 Claude：

```bash
./gen_configs.sh --agent claude input/jobs/my-job.json
```

指定模型：

```bash
./gen_configs.sh --model claude-opus-4-8 input/jobs/my-job.json
```

远程压测和排名：

```bash
./run_executor.sh input/jobs/my-job.json input/targets/my-target.json
```

生成报告：

```bash
./gen_report.sh outputs/<job_id>
```

就这三条主命令。结果会同时出现在容器 `/app/outputs` 和宿主机 `~/llm-tuner-work/outputs`。

### 5. 容器停了以后

```bash
docker start llm-infer-tuner
docker exec -it llm-infer-tuner bash
```

输入、输出和登录态都在宿主机，不会因为容器停止而丢失。

## 输入文件

容器内可以直接用 `nano` 编辑；也可以在宿主机的 `~/llm-tuner-work/input` 中编辑，两个位置看到的是同一批文件。

### JobSpec：`input/jobs/my-job.json`

JobSpec 描述“要测试什么”：

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

| 字段 | 含义 |
| --- | --- |
| `gpu_model` | `catalogs/gpu.yaml` 中的 GPU ID |
| `gpu_count` / `gpu_memory_gb` | GPU 数量和单卡显存 |
| `model` | `catalogs/models.yaml` 中的模型 ID |
| `image` | `catalogs/sglang-images.yaml` 中的 SGLang 镜像 ID |
| `workload` | `catalogs/workloads.yaml` 中的输入/输出长度和压测配置 |
| `sla` | 平均 TTFT、平均 TPOT 和最低成功率 |
| `search.max_candidates` | 非 baseline 候选上限 |
| `search.baseline` | 可选用户基线，不占 `max_candidates` 名额 |
| `search.baseline_threshold_pct` | 相对 baseline 的 goodput 比较门槛 |

### TargetSpec：`input/targets/my-target.json`

TargetSpec 描述“去哪里测试”：

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

`image_ref` 是目标 GPU 机实际运行的 SGLang 服务镜像，不是本项目的 tuner 镜像。优先使用 SSH key；如果填写 `ssh_password`，不要把带密码的文件提交到 Git。

### 不使用 AI：单文件模式

如果已有参数，可以把目标信息和候选写在同一个 `.json` 文件中，直接运行：

```bash
./run_executor.sh input/configs/my-config.json
```

文件结构：

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

`params` 的下划线会自动转换为 SGLang 的连字符参数，布尔 `true` 会变成裸 flag；模型路径、host、port 和强制的 radix off 由执行器补齐或校正。

## 可扩展规则库

`knowledge.md` 现在只是入口索引，具体规则按主题拆开，便于一条一条追加：

```text
.claude/skills/sglang-server-config-gen/
├── knowledge.md                 # 读取顺序和规则入口
└── references/rules/
    ├── attention.yaml           # attention 后端
    ├── parallelism.yaml         # TP/PP/EP
    ├── memory.yaml              # 显存、KV、混合架构
    ├── speculative.yaml         # 投机解码
    ├── scheduling.yaml          # 调度和搜索轴
    └── fairness.yaml            # 公平性和报告口径
```

新增经验时，只需在对应 YAML 的 `rules` 数组末尾追加一条，写清 `id`、适用条件、指导意见和证据；不需要修改生成器代码。新增主题也会被自动发现。提交前可运行：

```bash
uv run python scripts/validate_knowledge.py
```

这个校验只检查规则文件格式、重复 ID 和证据完整性，不检查或拦截 AI 生成的候选命令。

## 常用高级开关

这些都是可选项，普通用户可以跳过。

```bash
# AI 单次超时 900 秒，失败不重试
GEN_TIMEOUT_SECONDS=900 GEN_MAX_RETRIES=0 \
  ./gen_configs.sh input/jobs/my-job.json

# 第二轮按 floor(gpu_count / tp_size) 启动多实例做整机真实吞吐
FILL_HOST=1 ./run_executor.sh input/jobs/my-job.json input/targets/my-target.json

# 调整服务启动无进展超时、硬超时和尝试次数
STARTUP_STALL_TIMEOUT_SECONDS=600 \
STARTUP_HARD_TIMEOUT_SECONDS=1800 \
STARTUP_MAX_ATTEMPTS=3 \
  ./run_executor.sh input/jobs/my-job.json input/targets/my-target.json
```

## 排名和结果

只有同时满足以下条件的并发点才参与排名：

- 至少有一个请求完成。
- 成功率满足 SLA。
- 平均 TTFT 和平均 TPOT 满足 SLA。
- 平均输出长度达到目标长度的 90%，没有明显截断。

默认归一化指标：

```text
goodput_per_host = total_throughput × (gpu_count / tp_size)
```

输出目录：

```text
outputs/<job_id>/
├── configs.jsonl
├── best_config.md
└── results/
    ├── task_status.json
    ├── candidate_results.jsonl
    ├── ranking.json
    └── <candidate-id>/            # 服务日志、压测日志和原始结果
```

## 常见问题

- `outputs` permission denied：执行 `sudo chown -R 1000:1000 ~/llm-tuner-work/outputs ~/llm-tuner-work/claude-raw-outputs`。
- Docker socket permission denied：让宿主机用户获得 Docker 权限并重新登录；这不是容器内脚本错误。
- `tclaude` 每次要求登录：确认创建容器时挂载了 `$HOME/.tclaude:/home/runner/.tclaude`。
- 有 AI 原始日志但没有 `configs.jsonl`：先检查输出目录写权限；正式文件只会在结构化结果解析成功后原子写入。
- 远程 `/health` 不通过：查看 `outputs/<job_id>/results/<candidate-id>/server*.log`，重点检查模型路径、SGLang 镜像、TP/EP、显存和模型专属参数。
- 本地没有 GPU：仍可生成配置和报告；压测必须能 SSH 到有 GPU、模型权重和 SGLang 镜像的目标机。
- 目标容器会被清理：执行器按独占整机设计，会清理目标 GPU 机上的容器、GPU 进程和测试端口，请只在专用压测机器上运行。

## 使用源码开发

只有需要修改代码或知识库时才需要下载项目：

```bash
git clone -b main https://github.com/wuhangxian/llm-infer-tuner.git
cd llm-infer-tuner
uv sync

uv run python -m pytest tests/ -q
```

自动同步 SGLang 和模型目录：

```bash
./update.sh
```

主要目录：

```text
catalogs/                         GPU、模型、workload 和 SGLang 镜像事实
.claude/skills/                   AI 配置生成流程和调优知识
runners/                          SSH、容器、压测、并发搜索和排名
schemas/                          JobSpec 等数据契约
scripts/                          catalog 同步和报告生成
input/                            job、target 和手写配置
outputs/                          配置、排名、日志和报告
```

GHCR 镜像通过下面的 OCI 标签关联到本项目：

```dockerfile
LABEL org.opencontainers.image.source="https://github.com/wuhangxian/llm-infer-tuner"
```

镜像不包含模型权重、SSH 私钥或 AI 登录 token；这些信息始终由使用者在运行时提供。
