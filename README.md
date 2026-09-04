# llm-infer-tuner

给定模型、GPU、SGLang 镜像、输入/输出长度和 SLA，自动生成候选启动参数，在真实 GPU 机器上压测并找出最优配置，最后生成可复现的部署报告。

核心思路是：**AI 决定测试哪些候选，确定性执行器负责校验、实测和排名。**

## 工作流

```text
job.json
   │
   ├─ ./gen_configs.sh
   │    tclaude/claude 读取知识库和 catalogs
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

## 先照着跑：容器快速开始

不需要下载源码，也不需要在宿主机安装 Python 或 Node.js。下面的命令只在宿主机执行一次，创建好容器后，后续操作都在容器里完成。

公开镜像地址：`ghcr.nju.edu.cn/wuhangxian/llm-infer-tuner:0903-dorianwu`。镜像内已经包含运行环境、两个 AI CLI、脚本、两文件版 skill，以及 `input/` 下的 job、target、config 示例；不包含模型权重、SSH 私钥或登录 token。

```bash
# 1. 准备工作目录并拉取公开镜像
mkdir -p ~/llm-tuner-work/{input,outputs,claude-raw-outputs}
mkdir -p "$HOME/.tclaude" "$HOME/.claude"
cd ~/llm-tuner-work
docker pull ghcr.nju.edu.cn/wuhangxian/llm-infer-tuner:0903-dorianwu

# 2. 创建一个长期使用的容器（只需执行一次）
docker run -dit --name llm-infer-tuner --restart unless-stopped \
  -v "$PWD/input:/app/input" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$PWD/claude-raw-outputs:/app/claude-raw-outputs" \
  -v "$HOME/.tclaude:/home/runner/.tclaude" \
  -v "$HOME/.claude:/home/runner/.claude" \
  -v "$HOME/.ssh:/home/runner/.ssh:ro" \
  ghcr.nju.edu.cn/wuhangxian/llm-infer-tuner:0903-dorianwu bash

# 3. 进入容器
docker exec -it llm-infer-tuner bash
```

如果宿主机之前用 root 创建过输出目录，执行一次：

```bash
sudo chown -R 1000:1000 ~/llm-tuner-work/outputs ~/llm-tuner-work/claude-raw-outputs
```

进入容器后，镜像会自动把这 6 个示例补到 `input/`（不会覆盖同名文件）：

```text
input/jobs/0_EXAMPLE.json
input/jobs/qwen35-27-input8k-output1k-cand16.json
input/targets/0_EXAMPLE.json
input/targets/qwen35-27-input8k-output1k-cand16-100-67-146-43.json
input/configs/0_EXAMPLE.json
input/configs/qwen35-27-input8k-output1k-cookbook.json
```

然后只需要按顺序执行下面几行：

```bash
tclaude login                                      # tclaude 首次执行一次；已有登录态可跳过
# 如果使用公开 Claude，则改用：claude login
./gen_configs.sh input/jobs/qwen35-27-input8k-output1k-cand16.json
./run_executor.sh input/jobs/qwen35-27-input8k-output1k-cand16.json input/targets/qwen35-27-input8k-output1k-cand16-100-67-146-43.json
./gen_report.sh outputs/qwen35-27b-fp8_pro5000_8x72g_input4k-output1k-cand16
```

其中 `gen_configs.sh` 生成候选，`run_executor.sh` 连接真实 GPU 机器压测，`gen_report.sh` 生成报告。若使用公开 Claude，把 `tclaude login` 改成 `claude login`，并把生成命令改成 `./gen_configs.sh --agent claude input/jobs/qwen35-27-input8k-output1k-cand16.json`。若只想测试已有配置，可先复制并修改 `input/configs/0_EXAMPLE.json` 中的机器、模型和镜像信息，再运行 `./run_executor.sh input/configs/your-config.json`。报告在 `outputs/<job_id>/best_config.md`，宿主机对应目录也能直接看到。

容器停止后再次使用：

```bash
docker start llm-infer-tuner
docker exec -it llm-infer-tuner bash
```

如果镜像标签更新过，已有容器不会自动更新，需要先 `docker pull`，再删除并按上面的第 2 步重新创建容器。

| 命令 | 作用 | AI 登录 | 目标 GPU 机 |
| --- | --- | --- | --- |
| `./gen_configs.sh` | 生成候选配置 | 需要 | 不需要 |
| `./run_executor.sh` | 远程压测和排名 | 不需要 | 需要 |
| `./gen_report.sh` | 从已有结果生成报告 | 不需要 | 不需要 |

## 主要功能

- 支持腾讯内网 `tclaude` 和公开 `claude` CLI，可用 `--agent`、`--model` 切换。
- AI 读取 `SKILL.md`、`knowledge.md` 和 GPU/模型/负载/镜像 catalogs，生成带理由的结构化候选。
- 搜索 TP/EP、attention 后端、显存比例、KV cache、chunked prefill、调度策略和投机解码等参数。
- 按 GPU、CUDA、模型结构和 SGLang 镜像参数白名单过滤必然无法启动的组合。
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
- 投机解码是否生成由模型卡片的 `mtp_params` 决定。自带 MTP 权重的 EAGLE/NEXTN 类方案可以候选；需要外挂 draft 模型的算法必须提供 `speculative_draft_model_path`。
- 不会根据 workload 人为写紧 `--context-length`，服务上下文长度由模型和 SGLang 配置决定。

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
