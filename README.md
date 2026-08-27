# llm-infer-tuner

大模型推理参数寻优 Agent — 在指定模型 / 卡型 / 工作负载 / SLA 下,自动生成一批 SGLang 启动配置候选,拿到目标机器上真机压测,在 SLA 约束内按 goodput(有效吞吐)排名,输出可复现的最佳部署配置。

核心理念:**AI 定考什么题,确定性代码当阅卷老师。**

---

## 快速开始

### 0. 前置

- Python 3.11+ + [uv](https://github.com/astral-sh/uv)
- `jq`、`claude` CLI
- 目标 GPU 机器:SSH 可达、已装 docker、已备好模型权重和 sglang 镜像

```bash
uv sync
```

### 1. 配置凭证

```bash
cp .env.example .env
vim .env           # 填 ANTHROPIC_BASE_URL 和 ANTHROPIC_AUTH_TOKEN
chmod 600 .env
```

`.env` 已被 `.gitignore` 排除。脚本启动时自动加载。

---

## 两种使用方式

### 方式 A: AI 生成 + 自动压测(两步)

**第一步:AI 生成候选配置**

```bash
./gen_configs.sh input/jobs/<job>.json
```

AI 读 knowledge.md + catalogs(gpu/models/workloads/images),推导 TP/attention/mem-fraction/投机解码等参数,生成候选配置到 `outputs/<job_id>/configs.json`。

**第二步:远程压测 + 排名**

```bash
./run_executor.sh input/jobs/<job>.json input/targets/<target>.json
```

SSH 到远程机器,起容器 → 起服务 → 自适应并发搜索 → 压测 → 排名 → 输出 `outputs/<job_id>/results/ranking.json`。

### 方式 B: 手写配置 + 直接压测(一步)

写一个 JSON 文件(含 _meta + candidates),直接跑:

```bash
./run_executor.sh input/configs/<config>.json
```

JSON 格式:

```json
{
  "_meta": {
    "job_id": "qwen38-27b-test",
    "gpu_model": "G24_pro5000",
    "gpu_count": 8,
    "gpu_memory_gb": 72,
    "workload": "W04_input-4k-output-1k",
    "benchmark_method": "sglang-bench-serving",
    "sla": {
      "max_avg_ttft_ms": 2000,
      "max_avg_tpot_ms": 80,
      "min_success_rate": 0.99
    },
    "ssh_target": "ubuntu@122.51.115.16",
    "ssh_password": "",
    "model_host_dir": "/data/models/your-model/",
    "model_container_path": "/data/models/your-model/",
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
        "trust_remote_code": true,
        "disable_radix_cache": true
      },
      "reasons": ["TP4+flashinfer+mf0.88"]
    }
  ]
}
```

**只需填 `params`,执行器自动拼成完整启动命令。** 字段名 `tp_size` → flag `--tp-size`(下划线转连字符),布尔 `true` → 裸 flag,`false` → 不写。`--model-path`、`--host`、`--port` 自动补全。

可填的参数包括但不限于:
- 基础:`tp_size`/`pp_size`/`ep_size`/`attention_backend`/`mem_fraction_static`/`kv_cache_dtype`/`chunked_prefill_size`/`schedule_conservativeness`/`page_size`/`disable_radix_cache`
- 投机解码:`speculative_algorithm`/`speculative_num_steps`/`speculative_eagle_topk`/`speculative_num_draft_tokens`/`speculative_draft_model_path`
- Mamba/GDN:`mamba_radix_cache_strategy`/`mamba_full_memory_ratio`/`mamba_ssm_dtype`
- 模型专属:`trust_remote_code`/`reasoning_parser`/`tool_call_parser`
- 任何 SGLang launch_server 认的参数

---

## 功能特性

### 基线配置 (Baseline)

在 job.json 的 `search` 里加 `baseline` 字段(可选):

```json
"search": {
  "max_candidates": 16,
  "max_runtime_minutes": 180,
  "baseline": {
    "tp_size": 4,
    "attention_backend": "flashinfer",
    "mem_fraction_static": 0.88
  },
  "baseline_threshold_pct": 5
}
```

- `baseline` 里填任意参数,AI 补全其余,生成基线候选(id="baseline")放在第一条
- 基线不算在 max_candidates 名额里(总数 = N+1)
- `baseline_threshold_pct: 5` → 只保留比基线 goodput 高 5% 以上的候选
- 不写 baseline 时行为不变

### NUMA 感知 GPU 分配

执行器自动读 `nvidia-smi topo -m` 检测 NUMA 分组,同一个候选的 GPU 必须在同一 NUMA 节点内,避免跨 NUMA 通信导致性能下降。

例如 8 卡双 NUMA(GPU 0-3 = NUMA 0,GPU 4-7 = NUMA 1):
- TP1 × 4 → GPU 0,1,2,3(NUMA 0)
- TP1 × 4 → GPU 4,5,6,7(NUMA 1)
- TP2 × 2 → GPU 0,1 + GPU 2,3(NUMA 0)
- TP4 × 1 → GPU 0,1,2,3(整个 NUMA 0)

### 并行压测

多个候选可以同时跑在不同 GPU 上(每个候选独立容器,独立端口),不浪费空闲卡。

### 关闭 Radix Cache

默认所有候选 pin `--disable-radix-cache`,测纯推理性能。投机解码 + `no_buffer` + `--disable-radix-cache` 是合法组合。

### Goodput 归一化排名

排名按 `goodput_per_host = total_throughput × (gpu_count / tp_size)`,不是单实例吞吐。TP2 跑 300 tok/s × 4 实例 = 1200 > TP8 跑 1000 tok/s × 1 实例 = 1000。

### SSH 密码支持

target.json 里可选填 `ssh_password`,有密码用 sshpass,留空走 key 免密。

### 服务启动失败分析

服务起不来时自动提取 server.log 里的错误信息(ValueError/AssertionError 等)打印到 stderr。

### Preflight 校验

启动前检查:SSH 连通性、模型目录存在、镜像可用(没有就自动 pull)、清理同名旧容器。

---

## 自动同步(每日更新)

### 一键更新

```bash
./update.sh
```

等价于:
```bash
./scripts/run_daily_sync.sh
```

### 同步内容

1. **SGLang 参数同步**(`scripts/sync_sglang_params.py`)
   - 自动扫描 SGLang 所有 git tag(≥ v0.5.13)
   - 发现新版本 → clone → AST 解析 server_args.py → 更新 `catalogs/sglang-images.yaml` 的 `valid_flags` 和 `attention_backends`
   - 已有版本检查参数是否有变化
   - 生成 `reports/constraints_*.md` 约束报告供人工审核

2. **模型信息同步**(`scripts/sync_hf_models.py`)
   - 扫描 SGLang cookbook(2026-05-01 之后的新模型)发现新模型
   - 从 HuggingFace API 拉 config.json → 提取架构/量化/MoE 信息
   - 从 cookbook mdx 提取 default_flags(parser 名)和 mtp_params(投机解码参数)
   - 从 HuggingFace API 估算权重大小
   - 已有模型检查 config.json 是否有变化并自动补全缺失字段
   - 写入 `catalogs/models.yaml`,自动更新 version/updated/total

3. **SGLang 仓库自动更新**
   - 同步前自动 `git pull` SGLang 仓库到最新,确保 cookbook 不过期

### 设置定时任务

```bash
crontab -e
# 每天早上 9 点自动跑
0 9 * * * cd /data/home/dorianwu/aaawhx-study/llm-infer-tuner && ./update.sh >> logs/sync.log 2>&1
```

有变化时打印 diff 摘要,你手动 `git push`。

---

## 目录结构

```
llm-infer-tuner/
├── gen_configs.sh              # 第一步入口:AI 生成候选配置
├── run_executor.sh             # 第二步入口:远程压测 + 排名(支持单文件模式)
├── update.sh                   # 一键更新脚本(同步 SGLang 参数 + 模型信息)
├── .env.example                # 凭证模板
├── .env                        # 你的凭证(gitignore)
│
├── input/
│   ├── jobs/                   # AI 模式输入(job.json + target.json)
│   ├── targets/
│   └── configs/                # 手写模式输入(单文件 JSON,含 _meta + candidates)
│
├── catalogs/                   # 引擎无关的共享事实
│   ├── gpu.yaml                #   显卡信息(G01-G32, 含 sm_major/nvlink)
│   ├── models.yaml             #   模型信息(M01-M225, 含架构/量化/MoE/parser/mtp)
│   ├── workloads.yaml          #   负载场景(W01-W10, 输入/输出长度)
│   └── sglang-images.yaml      #   SGLang 镜像信息(I01-I03, 含 attention_backends/valid_flags)
│
├── .claude/skills/             # AI 读的知识库
│   └── sglang-server-config-gen/
│       ├── SKILL.md            #   流程入口
│       ├── knowledge.md        #   全部调优经验(§0-§12)
│       └── (images.yaml 已移到 catalogs/)
│
├── runners/                    # 第二步执行器(全确定性,无 AI)
│   ├── executor.py             #   主编排
│   ├── remote.py               #   SSH + sshpass
│   ├── container.py            #   docker 生命周期
│   ├── readiness.py            #   /health 轮询
│   ├── bench_runner.py         #   压测命令生成(调 claude)
│   ├── concurrency_search.py   #   自适应并发搜索
│   ├── metrics.py              #   压测结果解析
│   └── ranker.py               #   goodput 排名
│
├── planner/
│   └── claude_code_client.py   # claude CLI 封装
│
├── schemas/
│   └── job_spec.py             # JobSpec 校验(含 baseline 字段)
│
├── scripts/                    # 自动同步脚本
│   ├── sync_sglang_params.py   #   SGLang 源码参数提取(AST)
│   ├── sync_hf_models.py       #   HuggingFace 模型信息同步
│   └── run_daily_sync.sh       #   定时同步入口
│
├── references/
│   └── benchmark_methods/      # 压测方法契约
│
├── outputs/                    # 产物(gitignore)
├── claude-raw-outputs/         # claude 原始返回(gitignore)
├── reports/                    # 同步报告(gitignore)
└── logs/                       # 日志(gitignore)
```

---

## 输入文件说明

### job.json(AI 模式)

```json
{
  "job_id": "qwen38-27b_pro5000_8x72g_input-4k-output-1k_cand16",
  "engine": "sglang",
  "gpu_model": "G24_pro5000",
  "gpu_count": 8,
  "gpu_memory_gb": 72,
  "model": "M55_qwen3-8-27b-fp8",
  "image": "I03_sglang-v0.5.16",
  "workload": "W04_input-4k-output-1k",
  "benchmark_method": "sglang-bench-serving",
  "sla": {
    "max_avg_ttft_ms": 2000,
    "max_avg_tpot_ms": 80,
    "min_success_rate": 0.99
  },
  "search": {
    "max_candidates": 16,
    "max_runtime_minutes": 180
  }
}
```

### target.json

```json
{
  "gpu_model": "G24_pro5000",
  "gpu_count": 8,
  "gpu_memory_gb": 72,
  "ssh_target": "ubuntu@122.51.115.16",
  "ssh_password": "",
  "model_host_dir": "/data/models/Qwen3.8-27B-FP8/",
  "model_container_path": "/data/models/Qwen3.8-27B-FP8/",
  "image_ref": "hai-beijing.tencentcloudcr.com/ai/sglang:v0.5.16-cu129",
  "port": 30000
}
```

### 手写配置 JSON(单文件模式)

文件名格式:`{模型}_{显卡}_{数量}x{规格}_{日期}_{序号}.json`

示例:`qwen38-27b_pro5000_8x72g_20260826_01.json`

见上方「方式 B」的格式说明。

---

## 测试

```bash
uv run python -m pytest tests/ -q
```
