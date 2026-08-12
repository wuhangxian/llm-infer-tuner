# HAI 大模型推理最佳参数寻优 Agent

## 1. 项目背景

HAI 团队在不同模型、GPU 卡型和推理场景下，需要反复调整推理服务参数。当前调优过程比较依赖个人经验，存在以下问题：

- 参数调优耗时长，重复劳动多；
- 模型、卡型、SGLang 版本变化后，历史经验不一定仍然适用；
- 经验缺少结构化沉淀，难以复用和追溯；
- 测试范围、失败原因、SLA 判断和最终配置缺少统一规范。

本项目希望构建一个推理参数寻优 Agent，将 HAI 团队的硬件、模型、引擎和实测经验沉淀为知识库，并在指定模型、卡型、工作负载和 SLA 下，自动生成候选参数、执行测试、分析结果，最终输出可复现的最佳部署配置。

## 2. 项目目标

### 2.1 第一版目标

第一版聚焦：

```text
推理引擎：SGLang
硬件平台：NVIDIA GPU
部署形态：单机多卡，优先支持容器环境
典型场景：日常 QA Chat、AI Coding
核心能力：参数范围建议 + 候选测试 + SLA 约束下的最优配置选择
```

第一版不追求覆盖所有模型和所有参数，而是先建立一个真实可运行的闭环：

```text
测试需求
  ↓
知识库检索和事实校验
  ↓
生成参数搜索计划
  ↓
生成候选启动/压测命令
  ↓
批量运行候选
  ↓
采集指标、日志和失败原因
  ↓
SLA 过滤和性能排序
  ↓
输出最佳配置和完整证据
```

当前开发阶段进一步收窄为：先完成 SGLang + NVIDIA 的离线测试范围生成器。固定 JSON JobSpec 作为输入，由 Claude Code CLI 渐进式读取 `specs/` 和 `references/` 并分析生成 SearchPlan，再由本地程序校验并输出候选参数和 server/benchmark 命令；不启动服务、不执行 benchmark。初期直接在宿主机调试，稳定后再封装 LLMOptAgent Agent 容器。详细实施计划见 [`docs/development-plan.md`](docs/development-plan.md)。

当前已有一个 Pro5000 离线验收示例：

Claude Code 的第三方 API 配置可以放在用户级私密文件中，不需要每次手动 `export`：

```bash
mkdir -p ~/.config/llmopt-agent
cp .env.example ~/.config/llmopt-agent/claude.env
chmod 600 ~/.config/llmopt-agent/claude.env
vim ~/.config/llmopt-agent/claude.env
```

`llmopt plan` 会自动读取该文件；也可以显式指定 `--claude-env-file PATH`。Shell 中已经
存在的环境变量优先级更高。不要把真实 token 写入 Git，已暴露的 token 应立即轮换。

```bash
cd /data/LLMOptAgent
uv run llmopt validate-job examples/jobs/qwen36_pro5000_random_v1.json
uv run llmopt plan examples/jobs/qwen36_pro5000_random_v1.json \
  --search-plan tests/fixtures/search_plan_qwen36_pro5000.json \
  --output outputs/qwen36_pro5000_random_v1
uv run llmopt render outputs/qwen36_pro5000_random_v1/search_plan.json \
  --job examples/jobs/qwen36_pro5000_random_v1.json \
  --model-path /data1/model/Qwen3.6-27B-FP8/ \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --output outputs/qwen36_pro5000_random_v1/rendered
```

上述命令只生成 `candidates.jsonl`、`commands.sh` 和报告，不会启动 SGLang 服务或执行压测。
如果已经从目标镜像保存了帮助文本，可以额外进行参数名校验：

```bash
uv run llmopt render outputs/qwen36_pro5000_random_v1/search_plan.json \
  --job examples/jobs/qwen36_pro5000_random_v1.json \
  --model-path /data1/model/Qwen3.6-27B-FP8/ \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --output outputs/qwen36_pro5000_random_v1/rendered \
  --server-help references/sglang/snapshots/<version>/launch_server_help.txt \
  --benchmark-help references/sglang/snapshots/<version>/bench_serving_help.txt
```

### 2.2 长期目标

未来扩展到：

- vLLM 推理引擎；
- 昆仑芯、海光国产硬件平台；
- 多机、多节点、PD 分离、EP、Prefill/Decode 分离等部署形态；
- 基于历史测试结果的经验迁移和自适应搜索；
- 持续运行的参数回归、版本对比和配置推荐服务。

长期目标不要求第一版提前实现。第一版只需要把数据契约和扩展边界定义稳定。

## 3. 设计原则

### 3.1 Agent 负责推理，程序负责确定性执行

Agent 可以根据知识库理解场景并提出搜索计划，但不能直接自由发挥生成一堆未经校验的命令。

推荐分工：

```text
Agent：理解需求、检索知识、提出参数空间和理由
规则引擎：检查硬件/模型/版本/显存/参数兼容性
轻量 Adapter：把通用候选翻译成 SGLang-NVIDIA 命令
执行器：启动服务、压测、采集日志和结果
评估器：执行 SLA 判断、排序、剪枝和报告
```

### 3.2 区分事实、经验和推测

知识库中的内容必须区分：

- `hard_constraint`：硬约束，例如显存不足、参数不存在、后端不支持；
- `official`：来自官方文档或源码；
- `measured`：在明确硬件、模型、镜像和版本下的实测结果；
- `heuristic`：团队经验或启发式规则；
- `judgment`：尚未充分验证的人工判断。

Agent 输出的每个关键搜索轴都应记录理由、证据和置信度，避免把一次实测结论误认为普适规律。

### 3.3 “最优”必须绑定场景和目标函数

不存在脱离工作负载的通用最优参数。最终配置必须明确对应：

- 模型和权重版本；
- GPU 型号、显存、卡数和互联拓扑；
- SGLang 镜像或 commit；
- 输入/输出长度分布；
- 并发或请求速率；
- Cache 命中情况；
- SLA 和主优化目标。

默认目标为：

```text
在满足 TTFT、TPOT、成功率等 SLA 的候选中，最大化 token throughput。
```

如果用户更关心 request throughput、成本、显存余量或尾延迟，必须在需求中显式指定。

## 4. 总体架构

```text
┌──────────────────────┐
│  JobSpec 测试需求     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Specs + References    │  硬件、模型、workload、引擎参数和调优经验
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Search Planner         │  生成参数轴、约束、预算和候选范围
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ SGLangNvidiaAdapter    │  兼容性检查、命令编译、日志解析
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Executor              │  启动、健康检查、压测、清理、采集 artifact
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Evaluator             │  指标归一化、SLA、排序、剪枝、报告
└──────────────────────┘
```

### 4.1 轻量 Adapter

第一版不实现复杂的插件系统，也不提前拆成多个 `EngineAdapter` 和 `HardwareAdapter`。只实现一个轻量模块：

```text
SGLangNvidiaAdapter
```

它是一个代码边界，而不是独立服务，主要负责：

1. 将通用候选参数翻译为 SGLang CLI 参数；
2. 结合目标镜像的 `--help` 检查参数是否真实存在；
3. 生成 SGLang 启动命令和 benchmark 命令；
4. 根据 NVIDIA GPU、CUDA、NCCL 和显存信息做兼容性校验；
5. 解析 SGLang 启动日志中的显存、KV Cache 和并发容量信息；
6. 将启动失败、OOM、CUDA/NCCL 错误转换为统一失败类型。

它不负责搜索算法、知识库管理、任务调度和最终排名。

未来增加 vLLM 时，再增加独立模块：

```text
adapters/
├── sglang_nvidia.py
├── vllm_nvidia.py
├── sglang_kunlunxin.py
├── sglang_hygon.py
├── vllm_kunlunxin.py
└── vllm_hygon.py
```

国产卡支持初期也采用同样的轻量方式，分别实现昆仑芯和海光的引擎/硬件组合适配，不要求一开始建立复杂的多层抽象。

## 5. 核心数据契约

系统内部使用引擎无关的数据结构，具体命令由测试方法定义生成。详细的硬件、模型和 workload 档案放在 `specs/`，引擎参数知识和调优经验放在 `references/`；测试方法的规则和参数映射放在 `references/benchmark_methods/`。

### 5.1 JobSpec：测试需求

```json
{
  "job_id": "qwen3_hccpk1_qa_v1",
  "engine": "sglang",
  "instance_type": "HCCPK1.96XLARGE1536",
  "model": "qwen3-235b-a22b-fp8",
  "workload": "qa-chat-3.5k-1k",
  "benchmark_method": "sglang-bench-serving",
  "sla": {
    "max_avg_ttft_ms": 2000,
    "max_avg_tpot_ms": 80,
    "min_success_rate": 0.99
  },
  "search": {
    "max_candidates": 30,
    "max_runtime_minutes": 180
  }
}
```

JobSpec 只表达本次任务需要组合哪些对象。`benchmark_method` 用于选择测试方法，测试方法的规则和参数映射维护在 `references/benchmark_methods/`。GPU 显存、卡数、模型版本、镜像版本、量化方案和 workload 详细参数由对应的 `specs/` 和 `references/` 文件提供。解析时如果引用不存在、版本不明确或信息冲突，Agent 必须报告问题，不能静默猜测。

第一阶段不生成独立的 `ResolvedJobSpec` 文件。后续真实执行阶段如需复现，再在 run artifact 中生成 `run_manifest.json`，保存本次使用的 JobSpec、specs、references 和候选。

### 5.2 SearchPlan：参数搜索计划

```json
{
  "pinned": {
    "model_path": "...",
    "tp_size": 8,
    "reasoning_parser": "qwen3"
  },
  "axes": {
    "attention_backend": {
      "values": ["fa3", "flashinfer"],
      "source": "official",
      "reason": "目标 GPU 和 SGLang 版本支持",
      "hard_constraint": false
    },
    "chunked_prefill_size": {
      "values": [4096, 8192, 16384],
      "source": "heuristic",
      "failure_modes": ["latency_regression", "runtime_oom"]
    }
  },
  "constraints": [
    {
      "exclude": {"tp_size": 1},
      "reason": "模型权重或 KV Cache 无法满足显存要求"
    }
  ],
  "search_policy": {
    "strategy": "bounded_then_adaptive",
    "max_candidates": 30
  }
}
```

SearchPlan 需要同时表达：

- 固定参数；
- 待搜索参数轴；
- 参数取值；
- 组合约束；
- 失败模式；
- 证据来源；
- 搜索预算；
- 是否允许后续自适应扩展。

SearchPlan 的组织方式参考 `AI-Infra-Auto-Driven-SKILLS` 的 benchmark 配置：将固定的 `base_server_flags` 与待搜索的 `search_space` 分开。区别在于，本项目的 Planner 负责根据 JobSpec 和 References 自动推导这两部分，而不是要求用户手工完整填写。

```json
{
  "base_server_flags": {
    "model_path": "...",
    "tp_size": 8,
    "reasoning_parser": "qwen3"
  },
  "search_space": {
    "attention_backend": ["fa3", "flashinfer"],
    "chunked_prefill_size": [4096, 8192, 16384]
  }
}
```

### 5.3 Candidate：可执行候选

```json
{
  "candidate_id": "sglang-c001",
  "params": {
    "tp_size": 8,
    "attention_backend": "fa3",
    "chunked_prefill_size": 8192
  },
  "server_command": "python -m sglang.launch_server ...",
  "benchmark_command": "python -m sglang.bench_serving ...",
  "expected_risk": "medium"
}
```

候选命令必须可复现，包含模型、镜像、参数、workload、端口、GPU 可见设备和 artifact 输出位置。

### 5.4 RunResult：测试结果

每个候选都要保留结果，包括失败候选：

```json
{
  "candidate_id": "sglang-c001",
  "status": "ok",
  "failure_reason": null,
  "metrics": {
    "request_throughput": 15.8,
    "output_token_throughput": 12500.0,
    "total_token_throughput": 42000.0,
    "p50_ttft_ms": 410.0,
    "p99_ttft_ms": 1550.0,
    "p50_tpot_ms": 24.0,
    "p99_tpot_ms": 72.0,
    "success_rate": 0.995
  },
  "memory": {
    "available_gpu_mem_gb": 7.2,
    "max_total_num_tokens": 131072,
    "max_num_reqs": 128
  },
  "artifacts": {
    "server_log": "...",
    "benchmark_result": "...",
    "server_help": "...",
    "benchmark_help": "..."
  }
}
```

失败状态至少包括：

```text
bad_args
startup_oom
runtime_oom
cuda_error
nccl_error
health_check_failed
benchmark_timeout
infra_error
sla_failed
```

## 6. SOP 和模块职责

### Stage 0：Specs 与 References 准备

建立可被 Agent 检索、可被程序校验的详细档案和参数知识。

#### 0.1 硬件知识

第一版支持 NVIDIA GPU，记录：

- GPU 型号、显存和计算能力；
- CUDA、驱动、NCCL 兼容性；
- NVLink/PCIe/NVSwitch 拓扑；
- 单机卡数、CPU、内存、SSD、RDMA；
- 常见 TP/PP/EP 组合和已知限制。

后续增加国产卡知识时，优先覆盖昆仑芯和海光，分别记录其显存、计算能力、驱动/运行时、卡间互联、通信库、SGLang/vLLM 支持状态和已验证参数。

#### 0.2 模型知识

记录：

- ModelCard、模型架构和参数规模；
- dense/MoE、MLA/GQA、Hybrid/Mamba 等结构；
- 权重精度和量化格式；
- tokenizer、reasoning parser、tool-call parser；
- context length、KV Cache 特征和已知 SGLang 配置。

#### 0.3 Engine References

`references/sglang/` 不复制整篇官方参数文档。经常更新的官方资料通过链接引用，Agent 在生成 SearchPlan 时按目标 SGLang 版本检索；项目自己的搜索策略和约束则固定保存在本地。

当前采用“官方来源 + 项目策略 + 目标环境校验”的方式：

- `sources.json`：官方文档、SGLang 源码和外部参考项目的链接；
- `parameter_policy.json`：哪些参数第一轮允许搜索、哪些参数默认固定，以及证据要求；
- `tuning_principles.md`：TP、上下文长度等调优经验、反模式和证据要求；
- 目标镜像中的 `python -m sglang.launch_server --help`：确认当前参数名、别名和可用性的最终依据。

不维护一份脱离版本的完整 SGLang 参数字典，也不把最新网页参数直接当作目标镜像一定支持的参数。

当前重要调优原则是：在显存、KV Pool 和 SLA 都满足的前提下优先使用最少卡数；
PCIe 多卡优先评估 PP 以减少通信，NVLink 多卡重点评估 TP 的延迟收益。互联拓扑
只决定候选优先级，不直接决定最终最优方案，最终结果必须来自相同 workload 下的实测。

SearchPlan 中 `pinned` 和 `search_space` 只能放 SGLang 参数名；基线说明、经验备注
应放到 `constraints`、`axes.*.reason`、`axes.*.source` 或顶层 `notes`，不能伪装成
`baseline_tp_size` 等参数。

后续如果需要保存版本化证据，只保存目标镜像的 CLI 快照：

```text
references/sglang/sources.json
references/sglang/parameter_policy.json
references/sglang/tuning_principles.md
references/sglang/snapshots/<sglang-version>/launch_server_help.txt
```

#### 0.4 Workload 知识

第一版固定两类标准场景：

| 场景 | 默认长度 | Cache | 测试方式 |
|---|---:|---|---|
| 日常 QA Chat | 输入 3.5k / 输出 1k | 不考虑命中 | bench serving |
| AI Coding | 输入 100k / 输出 1k | 考虑命中 | multiturn / cache-aware benchmark |

实际生产流量可以覆盖默认值，但必须保存 p50/p95/p99 长度和 Cache 分布。

### Stage 1：生成测试需求

输入简化的任务描述，引用 `specs/` 中的硬件、模型和 workload 档案，生成规范化 `JobSpec`。

解析和展开时校验：

- `instance_type`、模型和 workload 引用是否存在；
- 硬件档案中的显存、卡数和互联信息是否完整；
- 模型档案中的权重、量化、Tokenizer 和上下文信息是否完整；
- workload 档案是否包含输入/输出长度、并发或请求速率和测试方法；
- SGLang 镜像、版本和必要运行时信息是否明确；
- SLA 是否明确统计方式、成功率和目标值；
- 测试预算是否足以完成候选搜索。

缺失事实时，Agent 必须报告缺口，不能静默填默认值。

### Stage 2：生成参数搜索计划

流程：

1. Python 程序校验 `JobSpec`；
2. Claude Code CLI 读取 `JobSpec`、相关 specs、SGLang references 和必要源码；
3. Claude Code 识别硬约束和可调参数；
4. Claude Code 估算显存、TP 和上下文可行性；
5. Claude Code 输出结构化 `SearchPlan`；
6. 本地规则引擎执行参数合法性和组合约束校验；
7. 本地命令渲染器编译候选命令；
8. 输出候选清单和每个参数的理由、证据、风险。

第一版优先搜索高影响且容易验证的参数：

- `tp_size`；
- `attention_backend`；
- `prefill_attention_backend` / `decode_attention_backend`；
- `mem_fraction_static`；
- `kv_cache_dtype`；
- `chunked_prefill_size`；
- `max_running_requests`；
- `max_total_tokens`；
- CUDA Graph 相关开关；
- 模型需要的 reasoning/tool parser 应固定启用；speculative decoding 只有在模型和运行时均支持时才纳入搜索。

不建议第一轮同时搜索所有参数。模型路径、精度、必要 parser、拓扑相关参数等应优先固定。

### Stage 3：候选执行与自适应搜索（后续阶段）

#### 3.1 候选执行生命周期

```text
候选准备
  ↓
检查 GPU 是否空闲
  ↓
启动 SGLang
  ↓
健康检查和模型检查
  ↓
单请求 streaming smoke test
  ↓
短 benchmark
  ↓
正式 benchmark
  ↓
采集 server log / benchmark result / nvidia-smi
  ↓
停止服务并清理资源
```

每个候选必须隔离端口、GPU、日志和结果目录，避免前一个候选影响后一个候选。

#### 3.2 搜索策略

第一版采用两阶段搜索，而不是无限制笛卡尔积：

```text
第一轮：baseline + 有界高影响参数组合
第二轮：保留 SLA 通过且表现靠前的候选，围绕其扩展或细化
```

第一轮建议默认不超过 10～30 个候选。后续可以增加：

- Successive Halving；
- Bayesian Optimization；
- ASHA/Hyperband；
- 基于 OOM 边界的二分或单调剪枝。

#### 3.3 失败处理

失败候选不能直接丢弃。执行器需要区分：

- `bad_args`：参数不存在或组合非法；
- `startup_oom`：启动阶段显存不足；
- `runtime_oom`：压测峰值或高并发阶段 OOM；
- `cuda_error` / `nccl_error`：运行时或分布式错误；
- `health_check_failed`：服务未正常提供请求；
- `benchmark_timeout`：测试过程超时；
- `infra_error`：环境、镜像、网络或机器问题；
- `sla_failed`：完成测试但未满足目标。

不同失败类型对下一轮搜索的处理不同，不能统一当作“性能差”。

### Stage 4：结果评估与报告（后续阶段）

评估步骤：

1. 校验 benchmark 是否完成，平均输出长度是否接近目标；
2. 校验结果是否存在截断、网关超时或异常成功率；
3. 重新计算 SLA，而不是完全信任外部脚本的标记；
4. 过滤 SLA 不通过的候选；
5. 按目标函数排序；
6. 输出最佳候选、次优候选和失败候选；
7. 保存完整命令、版本、日志和原始结果。

报告至少包含：

- 最佳启动命令和 benchmark 命令；
- 参数解释和选择理由；
- SLA 指标和吞吐指标；
- 显存和 KV Cache 信息；
- 所有候选的状态和失败原因；
- 测试环境和版本清单；
- 结果适用范围和已知局限。

## 7. 第一版建议目录

```text
LLMOptAgent/
├── README.md
├── docs/
│   └── development-plan.md
├── pyproject.toml
├── schemas/
│   ├── job_spec.py
│   ├── search_plan.py
│   └── candidate.py
├── specs/
│   ├── hardware/
│   ├── models/
│   └── workloads/
├── references/
│   ├── sglang/
│   └── parallelism/
│       ├── principles.md
│       ├── topology.md
│       └── memory-and-kv-pool.md
├── skills/
│   └── parallelism-search/
│       └── SKILL.md
├── jobs/
├── outputs/
├── planner/
│   ├── job_validator.py
│   ├── spec_loader.py
│   ├── reference_loader.py
│   ├── rule_checker.py
│   ├── search_planner.py
│   ├── claude_code_client.py
│   ├── plan_validator.py
│   ├── candidate_generator.py
│   └── command_renderer.py
├── prompts/
│   └── search_plan.md
├── cli/
│   └── main.py
├── examples/
│   └── jobs/
└── tests/
```

第一阶段只实现离线测试范围生成，不实现 `requirement_parser.py`、`ResolvedJobSpec`、Executor、Runner、结果评估和 vLLM Adapter。

## 8. 与 AI-Infra-Auto-Driven-SKILLS 的关系

该项目参考并复用 `AI-Infra-Auto-Driven-SKILLS` 的 benchmark 设计，但增加自动规划搜索空间和候选调度能力。

可复用内容包括：

- `llm-serving-auto-benchmark` 的 `base_server_flags/search_space` 配置思路、候选命令、结果 JSONL 和 SLA 规范；
- SGLang Cookbook 中的模型专属启动 recipe；
- 目标环境 `--help` 的 CLI 参数和版本校验方法；
- `llm-serving-capacity-planner` 的启动日志和显存分析；
- profiler、pipeline 和模型优化历史相关能力。

两者边界如下：

```text
AI-Infra-Auto-Driven-SKILLS
  提供 benchmark 配置模式、结果规范、知识和分析技能

HAI 参数寻优 Agent
  负责需求理解、自动生成 base_flags/search_space、候选调度、自适应搜索和 HAI 经验沉淀
```

其中，`AI-Infra-Auto-Driven-SKILLS` 当前的 benchmark skill 是配置和分析工具，不是完整的服务启动和候选调度器，也不会仅凭模型和卡型自动推导完整搜索范围。因此本项目需要实现自己的 Stage2 Planner、Executor 和搜索闭环。

## 9. 第一版验收标准

满足以下条件，视为第一版闭环完成：

1. 能校验固定 JSON JobSpec 及其 specs 引用；
2. 能在至少一种 NVIDIA GPU 规格上生成 SGLang SearchPlan；
3. 能覆盖 QA Chat 和 AI Coding 两类 workload spec；
4. 能基于 references 生成带理由和证据的参数范围；
5. 能生成有界的候选参数组合；
6. 能生成经过 CLI 规则校验的 SGLang server/benchmark 命令；
7. 能输出 `search_plan.json`、`candidates.jsonl` 和 Markdown 计划报告；
8. 能说明被硬约束排除的候选及原因；
9. 全流程不启动服务、不执行 benchmark、不依赖 GPU。

## 10. 后续路线图

### Phase 1：SGLang + NVIDIA 测试范围生成器

- 完成 JobSpec、SearchPlan、Candidate schema；
- 建立 NVIDIA、模型和 SGLang 知识库；
- 实现离线 `SGLangNvidiaAdapter` 命令渲染；
- 完成 QA Chat、AI Coding 两类场景；
- 输出候选参数范围、命令和计划报告。

### Phase 2：SGLang 执行和结果评估

- 实现本地/容器 Executor；
- 完成 SLA 评估和报告；
- 引入 OOM 边界剪枝；
- 增加 Successive Halving 或 Bayesian Optimization；
- 将每次测试结果回写到 measured-results；
- 支持同模型、同卡型、不同 SGLang 版本的回归对比；
- 增加容量分析和 profiler 自动触发。

### Phase 3：vLLM 支持

- 增加 `vllm_nvidia.py`；
- 建立 SGLang/vLLM 参数族映射，但保留各引擎独立语义；
- 支持 SGLang 和 vLLM 在同一 workload 下的公平比较；
- 增加 vLLM 独立的 CLI 校验、启动日志解析和 benchmark 结果适配。

### Phase 4：昆仑芯/海光和复杂部署

- 增加昆仑芯和海光硬件知识；
- 增加 `sglang_kunlunxin.py`、`sglang_hygon.py`；
- 根据实际支持情况增加 `vllm_kunlunxin.py`、`vllm_hygon.py`；
- 建立 NVIDIA、昆仑芯、海光之间的参数兼容性边界，不能简单复用 CUDA 参数；
- 支持多机、RDMA、PD 分离、EP 和异构集群；
- 增加 K8s 调度器和资源队列；
- 建立跨平台经验迁移和配置推荐能力。

## 11. 当前待确认事项

在开始实现前，需要进一步确定：

1. 第一阶段直接在宿主机调试 Claude Code CLI；后续再确定 Agent 容器的基础镜像和源码挂载方式；
2. 第一批优先支持的 NVIDIA GPU 型号；
3. 第一批模型和对应镜像版本；
4. QA Chat 和 AI Coding 的真实 workload 数据来源；
5. 最终结果是写本地文件、数据库，还是接入现有 HAI 平台；
6. 是否需要保留人工审核 SearchPlan 的环节；
7. 第一版的最大候选数和单次调优时间预算。
