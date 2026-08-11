# SGLang + NVIDIA 测试范围生成器开发计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or an equivalent task-by-task workflow to implement this plan. Each task uses checkbox syntax for tracking.

**Goal:** 实现一个基于固定 JobSpec、硬件/模型/workload specs 和 SGLang references 的离线测试范围生成器，输出可解释的 SearchPlan、候选参数和测试命令，不启动服务、不执行 benchmark。

**Architecture:** 第一阶段采用“Claude Code 分析 + 本地确定性校验”的流水线：JobSpec 校验 → Claude Code 渐进式读取 specs/references → 输出结构化 SearchPlan → 本地规则校验 → 候选生成 → SGLang 命令渲染。服务执行器、Kubernetes 和结果评估延期到后续阶段。

**Tech Stack:** Python 3.11+、Pydantic v2、Typer、PyYAML、packaging、Claude Code CLI、pytest、Ruff、Pyright、uv。

---

## 1. 第一阶段边界

### 纳入范围

- 固定 JSON JobSpec 输入；
- NVIDIA 硬件、模型和 workload specs；
- SGLang 参数和兼容性 references；
- JobSpec/schema 校验；
- specs 和 references 解析；
- 硬约束和参数兼容性检查；
- 根据规则生成有限候选搜索空间；
- 通过 Claude Code CLI 分析 JobSpec 和 references，生成结构化 SearchPlan；
- 生成 SGLang server/benchmark 命令文本；
- 输出 JSON、JSONL、Markdown 计划报告；
- 离线单元测试和 fixture 测试。

### 明确不做

- 自然语言需求解析；
- ResolvedJobSpec 独立文件；
- LLM/API 直接生成最终 shell 命令；
- 启动 SGLang 服务；
- Docker、SSH、Kubernetes 执行；第一阶段允许宿主机直接调试；
- GPU 实测、OOM 反馈和自适应搜索；
- benchmark 结果评估和最佳参数选择；
- vLLM 和国产卡 Adapter。

后续执行阶段可以生成 `run_manifest.json`，记录一次实际运行使用的 JobSpec、specs、references 和候选，但它不是第一阶段的核心对象。

## 1.1 运行形态

第一阶段优先直接在开发机上运行 Claude Code CLI 和 Python 程序，缩短调试链路，不要求一开始构建容器。调试时可以在受控工作目录中使用 `--dangerously-skip-permissions`，但不得把密钥、生产配置或无关宿主机目录暴露给 Claude Code。

稳定后再构建 `LLMOptAgent` 专用 Agent 容器，容器包含：

- LLMOptAgent 代码；
- Claude Code CLI；
- specs 和 references；
- SGLang/vLLM 源码快照，用于版本相关的源码分析；
- Python 依赖和计划生成工具。

该容器只负责 Agent 分析和命令生成，不启动推理服务。真正的 SGLang/vLLM 服务仍由后续独立运行环境负责。容器化后仍然需要保留目录白名单、执行超时和敏感信息隔离。

## 2. 第一阶段目录和文件职责

```text
/data/LLMOptAgent/
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
│   │   ├── sources.json
│   │   ├── parameter_policy.json
│   │   └── tuning_principles.md
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
│   ├── __init__.py
│   ├── job_validator.py
│   ├── spec_loader.py
│   ├── reference_loader.py
│   ├── rule_checker.py
│   ├── search_planner.py
│   ├── claude_code_client.py
│   ├── plan_validator.py
│   ├── candidate_generator.py
│   └── command_renderer.py
├── cli/
│   ├── __init__.py
│   └── main.py
└── tests/
    ├── fixtures/
    ├── test_job_validator.py
    ├── test_spec_loader.py
    ├── test_rule_checker.py
    ├── test_search_planner.py
    ├── test_candidate_generator.py
    └── test_command_renderer.py
```

第一阶段暂时不新增 `agent/`、`runners/`、`resolved/` 的业务代码。保留这些目录作为后续执行阶段的扩展位置即可。

## 2.1 推荐开发顺序

虽然最终链路包含 Claude Code，但开发时不先接 AI。先用固定 fixture 模拟一份合法的 SearchPlan，把本地校验、候选生成和命令渲染跑通，再接入 Claude Code。这样可以把“AI 是否正确理解 references”和“本地程序是否能安全生成命令”分开验证。

推荐顺序如下：

```text
项目基础
  → 数据模型
  → specs/references 加载
  → 硬约束检查
  → fixture SearchPlan
  → 候选生成
  → SGLang 命令渲染
  → Claude Code Search Planner
  → CLI 串联
  → 端到端离线验收
```

每一步都应形成一个可运行的小闭环：

1. **项目基础**：建立 Python 包、依赖、测试和 CLI 空入口。
2. **数据模型**：用 Pydantic 固化 JobSpec、SearchPlan 和 Candidate 的 JSON 契约。
3. **加载层**：按 ID 加载 HardwareSpec、ModelSpec、WorkloadSpec、benchmark method 和 SGLang policy。
4. **规则层**：检查引用、TP、上下文长度、参数类别和硬件/模型约束，并保留排除理由。
5. **本地规划链路**：使用 fixture SearchPlan 验证候选上限、baseline-first 和命令渲染，不依赖 Claude Code。
6. **Claude Code 接入**：让 Claude 只负责分析资料并输出 SearchPlan，禁止直接输出最终 Shell 命令。
7. **CLI 和验收**：串联 `validate-job`、`plan`、`render`，确认全流程不启动服务、不访问 GPU。

其中第 5 步是关键里程碑。完成后，即使暂时没有 Claude Code，也应能用人工编写的 SearchPlan 生成合法候选；完成第 6 步后，才实现自动分析能力。

## 3. 数据契约

### 3.1 JobSpec

JobSpec 保持简洁，只引用对象 ID：

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
    "max_avg_tpot_ms": 80
  },
  "search": {
    "max_candidates": 30,
    "max_runtime_minutes": 180
  }
}
```

第一阶段要求：

- `engine` 只能为 `sglang`；
- `instance_type` 必须能在 `specs/hardware/` 找到；
- `model` 必须能在 `specs/models/` 找到；
- `workload` 必须能在 `specs/workloads/` 找到；
- `benchmark_method` 必须能在 `references/benchmark_methods/` 找到；
- `max_candidates` 必须为正整数；
- SLA 字段必须使用明确的统计语义。

### 3.2 SearchPlan

SearchPlan 包含：

- `base_server_flags`：固定的服务参数；
- `search_space`：待搜索参数及候选值；
- `constraints`：被排除的参数组合和原因；
- `axes`：参数轴的语义、来源和风险；
- `search_policy`：候选上限和生成策略。

```json
{
  "job_id": "qwen3_hccpk1_qa_v1",
  "base_server_flags": {
    "model_path": "Qwen/Qwen3-235B-A22B-FP8",
    "tp_size": 8,
    "reasoning_parser": "qwen3"
  },
  "search_space": {
    "attention_backend": ["fa3", "flashinfer"],
    "chunked_prefill_size": [4096, 8192, 16384]
  },
  "constraints": [],
  "search_policy": {
    "strategy": "baseline_first_bounded_product",
    "max_candidates": 30
  }
}
```

### 3.3 Candidate

Candidate 是一个可以交给未来执行器的命令描述，但第一阶段只渲染，不执行：

```json
{
  "candidate_id": "sglang-c001",
  "params": {
    "tp_size": 8,
    "attention_backend": "flashinfer",
    "chunked_prefill_size": 8192
  },
  "server_command": ["python", "-m", "sglang.launch_server", "..."],
  "benchmark_command": ["python", "-m", "sglang.bench_serving", "..."],
  "benchmark_commands": [
    ["python", "-m", "sglang.bench_serving", "..."],
    ["python", "-m", "sglang.bench_serving", "..."]
  ],
  "reasons": ["..."],
  "expected_risk": "medium"
}
```

## 4. 模块开发计划

### Task 1：项目基础和开发工具

**Files:**

- Create: `pyproject.toml`
- Create: `planner/__init__.py`
- Create: `cli/__init__.py`
- Create: `cli/main.py`
- Create: `tests/test_smoke.py`

- [x] 创建 Python package 和 `llmopt` CLI 入口；
- [x] 配置 Python 版本、Pydantic、Typer、PyYAML、packaging；
- [x] 配置 Ruff、Pyright 和 pytest；
- [x] 增加 `llmopt --help` smoke test；
- [x] 运行 `uv run pytest`、`uv run ruff check .`、`uv run pyright`。

### Task 2：定义 Pydantic 数据模型

**Files:**

- Create: `schemas/job_spec.py`
- Create: `schemas/search_plan.py`
- Create: `schemas/candidate.py`
- Create: `tests/test_schemas.py`

- [x] 定义 `JobSpec`、SLA、SearchBudget；
- [x] 定义 Hardware/Model/Workload 引用字段；
- [x] 定义 `SearchPlan`、ParameterAxis、Constraint；
- [x] 定义 `Candidate` 和命令字段；
- [x] 为非法字段、非法数值和缺失引用编写失败测试；
- [x] 保证模型可以导出 JSON Schema。

### Task 3：实现 specs 加载

**Files:**

- Create: `planner/spec_loader.py`
- Create: `tests/test_spec_loader.py`
- Add fixtures: `tests/fixtures/specs/`

- [x] 实现按 ID 查找 hardware/model/workload spec；
- [x] 统一处理 JSON 和 YAML；
- [x] 对不存在、重复和格式错误的 spec 给出明确错误；
- [x] 将加载结果转换为内部 Pydantic 对象；
- [x] 测试示例 JobSpec 可以加载三类 spec。

### Task 4：实现 SGLang references 加载

**Files:**

- Create: `planner/reference_loader.py`
- Add: `references/sglang/sources.json`
- Add: `references/sglang/parameter_policy.json`
- Add: `references/sglang/tuning_principles.md`
- Create: `tests/test_reference_loader.py`

- [x] 定义来源链接和参数策略的 JSON 结构；
- [x] 支持加载官方来源链接和外部参考链接；
- [x] 支持加载搜索参数、固定参数和条件搜索参数；
- [x] 将调优经验和判断原则使用 Markdown 保存，避免与机器可校验策略混杂；
- [x] 要求候选参数通过目标镜像 `launch_server --help` 校验；
- [x] 用少量 NVIDIA/SGLang 参数建立最小可用策略集。

第一版不要求一次性录入所有 SGLang 参数，也不复制官方文档。优先支持 TP、attention backend、KV cache、chunked prefill、并发和 parser；官方文档只保存链接，版本事实以后由目标镜像的 `--help` 快照提供。

### Task 5：实现规则检查器

**Files:**

- Create: `planner/rule_checker.py`
- Create: `tests/test_rule_checker.py`

- [x] 检查 TP 不超过硬件 GPU 数；
- [x] 检查 context length 覆盖 workload 长度；
- [x] 检查并标记 attention backend 的 GPU/版本兼容性；缺少目标环境事实时输出 warning，不静默猜测或过滤；
- [x] 检查参数是否存在、类型是否正确；
- [x] 校验当前策略已定义的参数分类和组合约束；未定义的版本/模型互斥关系保留为后续 reference 扩展；
- [x] 将每个排除项记录为 `constraint + reason + source`；
- [x] 区分 hard constraint 和 heuristic warning。

### Task 6：实现 Claude Code Search Planner

**Files:**

- Create: `planner/search_planner.py`
- Create: `planner/claude_code_client.py`
- Create: `planner/plan_validator.py`
- Create: `prompts/search_plan.md`
- Create: `tests/test_search_planner.py`
- Create: `tests/test_claude_code_client.py`

- [x] 用 `claude -p` 启动非交互式 Claude Code；
- [x] 通过 `--add-dir` 暴露 LLMOptAgent、specs、references 和可选源码目录；
- [x] 通过 `--output-format json` 和 `--json-schema` 要求结构化 SearchPlan；
- [x] 初期调试支持在受控目录使用 `--dangerously-skip-permissions`；正式运行提供只读工具白名单模式；
- [x] prompt 明确不启动服务、不修改 specs/references、不执行 benchmark；
- [x] 在 prompt 中明确当前只生成 SearchPlan，不执行 benchmark；
- [x] 固定模型路径、精度和模型专属 parser；
- [x] 由 Claude Code 根据硬件和模型分析 TP、attention backend、KV cache、chunked prefill 和并发范围；
- [x] 由 `plan_validator.py` 校验 Claude 输出，拒绝非法参数和超预算候选；
- [x] 输出参数选择理由、来源和风险；
- [x] 对 Claude CLI 超时、非零退出码和非法 JSON 做明确错误处理。

第一版采用 `baseline_first_bounded_product`，不引入 Bayesian Optimization 或 Optuna；LLM 只负责分析 references 和生成 SearchPlan，不直接生成最终命令。

### Task 7：实现候选生成器

**Files:**

- Create: `planner/candidate_generator.py`
- Create: `tests/test_candidate_generator.py`

- [x] 生成 baseline candidate；
- [x] 根据 `search_space` 生成有界组合；
- [x] 支持应用上游 rule checker 提供的 candidate filter 排除结果；
- [x] 为候选生成稳定 ID；
- [x] 保留每个候选的坐标、来源和风险；
- [x] 测试候选数量上限和组合顺序。

### Task 8：实现 SGLang 命令渲染器

**Files:**

- Create: `planner/command_renderer.py`
- Create: `tests/test_command_renderer.py`

- [x] 将固定参数和候选参数转换为 argv list；
- [x] 生成 `python -m sglang.launch_server` 命令；
- [x] 生成 `python -m sglang.bench_serving` 命令；
- [x] 根据 workload 生成输入/输出长度、请求数和并发参数；
- [x] 支持使用目标镜像 help snapshot 校验参数；
- [x] 避免 shell 字符串拼接，优先输出 argv list；
- [x] 同时提供 shell 渲染格式供人工复制。

这里的 Adapter 只负责离线命令转换，不启动服务。

### Task 9：实现 CLI 和输出文件

**Files:**

- Modify: `cli/main.py`
- Create: `tests/test_cli.py`

第一版 CLI：

```bash
llmopt validate-job jobs/example.json
llmopt plan jobs/example.json --output outputs/example
llmopt render outputs/example/search_plan.json
```

- [x] `validate-job` 只做 JobSpec 和引用校验；
- [x] `plan` 生成 `search_plan.json` 和 `candidates.jsonl`；
- [x] `render` 输出 `commands.sh` 和 `plan_report.md`；
- [x] 输出目录包含输入 JobSpec 的副本；
- [x] 失败时返回非零 exit code 和可读错误；
- [x] 不启动任何服务，不调用 GPU。

### Task 10：端到端离线验收

**Files:**

- Add: `examples/jobs/qwen36_pro5000_random_v1.json`
- Add: `tests/test_end_to_end_plan.py`
- Modify: `README.md`

- [x] 使用示例 JobSpec 完成全流程；
- [x] 输出固定数量的候选；
- [x] 每个候选包含 server/benchmark 命令；
- [x] 非法 GPU、模型、workload 引用能够失败；
- [x] 不支持的 backend 能被排除并说明原因；
- [x] 记录当前第一阶段验收结果和已知限制。

Task 10 的离线验收使用 Pro5000、Qwen3.6-27B-FP8 和 `random-32k-1k` 示例，
通过 SearchPlan fixture 生成 2 个候选，并渲染每个候选的 SGLang server 命令和
`bench_serving` 命令。测试只检查 JSON、规则和命令产物，不启动服务、不访问 GPU。
当前限制是 SearchPlan fixture 仍由人工提供；Claude Code 规划链路已有单元测试，
但真实 Claude 调用和 benchmark 执行留到后续阶段。

### Task 11：接入目标镜像 help 快照校验

**Files:**

- Add: `planner/help_snapshot.py`
- Add: `tests/test_help_snapshot.py`
- Modify: `cli/main.py`
- Modify: `tests/test_cli.py`

- [x] 从 `launch_server --help` 和 `bench_serving --help` 文本中提取长参数名；
- [x] `render` 支持 `--server-help` 和 `--benchmark-help`；
- [x] 目标快照缺少候选参数时，在渲染阶段返回明确错误；
- [x] 空快照或不存在的快照返回明确错误；
- [x] 不执行任何外部命令，只读取用户显式提供的文本快照。

快照校验只确认参数名是否存在，不从帮助文本推断参数类型、默认值或硬件兼容性。
这些信息仍由目标镜像实际行为、references 和后续运行结果共同确认。

上下文长度有特殊语义：WorkloadSpec 的输入/输出长度只描述本轮压测请求，
不等于服务应该暴露的上下文窗口。若 ModelSpec 提供 `native_context_length`，
SearchPlan 默认应保留该能力；只有 JobSpec 显式要求服务 cap，或已有实测证据证明
目标硬件/运行时无法承载时，才允许缩短 `context_length`，并且必须说明依据。

### Task 12：并行策略候选生成

**Files:**

- Add: `references/parallelism/*.md`
- Add: `skills/parallelism-search/SKILL.md`
- Modify: `schemas/search_plan.py`
- Modify: `planner/candidate_generator.py`
- Modify: `planner/rule_checker.py`

- [x] 用 `parallelism_candidates` 表达完整 TP/PP 部署方案；
- [x] 强制校验 `gpu_count = tp_size * pp_size`；
- [x] 强制校验候选总卡数不超过硬件 GPU 数；
- [x] 禁止在存在并行方案时把 TP/PP 作为独立搜索轴；
- [x] 将并行部署方案与 backend/prefill 等运行参数分层生成；
- [x] 建立 PCIe/NVLink、KV Pool 和最小可行卡数知识文档。

## 5. 后续阶段，不纳入当前实现

### Phase 2：SGLang 执行器

- 本地 Docker/容器 Runner；
- 健康检查、smoke test、benchmark；
- 进程和 GPU 资源隔离；
- server log、benchmark result 和 artifact 管理；
- OOM/CUDA/NCCL/timeout 分类。

### Phase 3：结果评估和搜索反馈

- RunResult schema；
- SLA 计算和候选排序；
- 失败候选保留；
- OOM 边界剪枝；
- Successive Halving 或其他自适应搜索。

### Phase 4：容器化、Agent 扩展和多平台

- 构建 `Dockerfile.agent`，封装 Claude Code、LLMOptAgent 和源码快照；
- 可选的自然语言 JobSpec 生成；
- 直接 LLM API Planner 作为 Claude Code CLI 的替代后端；
- vLLM NVIDIA Adapter；
- 昆仑芯/海光 Adapter；
- K8s/SSH Runner；
- 历史结果推荐和回归测试。

## 6. 开发验收命令

```bash
cd /data/LLMOptAgent
uv sync
uv run pytest
uv run ruff check .
uv run pyright
uv run llmopt --help
uv run llmopt plan jobs/example.json --output outputs/example
```

第一阶段完成的最低标准：

```text
固定 JSON JobSpec
  → specs/references 校验
  → SearchPlan
  → 候选参数 JSONL
  → SGLang server/benchmark 命令
```

全流程不得启动真实服务，也不得依赖 GPU。
