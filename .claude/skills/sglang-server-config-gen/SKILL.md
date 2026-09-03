---
name: sglang-server-config-gen
description: 输入一个固定 JSON(JobSpec:机型/模型/镜像/负载/SLA/预算),输出一批(十几条)可直接执行的 SGLang 服务端启动配置 + 分并发压测命令,供下游执行器批量实测。engine=sglang 时路由到这里。当用户提到「生成 sglang 启动配置」「参数寻优」「扫参数」「给我一批配置去跑」时使用。本 skill 只输出服务端(launch_server)启动配置;分并发压测命令由姊妹 skill `sglang-client-config-gen` 负责。
---

# SGLang 启动配置生成(第一阶段)

## 这个 skill 干什么、不干什么

```
你给 1 个 JSON            本 skill(AI 读规则生成)        下游(不在本 skill)
┌──────────────┐        ┌──────────────────┐        ┌──────────────┐
│ JobSpec      │        │ AI 读 rules +     │        │ 执行器        │
│ (机型/模型/  │───────▶│ catalogs + images │───────▶│ 拉镜像/起服务 │
│  镜像/负载/  │        │ → 十几条候选配置    │        │ 压测/剪枝     │
│  SLA/预算)   │        │ (按规则生成候选)  │        │ 出报告/回最优 │
└──────────────┘        └──────────────────┘        └──────────────┘
```

**干**:把「机型 × 模型 × 镜像 × 负载 × SLA」翻译成一批可直接执行的启动命令 + 分并发压测命令,附每条的理由和风险,让执行器有依据自己做决策。

**不干**:起服务、压测、剪枝、入库、出报告(下游第二阶段的事)。

**核心原则:加新经验 = 在 `references/rules/` 对应主题 YAML 的 `rules` 末尾追加一条；加模型/镜像事实才改 `catalogs/*.yaml`。普通规则不需要改生成器代码。**

---

## 输入契约(INPUT)

一个 JSON 文件 = `JobSpec`(严格接口,多字段报错;schema 见 `schemas/job_spec.py`)。全是 ID 引用,细节在 catalog 卡片里:

```json
{
  "job_id": "qwen36_pro5000_random_v1",
  "engine": "sglang",
  "gpu_model": "pro5000",
  "gpu_count": 8,
  "gpu_memory_gb": 72,
  "model": "qwen36-27b-fp8",
  "image": "I01_sglang-v0.5.10",
  "workload": "qa-chat-3.5k-1k",
  "sla": { "max_avg_ttft_ms": 2000, "max_avg_tpot_ms": 80, "min_success_rate": 0.99 },
  "search": { "max_candidates": 16, "max_runtime_minutes": 180 }
}
```

| 字段 | 指向 | 提供什么 |
|---|---|---|
 | `gpu_model` | `catalogs/gpu.yaml` | 算力(sm_major) / NVLink |
 | `gpu_count` / `gpu_memory_gb` | inline (JobSpec) | 卡数(TP/EP 上限) / 单卡显存(TP 准入判据) |
| `model` | `catalogs/models.yaml` | 架构 / 是否 MoE / 权重大小 / parser 名 / KV 粗估 |
| `image` | `catalogs/sglang-images.yaml` | CUDA 版本 / attention 菜单 / speculative_algorithms / flag 别名 / valid_flags 白名单 |
| `workload` | `catalogs/workloads.yaml` | 输入/输出长度 / 并发梯度 / 采样参数 |
| `sla` / `search` | inline | SLA 阈值 / 候选数与时间预算 |

**⚠️ 输入里没有、也永远不该有 `context_length` 字段**(见 `references/rules/memory.yaml`)。

---

## AI 生成时读哪些文件、按什么步骤

**读(按序,路径均相对项目根)**:① 本 `SKILL.md` → ② `.claude/skills/sglang-server-config-gen/knowledge.md`(规则索引) → ③ `references/rules/README.md`(规则格式) → ④ 按 JobSpec 读取相关主题 YAML → ⑤ `catalogs/*.yaml`(按 ID 取 GPU/model/workload/image 卡片)。

> 路径说明:`knowledge.md` 与本 SKILL.md 同目录(`.claude/skills/sglang-server-config-gen/`);`catalogs/`(含 `sglang-images.yaml`)、`schemas/` 在项目根下。`claude` 从项目根运行,按上述相对路径即可读到。

**步骤**:

1. **解析 JobSpec** → 取出 5 张卡片(gpu / model / workload / image + inline sla/search)。
2. **定 attention 轴**:按 `references/rules/attention.yaml` 取 GPU 短名单 ∩ image.attention_backends ∩ CUDA 达标。
   - SM120 → 只有 `flashinfer / triton`;**prefill 永不用 trtllm_mha**(会 raise);fa3 排除。
3. **定并行度**:按 `references/rules/parallelism.yaml`:
   - TP 候选 = 权重放得下的 2 的幂(≤ gpu_count);默认全保留。长输入把 TP1 排末尾。
   - PP 恒 1(单机排除 pp>1)。
   - EP:仅 MoE 生成;dense 模型(如 qwen36-27b)**不生成 ep 轴**。
4. **铺搜索轴**:按 `references/rules/memory.yaml` 和 `scheduling.yaml` 的默认档铺 mem-fraction / chunked-prefill / max-running-requests / kv-cache-dtype / schedule-conservativeness，并按算力/CUDA 过滤搜不了的值。
5. **pin 与排除**:读取相关主题规则；模型 default_flags 原样取，lpm 仅共享前缀场景。实验规则可以生成并标风险，不把经验判断误当成候选硬闸。
   - **本 case workload=qa-chat 请求独立 → 用默认 fcfs,不 pin lpm。**
6. **投机解码**:按 `references/rules/speculative.yaml` 把模型卡 `speculative_options`/`mtp_params` 与镜像卡 `speculative_algorithms` 取交集；NONE 保留为对照，不能只写死 EAGLE。
7. **混合架构**:按 `references/rules/memory.yaml` 的 hybrid mamba 规则生成缓存策略。
8. **绝不写 `--context-length`**:遵守 `memory.yaml` 的 context 规则。
9. **组装候选**:笛卡尔积后按 `search.max_candidates` 截断(基线优先:先出低风险基线,再铺高风险轴);每条给 server_command + 分并发 benchmark_commands + reasons + expected_risk。

---

## 输出契约(OUTPUT)

一个配置文件,`candidates` 数组装十几条(~10-16),每条(schema 见 gen_configs.sh 内嵌的 JSON Schema):

| 字段 | 内容 |
|---|---|
| `candidate_id` | 编号(如 `c001`) |
| `params` | 这条改了哪些轴的结构化字典(如 `{attention_backend, chunked_prefill_size, ep_size, mem_fraction_static}`) |
| `server_command` | 完整 `python -m sglang.launch_server ...`(**不带 `--context-length`**) |
| `benchmark_commands` | 分并发压测命令已拆出到姊妹 skill `sglang-client-config-gen`,本 skill 输出可留空 `[]` 或仅给占位;真正的压测命令由客户端 skill 依据同一 JobSpec 的 workload+benchmark_method 生成(字段仍保留,向后兼容) |
| `reasons` | AI 为什么给这条(引用命中的 rule id、模型/镜像事实和风险) |
| `expected_risk` | `low` / `medium` / `high` |

下游:执行器逐条起服务、分并发压测、二分找 SLA 下最大吞吐(goodput),回 3 条最优。

---

## 生成和验证边界

AI 根据规则和 catalogs 生成候选，生成阶段只保证 JSON 结构完整、候选数量不超过
`max_candidates`，并尽量在 `reasons` 中说明适用条件和风险。不会把知识规则做成一个
“候选硬闸”：`experimental` 或资料不完整的候选仍可交给执行器，能否启动和性能如何以
目标机器实测为准。这符合参数寻优的探索目的，也避免规则写错时整批候选被提前吞掉。

确定性校验只用于两件事：`scripts/validate_knowledge.py` 检查 YAML 规则库没有坏格式、
重复 ID 或缺证据；输出 JSON Schema 检查 AI 返回能被下游读取。新增普通规则不需要改
生成器代码，新增模型/镜像事实才更新对应 catalogs。

---

## 目录

```
catalogs/                          # 共享、引擎无关(vllm skill 也复用),在项目根
  gpu.yaml  models.yaml  workloads.yaml
schemas/                           # Pydantic 契约,在项目根
  job_spec.py
.claude/skills/sglang-server-config-gen/  # 本 skill(claude 从这里加载)
  SKILL.md        # 本文件:入口(读什么、步骤、输出)
  knowledge.md    # 规则索引和新增流程
  references/rules/  # 按主题逐条追加的 YAML 规则
    README.md
    attention.yaml
    parallelism.yaml
    memory.yaml
    speculative.yaml
    scheduling.yaml
    fairness.yaml
  # sglang-images.yaml 在 catalogs/ 下(引擎相关镜像事实)
```

**engine 路由**:JobSpec 的 `engine` 字段决定用哪个 skill —— `engine:"sglang"` → 本 skill(`.claude/skills/sglang-server-config-gen/`);将来 `engine:"vllm"` → 新建 `.claude/skills/vllm-config-gen/`,复用同一套 `catalogs/`。

## 姊妹 skill

服务端与客户端配对使用,共享同一套 `catalogs/` 与同一个 JobSpec:

- **本 skill `sglang-server-config-gen`(服务端)**:出 `python -m sglang.launch_server ...` 启动配置(TP/PP/EP、attention 后端、mem-fraction 等)。
- **`sglang-client-config-gen`(客户端)**:依据同一 JobSpec 的 workload + benchmark_method,出 `python -m sglang.bench_serving ...` 分并发压测命令(1,2,4,8,16,32…)。

两者读同一套 `catalogs/*.yaml` 和同一个 JobSpec:服务端决定「怎么起」,客户端决定「怎么压」,下游执行器把两者拼起来跑。
