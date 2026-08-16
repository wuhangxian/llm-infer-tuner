---
name: sglang-server-config-gen
description: 输入一个固定 JSON(JobSpec:机型/模型/镜像/负载/SLA/预算),输出一批(十几条)可直接执行的 SGLang 服务端启动配置 + 分并发压测命令,供下游执行器批量实测。engine=sglang 时路由到这里。当用户提到「生成 sglang 启动配置」「参数寻优」「扫参数」「给我一批配置去跑」时使用。本 skill 只输出服务端(launch_server)启动配置;分并发压测命令由姊妹 skill `sglang-client-config-gen` 负责。
---

# SGLang 启动配置生成(第一阶段)

## 这个 skill 干什么、不干什么

```
你给 1 个 JSON            本 skill(AI 读规则生成)        下游(不在本 skill)
┌──────────────┐        ┌──────────────────┐        ┌──────────────┐
│ JobSpec      │        │ AI 读 catalogs +  │        │ 执行器        │
│ (机型/模型/  │───────▶│ knowledge + images │───────▶│ 拉镜像/起服务 │
│  镜像/负载/  │        │ → 十几条候选配置    │        │ 压测/剪枝     │
│  SLA/预算)   │        │ (代码过 3 道硬闸)  │        │ 出报告/回最优 │
└──────────────┘        └──────────────────┘        └──────────────┘
```

**干**:把「机型 × 模型 × 镜像 × 负载 × SLA」翻译成一批可直接执行的启动命令 + 分并发压测命令,附每条的理由和风险,让执行器有依据自己做决策。

**不干**:起服务、压测、剪枝、入库、出报告(下游第二阶段的事)。

**核心原则:加新经验 = 只改 `knowledge.md` / `catalogs/*.yaml` / `images.yaml` 的文本,不改生成器代码。**

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
  "image": "sglang-v0.5.10",
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
| `image` | `images.yaml`(本 skill 内) | CUDA 版本 / attention 菜单 / flag 别名 / valid_flags 白名单 |
| `workload` | `catalogs/workloads.yaml` | 输入/输出长度 / 并发梯度 / 采样参数 |
| `sla` / `search` | inline | SLA 阈值 / 候选数与时间预算 |

**⚠️ 输入里没有、也永远不该有 `context_length` 字段**(见 knowledge.md §0)。

---

## AI 生成时读哪些文件、按什么步骤

**读(按序,路径均相对项目根)**:① 本 `SKILL.md` → ② `.claude/skills/sglang-server-config-gen/knowledge.md`(全部经验判据)→ ③ 三张 `catalogs/*.yaml`(按 JobSpec 里的 ID 取对应卡片)→ ④ 本 skill 的 `.claude/skills/sglang-server-config-gen/images.yaml`(按 image ID 取镜像卡片)。

> 路径说明:`knowledge.md` 和 `images.yaml` 与本 SKILL.md 同目录(`.claude/skills/sglang-server-config-gen/`);`catalogs/`、`schemas/` 在项目根下。`claude` 从项目根运行,按上述相对路径即可读到。

**步骤**:

1. **解析 JobSpec** → 取出 5 张卡片(gpu / model / workload / image + inline sla/search)。
2. **定 attention 轴**:`knowledge.md §1` 短名单[gpu_model 查表得 sm_major] ∩ image.attention_backends ∩ CUDA 达标。
   - SM120 → 只有 `flashinfer / triton`;**prefill 永不用 trtllm_mha**(会 raise);fa3 排除。
3. **定并行度**(`knowledge.md §2`):
   - TP 候选 = 权重放得下的 2 的幂(≤ gpu_count);默认全保留。长输入把 TP1 排末尾。
   - PP 恒 1(单机排除 pp>1)。
   - EP:仅 MoE 生成;dense 模型(如 qwen36-27b)**不生成 ep 轴**。
4. **铺搜索轴**(`knowledge.md §3`):mem-fraction / chunked-prefill / max-running-requests / kv-cache-dtype / schedule-conservativeness,按默认档;按算力/CUDA 过滤搜不了的值。
5. **pin 与排除**(`knowledge.md §4-5`):模型 default_flags 原样取;lpm 仅共享前缀场景;排除项不生成。
   - **本 case workload=qa-chat 请求独立 → 用默认 fcfs,不 pin lpm。**
6. **混合架构**(`knowledge.md §7`):qwen3.6 `hybrid_mamba` → 缓存策略两分支值得 A/B(输入仅 3.5k 非长上下文,ratio 极端逃生用不到)。
7. **绝不写 `--context-length`**(`knowledge.md §0`)。
8. **组装候选**:笛卡尔积后按 `search.max_candidates` 截断(基线优先:先出低风险基线,再铺高风险轴);每条给 server_command + 分并发 benchmark_commands + reasons + expected_risk。

---

## 输出契约(OUTPUT)

一个配置文件,`candidates` 数组装十几条(~10-16),每条(schema 见 `schemas/candidate.py`):

| 字段 | 内容 |
|---|---|
| `candidate_id` | 编号(如 `c001`) |
| `params` | 这条改了哪些轴的结构化字典(如 `{attention_backend, chunked_prefill_size, ep_size, mem_fraction_static}`) |
| `server_command` | 完整 `python -m sglang.launch_server ...`(**不带 `--context-length`**) |
| `benchmark_commands` | 分并发压测命令已拆出到姊妹 skill `sglang-client-config-gen`,本 skill 输出可留空 `[]` 或仅给占位;真正的压测命令由客户端 skill 依据同一 JobSpec 的 workload+benchmark_method 生成(字段仍保留,`schemas/candidate.py` 向后兼容) |
| `reasons` | AI 为什么给这条(可追溯到 knowledge.md 哪条经验) |
| `expected_risk` | `low` / `medium` / `high` |

下游:执行器逐条起服务、分并发压测、二分找 SLA 下最大吞吐(goodput),回 3 条最优。

---

## 代码只做的 3 道硬校验闸(不是生成器,是验证器)

AI 负责分类+派生候选;代码在生成后只拦「一定起不来」的:

1. **plan 期**:每条 attention 后端 ∈ image 菜单 ∩ sm 短名单;`tp % ep == 0` 且 `num_experts % ep == 0`;dense 禁 ep;SM120 禁 prefill=trtllm_mha。
2. **schema 期**:`gpu_count == tp_size × pp_size`(Pydantic 校验)。
3. **render 期**:每条命令的 flag 按 image.flag_aliases 翻译成本版真名后,∈ image.valid_flags 白名单;不在即拒绝。

**边界(如实)**:加轴/加档/加卡/加镜像/加模型 = 零代码改动;但全新一类硬约束(如将来上 SM100 要拦 cutlass)仍需改 rule_checker。

---

## 目录

```
catalogs/                          # 共享、引擎无关(vllm skill 也复用),在项目根
  gpu.yaml  models.yaml  workloads.yaml
schemas/                           # Pydantic 契约,在项目根
  job_spec.py  candidate.py  search_plan.py
.claude/skills/sglang-server-config-gen/  # 本 skill(claude 从这里加载)
  SKILL.md        # 本文件:入口(读什么、步骤、输出、闸)
  knowledge.md    # 全部调优经验+判据(改经验改这里)
  images.yaml     # sglang 各版镜像事实(引擎相关)
```

**engine 路由**:JobSpec 的 `engine` 字段决定用哪个 skill —— `engine:"sglang"` → 本 skill(`.claude/skills/sglang-server-config-gen/`);将来 `engine:"vllm"` → 新建 `.claude/skills/vllm-config-gen/`,复用同一套 `catalogs/`。

## 姊妹 skill

服务端与客户端配对使用,共享同一套 `catalogs/` 与同一个 JobSpec:

- **本 skill `sglang-server-config-gen`(服务端)**:出 `python -m sglang.launch_server ...` 启动配置(TP/PP/EP、attention 后端、mem-fraction 等)。
- **`sglang-client-config-gen`(客户端)**:依据同一 JobSpec 的 workload + benchmark_method,出 `python -m sglang.bench_serving ...` 分并发压测命令(1,2,4,8,16,32…)。

两者读同一套 `catalogs/*.yaml` 和同一个 JobSpec:服务端决定「怎么起」,客户端决定「怎么压」,下游执行器把两者拼起来跑。
