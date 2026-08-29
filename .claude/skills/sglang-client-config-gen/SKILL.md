---
name: sglang-client-config-gen
description: 输入一个 JobSpec(用其 workload + benchmark_method)+ 一个已起好的服务端地址(host:port)+ 模型路径,输出每个并发档一条可直接执行的 `python -m sglang.bench_serving` 压测命令,供下游执行器逐档跑、收指标、算 goodput。engine=sglang 的压测阶段路由到这里。当用户提到「生成压测命令」「bench」「压测客户端」「分并发扫」时使用。
---

# SGLang 压测命令生成(客户端)

## 这个 skill 干什么、不干什么

```
输入(JobSpec + 服务地址)     本 skill(AI 读规则生成)        下游(不在本 skill)
┌──────────────────┐        ┌──────────────────┐        ┌──────────────┐
│ JobSpec           │        │ AI 读 workloads +  │        │ 执行器        │
│  .workload        │        │ benchmark_method + │        │ 逐档跑 bench  │
│  .benchmark_method│───────▶│ 本 knowledge       │───────▶│ 收指标        │
│ + host:port(运行期)│        │ → 每并发档一条命令 │        │ 数据体检      │
│ + model_path(运行期)│       │  (严格照 mapping)  │        │ 算 goodput/回最优│
└──────────────────┘        └──────────────────┘        └──────────────┘
```

**干**:把「workload × benchmark_method × 并发梯度」翻译成一串**分并发**的 `python -m sglang.bench_serving` 压测命令,每档一条,附每条的理由。

**不干**:起服务(那是 server skill 的事)、跑压测、算 SLA、剪枝、出报告(下游执行器第二阶段的事)。本 skill 只出命令文本。

**核心原则:加新经验 = 只改 `knowledge.md` / `catalogs/*.yaml` / `references/benchmark_methods/*.json` 的文本,不改任何代码。**

---

## 输入契约(INPUT)

不新造接口——复用 server skill 用的同一个 `JobSpec`,只取其中压测相关的两个 ID 字段,外加服务端起来后才知道的三个运行期注入值:

| 字段 | 指向 | 提供什么 |
|---|---|---|
| `JobSpec.workload` | `catalogs/workloads.yaml` | 输入长度(`input_tokens.value`)/ 输出长度(`output_tokens.value`)/ 并发梯度(`traffic.values`)/ 采样 |
| `JobSpec.benchmark_method` | `references/benchmark_methods/<method>.json` | 方法定义:`entrypoint` / `fixed_args` / `argument_mapping` / `traffic.num_prompts_multiplier` / `derived_args` / `notes` |
| `host`(运行期注入) | 服务端起来后才知道 | bench 连的目标主机,命令里用占位 `${BENCHMARK_HOST}` |
| `port`(运行期注入) | 同上 | 目标端口,占位 `${BENCHMARK_PORT}` |
| `model_path`(运行期注入) | 同上 | `--model` 用的模型标识,占位 `${MODEL_PATH}` |

**⚠️ 两条铁律(见 `knowledge.md`):**
- 压测命令里**绝不写 `--context-length`**(§0)——压测复刻生产,生产不写则压测不写。
- **`--backend` 恒为 `sglang`**(§1)——走 `/generate`;走 `/v1/chat/completions` 会把 thinking 模型的 decode 吞吐算成 0。

---

## AI 生成时读哪些文件、按什么步骤

**读(按序,路径均相对项目根)**:① 本 `SKILL.md` → ② `.claude/skills/sglang-client-config-gen/knowledge.md`(全部压测经验判据)→ ③ `catalogs/workloads.yaml`(按 `JobSpec.workload` 取对应卡片)→ ④ `references/benchmark_methods/<JobSpec.benchmark_method>.json`(取方法定义与 `argument_mapping`)。

> 路径说明:`knowledge.md` 与本 SKILL.md 同目录(`.claude/skills/sglang-client-config-gen/`);`catalogs/`、`references/` 在项目根下,与 server skill **共享同一份**。`claude` 从项目根运行,按上述相对路径即可读到。

**步骤**:

1. **解析 JobSpec** → 取 `workload` 与 `benchmark_method` 两个 ID。
2. **取输入/输出长度**:从 workload 卡读 `input_tokens.value`(→ `--random-input-len`)与 `output_tokens.value`(→ `--random-output-len`)。
3. **取并发梯度**:优先取 workload 卡的 `traffic.values`;缺失时回落到 benchmark_method 的 `traffic.concurrency_values`(默认 `[1,2,4,8,16,32]`)。
4. **每档算 num_prompts**:`num_prompts = concurrency × num_prompts_multiplier`(multiplier 从 benchmark_method 的 `traffic.num_prompts_multiplier` 取,默认 4)。见 `knowledge.md §3`——**每档请求量随并发等比,结果才可比;别所有档固定同一 num_prompts。**
5. **按 `argument_mapping` 拼命令**:严格照 benchmark_method json 的 `argument_mapping` / `fixed_args` / `runtime_args`,一个 flag 一个 flag 落位(见 `knowledge.md §6`)。**别自己发明 flag**——mapping 里没有的 flag 不许出现。
6. **附 reason**:每条写清这一档为什么这么设(并发/num_prompts 的来由、可追溯到 knowledge.md 哪条)。

---

## 输出契约(OUTPUT)

一个 `benchmark_commands` 数组,**每个并发档一条**,每条:

| 字段 | 内容 |
|---|---|
| `concurrency` | 该档并发度(→ `--max-concurrency`) |
| `num_prompts` | `concurrency × multiplier`(→ `--num-prompts`) |
| `command` | 完整一行 `python -m sglang.bench_serving ...`,含 `--backend sglang` `--host` `--port` `--model` `--dataset-name random-ids` `--random-input-len` `--random-output-len` `--max-concurrency` `--num-prompts` `--random-range-ratio 1.0` `--output-file`(**绝不含 `--context-length`**) |
| `reason` | 这一档为什么这么设(可追溯到 knowledge.md) |

下游:执行器逐档跑这些命令、收指标、做数据体检(`knowledge.md §5`)、在 SLA 约束下算 goodput(§4),回最优。

### 真实样例(workload=`qa-chat-3.5k-1k`,input-len 3500 / output-len 1000,multiplier 4)

并发 1(num_prompts = 1×4 = 4):

```
python -m sglang.bench_serving --backend sglang --host ${BENCHMARK_HOST} --port ${BENCHMARK_PORT} --model ${MODEL_PATH} --dataset-name random-ids --random-input-len 3500 --random-output-len 1000 --random-range-ratio 1.0 --max-concurrency 1 --num-prompts 4 --output-file result_${JOB_ID}_${TIMESTAMP}.jsonl
```

并发 32(num_prompts = 32×4 = 128):

```
python -m sglang.bench_serving --backend sglang --host ${BENCHMARK_HOST} --port ${BENCHMARK_PORT} --model ${MODEL_PATH} --dataset-name random-ids --random-input-len 3500 --random-output-len 1000 --random-range-ratio 1.0 --max-concurrency 32 --num-prompts 128 --output-file result_${JOB_ID}_${TIMESTAMP}.jsonl
```

> 完整梯度 `[1,2,4,8,16,32]` → num_prompts `[4,8,16,32,64,128]`,共 6 条命令,逐档各执行一次(benchmark_method `notes`:每个并发档位执行一次 bench_serving)。

---

## 硬约束 / 红线

1. **绝不 `--context-length`**——压测复刻生产,生产不写则压测不写(`knowledge.md §0`)。
2. **`--backend` 恒 `sglang`**——thinking 模型走 `/v1/chat/completions` 时 `delta.content` 可能为空,会把 decode 吞吐算成 0(`knowledge.md §1` 数据体检)。
3. **续行反斜杠后不能有空格**——若把命令折成多行,`\` 后紧跟换行,末尾一个空格就断(benchmark_method `notes` 第三条)。样例给成单行即为规避此坑。
4. **`num_prompts` 必须 = 并发 × multiplier**——否则每档实际请求数不可比,梯度失去意义(`knowledge.md §3`)。
5. **host / port / model 是运行期注入**——服务端起来后才知道,skill 用占位 `${BENCHMARK_HOST}` / `${BENCHMARK_PORT}` / `${MODEL_PATH}`(或明确标注「需注入」),不许在 skill 里写死某个具体地址。
6. **flag 必须在 benchmark_method 的 `argument_mapping` 里有对应**——不在 mapping 里的 flag 一律不生成(`knowledge.md §6`)。

---

## 目录

```
catalogs/                                 # 共享、引擎无关(server skill 也复用),在项目根
  gpu.yaml  models.yaml  workloads.yaml
references/benchmark_methods/             # 压测方法权威定义,在项目根
  sglang_bench_serving.json
.claude/skills/sglang-client-config-gen/  # 本 skill(claude 从这里加载)
  SKILL.md        # 本文件:入口(读什么、步骤、输出、红线)
  knowledge.md    # 全部压测经验+判据(改经验改这里)
```

## 姊妹 skill

- **`sglang-server-config-gen`**(出服务端启动配置):同一个 `JobSpec`、同一套 `catalogs/`;它出 `server_command`(带 tp/attention/mem-fraction 等启动轴),本 skill 出 `benchmark_commands`(分并发压测)。
- **engine 路由 + 阶段路由**:`JobSpec.engine == "sglang"` → 这两个 sglang skill;其中**起服务阶段**用 `sglang-server-config-gen`,**压测阶段**(服务端已就绪、host:port 已知)路由到**本 skill**。将来 `engine:"vllm"` → 新建对应 skill,复用同一套 `catalogs/`。
