#!/bin/bash
# gen_configs.sh —— 第一阶段:生成 SGLang 启动配置候选。
#
# 用法:  ./gen_configs.sh <job.json> [out.jsonl]
#        默认输出 outputs/<job_id>/configs.jsonl。
#
# 脚本分 6 步,其中只有第 4 步调 AI(claude)做调优决策,其余全是确定性代码。
# cmd 里的 --model-path 恒为占位符 ${MODEL_PATH}(路径是机器事实,非调优决策),
# 第二步由 targets.json 填入。故 configs.jsonl 机器无关,可跨机搬运。
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 步:前置检查
# ─────────────────────────────────────────────────────────────────────────
# 脚本启动前先确认四样东西在不在,任何一个缺了就直接退出,不往下跑:
#   • jq        — 命令行 JSON 处理工具,后面第 2/5/6 步都要用它读/写/解析 JSON
#   • job.json  — 你传进来的参数路径对不对,文件在不在
#   • SKILL.md  — AI 的流程说明书,在 .claude/skills/sglang-server-config-gen/ 下
#   • claude    — AI CLI,第 4 步要用它生成配置,没装就跑不了
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOB="${1:?用法: ./gen_configs.sh <job.json> [out.jsonl]}"
command -v jq >/dev/null || { echo "❌ 需要 jq" >&2; exit 1; }
[ -f "$JOB" ] || { echo "❌ job 文件不存在: $JOB" >&2; exit 1; }

SKILL_DIR=".claude/skills/sglang-server-config-gen"
[ -f "$SKILL_DIR/SKILL.md" ] || { echo "❌ 找不到 skill: $SKILL_DIR/SKILL.md(请在 repo 根运行)" >&2; exit 1; }
command -v claude >/dev/null || { echo "❌ claude 不在 PATH" >&2; exit 1; }

# 模型路径恒为占位符 —— 第一步机器无关,不绑死任何机器的物理路径。
MODEL_PATH='${MODEL_PATH}'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 步:解析 job.json
# ─────────────────────────────────────────────────────────────────────────
# 用 jq 从 job.json 里读出 job_id 字段(如 qwen36-27b-...cand4),
# 拼成输出路径 outputs/<job_id>/configs.jsonl,然后 mkdir -p 创建输出目录。
# 如果用户传了第二个参数,就用那个路径覆盖默认路径。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOB_ID="$(jq -r '.job_id' "$JOB")"
OUT="${2:-outputs/${JOB_ID}/configs.jsonl}"
mkdir -p "$(dirname "$OUT")"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 步:拼接 prompt
# ─────────────────────────────────────────────────────────────────────────
# 把三部分组装成一段完整的 prompt,后面第 4 步把它喂给 claude:
#   1. job.json 的完整内容(模型/卡型/负载/SLA/镜像)
#   2. 调优硬约束(块量化×TP 整除、绝不写 context-length、模型 flag 原样取等)
#   3. 输出 JSON Schema(约束 claude 返回 {candidates:[...]} 结构)
# prompt 里还告诉 claude 该按顺序读 4 个知识库文件:
#   SKILL.md → knowledge.md → catalogs/*.yaml → images.yaml
# --model-path 在这里写死为 ${MODEL_PATH} 占位符,不绑定任何机器路径。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOB_JSON="$(cat "$JOB")"

# 输出结构约束:一个 {candidates:[...]} 对象,第 5 步 jq 再拆成 jsonl
SCHEMA='{"type":"object","required":["candidates"],"properties":{"candidates":{"type":"array","items":{"type":"object","required":["id","params","cmd","reasons"],"properties":{"id":{"type":"string"},"params":{"type":"object"},"cmd":{"type":"string"},"reasons":{"type":"array","items":{"type":"string"}}}}}}}'

read -r -d '' PROMPT <<EOF || true
# llm-infer-tuner 一步出 SGLang 启动配置(JSONL)

你要为下面这个 job 生成一组**可直接执行**的 SGLang 启动配置候选。

## JobSpec
\`\`\`json
${JOB_JSON}
\`\`\`

## 权威知识库(必须按顺序读,再产出任何参数)
1. \`${SKILL_DIR}/SKILL.md\` —— 流程入口:该读什么、推导步骤、输出契约、3 道硬闸。
2. \`${SKILL_DIR}/knowledge.md\` —— **所有调优判断都在这**(§0 绝不写 context-length;§1 按算力选 attention;§2 TP/PP/EP 推导,**含块量化×TP 整除硬约束**;§3 搜索空间;§4 pin;§5 排除项;§6 阶段顺序;§7 混合 mamba;§8 调度;§9 并发)。每条 reason 里注明你依据的章节。
3. \`catalogs/*.yaml\` —— 按 JobSpec 点名的卡:gpu.yaml(gpu_model→sm/nvlink;gpu_count/gpu_memory_gb 在 JobSpec 里)、models.yaml(model→arch/is_moe/num_experts/moe_intermediate_size/quantization.block_size/weight_gb/default_flags)、workloads.yaml(workload→输入输出长度/并发梯度)。
4. \`${SKILL_DIR}/images.yaml\` —— JobSpec.image 点名的镜像卡(CUDA、attention 菜单、valid_flags 白名单)。

## 硬要求
- **块量化×TP 整除**(knowledge.md §2):块量化(fine-grained FP8/AWQ/GPTQ)下,MoE 每卡专家维度 = moe_intermediate_size/tp,须整除 block_size;\`tp_max=floor(moe_intermediate_size/block_size)\`。**超出 tp_max 的 tp 直接不生成**(会在加载权重时崩,不是压测 OOM)。
- **绝不写 \`--context-length\`**(§0 红线),pin 与搜索空间里都不许出现。
- 模型专属 flag(reasoning-parser/tool-call-parser/trust-remote-code)从 models.yaml 的 default_flags 原样取,别自己编。
- 只用 SGLang 真实参数名,别造 flag;attention 后端须在该 SM 的 shortlist 内(§1)。
- 候选数 ≤ JobSpec 的 max_candidates。

## 输出(严格按 schema,只返回 JSON 对象,不要 markdown)
返回 \`{"candidates":[...]}\`。每个候选:
- \`id\`:如 "c001"。
- \`params\`:结构化决策值(tp_size/pp_size/attention_backend/mem_fraction_static + pin 的模型 flag)。
- \`cmd\`:**完整可直接执行的一行命令**,格式:
  \`python -m sglang.launch_server --model-path ${MODEL_PATH} <各 flag> --host 0.0.0.0 --port 30000\`
  store_true 类 flag(如 --trust-remote-code)为 true 时写裸 flag、false 时省略;--model-path 用上面给的路径。
- \`reasons\`:每条注明依据的 knowledge.md 章节(尤其 tp 为何是这些、为何砍掉更大的 tp)。
EOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 步:调 claude(AI 决策,约 2 分钟)
# ─────────────────────────────────────────────────────────────────────────
# 用 claude -p "$PROMPT" 非交互模式调用 AI。关键参数:
#   --add-dir .           让 claude 能读项目根下的 catalogs/ 等
#   --add-dir $SKILL_DIR  让 claude 能读 SKILL.md/knowledge.md/images.yaml
#   --json-schema         约束 claude 返回 {candidates:[...]} 结构
#   --output-format json  让 claude 输出 JSON(而非纯文本)
# AI 读知识库 → 查表(gpu/model/workload/image)→ 推导 TP/attention/mem-fraction
# → 生成一批候选,每条含 params + cmd + reasons。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "ℹ️  模型路径在命令里保留占位符 \${MODEL_PATH}(机器无关);第二步由 targets.json 填入实际路径。" >&2
echo "ℹ️  输出 → $OUT" >&2
echo "▶ 调 claude 生成配置(读 skill+knowledge+catalogs,几分钟)…" >&2
RAW="$(claude -p "$PROMPT" \
  --output-format json \
  --json-schema "$SCHEMA" \
  --add-dir . \
  --add-dir "$SKILL_DIR" \
  --dangerously-skip-permissions)"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 步:拆 JSONL
# ─────────────────────────────────────────────────────────────────────────
# claude 返回的是一整个 JSON 对象,需要 jq 拆成一行一候选(JSONL 格式):
#   • claude --output-format json 把结果包在 .result 或 .structured_output 里
#   • .result 可能是字符串(需 fromjson 解开),也可能是对象(直接用)
#   • .structured_output 是结构化输出(优先取)
#   • .candidates[] 遍历数组每个元素逐个输出,-c 压成一行
#   • -e 如果结果为 null/false 就报错退出(校验 claude 返回了有效数据)
# 最终写入 configs.jsonl,每行一个独立 JSON,方便第二步 executor 逐行读取。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "$RAW" | jq -e '
  ( .structured_output // (.result | if type=="string" then fromjson else . end) // . )
  | .candidates[]
' -c > "$OUT"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 6 步:预览输出
# ─────────────────────────────────────────────────────────────────────────
# 用 wc -l 数生成了多少条候选,再用 jq 从每行里取 id/tp/attention/
# mem-fraction 拼成一行预览打印到 stderr,让用户一眼看到结果概貌,
# 不用手动 cat 文件。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
N="$(wc -l < "$OUT" | tr -d ' ')"
echo "✅ 已生成 $N 条候选 → $OUT" >&2
echo "── 预览(id / tp / attention / mem-fraction)──" >&2
jq -r '"  \(.id)  tp=\(.params.tp_size)  \(.params.attention_backend)  mf=\(.params.mem_fraction_static)"' "$OUT" >&2
