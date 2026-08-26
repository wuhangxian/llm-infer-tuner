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

# Load .env for claude API credentials
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

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
#
#   1. job.json 的完整内容
#      — 模型/卡型/负载/SLA/镜像等,让 AI 知道这次要调什么
#
#   2. 调优硬约束
#      — 块量化×TP 必须整除,超出 tp_max 的 tp 不生成
#      — 绝不写 --context-length(§0 红线)
#      — 模型专属 flag(reasoning-parser 等)从 models.yaml 原样取,不自己编
#      — attention 后端必须在 SM 短名单内
#      — 候选数 ≤ max_candidates
#
#   3. 输出 JSON Schema
#      — 约束 claude 返回 {candidates:[...]} 结构
#      — 每条候选含 id + params + cmd + reasons 四个字段
#      — 第 5 步 jq 再拆成一行一候选的 JSONL
#
# prompt 里还告诉 claude 该按顺序读 4 个知识库文件:
#   ① SKILL.md     — 流程入口:该读什么、推导步骤、输出契约
#   ② knowledge.md — 全部调优经验(§0-§9)
#   ③ catalogs/    — gpu.yaml + models.yaml + workloads.yaml(按 job 里的 ID 查表)
#   ④ catalogs/sglang-images.yaml  — 镜像信息(CUDA 版本、支持的 attention 后端)
#
# --model-path 在这里写死为 ${MODEL_PATH} 占位符,不绑定任何机器路径,
# 第二步由 targets.json 填入实际路径。
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

## 基线配置(可选)

如果 JobSpec 里有 \`baseline\` 字段,你必须在 candidates 数组最前面插入一条基线候选:
- id 为 "baseline"
- params 里加 \`"is_baseline": true\`
- params 的 tp_size/attention_backend/mem_fraction_static 取 baseline 字段指定的值
- 其余参数取默认值(和基线候选的 §4 pin 一致)
- cmd 拼成完整的启动命令
- 如果没有 baseline 字段,正常生成候选即可,不插基线

## 执行方式

请按 \`${SKILL_DIR}/SKILL.md\` 的流程执行:读 knowledge.md + catalogs/*.yaml(含 sglang-images.yaml),
按其中的约束和推导步骤生成候选配置。所有调优判据、硬约束、输出格式
都在 SKILL.md 和 knowledge.md 里,这里不重复。

## 运行时信息(知识库里没有的)

- --model-path 用占位符 \`${MODEL_PATH}\`,不绑定任何机器路径(第二步由 targets.json 填入)。
- --host 0.0.0.0 --port 30000。
- store_true 类 flag(如 --trust-remote-code)为 true 时写裸 flag、false 时省略。
- 候选数 ≤ JobSpec 的 max_candidates。
EOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 步:调 claude(AI 决策,约 2 分钟)
# ─────────────────────────────────────────────────────────────────────────
# 用 claude -p "$PROMPT" 非交互模式调用 AI。关键参数:
#   --add-dir .           让 claude 能读项目根下的 catalogs/ 等
#   --add-dir $SKILL_DIR  让 claude 能读 SKILL.md/knowledge.md
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

# 保存 claude 原始返回到 claude-raw-outputs/(方便调试和回溯,不进 git)
RAW_DIR="claude-raw-outputs"
mkdir -p "$RAW_DIR"
echo "$RAW" > "$RAW_DIR/${JOB_ID}.json"
echo "ℹ️  原始返回 → $RAW_DIR/${JOB_ID}.json" >&2

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
# 用 wc -l 数生成了多少条候选,再用 jq 从每行里取 id/tp/ep/attention/
# mem-fraction 拼成一行预览打印到 stderr,让用户一眼看到结果概貌,
# 不用手动 cat 文件。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
N="$(wc -l < "$OUT" | tr -d ' ')"
echo "✅ 已生成 $N 条候选 → $OUT" >&2
echo "── 预览(id / tp / ep / att / mf / mamba / page / spec / chunk / kv / sched / radix)──" >&2
jq -r '
  "  " + .id +
  "  tp=" + (.params.tp_size|tostring) +
  "  ep=" + (.params.ep_size // 1 | tostring) +
  "  " + (.params.attention_backend // "-(default)") +
  "  mf=" + (.params.mem_fraction_static|tostring) +
  "  mamba=" + (.params.mamba_radix_cache_strategy // .params.mamba_scheduler_strategy // .params["mamba-radix-cache-strategy"] // "no_buffer(default)") +
  "  page=" + (.params.page_size // .params["page-size"] // "1(default)" | tostring) +
  "  spec=" + (.params.speculative_algorithm // .params["speculative-algorithm"] // .params.speculative // "none(default)") +
  "  chunk=" + (.params.chunked_prefill_size // .params["chunked-prefill-size"] // "8192(default)" | tostring) +
  "  kv=" + (.params.kv_cache_dtype // .params["kv-cache-dtype"] // "auto(default)") +
  "  sched=" + (.params.schedule_conservativeness // .params["schedule-conservativeness"] // "1.0(default)" | tostring) +
  "  radix=off"
' "$OUT" >&2
