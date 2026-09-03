#!/bin/bash
# gen_configs.sh —— 第一阶段:生成 SGLang 启动配置候选。
#
# 用法:  ./gen_configs.sh [--agent tclaude|claude] [--model MODEL] <job.json> [out.json]
#        默认输出 outputs/<job_id>/configs.jsonl。
#
# --agent 选用哪个 Claude Code CLI:
#   tclaude(默认)—— 腾讯内网网关,默认模型别名 claude-hy3(仅腾讯员工可用)
#   claude        —— 公开 CLI,走 `claude login` 或 ANTHROPIC_API_KEY;不指定
#                    --model 时用 claude 自身默认模型(不硬塞腾讯别名)
# tclaude 与 claude 命令行契约完全一致(tclaude 内嵌 @anthropic-ai/claude-code),
# 故只需切 binary 名与默认模型,其余逻辑(stream-json / json-schema 解析)通用。
#
# 脚本分 6 步,其中只有第 4 步调 AI 做调优决策,其余全是确定性代码。
# cmd 里的 --model-path 恒为占位符 ${MODEL_PATH}(路径是机器事实,非调优决策),
# 第二步由 targets.json 填入。故 configs.jsonl 机器无关,可跨机搬运。
set -euo pipefail

# 记录整个脚本的墙钟起点(从这里到第 6 步预览完的总耗时)。
# 用 bash 内置 SECONDS:赋 0 即从此刻开始按秒累计,零依赖、不受子进程影响。
SECONDS=0

usage() {
  echo "用法: ./gen_configs.sh [--agent tclaude|claude] [--model MODEL] <job.json> [out.json]" >&2
}

# Load .env for optional API-related environment settings.
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 步:前置检查
# ─────────────────────────────────────────────────────────────────────────
# 脚本启动前确认依赖和输入都在,任何一个缺了就直接退出,不往下跑:
#   • jq        — 命令行 JSON 处理工具,后面第 2/5/6 步都要用它读/写/解析 JSON
#   • job.json  — 你传进来的参数路径对不对,文件在不在
#   • SKILL.md  — AI 的流程说明书,在 .claude/skills/sglang-server-config-gen/ 下
#   • tclaude   — AI CLI,第 4 步要用它生成配置,没装就跑不了
#   • python3 + tclaude_guard.py — 超时/重试/信号与子进程回收
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT="tclaude"          # 默认走腾讯内网 CLI,保持既有行为不变
MODEL=""                 # 空 = 用户未显式指定,解析后按 agent 决定默认值
MODEL_SOURCE=""
POSITIONAL=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --agent)
      if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        echo "❌ --agent 需要值(tclaude 或 claude)" >&2
        usage
        exit 2
      fi
      AGENT="$2"
      shift 2
      ;;
    --agent=*)
      AGENT="${1#--agent=}"
      if [ -z "$AGENT" ]; then
        echo "❌ --agent 需要值(tclaude 或 claude)" >&2
        usage
        exit 2
      fi
      shift
      ;;
    --model)
      if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        echo "❌ --model 需要非空模型名" >&2
        usage
        exit 2
      fi
      MODEL="$2"
      MODEL_SOURCE="命令行"
      shift 2
      ;;
    --model=*)
      MODEL="${1#--model=}"
      if [ -z "$MODEL" ]; then
        echo "❌ --model 需要非空模型名" >&2
        usage
        exit 2
      fi
      MODEL_SOURCE="命令行"
      shift
      ;;
    --)
      shift
      while [ "$#" -gt 0 ]; do
        POSITIONAL+=("$1")
        shift
      done
      ;;
    -*)
      echo "❌ 未知选项: $1" >&2
      usage
      exit 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [ "${#POSITIONAL[@]}" -lt 1 ] || [ "${#POSITIONAL[@]}" -gt 2 ]; then
  usage
  exit 2
fi

JOB="${POSITIONAL[0]}"
OUT_ARG="${POSITIONAL[1]:-}"

# 校验 --agent 只能是 tclaude / claude(挡住 codex 等尚未支持的值,免得组出无效命令)
case "$AGENT" in
  tclaude|claude) ;;
  *)
    echo "❌ 未知 --agent: $AGENT(仅支持 tclaude | claude)" >&2
    usage
    exit 2
    ;;
esac

# 默认模型按 agent 决定:
#   tclaude —— 补腾讯网关别名 claude-hy3(公开 claude 不认这个别名)
#   claude  —— 不补默认模型,交给 claude 自身默认(用户可用 --model 覆盖)
if [ -z "$MODEL_SOURCE" ]; then
  if [ "$AGENT" = "tclaude" ]; then
    MODEL="claude-hy3"
    MODEL_SOURCE="默认"
  else
    MODEL_SOURCE="claude 默认"
  fi
fi

# 只有拿到具体模型名时才传 --model;claude 未指定时留空,用其自身默认
AGENT_MODEL_ARGS=()
if [ -n "$MODEL" ]; then
  AGENT_MODEL_ARGS=(--model "$MODEL")
fi

command -v jq >/dev/null || { echo "❌ 需要 jq" >&2; exit 1; }
command -v python3 >/dev/null || { echo "❌ 需要 python3" >&2; exit 1; }
[ -f "$JOB" ] || { echo "❌ job 文件不存在: $JOB" >&2; exit 1; }

SKILL_DIR=".claude/skills/sglang-server-config-gen"
[ -f "$SKILL_DIR/SKILL.md" ] || { echo "❌ 找不到 skill: $SKILL_DIR/SKILL.md(请在 repo 根运行)" >&2; exit 1; }
RULES_DIR="$SKILL_DIR/references/rules"
command -v "$AGENT" >/dev/null || { echo "❌ $AGENT 不在 PATH" >&2; exit 1; }
TCLAUDE_GUARD="runners/tclaude_guard.py"
[ -f "$TCLAUDE_GUARD" ] || { echo "❌ 找不到 guard: $TCLAUDE_GUARD" >&2; exit 1; }

# 只校验知识库格式，不把实验候选提前拦掉；是否能启动以目标 GPU 实测为准。
if [ -f scripts/validate_knowledge.py ]; then
  python3 scripts/validate_knowledge.py >/dev/null || {
    echo "❌ 规则库校验失败，请先修复 references/rules/*.yaml" >&2
    exit 1
  }
fi

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
# 默认输出名必须与 run_executor.sh 默认查找的路径(outputs/<job_id>/configs.jsonl)一致,
# 否则默认两步流程会在 run_executor 的存在性检查处报「configs 不存在」直接退出。
# 内容是单个 {candidates:[...]} JSON 对象,executor._load_candidates 两种格式都能读。
OUT="${OUT_ARG:-outputs/${JOB_ID}/configs.jsonl}"
mkdir -p "$(dirname "$OUT")"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 步:拼接 prompt
# ─────────────────────────────────────────────────────────────────────────
# 把三部分组装成一段完整的 prompt,后面第 4 步把它喂给 tclaude:
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
#      — 约束 tclaude 返回 {candidates:[...]} 结构
#      — 每条候选含 id + params + cmd + reasons 四个字段
#      — 第 5 步 jq 再拆成一行一候选的 JSONL
#
# prompt 里还告诉 agent 按顺序读入口、规则索引、主题规则和 catalogs:
#   ① SKILL.md → ② knowledge.md + references/rules/README.md
#   ③ 按 JobSpec 读取相关 references/rules/*.yaml
#   ④ catalogs/ 下按 ID 查 GPU/model/workload/image 事实
#
# --model-path 在这里写死为 ${MODEL_PATH} 占位符,不绑定任何机器路径,
# 第二步由 targets.json 填入实际路径。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOB_JSON="$(cat "$JOB")"

# 输出结构约束:一个 {candidates:[...]} 对象
SCHEMA='{"type":"object","required":["candidates"],"properties":{"candidates":{"type":"array","items":{"type":"object","required":["id","params","cmd","reasons"],"properties":{"id":{"type":"string"},"params":{"type":"object"},"cmd":{"type":"string"},"reasons":{"type":"array","items":{"type":"string"}}}}}}}'

read -r -d '' PROMPT <<EOF || true
# llm-infer-tuner 一步出 SGLang 启动配置(JSON)

你要为下面这个 job 生成一组**可直接执行**的 SGLang 启动配置候选。

## JobSpec
\`\`\`json
${JOB_JSON}
\`\`\`

## 基线配置(可选)

如果 JobSpec 里有 \`baseline\` 字段(在 search.baseline 下):
- 你必须在 candidates 数组**最前面**插入一条基线候选,id 为 "baseline",params 里加 \`"is_baseline": true\`
- **严格模式(重要)**:基线是"用户基线复现",不是调优候选。params **只放** baseline 字段里用户显式写的参数,原样保留、不增不改(\`is_baseline\` 标记除外)。
  用户没写的参数**一律不补**——不补 §4 pin(含 --schedule-policy 等)、不补 §3 各轴搜索默认档、不补 §4 default_flags 里的 parser 类 flag。让 SGLang 用它自己的内部默认(即命令里根本不出现该 flag)。
- **唯一例外(硬启动依赖,不补则 server 直接崩,故必须补)**:
  1. 若 catalogs/models.yaml 该模型 \`default_flags\` 含 \`trust-remote-code: true\`,即使用户没写也必须补 \`--trust-remote-code\`(CLI 安全开关,模型默认无法翻成 true)。
  2. 若用户写了 \`mamba_radix_cache_strategy: extra_buffer\` 但没写 \`page_size\`,必须补 \`--page-size 64\`(\`FLA_CHUNK_SIZE(64) % page_size == 0\` 硬约束,否则启动报错)。
  这两个例外若要补,必须同时写进 baseline 的 params(不能只写进 cmd),使 executor 兜底路径也一致。除此之外不补任何默认 flag。
- 基线的 cmd 严格 = {用户在 baseline 里列的 flag} + {上面必要的硬启动依赖例外} + {运行时占位符:\`--model-path \${MODEL_PATH}\` \`--host 0.0.0.0\` \`--port 30000\`}。用户写的某 flag 值恰好等于模型默认时,仍照写(尊重显式意图)。
- **基线不算在 max_candidates 名额里**,总候选数 = max_candidates + 1
- 你正常生成 max_candidates 条候选,排在基线后面(**这 max_candidates 条是正常调优候选,仍按 §3/§4 正常补全 pin 和 default_flags,严格模式只约束 baseline 那一条**)

如果没有 baseline 字段:正常生成 max_candidates 条候选(第一条是基线锚点),不额外插基线。

## 公平性硬约束

- 所有候选(含 baseline)实际执行时统一关闭 Radix/Prefix Cache；executor 会覆盖用户输入并确保启动命令带且仅带一个 \`--disable-radix-cache\`。
- \`mamba_radix_cache_strategy\` 可保留为用户请求值用于审计，但在 Radix Cache 关闭时记为 effective=inactive，不作为候选间有效调优轴。

## 执行方式

请按 \`${SKILL_DIR}/SKILL.md\` 的流程执行:先读 knowledge.md 和
references/rules/README.md，再按 JobSpec 读取 attention/parallelism/memory/speculative/
scheduling/fairness 相关 YAML，最后读取 catalogs/*.yaml(含 sglang-images.yaml)。
投机解码必须把模型卡的 speculative_options/mtp_params 与镜像卡的
speculative_algorithms 取交集，NONE 保留为对照，不能只写死 EAGLE。
所有普通调优判据和输出格式都在 SKILL.md、规则文件和 catalogs 里，这里不重复。
规则是决策依据和风险说明，不是生成阶段的候选硬闸；experimental 或资料不完整的候选
可以生成并交给执行器实测。

## 探索边界(必须遵守,直接影响耗时)

- **输出格式的唯一权威是本 prompt 末尾的 JSON Schema 和 SKILL.md**。禁止为"对齐格式"去读 \`outputs/\`、\`claude-raw-outputs/\` 里任何历史生成结果(configs.json/configs.jsonl/ranking.json/*.jsonl raw)。那些是过往产物、可能过时或来自不同 job,不是格式标准,参照它们只会拖慢并引入不一致。
- **只读你推导所必需的输入**:SKILL.md、knowledge.md、references/rules/*.yaml、catalogs/*.yaml(按 job 里的 ID 查表)。不要浏览 sibling job、不要翻别的 job 的 job.json/结果、不要 git log。本 job 的 JobSpec 已在上文给全。
- **不要参考任何"之前跑过的类似 job"的候选或压测排名来做本次决策**。每个 job 独立按知识库推导;历史结果不构成本次判据。
- 拿到查表所需事实后**尽快进入推导与产出**,不要做超出上述范围的探索性文件浏览。

## 运行时信息(知识库里没有的)

- --model-path 用占位符 \`${MODEL_PATH}\`,不绑定任何机器路径(第二步由 targets.json 填入)。
- --host 0.0.0.0 --port 30000。
- store_true 类 flag(如 --trust-remote-code)为 true 时写裸 flag、false 时省略。
- 候选数 ≤ JobSpec 的 max_candidates。
EOF

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 步:调 tclaude(AI 决策,约 2 分钟)
# ─────────────────────────────────────────────────────────────────────────
# 用 tclaude -p "$PROMPT" 非交互模式调用 AI。关键参数:
#   --add-dir .           让 tclaude 能读项目根下的 catalogs/ 等
#   --add-dir $SKILL_DIR  让 tclaude 能读 SKILL.md/knowledge.md
#   --json-schema         约束 tclaude 返回 {candidates:[...]} 结构
#   --output-format json  让 tclaude 输出 JSON(而非纯文本)
# AI 读知识库 → 查表(gpu/model/workload/image)→ 推导 TP/attention/mem-fraction
# → 生成一批候选,每条含 params + cmd + reasons。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEN_STREAM=1(默认)走真流式 stream-json:边生成边把「读了哪个文件 / 在想什么 /
#   正在产出候选」实时打到命令行,不再是黑盒;完整事件流(NDJSON)落盘便于回溯。
# GEN_STREAM=0 回退旧行为:--output-format json,一次性返回单个 JSON 对象。
# 两条路径都保留 --json-schema(已实测:stream-json 末尾 result 事件仍带 structured_output)。
GEN_STREAM="${GEN_STREAM:-1}"
GEN_TIMEOUT_SECONDS="${GEN_TIMEOUT_SECONDS:-600}"
GEN_TIMEOUT_GRACE_SECONDS="${GEN_TIMEOUT_GRACE_SECONDS:-10}"
GEN_MAX_RETRIES="${GEN_MAX_RETRIES:-1}"

validate_decimal_range() {
  local name="$1" value="$2" minimum="$3" maximum="$4"
  if ! [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "❌ $name 必须是十进制整数(不允许前导零): $value" >&2
    return 2
  fi
  local decimal_value=$((10#$value))
  if [ "$decimal_value" -lt "$minimum" ] || [ "$decimal_value" -gt "$maximum" ]; then
    echo "❌ $name 必须在 $minimum..$maximum 之间: $value" >&2
    return 2
  fi
}

validate_decimal_range GEN_TIMEOUT_SECONDS "$GEN_TIMEOUT_SECONDS" 1 86400 || exit $?
validate_decimal_range GEN_TIMEOUT_GRACE_SECONDS "$GEN_TIMEOUT_GRACE_SECONDS" 1 300 || exit $?
validate_decimal_range GEN_MAX_RETRIES "$GEN_MAX_RETRIES" 0 10 || exit $?
MAX_ATTEMPTS=$((10#$GEN_MAX_RETRIES + 1))

RAW_DIR="claude-raw-outputs"
mkdir -p "$RAW_DIR"

SUCCESS_PATH_FILE="$(mktemp "$RAW_DIR/.${JOB_ID}.success.XXXXXX")"
rm -f -- "$SUCCESS_PATH_FILE"
OUT_TMP=""
INTERRUPTED=0

cleanup_temporary_files() {
  local rc=$?
  [ -z "$SUCCESS_PATH_FILE" ] || rm -f -- "$SUCCESS_PATH_FILE"
  [ -z "$OUT_TMP" ] || rm -f -- "$OUT_TMP"
  return "$rc"
}

record_interrupt() {
  INTERRUPTED=1
}

trap cleanup_temporary_files EXIT
trap record_interrupt INT

echo "ℹ️  模型路径在命令里保留占位符 \${MODEL_PATH}(机器无关);第二步由 targets.json 填入实际路径。" >&2
echo "ℹ️  $AGENT 模型 → ${MODEL:-<$AGENT 自身默认>} ($MODEL_SOURCE)" >&2
echo "ℹ️  防护 → 单次软超时 ${GEN_TIMEOUT_SECONDS}s, TERM→KILL 宽限 ${GEN_TIMEOUT_GRACE_SECONDS}s, 最多尝试 ${MAX_ATTEMPTS} 次" >&2
echo "ℹ️  输出 → $OUT" >&2

GUARD_ARGS=(
  python3 "$TCLAUDE_GUARD"
  --timeout-seconds "$GEN_TIMEOUT_SECONDS"
  --grace-seconds "$GEN_TIMEOUT_GRACE_SECONDS"
  --max-retries "$GEN_MAX_RETRIES"
  --raw-dir "$RAW_DIR"
  --job-id "$JOB_ID"
  --success-path-file "$SUCCESS_PATH_FILE"
)

if [ "$GEN_STREAM" != "0" ]; then
  # ── 默认:真流式 ─────────────────────────────────────────────────────
  echo "▶ 调 $AGENT 生成配置(流式,实时显示读库+推导过程,通常 2-6 分钟)…" >&2
  echo "────────────────────────────────────────────────────────────" >&2

  TCLAUDE_COMMAND=(
    "$AGENT" -p "$PROMPT" "${AGENT_MODEL_ARGS[@]}"
    --output-format stream-json
    --verbose
    --include-partial-messages
    --json-schema "$SCHEMA"
    --add-dir .
    --add-dir "$SKILL_DIR"
    --dangerously-skip-permissions
  )

  # set -euo pipefail 下临时关 -e；必须第一时间完整保存 PIPESTATUS，
  # 否则后续任何命令都会覆盖 guard/renderer 的真实退出码。
  set +e
  "${GUARD_ARGS[@]}" --stdout-suffix jsonl --forward-stdout -- "${TCLAUDE_COMMAND[@]}" \
  | jq -R --unbuffered -rj '
      # 每行一个事件 → 只渲染「进度骨架」到 stderr:一个动作一行,不刷屏。
      # -R:把每行当原始字符串读,再 fromjson —— tclaude 每行是完整 JSON,
      # 不加 -R 则 jq 直接解析,fromjson(只吃字符串)会全行失败被 // empty 吞掉。
      #
      # 刻意「只打工具动作 + 首尾标记」,不打模型思考文字:
      #   1) 旧版把 text_delta 增量逐字流出 → 文字一个字一个字蹦;
      #   2) 又用 💭 把同一段话前 140 字再回显一遍 → 每句出现两次、还被截断错行。
      # 两者叠加就是「乱码」观感。用户只想知道「生成到哪一步」,工具动作行(读了
      # 哪个库、写没写文件、有没有产出候选)本身就是步骤,故只留这些;思考文字全删。
      (fromjson? // empty) as $e | $e |
      if .type=="assistant" then
        ( .message.content[]? |
          if .type=="tool_use" then
            ( .name ) as $n |
            ( if   $n=="Read"             then "📖 读 "  + ((.input.file_path // "?") | sub(".*/";""))
              elif $n=="Grep"             then "🔎 搜 "  + (.input.pattern // "")
              elif $n=="Glob"             then "🗂  列 "  + (.input.pattern // "")
              elif $n=="Bash"             then "💻 "     + (.input.description // .input.command // "")
              elif $n=="Write"            then "📝 写候选文件"
              elif $n=="StructuredOutput" then "✍️  产出候选(结构化输出)…"
              else "🔧 " + $n end ) + "\n"
          else empty end )
      elif .type=="result" then
        ( if .is_error then "\n❌ 出错: " + (.subtype // "unknown") + "\n"
          else "\n✅ 生成完毕\n" end )
      else empty end
    ' >&2
  PIPELINE_STATUS=("${PIPESTATUS[@]}")
  set -e
  GUARD_RC="${PIPELINE_STATUS[0]}"
  RENDER_RC="${PIPELINE_STATUS[1]}"

  echo "────────────────────────────────────────────────────────────" >&2
  if [ "$INTERRUPTED" -ne 0 ] || [ "$GUARD_RC" -eq 130 ]; then
    echo "❌ 用户中断 tclaude 生成；不会重试或修改正式输出" >&2
    exit 130
  fi
  if [ "$GUARD_RC" -ne 0 ]; then
    echo "❌ tclaude guard 退出码 $GUARD_RC；上方已列出本次 attempt 日志" >&2
    [ ! -e "$OUT" ] || echo "ℹ️  已有输出未修改: $OUT" >&2
    exit "$GUARD_RC"
  fi
  if [ "$RENDER_RC" -ne 0 ]; then
    echo "❌ 流式进度渲染失败，退出码 $RENDER_RC；正式输出未修改" >&2
    exit "$RENDER_RC"
  fi
else
  # ── 回退:旧的单 JSON 对象行为(GEN_STREAM=0)────────────────────────
  echo "▶ 调 $AGENT 生成配置(非流式,读 skill+knowledge+catalogs,几分钟)…" >&2
  TCLAUDE_COMMAND=(
    "$AGENT" -p "$PROMPT" "${AGENT_MODEL_ARGS[@]}"
    --output-format json
    --json-schema "$SCHEMA"
    --add-dir .
    --add-dir "$SKILL_DIR"
    --dangerously-skip-permissions
  )
  set +e
  "${GUARD_ARGS[@]}" --stdout-suffix json -- "${TCLAUDE_COMMAND[@]}"
  GUARD_RC=$?
  set -e
  if [ "$INTERRUPTED" -ne 0 ] || [ "$GUARD_RC" -eq 130 ]; then
    echo "❌ 用户中断 tclaude 生成；不会重试或修改正式输出" >&2
    exit 130
  fi
  if [ "$GUARD_RC" -ne 0 ]; then
    echo "❌ tclaude guard 退出码 $GUARD_RC；上方已列出本次 attempt 日志" >&2
    [ ! -e "$OUT" ] || echo "ℹ️  已有输出未修改: $OUT" >&2
    exit "$GUARD_RC"
  fi
fi

if [ ! -s "$SUCCESS_PATH_FILE" ] || ! IFS= read -r RAW_FILE < "$SUCCESS_PATH_FILE"; then
  echo "❌ guard 成功但没有写入 success-path: $SUCCESS_PATH_FILE" >&2
  exit 1
fi
if [ ! -s "$RAW_FILE" ]; then
  echo "❌ guard 指向的成功 raw 不存在或为空: $RAW_FILE" >&2
  exit 1
fi
echo "ℹ️  成功 attempt 原始返回 → $RAW_FILE" >&2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 步:拆 JSONL(两种落盘格式统一处理,保留空/坏结果保护)
# ─────────────────────────────────────────────────────────────────────────
# 流式($RAW_FILE 是多行 NDJSON):逐行 fromjson 容错 → 取最后一条 type==result
#   事件 → 沿用旧取值链(structured_output 优先,退回 result 字符串则 fromjson)。
# 回退($RAW_FILE 是单个 JSON 对象):直接走旧取值链。
# 两条路径最终都产出 {candidates:[...]};-e / error() 保证空/坏结果非零退出,
# 先解析进同目录临时文件,成功后原子替换;失败时保留既有正式输出。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUT_DIR="$(dirname -- "$OUT")"
OUT_BASENAME="$(basename -- "$OUT")"
OUT_TMP="$(mktemp "$OUT_DIR/.${OUT_BASENAME}.tmp.XXXXXX")"

if [ "$GEN_STREAM" != "0" ]; then
  if [ ! -s "$RAW_FILE" ]; then
    echo "❌ tclaude 未产生任何输出(NDJSON 为空):$RAW_FILE" >&2
    exit 1
  fi
  # 软预检:找一下 result 事件,找不到只提示、不退出 —— 真正的成败由下面带 -e 的
  # 解析步骤裁决(它取不到 candidates 会非零退出)。
  #   为什么不在这里硬 exit:大候选集(如 cand32/cand64)的 result 事件是一行几万
  #   字符的巨型 JSON,常是全文件最长行。tclaude 流式写盘时,本检测可能卡在「成功
  #   标记已出、巨型行尚未落盘完整」的缝隙里跑,fromjson? 遇半截行被 ? 静默吞掉,
  #   于是误判「没有 result 事件」并 exit —— 结果候选明明生成成功,却丢了预览还报错。
  #   下面第 398 行的 `inputs | fromjson?` 解析是权威裁决,预检不必抢在它前面硬杀。
  if ! jq -R 'fromjson? // empty | select(.type=="result")' "$RAW_FILE" | grep -q .; then
    echo "⚠️  预检未在 NDJSON 里立刻看到 result 事件(大候选集落盘竞态常见),交由解析步骤裁决…" >&2
  fi
  set +e
  jq -R -e '
    [ inputs | fromjson? // empty | select(.type=="result") ] | last
    | if (.is_error == true) then error("tclaude 返回 is_error: \(.subtype)") else . end
    | ( .structured_output // (.result | if type=="string" then fromjson else . end) )
    | {candidates: .candidates}
  ' "$RAW_FILE" > "$OUT_TMP"
  PARSE_RC=$?
  set -e
  if [ "$INTERRUPTED" -ne 0 ] || [ "$PARSE_RC" -eq 130 ]; then
    echo "❌ 用户在候选解析阶段中断；已有输出未修改" >&2
    exit 130
  fi
  if [ "$PARSE_RC" -ne 0 ]; then
    echo "❌ 从最后一条 result 事件解析 candidates 失败(candidates 缺失/为 null?):$RAW_FILE" >&2
    [ ! -e "$OUT" ] || echo "ℹ️  已有输出未修改: $OUT" >&2
    exit 1
  fi
else
  set +e
  jq -e '
    ( .structured_output // (.result | if type=="string" then fromjson else . end) // . )
    | {candidates: .candidates}
  ' "$RAW_FILE" > "$OUT_TMP"
  PARSE_RC=$?
  set -e
  if [ "$INTERRUPTED" -ne 0 ] || [ "$PARSE_RC" -eq 130 ]; then
    echo "❌ 用户在候选解析阶段中断；已有输出未修改" >&2
    exit 130
  fi
  if [ "$PARSE_RC" -ne 0 ]; then
    echo "❌ 解析 candidates 失败:$RAW_FILE" >&2
    [ ! -e "$OUT" ] || echo "ℹ️  已有输出未修改: $OUT" >&2
    exit 1
  fi
fi

mv -f -- "$OUT_TMP" "$OUT"
OUT_TMP=""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 6 步:预览输出
# ─────────────────────────────────────────────────────────────────────────
# 用 wc -l 数生成了多少条候选,再用 jq 从每行里取 id/tp/ep/attention/
# mem-fraction 拼成一行预览打印到 stderr,让用户一眼看到结果概貌,
# 不用手动 cat 文件。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 候选数必须用 jq 数 .candidates 长度,不能用 wc -l:输出是多行 pretty-print JSON
# (一条候选占十几行),wc -l 会把总行数(如 33 条≈620 行)误当成候选数。
N="$(jq '.candidates | length' "$OUT")"
echo "✅ 已生成 $N 条候选 → $OUT" >&2
echo "── 预览(id / tp / ep / att / mf / requested_mamba / effective_mamba / page / spec / chunk / kv / sched / radix)──" >&2
jq -r '.candidates[] | 
  "  " + .id +
  "  tp=" + (.params.tp_size|tostring) +
  "  ep=" + (.params.ep_size // 1 | tostring) +
  "  " + (.params.attention_backend // "-(default)") +
  "  mf=" + (.params.mem_fraction_static|tostring) +
  "  requested_mamba=" + (.params.mamba_radix_cache_strategy // .params.mamba_scheduler_strategy // .params["mamba-radix-cache-strategy"] // "no_buffer(default)") +
  "  effective_mamba=inactive(radix_off)" +
  "  page=" + (.params.page_size // .params["page-size"] // "1(default)" | tostring) +
  "  spec=" + (.params.speculative_algorithm // .params["speculative-algorithm"] // .params.speculative // "none(default)") +
  "  chunk=" + (.params.chunked_prefill_size // .params["chunked-prefill-size"] // "8192(default)" | tostring) +
  "  kv=" + (.params.kv_cache_dtype // .params["kv-cache-dtype"] // "auto(default)") +
  "  sched=" + (.params.schedule_conservativeness // .params["schedule-conservativeness"] // "1.0(default)" | tostring) +
  "  radix=off"
' "$OUT" >&2

# 整个脚本墙钟耗时(从文件头 SECONDS=0 到此刻),纯展示、不写进 configs.jsonl。
# 大于 60s 时补一个「Xm Ys」的人类可读形式,方便一眼看出几分钟。
GEN_ELAPSED="$SECONDS"
if [ "$GEN_ELAPSED" -ge 60 ]; then
  echo "⏱  生成耗时 ${GEN_ELAPSED}s ($((GEN_ELAPSED / 60))m $((GEN_ELAPSED % 60))s,含前置检查/AI推导/落盘/预览全流程)" >&2
else
  echo "⏱  生成耗时 ${GEN_ELAPSED}s(含前置检查/AI推导/落盘/预览全流程)" >&2
fi
