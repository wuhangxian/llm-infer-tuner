#!/bin/bash
# run_executor.sh —— 第二阶段:远程压测 + 排名。
#
# 用法1(三参数模式,AI 生成后跑):
#   ./run_executor.sh <job.json> <target.json> [configs.jsonl] [results_dir]
#
# 用法2(单文件模式,手写配置):
#   ./run_executor.sh <bundle.json|bundle.jsonl>
#   单文件是一个顶层含 _meta + candidates 的 JSON wrapper。
#
# 压测命令由 JobSpec + workload + benchmark_method 确定性生成,第二阶段不调用 AI。
set -euo pipefail

command -v jq >/dev/null || { echo "❌ 需要 jq" >&2; exit 1; }
command -v setsid >/dev/null || { echo "❌ 需要 setsid" >&2; exit 1; }

usage() {
  echo "用法1: ./run_executor.sh <job.json> <target.json> [configs.jsonl] [results_dir]" >&2
  echo "用法2: ./run_executor.sh <bundle.json|bundle.jsonl>  (顶层含 _meta + candidates)" >&2
}

if [ "$#" -eq 0 ] || [ "$#" -gt 4 ]; then
  usage
  exit 2
fi

TMP_DIR=""
RUNNER_PID=""

stop_runner() {
  local signal_name="${1:-TERM}"
  local runner_group=""
  local watchdog_pid=""
  if [ -z "$RUNNER_PID" ]; then
    return 0
  fi
  runner_group="-$RUNNER_PID"
  if kill -0 -- "$runner_group" 2>/dev/null; then
    kill -s "$signal_name" -- "$runner_group" 2>/dev/null || true
    (
      sleep 3
      kill -s KILL -- "$runner_group" 2>/dev/null || true
    ) &
    watchdog_pid=$!
  fi
  wait "$RUNNER_PID" 2>/dev/null || true
  if [ -n "$watchdog_pid" ]; then
    if kill -0 -- "$runner_group" 2>/dev/null; then
      wait "$watchdog_pid" 2>/dev/null || true
    else
      kill "$watchdog_pid" 2>/dev/null || true
    fi
    wait "$watchdog_pid" 2>/dev/null || true
  fi
  if kill -0 -- "$runner_group" 2>/dev/null; then
    kill -s KILL -- "$runner_group" 2>/dev/null || true
  fi
  RUNNER_PID=""
  return 0
}

cleanup() {
  stop_runner TERM
  if [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ]; then
    rm -rf -- "$TMP_DIR"
  fi
}

handle_signal() {
  local signal_name="$1"
  local exit_code="$2"
  trap - "$signal_name"
  stop_runner "$signal_name"
  exit "$exit_code"
}

trap cleanup EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 检测模式:单文件 vs 三参数
# ─────────────────────────────────────────────────────────────────────────
SINGLE_FILE=""
if [ $# -eq 1 ] && { [[ "$1" == *.json ]] || [[ "$1" == *.jsonl ]]; }; then
  SINGLE_FILE="$1"
  [ -f "$SINGLE_FILE" ] || { echo "❌ 文件不存在: $SINGLE_FILE" >&2; exit 1; }
elif [ $# -ge 2 ] && [ $# -le 4 ] && [[ "$1" != -* ]] && [[ "$2" != -* ]]; then
  JOB="$1"
  TARGET="$2"
  [ -f "$JOB" ]     || { echo "❌ job 不存在: $JOB" >&2; exit 1; }
  [ -f "$TARGET" ]  || { echo "❌ target 不存在: $TARGET" >&2; exit 1; }
else
  usage
  exit 1
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 解析参数(单文件模式从 _meta 读,三参数模式从 job/target 读)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ -n "$SINGLE_FILE" ]; then
  # 单文件模式:先在 0700 临时目录里生成严格的 job/target/configs。
  # target 里绝不复制明文密码;兼容旧 bundle 时只通过当前进程环境传递。
  jq -e 'type == "object" and (._meta | type == "object") and (.candidates | type == "array")' \
    "$SINGLE_FILE" >/dev/null || {
      echo "❌ 单文件必须是顶层含 _meta 和 candidates 数组的 JSON wrapper" >&2
      exit 1
    }
  umask 077
  TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/llm-infer-tuner.XXXXXXXX")"
  chmod 700 "$TMP_DIR"
  JOB_ID="$(jq -r '._meta.job_id // "custom"' "$SINGLE_FILE")"
  CONFIGS="$TMP_DIR/configs.jsonl"
  RESULTS="outputs/${JOB_ID}/results"

  has_plain_password="$(jq -r '((._meta.ssh_password? // "") | length) > 0' "$SINGLE_FILE")"
  has_password_env="$(jq -r '((._meta.ssh_password_env? // "") | length) > 0' "$SINGLE_FILE")"
  if [ "$has_plain_password" = "true" ] && [ "$has_password_env" = "true" ]; then
    echo "❌ bundle 不能同时设置 ssh_password 和 ssh_password_env" >&2
    exit 1
  fi
  if [ "$has_plain_password" = "true" ]; then
    LLM_INFER_TUNER_BUNDLE_SSH_PASSWORD="$(jq -r '._meta.ssh_password' "$SINGLE_FILE")"
    export LLM_INFER_TUNER_BUNDLE_SSH_PASSWORD
  fi

  # max_candidates 只统计非 baseline;任一 baseline 标记都会启用严格 baseline 契约,
  # ID/标记不一致会继续由 CandidateSet 拒绝。
  jq '
    . as $root
    | ($root._meta) as $m
    | ([$root.candidates[] | select(.id == "baseline" or .params.is_baseline == true)]
       | length > 0) as $has_baseline
    | {
        job_id: ($m.job_id // "custom"),
        engine: "sglang",
        gpu_model: $m.gpu_model,
        gpu_count: $m.gpu_count,
        gpu_memory_gb: $m.gpu_memory_gb,
        model: ($m.model // "custom"),
        image: ($m.image // "custom"),
        workload: $m.workload,
        benchmark_method: $m.benchmark_method,
        sla: $m.sla,
        search: ({max_candidates: (($root.candidates | length) - (if $has_baseline then 1 else 0 end))}
                 + (if $has_baseline then {baseline: {}} else {} end))
      }
  ' "$SINGLE_FILE" > "$TMP_DIR/job.json"
  jq --arg runtime_password_env "LLM_INFER_TUNER_BUNDLE_SSH_PASSWORD" '
    ._meta as $m
    | {
        gpu_model: $m.gpu_model,
        gpu_count: $m.gpu_count,
        gpu_memory_gb: $m.gpu_memory_gb,
        ssh_target: $m.ssh_target,
        model_host_dir: $m.model_host_dir,
        model_container_path: $m.model_container_path,
        image_ref: $m.image_ref,
        port: ($m.port // 30000),
        remote_outputs_dir: ($m.remote_outputs_dir // ""),
        exclusive_host: ($m.exclusive_host // false)
      }
      + (if (($m.ssh_password_env? // "") | length) > 0
         then {ssh_password_env: $m.ssh_password_env}
         elif (($m.ssh_password? // "") | length) > 0
         then {ssh_password_env: $runtime_password_env}
         else {} end)
  ' "$SINGLE_FILE" > "$TMP_DIR/target.json"
  JOB="$TMP_DIR/job.json"
  TARGET="$TMP_DIR/target.json"
  # Extract candidates into JSONL for executor
  jq -c '.candidates[]' "$SINGLE_FILE" > "$TMP_DIR/configs.jsonl"
  MAX_CAND="$(jq -r '.search.max_candidates' "$JOB")"
else
  # 三参数模式:原来逻辑
  JOB_ID="$(jq -r '.job_id' "$JOB")"
  CONFIGS="${3:-outputs/${JOB_ID}/configs.jsonl}"
  RESULTS="${4:-outputs/${JOB_ID}/results}"
  [ -f "$CONFIGS" ] || { echo "❌ configs 不存在: $CONFIGS(先跑 ./gen_configs.sh $JOB)" >&2; exit 1; }
  MAX_CAND="$(jq -r '.search.max_candidates // 1' "$JOB")"
fi

CONTAINER_NAME="llm-infer-tuner-${JOB_ID}"

echo "▶ 执行器参数一览:" >&2
echo "    job_id=$JOB_ID  max_candidates=$MAX_CAND" >&2
echo "    bench_command=deterministic(no AI)" >&2
echo "    configs=$CONFIGS  results=$RESULTS" >&2
echo >&2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 可选:整机满载(round2)。用环境变量开,默认关,不影响原有单参数用法。
#   FILL_HOST=1 ./run_executor.sh <configs.json>   # round2 把 top-K 各复制成
#     floor(gpu/tp) 个实例整机满载、多端口并发实测求和(而非单实例纸面外推)。
#   MAX_PARALLEL=N  批内候选并发上限(满载下每候选独占整机,通常不用改)。
# ─────────────────────────────────────────────────────────────────────────
EXTRA_ARGS=()
if [ "${FILL_HOST:-0}" = "1" ] || [ "${FILL_HOST:-}" = "true" ]; then
  EXTRA_ARGS+=(--fill-host)
  echo "    ⚡ fill_host=ON:round2 整机满载实测(floor(gpu/tp) 实例并发求和)" >&2
fi
if [ -n "${MAX_PARALLEL:-}" ]; then
  EXTRA_ARGS+=(--max-parallel "$MAX_PARALLEL")
  echo "    max_parallel=$MAX_PARALLEL" >&2
fi

# 只限制服务拉起，不限制整个任务运行时间。可按机器/模型覆盖。
STARTUP_STALL_TIMEOUT_SECONDS="${STARTUP_STALL_TIMEOUT_SECONDS:-300}"
STARTUP_HARD_TIMEOUT_SECONDS="${STARTUP_HARD_TIMEOUT_SECONDS:-900}"
STARTUP_MAX_ATTEMPTS="${STARTUP_MAX_ATTEMPTS:-3}"
EXTRA_ARGS+=(
  --startup-stall-timeout "$STARTUP_STALL_TIMEOUT_SECONDS"
  --startup-hard-timeout "$STARTUP_HARD_TIMEOUT_SECONDS"
  --startup-max-attempts "$STARTUP_MAX_ATTEMPTS"
)
echo "    startup: stall=${STARTUP_STALL_TIMEOUT_SECONDS}s hard=${STARTUP_HARD_TIMEOUT_SECONDS}s attempts=${STARTUP_MAX_ATTEMPTS}; job_timeout=none" >&2

setsid uv run python -m runners.executor \
  --job "$JOB" \
  --target "$TARGET" \
  --configs "$CONFIGS" \
  --results "$RESULTS" \
  --container-name "$CONTAINER_NAME" \
  "${EXTRA_ARGS[@]}" &
RUNNER_PID=$!
if wait "$RUNNER_PID"; then
  RUNNER_STATUS=0
else
  RUNNER_STATUS=$?
fi
RUNNER_PID=""
exit "$RUNNER_STATUS"
