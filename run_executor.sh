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
RESULTS=""
JOB_ID=""
RUNNER_PID=""
RUNNER_STARTING=0
SHUTTING_DOWN=0
PENDING_SIGNAL=""
PENDING_EXIT_CODE=""
SHUTDOWN_GRACE_SECONDS=""
WRAPPER_CLEANUP_FAILURE=""
WAIT_INTERRUPTED=0

write_wrapper_status() {
  local task_status="$1"
  local signal_name="${2:-}"
  local exit_code="${3:-0}"
  local status_path=""
  local temporary_path=""
  local current_status=""
  if [ -z "$RESULTS" ]; then
    return 0
  fi
  if ! mkdir -p -- "$RESULTS"; then
    echo "❌ cannot prepare results directory for lifecycle status: $RESULTS" >&2
    return 1
  fi
  status_path="$RESULTS/task_status.json"
  temporary_path="$status_path.wrapper.$$"
  if [ -f "$status_path" ]; then
    current_status="$(jq -r '.task_status // ""' "$status_path" 2>/dev/null || true)"
  fi
  # Python writes richer interrupt evidence once its lifecycle is installed.
  # Never replace that record with the wrapper's minimal fallback.
  if [ "$task_status" = "INTERRUPTED" ] && [ "$current_status" = "INTERRUPTED" ]; then
    if [ -z "$WRAPPER_CLEANUP_FAILURE" ]; then
      return 0
    fi
    jq --arg failure "$WRAPPER_CLEANUP_FAILURE" \
      '.cleanup_failures = (((.cleanup_failures // []) + [$failure]) | unique)' \
      "$status_path" > "$temporary_path"
    mv -f -- "$temporary_path" "$status_path"
    return 0
  fi
  if [ "$task_status" = "INCOMPLETE" ] && [ "$current_status" != "RUNNING" ]; then
    return 0
  fi
  if [ "$task_status" = "INTERRUPTED" ]; then
    jq -n --arg job_id "$JOB_ID" --arg signal "SIG$signal_name" \
      --argjson exit_code "$exit_code" \
      --arg cleanup_failure "$WRAPPER_CLEANUP_FAILURE" \
      '{report_schema_version:1, job_id:$job_id, task_status:"INTERRUPTED",
        ranking_status:"PROVISIONAL", interrupted:true, signal:$signal,
        exit_code:$exit_code,
        cleanup_failures:(if $cleanup_failure == "" then [] else [$cleanup_failure] end),
        failure_type:null,
        failure_reason:null, status_source:"wrapper"}' > "$temporary_path"
  else
    jq -n --arg job_id "$JOB_ID" --arg task_status "$task_status" \
      '{report_schema_version:1, job_id:$job_id, task_status:$task_status,
        ranking_status:"PROVISIONAL", interrupted:false, signal:null,
        cleanup_failures:[], failure_type:null, failure_reason:null,
        status_source:"wrapper"}' > "$temporary_path"
  fi
  mv -f -- "$temporary_path" "$status_path"
}

group_state() {
  local process_group="$1"
  local rows=""
  if ! rows="$(ps -eo pgid=,stat= 2>/dev/null)"; then
    echo "UNKNOWN"
    return 0
  fi
  local awk_status=0
  if awk -v group="$process_group" '
      $1 == group && $2 !~ /^Z/ { found=1 }
      END { exit(found ? 0 : 1) }
    ' <<< "$rows"; then
    awk_status=0
  else
    awk_status=$?
  fi
  case "$awk_status" in
    0) echo "RUNNING" ;;
    1) echo "MISSING" ;;
    *) echo "UNKNOWN" ;;
  esac
}

stop_runner() {
  local signal_name="${1:-TERM}"
  local runner_group=""
  local state="UNKNOWN"
  local grace_whole=""
  local grace_fraction=""
  local grace_ms=0
  local grace_polls=0
  local poll_count=0
  if [ -z "$RUNNER_PID" ]; then
    return 0
  fi
  runner_group="-$RUNNER_PID"
  grace_whole="${SHUTDOWN_GRACE_SECONDS%%.*}"
  if [[ "$SHUTDOWN_GRACE_SECONDS" == *.* ]]; then
    grace_fraction="${SHUTDOWN_GRACE_SECONDS#*.}000"
    grace_fraction="${grace_fraction:0:3}"
  else
    grace_fraction="000"
  fi
  grace_ms=$((10#$grace_whole * 1000 + 10#$grace_fraction))
  grace_polls=$(((grace_ms + 49) / 50))
  state="$(group_state "$RUNNER_PID")"
  if [ "$state" != "MISSING" ]; then
    kill -s "$signal_name" -- "$runner_group" 2>/dev/null || true
  fi
  # Poll synchronously so a leader that exits before its cleaning descendants
  # cannot force us to wait the full (potentially hour-long) hard ceiling.  A
  # repeated signal merely interrupts one short sleep; the first signal/code
  # remains authoritative and the next poll continues cleanup.
  while [ "$state" != "MISSING" ] && [ "$poll_count" -lt "$grace_polls" ]; do
    sleep 0.05 || true
    state="$(group_state "$RUNNER_PID")"
    poll_count=$((poll_count + 1))
  done
  if [ "$state" != "MISSING" ]; then
    kill -s KILL -- "$runner_group" 2>/dev/null || true
    kill -s KILL "$RUNNER_PID" 2>/dev/null || true
  fi
  for _ in $(seq 1 50); do
    state="$(group_state "$RUNNER_PID")"
    if [ "$state" = "MISSING" ]; then
      break
    fi
    kill -s KILL -- "$runner_group" 2>/dev/null || true
    kill -s KILL "$RUNNER_PID" 2>/dev/null || true
    sleep 0.02
  done
  if [ "$state" != "MISSING" ]; then
    WRAPPER_CLEANUP_FAILURE="runner process group absence could not be proven: PGID=$RUNNER_PID state=$state"
    echo "❌ $WRAPPER_CLEANUP_FAILURE" >&2
  fi
  # Reap the group leader after the group is absent.  Repeated signals can
  # interrupt wait, so retry while the direct child PID still exists.
  if [ "$state" = "MISSING" ]; then
    local reap_attempt=0
    while [ "$reap_attempt" -lt 10 ]; do
      WAIT_INTERRUPTED=0
      wait "$RUNNER_PID" 2>/dev/null || true
      if [ "$WAIT_INTERRUPTED" -eq 0 ]; then
        break
      fi
      reap_attempt=$((reap_attempt + 1))
    done
    if [ "$WAIT_INTERRUPTED" -ne 0 ]; then
      WRAPPER_CLEANUP_FAILURE="${WRAPPER_CLEANUP_FAILURE:+$WRAPPER_CLEANUP_FAILURE; }runner reap repeatedly interrupted"
    fi
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
  if [ -z "$PENDING_SIGNAL" ]; then
    PENDING_SIGNAL="$signal_name"
    PENDING_EXIT_CODE="$exit_code"
  fi
  if [ "$RUNNER_STARTING" -eq 1 ] || [ "$SHUTTING_DOWN" -eq 1 ]; then
    WAIT_INTERRUPTED=1
    return 0
  fi
  SHUTTING_DOWN=1
  stop_runner "$PENDING_SIGNAL"
  write_wrapper_status "INTERRUPTED" "$PENDING_SIGNAL" "$PENDING_EXIT_CODE" || true
  exit "$PENDING_EXIT_CODE"
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
        exclusive_host: ($m.exclusive_host // false),
        allow_cross_numa: ($m.allow_cross_numa // false)
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
#     按 NUMA topology 复制实际可放置实例,整机满载、多端口并发实测求和。
#   MAX_PARALLEL=N  批内候选并发上限(满载下每候选独占整机,通常不用改)。
# ─────────────────────────────────────────────────────────────────────────
EXTRA_ARGS=()
if [ "${FILL_HOST:-0}" = "1" ] || [ "${FILL_HOST:-}" = "true" ]; then
  EXTRA_ARGS+=(--fill-host)
  echo "    ⚡ fill_host=ON:round2 整机满载实测(topology-aware 实例并发求和)" >&2
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

# Signal cleanup is bounded but scales with the largest legal active resource
# set: at most gpu_count server groups plus gpu_count benchmark groups and
# max_parallel containers.  The conservative 300s/GPU allowance covers exact
# server + benchmark callbacks, their manual/outer-lifecycle retry layers, and
# bounded remote observations around each 10s TERM grace.  The 60s/container
# allowance covers stop/remove/inspect; 300s covers broad fallback proof and
# status persistence. Tests/operators may override this shutdown-only ceiling.
GPU_COUNT="$(jq -r '.gpu_count' "$JOB")"
EFFECTIVE_MAX_PARALLEL="${MAX_PARALLEL:-8}"
if ! [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || ! [[ "$EFFECTIVE_MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "❌ gpu_count/MAX_PARALLEL 必须是正整数" >&2
  exit 2
fi
SHUTDOWN_GRACE_SECONDS="${LLM_INFER_TUNER_SHUTDOWN_GRACE_SECONDS:-$((GPU_COUNT * 300 + EFFECTIVE_MAX_PARALLEL * 60 + 300))}"
if ! [[ "$SHUTDOWN_GRACE_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "❌ LLM_INFER_TUNER_SHUTDOWN_GRACE_SECONDS 必须是非负秒数" >&2
  exit 2
fi

write_wrapper_status "RUNNING"
RUNNER_STARTING=1
RUNNER_GROUP_READY=0
# Non-interactive Bash starts asynchronous children with SIGINT ignored.  Reset
# both dispositions before exec so an early signal (before Python installs its
# lifecycle handlers) is still delivered instead of waiting for the watchdog.
setsid env --default-signal=INT --default-signal=TERM uv run python -m runners.executor \
  --job "$JOB" \
  --target "$TARGET" \
  --configs "$CONFIGS" \
  --results "$RESULTS" \
  --container-name "$CONTAINER_NAME" \
  "${EXTRA_ARGS[@]}" &
RUNNER_PID=$!
# ``setsid ... &`` publishes the child PID before the child necessarily calls
# setsid(2).  Keep signal handling in the pending state until PGID==PID so a
# signal cannot observe an unkillable fork→session-creation gap.
for _ in $(seq 1 500); do
  RUNNER_PGID="$(ps -o pgid= -p "$RUNNER_PID" 2>/dev/null | tr -d ' ')"
  if [ "$RUNNER_PGID" = "$RUNNER_PID" ]; then
    RUNNER_GROUP_READY=1
    break
  fi
  if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
    break
  fi
  sleep 0.01
done
if [ "$RUNNER_GROUP_READY" -ne 1 ] && kill -0 "$RUNNER_PID" 2>/dev/null; then
  echo "❌ runner 未在时限内建立独立 process group，强制终止" >&2
  kill -s KILL "$RUNNER_PID" 2>/dev/null || true
  kill -s KILL -- "-$RUNNER_PID" 2>/dev/null || true
  wait "$RUNNER_PID" 2>/dev/null || true
fi
RUNNER_STARTING=0
if [ -n "$PENDING_SIGNAL" ]; then
  handle_signal "$PENDING_SIGNAL" "$PENDING_EXIT_CODE"
fi
if wait "$RUNNER_PID"; then
  RUNNER_STATUS=0
else
  RUNNER_STATUS=$?
fi
RUNNER_PID=""
if [ "$RUNNER_STATUS" -ne 0 ]; then
  write_wrapper_status "INCOMPLETE" || true
fi
exit "$RUNNER_STATUS"
