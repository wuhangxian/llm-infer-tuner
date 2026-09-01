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

# Install a tiny signal guard before any jq/argument validation.  Bash can
# receive a signal in the few milliseconds before the full lifecycle functions
# below are defined; without this guard an old FINAL manifest remains visible.
# The guard uses only shell builtins/basic filesystem operations and is
# replaced by ``handle_signal`` once normal setup is complete.
EARLY_RESULTS=""
EARLY_SIGNAL_HANDLED=0
RESULTS_LOCK_FD=""
RESULTS_LOCK_PATH=""
RESULTS_LOCK_OWNED=0
RESULTS_LOCK_CONFLICT=0
if [ "$#" -ge 4 ]; then
  EARLY_RESULTS="${4:-}"
elif [ "$#" -ge 1 ]; then
  EARLY_JOB_NAME="${1##*/}"
  EARLY_JOB_NAME="${EARLY_JOB_NAME%.*}"
  [ -n "$EARLY_JOB_NAME" ] || EARLY_JOB_NAME="custom"
  EARLY_RESULTS="outputs/${EARLY_JOB_NAME}/results"
fi

acquire_results_lock_for() {
  # Only one wrapper may own a result directory at a time.  Without this
  # process lock, an older invocation that exits non-zero can race a newer
  # invocation and revoke the newer run's immutable FINAL manifest.
  local target_dir="$1"
  local lock_path=""
  local new_fd=""
  [ -n "$target_dir" ] || return 0
  lock_path="$target_dir/.run_executor.lock"
  if [ "$RESULTS_LOCK_OWNED" -eq 1 ] && [ "$RESULTS_LOCK_PATH" = "$lock_path" ]; then
    return 0
  fi
  if ! mkdir -p -- "$target_dir"; then
    echo "❌ cannot prepare results lock directory: $target_dir" >&2
    return 1
  fi
  if ! exec {new_fd}>"$lock_path"; then
    echo "❌ cannot open results lock: $lock_path" >&2
    return 1
  fi
  if ! flock -n "$new_fd"; then
    exec {new_fd}>&-
    RESULTS_LOCK_CONFLICT=1
    echo "❌ another executor already owns results directory: $target_dir" >&2
    return 75
  fi
  # Acquire the refined job-id path before releasing the filename-derived
  # preliminary path, so there is never an ownership-free publication window.
  if [ "$RESULTS_LOCK_OWNED" -eq 1 ] && [ -n "$RESULTS_LOCK_FD" ]; then
    flock -u "$RESULTS_LOCK_FD" 2>/dev/null || true
    exec {RESULTS_LOCK_FD}>&-
  fi
  RESULTS_LOCK_FD="$new_fd"
  RESULTS_LOCK_PATH="$lock_path"
  RESULTS_LOCK_OWNED=1
  RESULTS_LOCK_CONFLICT=0
}

revoke_manifest_file() {
  # Move the active pointer to a same-directory stale name whenever possible.
  # If the move itself fails (for example, a deliberately injected I/O error),
  # unlink the pointer as a last resort.  A stale FINAL pointer is unsafe: an
  # interrupted/failed invocation must never leave it authoritative merely
  # because preserving the old copy was impossible.  Callers hold the result
  # directory lock before reaching this helper.
  local manifest_path="$1"
  local stale_path="$2"
  if mv -f -- "$manifest_path" "$stale_path" 2>/dev/null; then
    return 0
  fi
  # Preserve a best-effort copy before unlinking.  The copy is diagnostic only
  # (the immutable generation remains in .report_generations); failure here
  # must not prevent the fail-closed unlink below.
  cp -p -- "$manifest_path" "$stale_path" 2>/dev/null || true
  if rm -f -- "$manifest_path" 2>/dev/null && [ ! -e "$manifest_path" ]; then
    echo "⚠️  manifest move failed; removed active pointer: $manifest_path" >&2
    return 0
  fi
  echo "❌ cannot revoke active report manifest: $manifest_path" >&2
  return 1
}

early_revoke_manifest() {
  local manifest_path=""
  local stale_path=""
  [ -n "${EARLY_RESULTS:-}" ] || return 0
  acquire_results_lock_for "$EARLY_RESULTS" || return $?
  manifest_path="$EARLY_RESULTS/report_manifest.json"
  [ -e "$manifest_path" ] || return 0
  stale_path="$EARLY_RESULTS/.report_manifest.stale.wrapper.$$.${EPOCHREALTIME//[^0-9]/}"
  revoke_manifest_file "$manifest_path" "$stale_path"
}

early_signal() {
  local signal_name="$1"
  local exit_code="$2"
  local status_path=""
  local temporary_path=""
  if [ "$EARLY_SIGNAL_HANDLED" -eq 1 ]; then
    return 0
  fi
  EARLY_SIGNAL_HANDLED=1
  # If another invocation owns this results directory, this process must not
  # revoke or overwrite that run merely because it received a signal.
  if ! early_revoke_manifest; then
    exit "$exit_code"
  fi
  if [ -n "${EARLY_RESULTS:-}" ]; then
    mkdir -p -- "$EARLY_RESULTS" 2>/dev/null || true
    status_path="$EARLY_RESULTS/task_status.json"
    temporary_path="$status_path.early.$$"
    # Signal names/codes are constants supplied by the traps, so this is safe
    # to emit without jq (which may itself be the failed preflight command).
    if printf '{"report_schema_version":1,"job_id":"","task_status":"INTERRUPTED","ranking_status":"PROVISIONAL","interrupted":true,"signal":"SIG%s","exit_code":%s,"cleanup_failures":[],"failure_type":null,"failure_reason":null,"status_source":"wrapper"}\n' \
      "$signal_name" "$exit_code" > "$temporary_path" 2>/dev/null; then
      mv -f -- "$temporary_path" "$status_path" 2>/dev/null || true
    else
      rm -f -- "$temporary_path" 2>/dev/null || true
    fi
  fi
  exit "$exit_code"
}

trap 'early_signal INT 130' INT
trap 'early_signal TERM 143' TERM

command -v jq >/dev/null || { echo "❌ 需要 jq" >&2; exit 1; }
command -v setsid >/dev/null || { echo "❌ 需要 setsid" >&2; exit 1; }
command -v flock >/dev/null || { echo "❌ 需要 flock" >&2; exit 1; }

usage() {
  echo "用法1: ./run_executor.sh <job.json> <target.json> [configs.jsonl] [results_dir]" >&2
  echo "用法2: ./run_executor.sh <bundle.json|bundle.jsonl>  (顶层含 _meta + candidates)" >&2
}

is_safe_identifier() {
  # Keep shell-derived output/container paths aligned with schemas.Identifier.
  # Never interpolate an untrusted job id before this check: ``..`` and path
  # separators would otherwise escape the outputs directory.
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]
}

if [ "$#" -eq 0 ] || [ "$#" -gt 4 ]; then
  usage
  exit 2
fi

TMP_DIR=""
RESULTS="${EARLY_RESULTS:-}"
JOB_ID=""
RUNNER_PID=""
RUNNER_STARTING=0
SHUTTING_DOWN=0
PENDING_SIGNAL=""
PENDING_EXIT_CODE=""
SHUTDOWN_GRACE_SECONDS=""
WRAPPER_CLEANUP_FAILURE=""
WAIT_INTERRUPTED=0
REPORT_MANIFEST_REVOKED="${REPORT_MANIFEST_REVOKED:-0}"

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
  if [ "$RESULTS_LOCK_CONFLICT" -eq 1 ]; then
    return 1
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
  # Preserve a richer provisional record written by Python, but replace a
  # stale/terminal FINAL compatibility record after its manifest was revoked.
  # The latter is common when jq fails before Python has started.
  if [ "$task_status" = "INCOMPLETE" ] \
    && { [ "$current_status" = "INCOMPLETE" ] || [ "$current_status" = "INTERRUPTED" ]; }; then
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

revoke_report_manifest() {
  # Each wrapper invocation owns a fresh report generation. Remove the old
  # active pointer before launching Python so a signal in the pre-handler
  # window cannot leave a previous FINAL report authoritative. Keep the
  # immutable generation recoverable under a unique same-directory name.
  local manifest_path=""
  local stale_path=""
  if [ -z "$RESULTS" ]; then
    return 0
  fi
  acquire_results_lock_for "$RESULTS" || return $?
  manifest_path="$RESULTS/report_manifest.json"
  if [ ! -e "$manifest_path" ]; then
    return 0
  fi
  stale_path="$RESULTS/.report_manifest.stale.wrapper.$$.${EPOCHREALTIME//[^0-9]/}"
  if ! revoke_manifest_file "$manifest_path" "$stale_path"; then
    return 1
  fi
  REPORT_MANIFEST_REVOKED=1
}

active_report_tuple() {
  # Read the status from the generation selected by the immutable pointer, not
  # from the loose compatibility task_status.json.  A child can crash between
  # committing those two views, so the loose copy is not authoritative here.
  local manifest_path=""
  local generation_id=""
  local status_path=""
  if [ -z "$RESULTS" ]; then
    return 1
  fi
  manifest_path="$RESULTS/report_manifest.json"
  [ -f "$manifest_path" ] || return 1
  if ! generation_id="$(jq -er '
      if type == "object"
         and (.report_schema_version | type) == "number"
         and .report_schema_version == 2
         and (.generation_id | type) == "string"
         and (.generation_id | test("^[0-9a-f]{32}$"))
         and .snapshot_id == .generation_id
      then .generation_id
      else empty
      end' "$manifest_path" 2>/dev/null)"; then
    # An unreadable/malformed manifest is unsafe.  Return an explicit non-zero
    # code (rather than propagating ``$?`` from a negated command) so callers
    # revoke the pointer instead of treating an empty tuple as provisional.
    return 1
  fi
  status_path="$RESULTS/.report_generations/$generation_id/task_status.json"
  [ -f "$status_path" ] || return 1
  jq -er '
    if type == "object"
       and (.report_schema_version | type) == "number"
       and .report_schema_version == 2
       and (.task_status | type) == "string"
       and (.ranking_status | type) == "string"
       and (.interrupted | type) == "boolean"
    then [.task_status, .ranking_status, (.interrupted | tostring)] | @tsv
    else empty
    end' "$status_path" 2>/dev/null
}

revoke_unsafe_report_manifest() {
  # A runner can write a schema-v2 FINAL pointer and then still exit non-zero
  # (for example, a post-report cleanup hook can fail).  Preserve only a
  # structurally valid provisional/interrupted generation; every final or
  # unreadable state is moved aside so it cannot remain authoritative.
  local manifest_path=""
  local tuple=""
  if [ -z "$RESULTS" ]; then
    return 0
  fi
  manifest_path="$RESULTS/report_manifest.json"
  [ -e "$manifest_path" ] || return 0
  if ! tuple="$(active_report_tuple 2>/dev/null)"; then
    revoke_report_manifest
    return $?
  fi
  case "$tuple" in
    $'INCOMPLETE\tPROVISIONAL\tfalse'|$'INTERRUPTED\tPROVISIONAL\ttrue')
      return 0
      ;;
    *)
      revoke_report_manifest
      ;;
  esac
}

revoke_final_report_manifest() {
  # Backward-compatible name used by the post-child failure path.
  revoke_unsafe_report_manifest
}

revoke_if_active_final_or_invalid() {
  local manifest_path=""
  local tuple=""
  if [ -z "$RESULTS" ]; then
    return 0
  fi
  manifest_path="$RESULTS/report_manifest.json"
  [ -e "$manifest_path" ] || return 0
  if ! tuple="$(active_report_tuple 2>/dev/null)"; then
    revoke_report_manifest
    return $?
  fi
  if [ "$tuple" = $'COMPLETED\tFINAL\tfalse' ]; then
    revoke_report_manifest
  fi
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

wrapper_exit() {
  # Preserve the original exit code while making sure a parse/launch failure
  # cannot leave the compatibility status at RUNNING.  This trap is only
  # enabled after a result path is known and an older manifest was actually
  # revoked, so ordinary argument-usage errors do not create output trees.
  local exit_code="$1"
  cleanup
  if [ "$exit_code" -ne 0 ] && [ -n "$RESULTS" ] \
    && [ "$RESULTS_LOCK_CONFLICT" -eq 0 ]; then
    write_wrapper_status "INCOMPLETE" || true
  fi
  return "$exit_code"
}

handle_signal() {
  local signal_name="$1"
  local exit_code="$2"
  local first_signal=0
  local revoke_on_signal=0
  if [ -z "$PENDING_SIGNAL" ]; then
    PENDING_SIGNAL="$signal_name"
    PENDING_EXIT_CODE="$exit_code"
    first_signal=1
  fi
  # The wrapper can receive a signal before Python has installed its own
  # lifecycle handler. Revoke the previous immutable report pointer at this
  # earliest known results path; the operation is idempotent once the pointer
  # has already been moved by the normal pre-launch guard. Repeated signals
  # must not move a newer schema-v2 INTERRUPTED/PROVISIONAL generation.
  if [ "$first_signal" -eq 1 ]; then
    # During argument/preflight processing RUNNER_PID is still empty, and
    # during the fork→setsid handshake RUNNER_STARTING remains set.  In either
    # window an active pointer can only be a stale prior generation.  Once the
    # runner is established, leave a Python-published PROVISIONAL/INTERRUPTED
    # generation alone; revoke only an actually old FINAL status.
    if [ "$RUNNER_STARTING" -eq 1 ] || [ -z "$RUNNER_PID" ]; then
      revoke_on_signal=1
    elif [ -e "$RESULTS/report_manifest.json" ]; then
      # Consult the immutable generation's task status.  The loose status can
      # lag a final commit (or be overwritten by a wrapper failure), so using
      # it here could leave a real FINAL pointer visible after SIGTERM.
      if ! revoke_if_active_final_or_invalid; then
        WRAPPER_CLEANUP_FAILURE="cannot revoke previous report manifest"
      fi
    fi
  fi
  if [ "$revoke_on_signal" -eq 1 ] && ! revoke_report_manifest; then
    WRAPPER_CLEANUP_FAILURE="cannot revoke previous report manifest"
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

trap 'wrapper_exit "$?"' EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

# Establish the best result path we can know before parsing any input.  This
# matters for malformed job/bundle files: a previous immutable FINAL pointer
# must not remain authoritative merely because jq failed before the normal
# argument-resolution block assigned RESULTS.  An explicit fourth argument is
# authoritative; otherwise use a safe filename-derived fallback and refine it
# after the job id is parsed below.
if [ "$#" -ge 4 ]; then
  RESULTS="$4"
else
  PRELIM_JOB_NAME="${1##*/}"
  PRELIM_JOB_NAME="${PRELIM_JOB_NAME%.*}"
  if [ -z "$PRELIM_JOB_NAME" ]; then
    PRELIM_JOB_NAME="custom"
  fi
  RESULTS="outputs/${PRELIM_JOB_NAME}/results"
  # A valid job/bundle may use an id different from its filename, and target
  # validation can fail before the normal jq block resolves that id.  Read
  # only the scalar id (guarded so malformed JSON remains on the filename
  # fallback) and revoke its default output pointer too.  Never turn an
  # untrusted value into a path unless it satisfies the same safe-id grammar as
  # the Python schemas.
  PRELIM_JOB_ID=""
  if [ "$#" -eq 1 ]; then
    PRELIM_JOB_ID="$(jq -r 'if type == "object" then (._meta.job_id // empty) else empty end' "$1" 2>/dev/null || true)"
  elif [ "$#" -ge 2 ]; then
    PRELIM_JOB_ID="$(jq -r 'if type == "object" then (.job_id // empty) else empty end' "$1" 2>/dev/null || true)"
  fi
  if [[ "$PRELIM_JOB_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    RESULTS="outputs/${PRELIM_JOB_ID}/results"
  fi
fi
revoke_report_manifest || exit 1

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
  if ! is_safe_identifier "$JOB_ID"; then
    echo "❌ job_id 必须是安全标识符 (ASCII 字母/数字/._-，最长128)" >&2
    exit 2
  fi
  CONFIGS="$TMP_DIR/configs.jsonl"
  RESULTS="outputs/${JOB_ID}/results"
  # The preliminary path above is based on the bundle filename.  If the
  # metadata supplies a different job id, revoke that path as well before any
  # generated files are published.
  revoke_report_manifest || exit 1

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
  if ! is_safe_identifier "$JOB_ID"; then
    echo "❌ job_id 必须是安全标识符 (ASCII 字母/数字/._-，最长128)" >&2
    exit 2
  fi
  CONFIGS="${3:-outputs/${JOB_ID}/configs.jsonl}"
  RESULTS="${4:-outputs/${JOB_ID}/results}"
  # A malformed job can fail the jq command above; valid-but-different job ids
  # can also change the default path.  Keep the active pointer fail-closed at
  # the resolved path before checking configs or doing any more work.
  revoke_report_manifest || exit 1
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
# Round-2 measurement mode defaults to measured full-host throughput.
# MEASUREMENT_MODE=estimated is the explicit single-instance extrapolation mode.
# Legacy FILL_HOST remains accepted: 1/true => full_host, 0/false => estimated.
#   MAX_PARALLEL=N  批内候选并发上限(满载下每候选独占整机,通常不用改)。
# ─────────────────────────────────────────────────────────────────────────
EXTRA_ARGS=()
RESOLVED_MEASUREMENT_MODE="full_host"
EXPLICIT_MEASUREMENT_MODE=""
LEGACY_MEASUREMENT_MODE=""
if [ "${MEASUREMENT_MODE+x}" = "x" ]; then
  case "$MEASUREMENT_MODE" in
    full_host|estimated) EXPLICIT_MEASUREMENT_MODE="$MEASUREMENT_MODE" ;;
    *)
      echo "❌ MEASUREMENT_MODE 必须是 full_host 或 estimated" >&2
      exit 2
      ;;
  esac
fi
if [ "${FILL_HOST+x}" = "x" ]; then
  case "$FILL_HOST" in
    1|true) LEGACY_MEASUREMENT_MODE="full_host" ;;
    0|false) LEGACY_MEASUREMENT_MODE="estimated" ;;
    *)
      echo "❌ FILL_HOST 必须是 1/true 或 0/false" >&2
      exit 2
      ;;
  esac
fi
if [ -n "$EXPLICIT_MEASUREMENT_MODE" ] && [ -n "$LEGACY_MEASUREMENT_MODE" ] \
   && [ "$EXPLICIT_MEASUREMENT_MODE" != "$LEGACY_MEASUREMENT_MODE" ]; then
  echo "❌ MEASUREMENT_MODE 与 legacy FILL_HOST 冲突" >&2
  exit 2
fi
RESOLVED_MEASUREMENT_MODE="${EXPLICIT_MEASUREMENT_MODE:-${LEGACY_MEASUREMENT_MODE:-full_host}}"
EXTRA_ARGS+=(--measurement-mode "$RESOLVED_MEASUREMENT_MODE")
echo "    measurement_mode=$RESOLVED_MEASUREMENT_MODE" >&2
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

# Revoke any prior immutable generation before the first runner process is
# spawned. This closes the shell-only signal window where Python has not yet
# installed its lifecycle handlers or published a schema-v2 tombstone.
revoke_report_manifest || exit 1

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
  # The child may have exited between fork and this observation.  Under
  # ``set -euo pipefail`` an unguarded ps pipeline would abort the wrapper
  # before it can reap the child and publish INCOMPLETE.
  if RUNNER_PGID="$(ps -o pgid= -p "$RUNNER_PID" 2>/dev/null | tr -d ' ')"; then
    :
  else
    RUNNER_PGID=""
  fi
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
  if ! revoke_final_report_manifest; then
    WRAPPER_CLEANUP_FAILURE="${WRAPPER_CLEANUP_FAILURE:+$WRAPPER_CLEANUP_FAILURE; }cannot revoke child FINAL report manifest"
  fi
  write_wrapper_status "INCOMPLETE" || true
elif [ -n "$RESULTS" ] && [ ! -e "$RESULTS/report_manifest.json" ]; then
  # A successful child must publish the schema-v2 report pointer.  Preserve
  # the historical zero exit code for compatibility, but never leave a
  # RUNNING status that could be mistaken for a completed report when a
  # crashed/mocked runner exits early without producing any evidence.
  echo "❌ runner exited successfully without report manifest" >&2
  write_wrapper_status "INCOMPLETE" || true
fi
exit "$RUNNER_STATUS"
