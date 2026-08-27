#!/bin/bash
# run_executor.sh —— 第二阶段:远程压测 + 排名。
#
# 用法1(三参数模式,AI 生成后跑):
#   ./run_executor.sh <job.json> <target.json> [configs.jsonl] [results_dir]
#
# 用法2(单文件模式,手写配置):
#   ./run_executor.sh <configs.jsonl>
#   configs.jsonl 第一行是 _meta(SLA/workload/target 信息),后面每行一条候选
#
# 脚本分 5 步确定性操作,不涉及 AI 决策。
set -euo pipefail

# Load .env for claude API credentials
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

command -v jq >/dev/null || { echo "❌ 需要 jq" >&2; exit 1; }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 检测模式:单文件 vs 三参数
# ─────────────────────────────────────────────────────────────────────────
SINGLE_FILE=""
if [ $# -eq 1 ] && [[ "$1" == *.jsonl ]]; then
  SINGLE_FILE="$1"
  [ -f "$SINGLE_FILE" ] || { echo "❌ 文件不存在: $SINGLE_FILE" >&2; exit 1; }
elif [ $# -ge 2 ]; then
  JOB="$1"
  TARGET="$2"
  [ -f "$JOB" ]     || { echo "❌ job 不存在: $JOB" >&2; exit 1; }
  [ -f "$TARGET" ]  || { echo "❌ target 不存在: $TARGET" >&2; exit 1; }
else
  echo "用法1: ./run_executor.sh <job.json> <target.json> [configs.jsonl] [results_dir]" >&2
  echo "用法2: ./run_executor.sh <configs.jsonl>  (第一行含 _meta)" >&2
  exit 1
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 解析参数(单文件模式从 _meta 读,三参数模式从 job/target 读)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ -n "$SINGLE_FILE" ]; then
  # 单文件模式:从第一行 _meta 读所有信息
  META="$SINGLE_FILE"
  JOB_ID="$(jq -s '.[0]._meta.job_id // "custom"' "$SINGLE_FILE")"
  CONFIGS="$SINGLE_FILE"
  RESULTS="outputs/${JOB_ID}/results"
  SSH_TARGET="$(jq -s '.[0]._meta.ssh_target' "$SINGLE_FILE")"
  SSH_PASSWORD="$(jq -s '.[0]._meta.ssh_password // ""' "$SINGLE_FILE")"
  MODEL_HOST_DIR="$(jq -s '.[0]._meta.model_host_dir' "$SINGLE_FILE")"
  MODEL_CONTAINER_PATH="$(jq -s '.[0]._meta.model_container_path' "$SINGLE_FILE")"
  IMAGE_REF="$(jq -s '.[0]._meta.image_ref' "$SINGLE_FILE")"
  PORT="$(jq -s '.[0]._meta.port // 30000' "$SINGLE_FILE")"
  REMOTE_OUTPUTS_DIR="$(jq -s '.[0]._meta.remote_outputs_dir // ""' "$SINGLE_FILE")"
  TARGET_GPU_MODEL="$(jq -s '.[0]._meta.gpu_model // ""' "$SINGLE_FILE")"
  TARGET_GPU_COUNT="$(jq -s '.[0]._meta.gpu_count // 0' "$SINGLE_FILE")"
  TARGET_GPU_MEM="$(jq -s '.[0]._meta.gpu_memory_gb // 0' "$SINGLE_FILE")"
  # 候选数 = 总行数 - 1(去掉 _meta 行)
  MAX_CAND=$(( $(wc -l < "$SINGLE_FILE" | tr -d ' ') - 1 ))

  # 写临时 job.json 和 target.json 给 executor.py
  TMP_DIR=$(mktemp -d)
  jq -s '.[0]._meta | {job_id, engine:"sglang", gpu_model, gpu_count, gpu_memory_gb, model:"custom", image:"custom", workload, benchmark_method, sla, search:{max_candidates:999, max_runtime_minutes:180}}' "$SINGLE_FILE" > "$TMP_DIR/job.json"
  jq -s '.[0]._meta | {gpu_model, gpu_count, gpu_memory_gb, ssh_target, ssh_password, model_host_dir, model_container_path, image_ref, port, remote_outputs_dir}' "$SINGLE_FILE" > "$TMP_DIR/target.json"
  JOB="$TMP_DIR/job.json"
  TARGET="$TMP_DIR/target.json"
else
  # 三参数模式:原来逻辑
  JOB_ID="$(jq -r '.job_id' "$JOB")"
  CONFIGS="${3:-outputs/${JOB_ID}/configs.jsonl}"
  RESULTS="${4:-outputs/${JOB_ID}/results}"
  [ -f "$CONFIGS" ] || { echo "❌ configs 不存在: $CONFIGS(先跑 ./gen_configs.sh $JOB)" >&2; exit 1; }
  SSH_TARGET="$(jq -r '.ssh_target' "$TARGET")"
  SSH_PASSWORD="$(jq -r '.ssh_password // ""' "$TARGET")"
  MODEL_HOST_DIR="$(jq -r '.model_host_dir' "$TARGET")"
  MODEL_CONTAINER_PATH="$(jq -r '.model_container_path' "$TARGET")"
  IMAGE_REF="$(jq -r '.image_ref' "$TARGET")"
  PORT="$(jq -r '.port // 30000' "$TARGET")"
  REMOTE_OUTPUTS_DIR="$(jq -r '.remote_outputs_dir // ""' "$TARGET")"
  TARGET_GPU_MODEL="$(jq -r '.gpu_model' "$TARGET")"
  TARGET_GPU_COUNT="$(jq -r '.gpu_count // 0' "$TARGET")"
  TARGET_GPU_MEM="$(jq -r '.gpu_memory_gb // 0' "$TARGET")"
  MAX_CAND="$(jq -r '.search.max_candidates // 1' "$JOB")"
fi

CONTAINER_NAME="llm-infer-tuner-${JOB_ID}"

echo "▶ 执行器参数一览:" >&2
echo "    job_id=$JOB_ID  max_candidates=$MAX_CAND" >&2
echo "    ssh=$SSH_TARGET" >&2
  echo "    image=$IMAGE_REF  container=$CONTAINER_NAME  port=$PORT" >&2
echo "    model(host)=$MODEL_HOST_DIR" >&2
echo "    model(container)=$MODEL_CONTAINER_PATH" >&2
echo "    configs=$CONFIGS  results=$RESULTS" >&2
echo >&2

exec uv run python -m runners.executor \\
  --job "$JOB" \\
  --configs "$CONFIGS" \\
  --results "$RESULTS" \\
  --ssh-target "$SSH_TARGET" \\
  --ssh-password "$SSH_PASSWORD" \\
  --image-ref "$IMAGE_REF" \\
  --model-host-dir "$MODEL_HOST_DIR" \\
  --model-container-path "$MODEL_CONTAINER_PATH" \\
  --container-name "$CONTAINER_NAME" \\
  --port "$PORT" \\
  --max-candidates "$MAX_CAND" \\
  --remote-outputs-dir "$REMOTE_OUTPUTS_DIR" \\
  --target-gpu-model "$TARGET_GPU_MODEL" \\
  --target-gpu-count "$TARGET_GPU_COUNT" \\
  --target-gpu-memory-gb "$TARGET_GPU_MEM"
