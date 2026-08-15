#!/bin/bash
# run_executor.sh —— 第二阶段执行器的一键封装。
# 从 input/targets/<target>.json 读「目标机器部署事实」(ssh/模型路径/镜像/容器名),
# 自动拼成 python -m runners.executor 的一长串参数,免得手敲。
#
# 用法(方案 A:机器信息全在 target.json,路径自动推):
#   ./run_executor.sh <job.json> <target.json> [configs.jsonl] [results_dir]
# 例:
#   ./run_executor.sh input/jobs/qwen36_35b_pro5000_0814.json input/targets/pro5000_s3.json
#   # configs 默认 outputs/<job_id>/configs.jsonl,results 默认 outputs/<job_id>/results
#
# 说明:targets.json 只存「非机密的部署事实」(IP/模型路径/镜像/端口)—— 换机器只换这个文件。
#       SSH 走 key 免密(remote.py 用 BatchMode=yes,不读密码),所以这里不存密码。
set -euo pipefail

JOB="${1:?用法: ./run_executor.sh <job.json> <target.json> [configs.jsonl] [results_dir]}"
TARGET="${2:?缺 target.json,例 input/targets/pro5000_s3.json}"

command -v jq >/dev/null || { echo "❌ 需要 jq" >&2; exit 1; }
[ -f "$JOB" ]     || { echo "❌ job 不存在: $JOB" >&2; exit 1; }
[ -f "$TARGET" ]  || { echo "❌ target 不存在: $TARGET" >&2; exit 1; }

# 路径全部按 job_id 自动推(第3/4参可覆盖)
JOB_ID="$(jq -r '.job_id' "$JOB")"
CONFIGS="${3:-outputs/${JOB_ID}/configs.jsonl}"
RESULTS="${4:-outputs/${JOB_ID}/results}"
[ -f "$CONFIGS" ] || { echo "❌ configs 不存在: $CONFIGS(先跑 ./gen_configs.sh $JOB)" >&2; exit 1; }

# 从 target.json 取「机器事实」
SSH_TARGET="$(jq -r '.ssh_target' "$TARGET")"
MODEL_HOST_DIR="$(jq -r '.model_host_dir' "$TARGET")"
MODEL_CONTAINER_PATH="$(jq -r '.model_container_path' "$TARGET")"
IMAGE_REF="$(jq -r '.image_ref' "$TARGET")"
PORT="$(jq -r '.port // 30000' "$TARGET")"
REMOTE_OUTPUTS_DIR="$(jq -r '.remote_outputs_dir // ""' "$TARGET")"

# 容器名按 job 命名(llmopt-<job_id>),不同 job 各起各的容器,永不撞名。
CONTAINER_NAME="llmopt-${JOB_ID}"
# 候选数从 job.json 的 search.max_candidates 读
MAX_CAND="$(jq -r '.search.max_candidates // 1' "$JOB")"

echo "▶ 执行器参数一览:" >&2
echo "    job=$JOB  job_id=$JOB_ID  max_candidates=$MAX_CAND" >&2
echo "    target=$TARGET  ssh=$SSH_TARGET" >&2
echo "    image=$IMAGE_REF  container=$CONTAINER_NAME  port=$PORT" >&2
echo "    model(host)=$MODEL_HOST_DIR" >&2
echo "    model(container)=$MODEL_CONTAINER_PATH" >&2
echo "    configs=$CONFIGS  results=$RESULTS" >&2
echo >&2

exec uv run python -m runners.executor \
  --job "$JOB" \
  --configs "$CONFIGS" \
  --results "$RESULTS" \
  --ssh-target "$SSH_TARGET" \
  --image-ref "$IMAGE_REF" \
  --model-host-dir "$MODEL_HOST_DIR" \
  --model-container-path "$MODEL_CONTAINER_PATH" \
  --container-name "$CONTAINER_NAME" \
  --port "$PORT" \
  --max-candidates "$MAX_CAND" \
  --remote-outputs-dir "$REMOTE_OUTPUTS_DIR"
