#!/bin/bash
# run_executor.sh —— 第二阶段:远程压测 + 排名。
#
# 用法:  ./run_executor.sh <job.json> <target.json> [configs.jsonl] [results_dir]
#        默认 configs = outputs/<job_id>/configs.jsonl
#        默认 results = outputs/<job_id>/results
#
# 脚本分 5 步确定性操作,不涉及 AI 决策:
#   1. 前置检查  — 验证 jq/job/target/configs 都就位
#   2. 解析路径  — 从 job.json 读 job_id,拼 configs/results 路径
#   3. 读取 target.json — 提取 SSH/模型路径/镜像/端口/GPU 等机器事实
#   4. 读取 job.json 额外字段 — 容器名 + max_candidates
#   5. 启动执行器  — 拼参数调 uv run python -m runners.executor
#
# 执行器(runners/executor.py)接管后:
#   preflight 校验 → 分配 GPU/端口 → 逐批起容器 → 起服务 → 压测 → 排名
#   → 输出 outputs/<job_id>/results/ranking.json
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 步:前置检查 — 验证依赖和输入文件都就位
# ─────────────────────────────────────────────────────────────────────────
#   • jq      — 读 job.json/target.json 需要
#   • job.json — 第 1 参数,测什么(模型/卡/负载/SLA)
#   • target.json — 第 2 参数,在哪测(IP/模型路径/镜像/端口)
#   • configs.jsonl — 第一步 gen_configs.sh 的产物,候选启动配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOB="${1:?用法: ./run_executor.sh <job.json> <target.json> [configs.jsonl] [results_dir]}"
TARGET="${2:?缺 target.json,例 input/targets/qwen36-35b-a3b-fp8_pro5000_8x72g_qa-chat-3.5k-1k_cand2__s3.json}"

command -v jq >/dev/null || { echo "❌ 需要 jq" >&2; exit 1; }
[ -f "$JOB" ]     || { echo "❌ job 不存在: $JOB" >&2; exit 1; }
[ -f "$TARGET" ]  || { echo "❌ target 不存在: $TARGET" >&2; exit 1; }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 步:解析路径 — 从 job.json 读 job_id,拼 configs 和 results 路径
# ─────────────────────────────────────────────────────────────────────────
#   • configs 默认 outputs/<job_id>/configs.jsonl(第一步生成的)
#   • results 默认 outputs/<job_id>/results(执行器写排名结果)
#   • 第 3/4 参数可覆盖默认路径
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOB_ID="$(jq -r '.job_id' "$JOB")"
CONFIGS="${3:-outputs/${JOB_ID}/configs.jsonl}"
RESULTS="${4:-outputs/${JOB_ID}/results}"
[ -f "$CONFIGS" ] || { echo "❌ configs 不存在: $CONFIGS(先跑 ./gen_configs.sh $JOB)" >&2; exit 1; }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 步:读取 target.json — 提取目标机器的部署事实
# ─────────────────────────────────────────────────────────────────────────
#   • ssh_target      — 远程服务器地址(如 ubuntu@122.51.115.16)
#   • ssh_password    — 选填,有密码用 sshpass,留空走 key 免密
#   • model_host_dir  — 模型权重在服务器宿主机上的路径(docker -v 挂载源)
#   • model_container_path — 模型权重在容器内看到的路径(launch_server --model-path 用这个)
#   • image_ref       — Docker 镜像完整地址(docker run 用)
#   • port            — 服务端口(默认 30000,并行时每候选自动递增)
#   • remote_outputs_dir — 远程输出目录(默认 $HOME/llm-infer-tuner-outputs/<job>)
#   • gpu_model/count/memory_gb — 硬件校验用,跟 job.json 比对防跑错机器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 步:读取 job.json 额外字段 — 容器名 + 候选数
# ─────────────────────────────────────────────────────────────────────────
#   • container_name — 按 job_id 命名,不同 job 各起各的容器,永不撞名
#   • max_candidates — 从 job.json 的 search.max_candidates 读,控制测多少条候选
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTAINER_NAME="llm-infer-tuner-${JOB_ID}"
MAX_CAND="$(jq -r '.search.max_candidates // 1' "$JOB")"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 步:启动执行器 — 拼参数调 runners.executor
# ─────────────────────────────────────────────────────────────────────────
#   uv run python -m runners.executor 接管后做:
#     preflight(SSH/模型/镜像/清理旧容器) → 硬件校验
#     → 分配 GPU/端口(按 tp_size) → 分批
#     → 逐批:起容器 → 起服务 → health 检查 → 自适应并发压测 → 停容器
#     → Round 2 精搜 top-K
#     → rank_candidates 按 goodput_per_host 排名
#     → 写 outputs/<job_id>/results/ranking.json
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
  --ssh-password "$SSH_PASSWORD" \
  --image-ref "$IMAGE_REF" \
  --model-host-dir "$MODEL_HOST_DIR" \
  --model-container-path "$MODEL_CONTAINER_PATH" \
  --container-name "$CONTAINER_NAME" \
  --port "$PORT" \
  --max-candidates "$MAX_CAND" \
  --remote-outputs-dir "$REMOTE_OUTPUTS_DIR" \
  --target-gpu-model "$TARGET_GPU_MODEL" \
  --target-gpu-count "$TARGET_GPU_COUNT" \
  --target-gpu-memory-gb "$TARGET_GPU_MEM"
