#!/bin/bash
# run_daily_sync.sh —— 每日同步 SGLang 参数和 HuggingFace 模型信息
#
# 用法: ./scripts/run_daily_sync.sh
#       或用 cron 每天自动跑:
#         crontab -e
#         0 9 * * * cd /data/home/dorianwu/aaawhx-study/llm-infer-tuner && ./scripts/run_daily_sync.sh >> logs/sync.log 2>&1
#
# 做两件事:
#   1. 从 SGLang 源码提取参数 → 更新 images.yaml + 生成约束报告
#   2. 从 HuggingFace API 拉模型 config.json → 对比 models.yaml + 生成差异报告
#
# 有变化时:自动更新 yaml 文件 + 打印 diff 摘要,你手动 git push。
# 没变化时:静默跳过。
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# SGLang 版本列表(每次新增 tag 时在这里加一行)
SGLANG_TAGS=("v0.5.10" "v0.5.13" "v0.5.16")

# 额外关注的新模型(HF model ID,空格分隔)
WATCH_MODELS=""

LOG_DIR="logs"
mkdir -p "$LOG_DIR" reports

TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo "Daily Sync: $TIMESTAMP"
echo "========================================"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 同步 SGLang 参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "[1/2] Syncing SGLang parameters..."
for tag in "${SGLANG_TAGS[@]}"; do
  echo "  --- $tag ---"
  python3 scripts/sync_sglang_params.py --tag "$tag" 2>&1 || echo "  [warn] $tag sync failed"
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 同步 HuggingFace 模型信息
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "[2/2] Syncing HuggingFace models..."
if [ -n "$WATCH_MODELS" ]; then
  python3 scripts/sync_hf_models.py --watch $WATCH_MODELS 2>&1 || echo "  [warn] hf sync failed"
else
  python3 scripts/sync_hf_models.py 2>&1 || echo "  [warn] hf sync failed"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 检查是否有变化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "Checking for changes..."
if git diff --quiet .claude/skills/sglang-server-config-gen/images.yaml catalogs/models.yaml 2>/dev/null; then
  echo "No changes detected."
else
  echo "*** Changes detected! ***"
  git diff --stat .claude/skills/sglang-server-config-gen/images.yaml catalogs/models.yaml
  echo ""
  echo "To push: git add -A && git commit -m 'Daily sync: update catalogs' && git push"
fi

echo ""
echo "Done. Reports in reports/ directory."
