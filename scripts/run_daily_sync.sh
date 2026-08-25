#!/bin/bash
# run_daily_sync.sh —— 每日自动同步 SGLang 参数和 HuggingFace 模型信息
#
# 用法:
#   ./scripts/run_daily_sync.sh
#
# cron 每天自动跑:
#   crontab -e
#   0 9 * * * cd /path/to/llm-infer-tuner && ./scripts/run_daily_sync.sh >> logs/sync.log 2>&1
#
# 做两件事:
#   1. 扫描 SGLang 所有 git tag,发现新版本自动提取参数写入 catalogs/sglang-images.yaml
#      已有版本检查参数是否有变化,有变化就更新
#   2. 扫描 SGLang cookbook 发现新模型,自动从 HuggingFace 拉 config.json
#      提取架构信息写入 models.yaml(标 [AUTO] 待人工补全)
#      已有模型检查 config.json 是否有变化
#
# 有变化时:自动更新 yaml 文件 + 打印 diff 摘要,你手动 git push
# 没变化时:静默跳过
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# SGLang 本地仓库路径(用于 cookbook 扫描,不用每次 clone)
SGLANG_REPO="${SGLANG_REPO:-/data/home/dorianwu/sglang-latest}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR" reports

TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo "Daily Sync: $TIMESTAMP"
echo "========================================"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 同步 SGLang 参数(自动扫描所有 tag)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "[1/3] Syncing SGLang parameters (auto-scan all tags)..."
python3 scripts/sync_sglang_params.py 2>&1 || echo "  [warn] sglang sync failed"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 同步 HuggingFace 模型信息(自动扫描 cookbook + 已有模型变更检测)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "[2/3] Syncing HuggingFace models (auto-scan cookbook)..."
python3 scripts/sync_hf_models.py --sglang-repo "$SGLANG_REPO" 2>&1 || echo "  [warn] hf sync failed"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 同步 GPU 信息
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "[3/3] Syncing GPU info..."
python3 scripts/sync_gpu.py 2>&1 || echo "  [warn] gpu sync failed"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 检查是否有变化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "Checking for changes..."
CHANGED_FILES=""
git diff --quiet catalogs/sglang-images.yaml 2>/dev/null || CHANGED_FILES="$CHANGED_FILES sglang-images.yaml"
git diff --quiet catalogs/models.yaml 2>/dev/null || CHANGED_FILES="$CHANGED_FILES models.yaml"

if [ -z "$CHANGED_FILES" ]; then
  echo "No changes detected."
else
  echo "*** Changes detected! ***"
  git diff --stat catalogs/sglang-images.yaml catalogs/models.yaml
  echo ""
  echo "To push: git add -A && git commit -m 'Daily sync: update catalogs' && git push"
fi

echo ""
echo "Done. Reports in reports/ directory."
echo "  - reports/models_diff.md (新模型 + 已有模型差异)"
echo "  - reports/constraints_*.md (每版 SGLang 的约束报告)"
