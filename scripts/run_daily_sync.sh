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
# 做三件事:
#   0. git pull SGLang 仓库到最新(确保 cookbook 和源码是最新版)
#   1. 扫描 SGLang 所有 git tag,发现新版本自动提取参数写入 catalogs/sglang-images.yaml
#   2. 扫描 SGLang cookbook 发现新模型,自动从 HuggingFace 拉 config.json
#      提取架构信息写入 models.yaml
#
# 有变化时:自动更新 yaml 文件 + 打印 diff 摘要,你手动 git push
# 没变化时:静默跳过
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# SGLang 本地仓库路径(用于 cookbook 扫描和源码提取)
SGLANG_REPO="${SGLANG_REPO:-/data/home/dorianwu/sglang-latest}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR" reports

TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo "Daily Sync: $TIMESTAMP"
echo "========================================"

# 记录各步失败,末尾统一以非零码退出——否则 `|| echo [warn]` 会把失败吞成
# exit 0,cron 永远以为同步成功,logs/sync.log 里的报错没人看。
SYNC_FAILURES=""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 0. 更新 SGLang 仓库到最新(确保 cookbook 和源码不过期)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "[0/3] Updating SGLang repo ($SGLANG_REPO)..."
if [ -d "$SGLANG_REPO/.git" ]; then
  cd "$SGLANG_REPO"
  git fetch origin --quiet 2>&1 || echo "  [warn] git fetch failed"
  git checkout main --quiet 2>&1 || echo "  [warn] git checkout main failed"
  git pull origin main --quiet 2>&1 || echo "  [warn] git pull failed"
  LATEST_COMMIT=$(git log --oneline -1)
  echo "  -> SGLang updated: $LATEST_COMMIT"
  cd "$PROJECT_ROOT"
else
  echo "  [warn] SGLang repo not found at $SGLANG_REPO, skipping update"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 同步 SGLang 参数(自动扫描所有 tag)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "[1/3] Syncing SGLang parameters (auto-scan all tags)..."
python3 scripts/sync_sglang_params.py 2>&1 || { echo "  [warn] sglang sync failed (exit $?)"; SYNC_FAILURES="$SYNC_FAILURES sglang-params"; }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 同步 HuggingFace 模型信息(自动扫描 cookbook + 已有模型变更检测)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "[2/3] Syncing HuggingFace models (auto-scan cookbook)..."
python3 scripts/sync_hf_models.py --sglang-repo "$SGLANG_REPO" 2>&1 || { echo "  [warn] hf sync failed (exit $?)"; SYNC_FAILURES="$SYNC_FAILURES hf-models"; }

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

if [ -n "$SYNC_FAILURES" ]; then
  echo ""
  echo "*** SYNC FAILED:${SYNC_FAILURES} —— 见上方日志,catalogs 可能未更新 ***" >&2
  exit 1
fi
