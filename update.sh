#!/bin/bash
# update.sh —— 一键更新 SGLang 参数 + HuggingFace 模型信息
#
# 用法:
#   ./update.sh
#
# 做三件事:
#   0. git pull SGLang 仓库到最新(确保 cookbook 和源码不过期)
#   1. 扫描 SGLang 所有 git tag,发现新版本自动提取参数写入 catalogs/sglang-images.yaml
#   2. 扫描 SGLang cookbook 发现新模型,自动从 HuggingFace 拉 config.json
#      提取架构信息写入 catalogs/models.yaml
#
# 有变化时:自动更新 yaml 文件 + 打印 diff 摘要,你手动 git push
# 没变化时:静默跳过
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# 加载 .env
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

exec ./scripts/run_daily_sync.sh "$@"
