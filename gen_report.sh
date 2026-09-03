#!/bin/bash
# gen_report.sh —— 从已有 benchmark output 一键生成便于人工阅读的 Markdown 报告。
#
# 用法：
#   ./gen_report.sh outputs/<job_id>
#   ./gen_report.sh outputs/<job_id>/results
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "用法: ./gen_report.sh <output目录> [--top-k N]" >&2
  echo "示例: ./gen_report.sh outputs/<job_id>" >&2
  exit 1
fi

command -v uv >/dev/null || {
  echo "❌ 需要 uv，请先安装或配置项目运行环境" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec uv run python "$SCRIPT_DIR/scripts/generate_best_config_md.py" "$@"
