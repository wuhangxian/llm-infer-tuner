#!/usr/bin/env bash
set -euo pipefail

# 首次启动时把镜像内置示例补到 /app/input。使用 -n，用户已编辑的同名文件不会被覆盖。
if [[ -d /opt/llm-infer-tuner-input && -d /app/input && -w /app/input ]]; then
  if ! cp -rn /opt/llm-infer-tuner-input/. /app/input/; then
    echo "⚠️  无法复制内置 input 示例，请检查 /app/input 写权限" >&2
  fi
fi

cd /app

if [[ "$#" -eq 0 ]]; then
  exec bash
fi

exec "$@"
