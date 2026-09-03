#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ "$#" -eq 0 ]]; then
  exec bash
fi

exec "$@"
