#!/usr/bin/env bash
# 初回のみ: コンテナが使うホスト側キャッシュを作成し、GUI 表示を許可する。
set -euo pipefail
BASE="$HOME/docker/isaac-sim-5.1"
mkdir -p "$BASE"/cache/{kit,ov,pip,glcache,computecache,wheel_legged_rl} "$BASE"/{logs,data,documents}
echo "created $BASE"
if command -v xhost >/dev/null && [ -n "${DISPLAY:-}" ]; then
  xhost +local:root >/dev/null && echo "xhost +local:root (GUI 表示用)"
fi
