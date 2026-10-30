#!/usr/bin/env bash
# コンテナ内で wheel_legged_rl のスクリプトを Isaac Lab の Python で実行する。
#
#   ./wl.sh joint_test.py --robot go2w --headless
#   ./wl.sh train.py --task WL-Go2W-Phase1-Flat-v0 --headless --num_envs 64 --max_iterations 5
#   ./wl.sh -m pytest                      # 引数が "-" で始まる場合は python.sh にそのまま渡す
#   ./wl.sh bash                           # コンテナのシェル
#
# コンテナが停止していれば起動する。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SERVICE=isaac-lab-wl

if [ -z "$(docker compose ps -q --status running "$SERVICE")" ]; then
  docker compose up -d "$SERVICE" >/dev/null
fi

TTY=()
[ -t 0 ] && [ -t 1 ] && TTY=(-it)

if [ "${1:-}" = "bash" ]; then
  exec docker compose exec "${TTY[@]}" "$SERVICE" bash
elif [[ "${1:-}" == -* ]]; then
  exec docker compose exec "${TTY[@]}" "$SERVICE" /workspace/isaaclab/isaaclab.sh -p "$@"
else
  script="$1"
  shift
  exec docker compose exec "${TTY[@]}" "$SERVICE" /workspace/isaaclab/isaaclab.sh -p "scripts/$script" "$@"
fi
