#!/usr/bin/env bash
# Fetch the Go2-W URDF + meshes (BSD-3, Unitree Robotics) into third_party/unitree_ros at a pinned commit.
# The meshes are ~30 MB, so they are not committed to this repository.
set -euo pipefail

COMMIT="${UNITREE_ROS_COMMIT:-ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee}"  # master @ 2026-09-16
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/unitree_ros"

if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_ros.git "$DEST"
fi
git -C "$DEST" sparse-checkout set robots/go2w_description
git -C "$DEST" fetch --quiet origin "$COMMIT" || true
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$COMMIT"

URDF="$DEST/robots/go2w_description/urdf/go2w_description.urdf"
test -f "$URDF"
echo "Go2-W description ready: $URDF (commit $COMMIT)"
