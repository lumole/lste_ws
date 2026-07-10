#!/usr/bin/env bash
set -euo pipefail

# Shared environment setup for tmux pipeline windows.
# - Loads ROS Noetic
# - Loads this workspace overlay (devel)
# - Initializes conda (without requiring `conda init`)

# Auto-detect workspace root (can override with LSTE_WS env var)
if [ -z "${WS:-}" ]; then
  WS="${LSTE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi

# Auto-detect pre_work workspace (for Gazebo plugins). Can override with LSTE_PRE_WS.
if [ -z "${LSTE_PRE_WS:-}" ]; then
  LSTE_PRE_WS="$(dirname "$WS")/pre_work"
  if [ ! -d "$LSTE_PRE_WS/devel/lib" ]; then
    LSTE_PRE_WS=""  # not found, leave empty
  fi
fi
export LSTE_PRE_WS
export LSTE_WS="$WS"

if [ -f /opt/ros/noetic/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
fi

if [ -f "$WS/devel/setup.bash" ]; then
  # shellcheck disable=SC1090
  source "$WS/devel/setup.bash"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "${CONDA_BASE:-}" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  else
    # fallback: try the shell hook
    eval "$(conda shell.bash hook 2>/dev/null || true)"
  fi
  # fix conda libffi.so.8 shadowing system libffi.so.7 (needed by apt opencv/PyKDL)
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:$LD_PRELOAD}
fi

# 统一 ROS master URI，避免 lste-env 和 lste session 各连不同的 roscore
export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311/}

