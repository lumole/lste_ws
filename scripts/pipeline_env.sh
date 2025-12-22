#!/usr/bin/env bash
set -euo pipefail

# Shared environment setup for tmux pipeline windows.
# - Loads ROS Noetic
# - Loads this workspace overlay (devel)
# - Initializes conda (without requiring `conda init`)

WS=${WS:-/home/zrz/lste_ws}

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
fi

