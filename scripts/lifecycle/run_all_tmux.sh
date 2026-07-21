#!/usr/bin/env bash
set -euo pipefail

WS="${LSTE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export LSTE_NO_ATTACH=1

if tmux has-session -t "=sappo-lste" 2>/dev/null; then
  echo "[all] Stopping the conflicting standalone SA-PPO session."
  tmux kill-session -t "=sappo-lste"
fi

"$WS/scripts/lifecycle/run_env_tmux.sh"

if ! tmux has-session -t "=lste" 2>/dev/null; then
  source "$WS/scripts/config/pipeline_env.sh"
  echo "[all] Waiting for Gazebo services..."
  for _ in $(seq 1 120); do
    if rosservice list 2>/dev/null | grep -qx /gazebo/spawn_urdf_model; then
      break
    fi
    sleep 1
  done
  if ! rosservice list 2>/dev/null | grep -qx /gazebo/spawn_urdf_model; then
    echo "[error] Gazebo spawn service did not become ready." >&2
    exit 1
  fi
  "$WS/scripts/lifecycle/run_nodes_tmux.sh"
else
  echo "[all] Reusing the existing 'lste' session."
  source "$WS/scripts/config/pipeline_env.sh"
  if rosnode list 2>/dev/null | grep -qx /StageEnv_0; then
    "$WS/scripts/lifecycle/switch_controller.sh" sappo
  elif rosnode list 2>/dev/null | grep -qx /teleop_twist_keyboard_reset; then
    "$WS/scripts/lifecycle/switch_controller.sh" teleop
  else
    "$WS/scripts/lifecycle/switch_controller.sh" sappo
  fi
fi

for session in lste-env lste lste-teleop; do
  if ! tmux has-session -t "=$session" 2>/dev/null; then
    echo "[error] Required tmux session was not created: $session" >&2
    exit 1
  fi
done

echo ""
echo "============================================"
echo "  LSTE complete system is ready"
echo "  Environment:  tmux attach -t lste-env"
echo "  Brain/RL:     tmux attach -t lste"
echo "  Keyboard:     teleop"
echo "============================================"
