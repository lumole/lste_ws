#!/usr/bin/env bash
set -euo pipefail

SESSIONS=(lste-teleop sappo-lste lste lste-env)

collect_residual_pids() {
  {
    pgrep -x gzclient || true
    pgrep -x gzserver || true
    pgrep -f '/opt/ros/noetic/bin/roscore' || true
    pgrep -f '/opt/ros/noetic/bin/rosmaster --core' || true
    pgrep -f 'sappo_pure.py' || true
    pgrep -f 'teleop_with_reset.py' || true
  } | sort -nu
}

for session in "${SESSIONS[@]}"; do
  if tmux has-session -t "=$session" 2>/dev/null; then
    echo "[stop] Stopping tmux session: $session"
    tmux kill-session -t "=$session"
  else
    echo "[stop] Session already stopped: $session"
  fi
done

remaining=()
for session in "${SESSIONS[@]}"; do
  if tmux has-session -t "=$session" 2>/dev/null; then
    remaining+=("$session")
  fi
done

if (( ${#remaining[@]} > 0 )); then
  echo "[error] Failed to stop sessions: ${remaining[*]}" >&2
  exit 1
fi

for _ in $(seq 1 30); do
  mapfile -t residual_pids < <(collect_residual_pids)
  if (( ${#residual_pids[@]} == 0 )); then
    break
  fi
  sleep 0.1
done

mapfile -t residual_pids < <(collect_residual_pids)
if (( ${#residual_pids[@]} > 0 )); then
  echo "[stop] Terminating residual ROS/Gazebo processes: ${residual_pids[*]}"
  kill -TERM "${residual_pids[@]}" 2>/dev/null || true
  sleep 1
fi

mapfile -t residual_pids < <(collect_residual_pids)
if (( ${#residual_pids[@]} > 0 )); then
  echo "[stop] Forcing residual processes to exit: ${residual_pids[*]}"
  kill -KILL "${residual_pids[@]}" 2>/dev/null || true
fi

echo "[stop] LSTE complete system is stopped."
