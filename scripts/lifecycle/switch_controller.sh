#!/usr/bin/env bash
set -euo pipefail

WS="${LSTE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SESSION="lste"
TELEOP_SESSION="lste-teleop"
MODE="${1:-}"
SAPPO_SPEED="${SAPPO_SPEED:-0.50}"
SAPPO_PYTHON="${SAPPO_PYTHON:-$HOME/miniconda3/envs/rlenvs/bin/python}"
RL_DIR="$WS/rl_navigation"
RUNTIME_DIR="$WS/runtime/sappo"
POLICY_DIR="$RL_DIR/policy"

if [[ "$MODE" != "sappo" && "$MODE" != "teleop" ]]; then
  echo "Usage: $0 {sappo|teleop}" >&2
  exit 2
fi

if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "[error] LSTE session '$SESSION' is not running." >&2
  exit 1
fi

ros_node_exists() {
  bash -lc "source \"$WS/scripts/config/pipeline_env.sh\"; rosnode list 2>/dev/null | grep -qx \"$1\""
}

wait_node_ready() {
  local node="$1"
  local i
  for i in $(seq 1 50); do
    if ros_node_exists "$node"; then
      return 0
    fi
    sleep 0.1
  done
  echo "[error] ROS node did not start: $node" >&2
  return 1
}

window_alive() {
  tmux list-panes -t "=$SESSION:$1" -F '#{pane_dead}' 2>/dev/null | grep -qx 0
}

start_rl_scan() {
  if window_alive rl_scan; then
    return
  fi
  tmux kill-window -t "=$SESSION:rl_scan" 2>/dev/null || true
  tmux new-window -t "=$SESSION:" -n rl_scan -c "$WS" \
    "bash -lc 'source \"$WS/scripts/config/pipeline_env.sh\"; \
roslaunch pointcloud_to_laserscan lste_pro3_to_scan.launch; exec bash'"
}

start_sappo() {
  if window_alive sappo && ros_node_exists /StageEnv_0; then
    return
  fi
  if [[ ! -x "$SAPPO_PYTHON" || ! -f "$RL_DIR/sappo_pure.py" || \
        ! -f "$POLICY_DIR/sa_peppo_1650.pth" ]]; then
    echo "[error] SA-PPO runtime is incomplete." >&2
    exit 1
  fi
  local python_site
  python_site="$($SAPPO_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  mkdir -p "$RUNTIME_DIR"
  ln -sfn "$POLICY_DIR" "$RUNTIME_DIR/policy"
  tmux kill-window -t "=$SESSION:sappo" 2>/dev/null || true
  tmux new-window -t "=$SESSION:" -n sappo -c "$RUNTIME_DIR" \
    "bash -lc 'source \"$WS/scripts/config/pipeline_env.sh\"; \
until rostopic list | grep -qx /pro3/wheel_odom && rostopic list | grep -qx /pro3/rlscan; do sleep 1; done; \
PYTHONPATH=\"$python_site:$WS/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3/dist-packages\" \
\"$SAPPO_PYTHON\" \"$RL_DIR/sappo_pure.py\" \
  _linear_speed_scale:=$SAPPO_SPEED \
  _goal_topic:=/lste/final_goal \
  _cmd_vel_topic:=/lste/cmd_vel/sappo \
  _wait_for_goal:=true; exec bash'"
  wait_node_ready /StageEnv_0
}

start_teleop() {
  if tmux has-session -t "=$TELEOP_SESSION" 2>/dev/null && \
     ros_node_exists /teleop_twist_keyboard_reset; then
    return
  fi
  local command
  command="bash -lc 'source \"$WS/scripts/config/pipeline_env.sh\"; \
python \"$WS/scripts/tools/teleop_with_reset.py\" cmd_vel:=/lste/cmd_vel/teleop; exec bash'"
  if tmux has-session -t "=$TELEOP_SESSION" 2>/dev/null; then
    tmux respawn-pane -k -t "=$TELEOP_SESSION:teleop" -c "$WS" "$command"
  else
    tmux new-session -d -s "$TELEOP_SESSION" -n teleop -c "$WS" "$command"
    tmux set-option -t "=$TELEOP_SESSION:" remain-on-exit on
  fi
  wait_node_ready /teleop_twist_keyboard_reset
}

select_controller() {
  source "$WS/scripts/config/pipeline_env.sh"
  python3 - "$MODE" <<'PY'
import sys
import time

import rospy
from std_msgs.msg import String

mode = sys.argv[1]
rospy.init_node("lste_controller_select_cli", anonymous=True, disable_signals=True)
publisher = rospy.Publisher("/lste/controller_select", String, queue_size=1)
deadline = time.time() + 5.0
while publisher.get_num_connections() == 0 and time.time() < deadline:
    time.sleep(0.05)
if publisher.get_num_connections() == 0:
    raise SystemExit("controller switch node is not available")
publisher.publish(String(data=mode))
time.sleep(0.15)
PY
}

start_rl_scan
start_sappo
start_teleop
select_controller
echo "[controller] Selected $MODE; SA-PPO and teleop remain running."
