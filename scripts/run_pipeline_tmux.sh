#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# 简易一键启动脚本：用 tmux 分窗启动仿真 + VLLM + prompt + DINO 检测 + score/state + 可视化。
# 可通过环境变量覆盖默认值，例如：
#   WS=/home/zrz/lste_ws TASK_JSON=... TASK_ID=... VLLM_URL=http://localhost:8000/v1 ./scripts/run_pipeline_tmux.sh

WS=${WS:-/home/zrz/lste_ws}
SESSION=${SESSION:-lste}
WORLD=${WORLD:-$WS/src/lste_core/worlds/topo_test2.world}
TASK_JSON=${TASK_JSON:-$WS/model/Data_exchange/vlm_prompt/lab/yellow_cup.json}
TASK_ID=${TASK_ID:-yellow_cup}
VLLM_URL=${VLLM_URL:-http://localhost:8000/v1}
VLLM_MODEL=${VLLM_MODEL:-$WS/model/MiniCPM/OpenBMB/MiniCPM4-0___5B}
VLLM_PROBE=${VLLM_PROBE:-${VLLM_URL%/}}
ACCESS_TOPO_CONFIG=${ACCESS_TOPO_CONFIG:-$WS/src/lste_topo_access/topo_tree/cfgs/access_topo.yaml}
ACCESS_TOPO_TEST_NAME=${ACCESS_TOPO_TEST_NAME:-default_test}
if [[ "$VLLM_PROBE" == */v1 ]]; then
  VLLM_PROBE="$VLLM_PROBE/models"
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux 未安装，请先安装 tmux。" >&2
  exit 1
fi

# 自动判断 world 是否已经包含 pro3 模型，包含则跳过二次 spawn
SPAWN_PRO3=true
if [[ -f "$WORLD" ]] && grep -q "<model name='pro3'>" "$WORLD"; then
  SPAWN_PRO3=false
fi

tmux_new_window() {
  local index="$1"
  local dir="$2"
  local title="$3"
  local cmd="$4"
  tmux new-window -t "$SESSION:$index" -n "$title" -c "$dir" \
    "bash -lc 'WS=\"$WS\"; source \"$WS/scripts/pipeline_env.sh\"; set -e; $cmd'; echo; echo '[EXIT] $title'; exec bash"
}

# 如果 session 已存在则复用，避免重复启动
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' 已存在，直接附着。" >&2
  exec tmux attach -t "$SESSION"
fi

# 新建 session：tmux 默认会创建 window 0，所以直接把 window 0 用作 roscore
tmux new-session -d -s "$SESSION" -c "$WS" -n "roscore" \
  "bash -lc 'WS=\"$WS\"; source \"$WS/scripts/pipeline_env.sh\"; roscore'; echo; echo '[EXIT] roscore'; exec bash"
tmux set-option -t "$SESSION" remain-on-exit on

# helper: wait for roscore
WAIT_ROSCORE='until rostopic list >/dev/null 2>&1; do echo \"waiting for roscore...\"; sleep 1; done'

# 1: 仿真 + 机器人（等待 master 就绪）
tmux_new_window 1 "$WS" "world" \
  "$WAIT_ROSCORE; roslaunch lste_core lab_with_pro3.launch world_name:=$WORLD spawn_pro3:=$SPAWN_PRO3"

# 2: 发布任务（latched，可随时替换 json/task_id）
tmux_new_window 2 "$WS" "task" \
  "$WAIT_ROSCORE; rosrun lste_core lste_task_node.py _json_path:=$TASK_JSON _task_id:=$TASK_ID"

# 3: 启动 VLLM 服务（MiniCPM）
tmux_new_window 3 "$WS/model/MiniCPM/test" "vllm" \
  "conda activate minicpm; bash start.sh"

# 4: 全局目标（/lste/final_goal）
tmux_new_window 4 "$WS" "goal" \
  "$WAIT_ROSCORE; rosrun lste_topo_access lste_goal_manager.py"

# 5: 等待 VLLM 就绪后启动 prompt 节点
tmux_new_window 5 "$WS" "prompt" \
  "$WAIT_ROSCORE; conda activate minicpm; \
   echo \"等待 VLLM 就绪...\"; \
   for i in \$(seq 1 120); do \
     if command -v curl >/dev/null 2>&1 && curl -sSf \"$VLLM_PROBE\" >/dev/null 2>&1; then echo \"VLLM ready\"; break; fi; \
     if ! command -v curl >/dev/null 2>&1; then \
       python -c \"import urllib.request; urllib.request.urlopen(\\\"$VLLM_PROBE\\\", timeout=1)\" >/dev/null 2>&1 && echo \"VLLM ready\" && break || true; \
     fi; \
     sleep 2; \
   done; \
   rosparam set /lste_prompt_node/vllm_stop_command \"pkill -f vllm.*serve\"; \
   rosrun lste_core lste_prompt_node.py _vllm_base_url:=$VLLM_URL _vllm_model_name:=$VLLM_MODEL"

# 6: DINO 检测
tmux_new_window 6 "$WS" "dino" \
  "$WAIT_ROSCORE; conda activate dino; export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7; rosrun lste_core lste_det_node.py"

# 7: score
tmux_new_window 7 "$WS" "score" \
  "$WAIT_ROSCORE; rosrun lste_core lste_score_node.py \
    _w_target:=0.2 _w_env:=0.3 _w_ctx:=0.5 \
    _lambda_neg:=0.7 _pos_midpoint:=0.25 _pos_steepness:=6 \
    _neg_midpoint:=0.15 _neg_steepness:=12"

# 8: state
tmux_new_window 8 "$WS" "state" \
  "$WAIT_ROSCORE; rosrun lste_core lste_state_node.py"

# 9: 可视化（叠加图 + RViz）
tmux_new_window 9 "$WS" "vis" \
  "$WAIT_ROSCORE; roslaunch lste_core lste_det_vis.launch"

# 10: oc_srfc（提供 /rbt_pose 等；goal_manager 依赖 /rbt_pose 才会发布 /lste/final_goal）
tmux_new_window 10 "$WS" "oc_srfc" \
  "$WAIT_ROSCORE; roslaunch lste_oc_srfc oc_srfc_proj.launch"

# 11: topo frontier（需 vsgp 环境）
tmux_new_window 11 "$WS" "gp_frontier" \
  "$WAIT_ROSCORE; conda activate vsgp; roslaunch lste_topo_access gp_frontier.launch \
    access_topo_config:=$ACCESS_TOPO_CONFIG access_topo_test_name:=$ACCESS_TOPO_TEST_NAME"

tmux attach -t "$SESSION"
