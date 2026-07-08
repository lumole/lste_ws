#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1


# Auto-detect workspace root (can override with LSTE_WS env var)
if [ -z "${WS:-}" ]; then
  WS="${LSTE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
LSTE_WS="$WS"
SESSION=${SESSION:-lste}
# 可选 YAML 配置：通过 PIPELINE_CONFIG 指定；存在时为未显式设置的变量提供默认值
PIPELINE_CONFIG=${PIPELINE_CONFIG:-$WS/scripts/pipeline_defaults.yaml}
if [[ -f "$PIPELINE_CONFIG" ]]; then
  eval "$(
    python - "$PIPELINE_CONFIG" <<'PY' || true
import sys, json
path = sys.argv[1]
try:
    import yaml
except ImportError:
    sys.exit(0)
with open(path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f) or {}
for k, v in data.items():
    if isinstance(v, (str, int, float)):
        # 简单输出 KEY=VALUE 供 bash eval，只支持标量
        print(f'CFG_{k}={v}')
PY
  )"
fi
SESSION=${SESSION:-${CFG_SESSION:-lste}}

# Helper: resolve a path — if relative, prepend $WS/
_resolve() {
  local val="$1"
  if [[ -z "$val" || "$val" == http://* || "$val" == https://* || "$val" == /* ]]; then
    echo "$val"
  else
    echo "$WS/$val"
  fi
}

WORLD=${WORLD:-$(_resolve "${CFG_WORLD:-worlds/place1.world}")}
TASK_JSON=${TASK_JSON:-$(_resolve "${CFG_TASK_JSON:-model/Data_exchange/vlm_prompt/lab/yellow_cup.json}")}
TASK_ID=${TASK_ID:-${CFG_TASK_ID:-yellow_cup}}
VLLM_URL=${VLLM_URL:-${CFG_VLLM_URL:-http://localhost:8000/v1}}
VLLM_MODEL=${VLLM_MODEL:-$(_resolve "${CFG_VLLM_MODEL:-model/MiniCPM/OpenBMB/MiniCPM4-0___5B}")}
VLLM_PROBE=${VLLM_PROBE:-${VLLM_URL%/}}
ACCESS_TOPO_CONFIG=${ACCESS_TOPO_CONFIG:-$(_resolve "${CFG_ACCESS_TOPO_CONFIG:-src/lste_topo_access/topo_tree/cfgs/access_topo.yaml}")}
ACCESS_TOPO_CONFIG_PASS=${ACCESS_TOPO_CONFIG_PASS:-$(_resolve "${CFG_ACCESS_TOPO_CONFIG_PASS:-src/lste_topo_access/topo_tree/cfgs/access_topo_pass.yaml}")}
ACCESS_TOPO_CONFIG_SUS_C=${ACCESS_TOPO_CONFIG_SUS_C:-$(_resolve "${CFG_ACCESS_TOPO_CONFIG_SUS_C:-src/lste_topo_access/topo_tree/cfgs/access_topo_sus_c.yaml}")}
ACCESS_TOPO_TEST_NAME=${ACCESS_TOPO_TEST_NAME:-${CFG_ACCESS_TOPO_TEST_NAME:-default_test}}
ACCESS_TOPO_RUN_NAME=${ACCESS_TOPO_RUN_NAME:-${CFG_ACCESS_TOPO_RUN_NAME:-$ACCESS_TOPO_TEST_NAME}}
GP_FRONTIER_RVIZ=${GP_FRONTIER_RVIZ:-$(_resolve "${CFG_GP_FRONTIER_RVIZ:-src/lste_topo_access/launch/gp_frontier.rviz}")}
SUSPICIOUS_WINDOW=${SUSPICIOUS_WINDOW:-${CFG_SUSPICIOUS_WINDOW:-5}}
FOLLOW_LOCKED_DONE_TIME=${FOLLOW_LOCKED_DONE_TIME:-${CFG_FOLLOW_LOCKED_DONE_TIME:-4.0}}
FRONTIER_LOG=${FRONTIER_LOG:-${CFG_FRONTIER_LOG:-true}}
GUI=${GUI:-${CFG_GUI:-true}}
if [[ "$VLLM_PROBE" == */v1 ]]; then
  VLLM_PROBE="$VLLM_PROBE/models"
fi

if [[ ! -f "$WORLD" ]]; then
  echo "[error] WORLD file not found: $WORLD" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux 未安装，请先安装 tmux。" >&2
  exit 1
fi

# 避免 TF 本地库缓存问题：跑 pipeline 前清理 vsgp 环境下的 TF pyc，并做一次导入自检
export PYTHONDONTWRITEBYTECODE=1
if command -v conda >/dev/null 2>&1; then
  echo "[preflight] 清理 vsgp 环境下的 TensorFlow 缓存并做自检..."
  CONDA_PREFIX="$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")"
  conda run -n vsgp bash -lc "find \"$CONDA_PREFIX/envs/vsgp/lib/python3.7/site-packages/tensorflow\" -name '*.pyc' -delete 2>/dev/null; find \"$CONDA_PREFIX/envs/vsgp/lib/python3.7/site-packages/tensorflow\" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null" || true
  conda run -n vsgp python - <<'PY' || true
import sys, tensorflow as tf
print("TF sanity:", sys.executable, tf.__version__)
PY
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
  # 如果已有 session 的 WORLD 与当前需求不同，则先杀掉重新建，避免复用旧 world
  EXISTING_WORLD="$(tmux show-environment -t "$SESSION" 2>/dev/null | awk -F= '/^LSTE_WORLD=/{print substr($0,length("LSTE_WORLD=")+1)}')"
  if [[ -n "$EXISTING_WORLD" && "$EXISTING_WORLD" != "$WORLD" ]]; then
    echo "[info] tmux session '$SESSION' exists with WORLD=$EXISTING_WORLD, restart with WORLD=$WORLD."
    tmux kill-session -t "$SESSION"
  else
    echo "tmux session '$SESSION' 已存在，直接附着。" >&2
    exec tmux attach -t "$SESSION"
  fi
fi

# 新建 session：tmux 默认会创建 window 0，所以直接把 window 0 用作 roscore
tmux new-session -d -s "$SESSION" -c "$WS" -n "roscore" \
  "bash -lc 'WS=\"$WS\"; source \"$WS/scripts/pipeline_env.sh\"; roscore'; echo; echo '[EXIT] roscore'; exec bash"
tmux set-environment -t "$SESSION" LSTE_WORLD "$WORLD"
tmux set-option -t "$SESSION" remain-on-exit on

# helper: wait for roscore
WAIT_ROSCORE='until rostopic list >/dev/null 2>&1; do echo \"waiting for roscore...\"; sleep 1; done'

# 1: 仿真 + 机器人（等待 master 就绪）
tmux_new_window 1 "$WS" "world" \
  "$WAIT_ROSCORE; roslaunch lste_core lab_with_pro3.launch world_name:=$WORLD spawn_pro3:=$SPAWN_PRO3 gui:=$GUI"

# 2: 发布任务（latched，可随时替换 json/task_id）
tmux_new_window 2 "$WS" "task" \
  "$WAIT_ROSCORE; rosrun lste_core lste_task_node.py _json_path:=$TASK_JSON _task_id:=$TASK_ID"

# 3: 启动 VLLM 服务（MiniCPM）
tmux_new_window 3 "$WS/model/MiniCPM/test" "vllm" \
  "conda activate minicpm; bash start.sh"

# 4: 全局目标（/lste/final_goal）
tmux_new_window 4 "$WS" "goal" \
  "$WAIT_ROSCORE; rosrun lste_topo_access lste_goal_manager.py _follow_locked_done_time:=$FOLLOW_LOCKED_DONE_TIME"

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
  "$WAIT_ROSCORE; rosrun lste_core lste_state_node.py _suspicious_window:=$SUSPICIOUS_WINDOW"

# 9: 可视化（叠加图 + RViz）
tmux_new_window 9 "$WS" "vis" \
  "$WAIT_ROSCORE; roslaunch lste_core lste_det_vis.launch"

# 10: oc_srfc（提供 /rbt_pose 等；goal_manager 依赖 /rbt_pose 才会发布 /lste/final_goal）
tmux_new_window 10 "$WS" "oc_srfc" \
  "$WAIT_ROSCORE; roslaunch lste_oc_srfc oc_srfc_proj.launch"

# 11: topo frontier（需 vsgp 环境）
tmux_new_window 11 "$WS" "gp_frontier" \
  "$WAIT_ROSCORE; conda activate vsgp; roslaunch lste_topo_access gp_frontier.launch \
    access_topo_config:=$ACCESS_TOPO_CONFIG \
    access_topo_config_pass:=$ACCESS_TOPO_CONFIG_PASS \
    access_topo_config_sus_c:=$ACCESS_TOPO_CONFIG_SUS_C \
    access_topo_test_name:=$ACCESS_TOPO_TEST_NAME \
    access_topo_run_name:=$ACCESS_TOPO_RUN_NAME \
    follow_locked_done_time:=$FOLLOW_LOCKED_DONE_TIME \
    frontier_log:=$FRONTIER_LOG"

# 12: frontier RViz
tmux_new_window 12 "$WS" "rviz_frontier" \
  "$WAIT_ROSCORE; rviz -d $GP_FRONTIER_RVIZ"

# 13: teleop keyboard
tmux_new_window 13 "$WS" "teleop" \
  "$WAIT_ROSCORE; rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/cmd_vel"

tmux attach -t "$SESSION"
