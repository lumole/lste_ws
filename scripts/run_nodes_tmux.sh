#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# ============================================
# 业务节点脚本：独立 session "lste"，依赖 lste-env 的 roscore + Gazebo
# 可反复重启调试，不影响环境 session
# 前置条件：先运行 ./scripts/run_env_tmux.sh
# ============================================

# ---- 路径解析（与 run_env_tmux.sh 完全一致） ----
if [ -z "${WS:-}" ]; then
  WS="${LSTE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
LSTE_WS="$WS"
ENV_SESSION=lste-env
SESSION=lste
PIPELINE_CONFIG=${PIPELINE_CONFIG:-$WS/scripts/pipeline_defaults.yaml}
if [[ -f "$PIPELINE_CONFIG" ]]; then
  eval "$(
    python - "$PIPELINE_CONFIG" <<'PY' || true
import sys
path = sys.argv[1]
try:
    import yaml
except ImportError:
    sys.exit(0)
with open(path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f) or {}
for k, v in data.items():
    if isinstance(v, (str, int, float)):
        print(f'CFG_{k}={v}')
PY
  )"
fi

_resolve() {
  local val="$1"
  if [[ -z "$val" || "$val" == http://* || "$val" == https://* || "$val" == /* ]]; then
    echo "$val"
  else
    echo "$WS/$val"
  fi
}

# ---- 所有变量解析 ----
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
DINO_MIN_INTERVAL=${DINO_MIN_INTERVAL:-${CFG_DINO_MIN_INTERVAL:-6.0}}
FRONTIER_LOG=${FRONTIER_LOG:-${CFG_FRONTIER_LOG:-true}}
if [[ "$VLLM_PROBE" == */v1 ]]; then
  VLLM_PROBE="$VLLM_PROBE/models"
fi

# ---- 前置检查 ----
if ! command -v tmux >/dev/null 2>&1; then
  echo "[error] tmux 未安装" >&2
  exit 1
fi

if ! tmux has-session -t "=$ENV_SESSION" 2>/dev/null; then
  echo "[error] 环境 session '$ENV_SESSION' 不存在，请先运行 ./scripts/run_env_tmux.sh" >&2
  exit 1
fi
echo "[nodes] 检测到 $ENV_SESSION 存活"

# ---- 如果节点 session 已存在，先杀 ----
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "[nodes] 发现旧 session '$SESSION'，杀掉重建..."
  tmux kill-session -t "=$SESSION"
fi
echo "[nodes] 清理旧 session 完毕"

# ---- 新建节点 session ----
echo "[nodes] 创建 session '$SESSION'..."
tmux new-session -d -s "$SESSION" -c "$WS" -n "pro3" \
  "bash -lc 'WS=\"$WS\"; source \"$WS/scripts/pipeline_env.sh\"; set -e; \
until rostopic list >/dev/null 2>&1; do echo \"waiting for rocore...\"; sleep 1; done; \
roslaunch lste_core spawn_pro3.launch spawn_model:=true; \
echo; echo \"[EXIT] pro3 spawn\"; exec bash'"
echo "[nodes] 创建后检查 sessions: $(tmux list-sessions 2>&1)"
tmux set-option -t "=$SESSION:" remain-on-exit on
echo "[nodes] pro3 spawned (window 0)"

# ---- 辅助函数 ----
tmux_new_window() {
  local index="$1"
  local dir="$2"
  local title="$3"
  local cmd="$4"
  tmux new-window -t "=$SESSION:$index" -n "$title" -c "$dir" \
    "bash -lc 'WS=\"$WS\"; source \"$WS/scripts/pipeline_env.sh\"; set -e; $cmd'; echo; echo '[EXIT] $title'; exec bash"
  echo "[nodes] 创建窗口 $index($title) 后检查: $(tmux list-sessions 2>&1)"
}

WAIT_ROSCORE='until rostopic list >/dev/null 2>&1; do echo \"waiting for roscore...\"; sleep 1; done'

# ---- 逐窗口启动（index 从 1 开始，0 已被 pro3 占用） ----
# 1: 发布任务
tmux_new_window 1 "$WS" "task" \
  "$WAIT_ROSCORE; rosrun lste_core lste_task_node.py _json_path:=$TASK_JSON _task_id:=$TASK_ID"

# 2: VLLM 服务
tmux_new_window 2 "$WS/model/MiniCPM/test" "vllm" \
  "conda activate minicpm; bash start.sh"

# 3: 全局目标
tmux_new_window 3 "$WS" "goal" \
  "$WAIT_ROSCORE; rosrun lste_topo_access lste_goal_manager.py _follow_locked_done_time:=$FOLLOW_LOCKED_DONE_TIME"

# 4: VLM Prompt 节点
tmux_new_window 4 "$WS" "prompt" \
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

# 5: DINO 检测
tmux_new_window 5 "$WS" "dino" \
  "$WAIT_ROSCORE; conda activate dino; export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7; rosrun lste_core lste_det_node.py _min_inference_interval:=$DINO_MIN_INTERVAL"

# 6: Score
tmux_new_window 6 "$WS" "score" \
  "$WAIT_ROSCORE; rosrun lste_core lste_score_node.py \
    _w_target:=0.2 _w_env:=0.3 _w_ctx:=0.5 \
    _lambda_neg:=0.7 _pos_midpoint:=0.25 _pos_steepness:=6 \
    _neg_midpoint:=0.15 _neg_steepness:=12"

# 7: State
tmux_new_window 7 "$WS" "state" \
  "$WAIT_ROSCORE; rosrun lste_core lste_state_node.py _suspicious_window:=$SUSPICIOUS_WINDOW"

# 8: 可视化（叠加图 + RViz）
tmux_new_window 8 "$WS" "vis" \
  "$WAIT_ROSCORE; roslaunch lste_core lste_det_vis.launch"

# 9: 球面投影
tmux_new_window 9 "$WS" "oc_srfc" \
  "$WAIT_ROSCORE; roslaunch lste_oc_srfc oc_srfc_proj.launch"

# 10: GP Frontier
tmux_new_window 10 "$WS" "gp_frontier" \
  "$WAIT_ROSCORE; conda activate vsgp; roslaunch lste_topo_access gp_frontier.launch \
    access_topo_config:=$ACCESS_TOPO_CONFIG \
    access_topo_config_pass:=$ACCESS_TOPO_CONFIG_PASS \
    access_topo_config_sus_c:=$ACCESS_TOPO_CONFIG_SUS_C \
    access_topo_test_name:=$ACCESS_TOPO_TEST_NAME \
    access_topo_run_name:=$ACCESS_TOPO_RUN_NAME \
    follow_locked_done_time:=$FOLLOW_LOCKED_DONE_TIME \
    frontier_log:=$FRONTIER_LOG"

# 11: Frontier RViz
tmux_new_window 11 "$WS" "rviz_frontier" \
  "$WAIT_ROSCORE; rviz -d $GP_FRONTIER_RVIZ"

# 12: 键盘遥控
tmux_new_window 12 "$WS" "teleop" \
  "$WAIT_ROSCORE; python \"$WS/scripts/teleop_with_reset.py\" cmd_vel:=/cmd_vel"

echo ""
echo "============================================"
echo "  节点全部启动完毕 (13 个窗口：pro3 + 12 节点)"
echo "  环境 session:  tmux attach -t lste-env"
echo "  节点 session:  tmux attach -t lste"
echo "============================================"

if [[ -t 0 ]]; then
  tmux attach -t "$SESSION"
fi
