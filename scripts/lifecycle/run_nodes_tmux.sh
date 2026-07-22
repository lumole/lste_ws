#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# ============================================
# 业务节点脚本：独立 session "lste"，依赖 lste-env 的 roscore + Gazebo
# 可反复重启调试，不影响环境 session
# 前置条件：先运行 ./scripts/lifecycle/run_env_tmux.sh
# ============================================

# ---- 路径解析（与 run_env_tmux.sh 完全一致） ----
if [ -z "${WS:-}" ]; then
  WS="${LSTE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
fi
LSTE_WS="$WS"
ENV_SESSION=lste-env
SESSION=lste
PIPELINE_CONFIG=${PIPELINE_CONFIG:-$WS/scripts/config/pipeline_defaults.yaml}
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
TASK_JSON_VALUE=${TASK_JSON:-${CFG_TASK_JSON:-}}
TASK_ID=${TASK_ID:-${CFG_TASK_ID:-}}
if [[ -z "$TASK_JSON_VALUE" || -z "$TASK_ID" ]]; then
  echo "[error] TASK_JSON and TASK_ID must be set in $PIPELINE_CONFIG or the environment" >&2
  exit 1
fi
TASK_JSON=$(_resolve "$TASK_JSON_VALUE")
if [[ ! -f "$TASK_JSON" ]]; then
  echo "[error] TASK_JSON file not found: $TASK_JSON" >&2
  exit 1
fi
VLLM_URL=${VLLM_URL:-${CFG_VLLM_URL:-http://localhost:8000/v1}}
VLLM_MODEL=${VLLM_MODEL:-$(_resolve "${CFG_VLLM_MODEL:-model/MiniCPM/OpenBMB/MiniCPM4-0___5B}")}
PROMPT_CACHE_ENABLED=${PROMPT_CACHE_ENABLED:-${CFG_PROMPT_CACHE_ENABLED:-true}}
PROMPT_CACHE_DIR=${PROMPT_CACHE_DIR:-$(_resolve "${CFG_PROMPT_CACHE_DIR:-runtime/prompt_cache}")}
VLLM_START_TIMEOUT=${VLLM_START_TIMEOUT:-${CFG_VLLM_START_TIMEOUT:-240}}
ACCESS_TOPO_CONFIG=${ACCESS_TOPO_CONFIG:-$(_resolve "${CFG_ACCESS_TOPO_CONFIG:-src/lste_topo_access/topo_tree/cfgs/access_topo.yaml}")}
ACCESS_TOPO_CONFIG_PASS=${ACCESS_TOPO_CONFIG_PASS:-$(_resolve "${CFG_ACCESS_TOPO_CONFIG_PASS:-src/lste_topo_access/topo_tree/cfgs/access_topo_pass.yaml}")}
ACCESS_TOPO_CONFIG_SUS_C=${ACCESS_TOPO_CONFIG_SUS_C:-$(_resolve "${CFG_ACCESS_TOPO_CONFIG_SUS_C:-src/lste_topo_access/topo_tree/cfgs/access_topo_sus_c.yaml}")}
ACCESS_TOPO_TEST_NAME=${ACCESS_TOPO_TEST_NAME:-${CFG_ACCESS_TOPO_TEST_NAME:-default_test}}
ACCESS_TOPO_RUN_NAME=${ACCESS_TOPO_RUN_NAME:-${CFG_ACCESS_TOPO_RUN_NAME:-$ACCESS_TOPO_TEST_NAME}}
GP_FRONTIER_RVIZ=${GP_FRONTIER_RVIZ:-$(_resolve "${CFG_GP_FRONTIER_RVIZ:-src/lste_topo_access/launch/gp_frontier.rviz}")}
SUSPICIOUS_WINDOW=${SUSPICIOUS_WINDOW:-${CFG_SUSPICIOUS_WINDOW:-5}}
FOLLOW_LOCKED_DONE_TIME=${FOLLOW_LOCKED_DONE_TIME:-${CFG_FOLLOW_LOCKED_DONE_TIME:-4.0}}
DINO_MIN_INTERVAL=${DINO_MIN_INTERVAL:-${CFG_DINO_MIN_INTERVAL:-1.5}}
DINO_INTERVAL_PASS=${DINO_INTERVAL_PASS:-${CFG_DINO_INTERVAL_PASS:-1.5}}
DINO_INTERVAL_SUSPICIOUS=${DINO_INTERVAL_SUSPICIOUS:-${CFG_DINO_INTERVAL_SUSPICIOUS:-1.5}}
DINO_INTERVAL_LOCKED=${DINO_INTERVAL_LOCKED:-${CFG_DINO_INTERVAL_LOCKED:-1.5}}
DINO_INTERVAL_EXHAUSTED=${DINO_INTERVAL_EXHAUSTED:-${CFG_DINO_INTERVAL_EXHAUSTED:-3.0}}
DETECTOR=${DETECTOR:-${CFG_DETECTOR:-wedetect-large}}
DETECTION_TRACKING_MODE=${DETECTION_TRACKING_MODE:-${CFG_DETECTION_TRACKING_MODE:-auto}}
DETECTION_VISUAL_TRACKING_ENABLED=${DETECTION_VISUAL_TRACKING_ENABLED:-${CFG_DETECTION_VISUAL_TRACKING_ENABLED:-true}}
DETECTION_DISPLAY_SYNC_MODE=${DETECTION_DISPLAY_SYNC_MODE:-${CFG_DETECTION_DISPLAY_SYNC_MODE:-latest_frame}}
DETECTION_DISPLAY_HISTORY_SIZE=${DETECTION_DISPLAY_HISTORY_SIZE:-${CFG_DETECTION_DISPLAY_HISTORY_SIZE:-90}}
WDETECT_SOURCE_DIR=${WDETECT_SOURCE_DIR:-$(_resolve "${CFG_WDETECT_SOURCE_DIR:-model/WeDetect}")}
WDETECT_VARIANT=${WDETECT_VARIANT:-${CFG_WDETECT_VARIANT:-base}}
WDETECT_CHECKPOINT=${WDETECT_CHECKPOINT:-$(_resolve "${CFG_WDETECT_CHECKPOINT:-model/WeDetect/checkpoints/wedetect_base.pth}")}
WDETECT_LANGUAGE_MODEL=${WDETECT_LANGUAGE_MODEL:-$(_resolve "${CFG_WDETECT_LANGUAGE_MODEL:-model/WeDetect/xlm-roberta-base}")}
WDETECT_SCORE_THRESHOLD=${WDETECT_SCORE_THRESHOLD:-${CFG_WDETECT_SCORE_THRESHOLD:-0.20}}
WDETECT_NMS_IOU=${WDETECT_NMS_IOU:-${CFG_WDETECT_NMS_IOU:-0.70}}
WDETECT_PRE_NMS_TOPK=${WDETECT_PRE_NMS_TOPK:-${CFG_WDETECT_PRE_NMS_TOPK:-3000}}
WDETECT_MAX_DETECTIONS=${WDETECT_MAX_DETECTIONS:-${CFG_WDETECT_MAX_DETECTIONS:-100}}
WDETECT_USE_FP16=${WDETECT_USE_FP16:-${CFG_WDETECT_USE_FP16:-true}}
WDETECT_RUNTIME=${WDETECT_RUNTIME:-${CFG_WDETECT_RUNTIME:-tensorrt}}
WDETECT_VISION_ONNX=${WDETECT_VISION_ONNX:-$(_resolve "${CFG_WDETECT_VISION_ONNX:-model/WeDetect/deploy/onnx_models/wedetect_base_vision.onnx}")}
WDETECT_TRT_ENGINE_CACHE=${WDETECT_TRT_ENGINE_CACHE:-$(_resolve "${CFG_WDETECT_TRT_ENGINE_CACHE:-model/WeDetect/deploy/trt_cache}")}
WDETECT_TRT_MAX_CLASSES=${WDETECT_TRT_MAX_CLASSES:-${CFG_WDETECT_TRT_MAX_CLASSES:-16}}
WDETECT_SPIN_HZ=${WDETECT_SPIN_HZ:-${CFG_WDETECT_SPIN_HZ:-10.0}}
WDETECT_MIN_INTERVAL=${WDETECT_MIN_INTERVAL:-${CFG_WDETECT_MIN_INTERVAL:-0.10}}
WDETECT_INTERVAL_PASS=${WDETECT_INTERVAL_PASS:-${CFG_WDETECT_INTERVAL_PASS:-0.10}}
WDETECT_INTERVAL_SUSPICIOUS=${WDETECT_INTERVAL_SUSPICIOUS:-${CFG_WDETECT_INTERVAL_SUSPICIOUS:-0.10}}
WDETECT_INTERVAL_LOCKED=${WDETECT_INTERVAL_LOCKED:-${CFG_WDETECT_INTERVAL_LOCKED:-0.10}}
WDETECT_INTERVAL_EXHAUSTED=${WDETECT_INTERVAL_EXHAUSTED:-${CFG_WDETECT_INTERVAL_EXHAUSTED:-0.30}}
case "$DETECTOR" in
  groundingdino)
    DETECTOR_BACKEND=groundingdino
    ;;
  wedetect-base)
    DETECTOR_BACKEND=wedetect
    WDETECT_VARIANT=base
    WDETECT_CHECKPOINT="$WS/model/WeDetect/checkpoints/wedetect_base.pth"
    WDETECT_LANGUAGE_MODEL="$WS/model/WeDetect/xlm-roberta-base"
    WDETECT_VISION_ONNX="$WS/model/WeDetect/deploy/onnx_models/wedetect_base_vision.onnx"
    WDETECT_TRT_ENGINE_CACHE="$WS/model/WeDetect/deploy/trt_cache"
    WDETECT_RUNTIME=tensorrt
    ;;
  wedetect-large)
    DETECTOR_BACKEND=wedetect
    WDETECT_VARIANT=large
    WDETECT_CHECKPOINT="$WS/model/WeDetect/checkpoints/wedetect_large.pth"
    WDETECT_LANGUAGE_MODEL="$WS/model/WeDetect/xlm-roberta-large"
    WDETECT_VISION_ONNX="$WS/model/WeDetect/deploy/onnx_models/wedetect_large_vision.onnx"
    WDETECT_TRT_ENGINE_CACHE="$WS/model/WeDetect/deploy/trt_cache_large"
    WDETECT_RUNTIME=tensorrt
    ;;
  *)
    echo "[error] DETECTOR must be groundingdino, wedetect-base, or wedetect-large (got '$DETECTOR')" >&2
    exit 1
    ;;
esac
if [[ "$DETECTOR_BACKEND" == "wedetect" ]]; then
  DETECTOR_MIN_INTERVAL=$WDETECT_MIN_INTERVAL
  DETECTOR_INTERVAL_PASS=$WDETECT_INTERVAL_PASS
  DETECTOR_INTERVAL_SUSPICIOUS=$WDETECT_INTERVAL_SUSPICIOUS
  DETECTOR_INTERVAL_LOCKED=$WDETECT_INTERVAL_LOCKED
  DETECTOR_INTERVAL_EXHAUSTED=$WDETECT_INTERVAL_EXHAUSTED
  DETECTOR_NODE=lste_wedetect_det_node.py
  DETECTOR_MODEL_ARGS="_wedetect_source_dir:=$WDETECT_SOURCE_DIR \
_wedetect_variant:=$WDETECT_VARIANT \
_wedetect_checkpoint:=$WDETECT_CHECKPOINT \
_wedetect_language_model:=$WDETECT_LANGUAGE_MODEL \
_wedetect_score_threshold:=$WDETECT_SCORE_THRESHOLD \
_wedetect_nms_iou:=$WDETECT_NMS_IOU \
_wedetect_pre_nms_topk:=$WDETECT_PRE_NMS_TOPK \
_wedetect_max_detections:=$WDETECT_MAX_DETECTIONS \
_wedetect_use_fp16:=$WDETECT_USE_FP16 \
_wedetect_runtime:=$WDETECT_RUNTIME \
_wedetect_vision_onnx:=$WDETECT_VISION_ONNX \
_wedetect_trt_engine_cache:=$WDETECT_TRT_ENGINE_CACHE \
_wedetect_trt_max_classes:=$WDETECT_TRT_MAX_CLASSES"
else
  DETECTOR_MIN_INTERVAL=$DINO_MIN_INTERVAL
  DETECTOR_INTERVAL_PASS=$DINO_INTERVAL_PASS
  DETECTOR_INTERVAL_SUSPICIOUS=$DINO_INTERVAL_SUSPICIOUS
  DETECTOR_INTERVAL_LOCKED=$DINO_INTERVAL_LOCKED
  DETECTOR_INTERVAL_EXHAUSTED=$DINO_INTERVAL_EXHAUSTED
  DETECTOR_NODE=lste_det_node.py
  DETECTOR_MODEL_ARGS=""
fi
FRONTIER_LOG=${FRONTIER_LOG:-${CFG_FRONTIER_LOG:-true}}
LSTE_CONTROLLER=${LSTE_CONTROLLER:-${CFG_LSTE_CONTROLLER:-teleop}}
SAPPO_SPEED=${SAPPO_SPEED:-0.50}
SAPPO_PYTHON=${SAPPO_PYTHON:-$HOME/miniconda3/envs/rlenvs/bin/python}
case "$LSTE_CONTROLLER" in
  teleop|sappo) ;;
  *)
    echo "[error] LSTE_CONTROLLER must be 'teleop' or 'sappo' (got '$LSTE_CONTROLLER')" >&2
    exit 1
    ;;
esac

# ---- 前置检查 ----
if ! command -v tmux >/dev/null 2>&1; then
  echo "[error] tmux 未安装" >&2
  exit 1
fi

if ! tmux has-session -t "=$ENV_SESSION" 2>/dev/null; then
  echo "[error] 环境 session '$ENV_SESSION' 不存在，请先运行 ./scripts/lifecycle/run_env_tmux.sh" >&2
  exit 1
fi
if tmux has-session -t "=sappo-lste" 2>/dev/null; then
  echo "[error] 独立 SA-PPO session 'sappo-lste' 仍在运行，请先停止它再启动 LSTE 节点" >&2
  exit 1
fi
echo "[nodes] 检测到 $ENV_SESSION 存活"

# ---- 如果节点 session 已存在，先杀 ----
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "[nodes] 发现旧 session '$SESSION'，杀掉重建..."
  tmux kill-session -t "=$SESSION"
fi
echo "[nodes] 清理旧 session 完毕"

# lste-env may keep roscore alive across node restarts, so remove the previous
# prompt decision before the gated vLLM window starts.
bash -lc "WS=\"$WS\"; source \"$WS/scripts/config/pipeline_env.sh\"; rosparam delete /lste_prompt_node/needs_vllm >/dev/null 2>&1 || true"

# ---- 新建节点 session ----
echo "[nodes] 创建 session '$SESSION'..."
tmux new-session -d -s "$SESSION" -c "$WS" -n "pro3" \
  "bash -lc 'WS=\"$WS\"; source \"$WS/scripts/config/pipeline_env.sh\"; set -e; \
until rostopic list >/dev/null 2>&1; do echo \"waiting for rocore...\"; sleep 1; done; \
if rosservice call /gazebo/get_world_properties 2>/dev/null | grep -q \"pro3\"; then \
  echo \"[pro3] Existing Gazebo model found; reuse it without respawning.\"; \
  SPAWN_MODEL=false; \
else \
  SPAWN_MODEL=true; \
fi; \
roslaunch lste_core spawn_pro3.launch spawn_model:=\$SPAWN_MODEL; \
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
    "bash -lc 'WS=\"$WS\"; source \"$WS/scripts/config/pipeline_env.sh\"; set -e; $cmd'; echo; echo '[EXIT] $title'; exec bash"
  echo "[nodes] 创建窗口 $index($title) 后检查: $(tmux list-sessions 2>&1)"
}

WAIT_ROSCORE='until rostopic list >/dev/null 2>&1; do echo \"waiting for roscore...\"; sleep 1; done'

# ---- 逐窗口启动（index 从 1 开始，0 已被 pro3 占用） ----
# 1: 发布任务
tmux_new_window 1 "$WS" "task" \
  "$WAIT_ROSCORE; rosrun lste_core lste_task_node.py _json_path:=$TASK_JSON _task_id:=$TASK_ID"

# 2: VLLM 服务。缓存命中时不加载 MiniCPM。
tmux_new_window 2 "$WS/model/MiniCPM/test" "vllm" \
  "$WAIT_ROSCORE; echo \"Waiting for prompt cache decision...\"; \
   while true; do \
     decision=\$(rosparam get /lste_prompt_node/needs_vllm 2>/dev/null || true); \
     case \"\$decision\" in \
       true) echo \"Prompt cache miss; starting MiniCPM.\"; conda activate minicpm; exec bash start.sh ;; \
       false) echo \"Prompt cache hit; MiniCPM was not started.\"; exit 0 ;; \
     esac; \
     sleep 0.5; \
   done"

# 3: 全局目标
tmux_new_window 3 "$WS" "goal" \
  "$WAIT_ROSCORE; rosrun lste_topo_access lste_goal_manager.py _follow_locked_done_time:=$FOLLOW_LOCKED_DONE_TIME"

# 4: VLM Prompt 节点。它先检查缓存，再通知窗口 2 是否启动 MiniCPM。
tmux_new_window 4 "$WS" "prompt" \
  "$WAIT_ROSCORE; conda activate minicpm; \
   rosparam set /lste_prompt_node/vllm_stop_command \"tmux kill-window -t =$SESSION:vllm\"; \
   rosrun lste_core lste_prompt_node.py \
     _vllm_base_url:=$VLLM_URL \
     _vllm_model_name:=$VLLM_MODEL \
     _cache_enabled:=$PROMPT_CACHE_ENABLED \
     _cache_dir:=$PROMPT_CACHE_DIR \
     _vllm_start_timeout:=$VLLM_START_TIMEOUT"

# 5: 独立检测节点：GroundingDINO 或 WeDetect，绝不在同一进程内混用。
tmux_new_window 5 "$WS" "detector" \
  "$WAIT_ROSCORE; conda activate dino; export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7; \
   export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib/python3.9/site-packages/tensorrt_libs:\$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cudnn/lib:\$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cublas/lib:\$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cufft/lib:\$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/curand/lib:\$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cusolver/lib:\$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cusparse/lib:\$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:\$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib:\$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/nvjitlink/lib:\$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:\${LD_LIBRARY_PATH:-}\"; \
   rosrun lste_core $DETECTOR_NODE \
     _min_inference_interval:=$DETECTOR_MIN_INTERVAL \
     _interval_pass:=$DETECTOR_INTERVAL_PASS \
     _interval_suspicious:=$DETECTOR_INTERVAL_SUSPICIOUS \
     _interval_locked:=$DETECTOR_INTERVAL_LOCKED \
     _interval_exhausted:=$DETECTOR_INTERVAL_EXHAUSTED \
     $DETECTOR_MODEL_ARGS \
     _spin_hz:=$WDETECT_SPIN_HZ"

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
  "$WAIT_ROSCORE; roslaunch lste_core lste_det_vis.launch \
    tracking_enabled:=$DETECTION_VISUAL_TRACKING_ENABLED \
    display_sync_mode:=$DETECTION_DISPLAY_SYNC_MODE \
    display_history_size:=$DETECTION_DISPLAY_HISTORY_SIZE \
    tracking_mode:=$DETECTION_TRACKING_MODE detector_backend:=$DETECTOR_BACKEND"

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

# 12: GUI 热切换请求处理（只切换 mux，不停止控制器进程）
tmux_new_window 12 "$WS" "controller_switch" \
  "$WAIT_ROSCORE; rosrun lste_core lste_controller_switch_node.py \
    _initial_mode:=$LSTE_CONTROLLER"

# 13: 唯一 /cmd_vel 发布者
tmux_new_window 13 "$WS" "cmd_vel_mux" \
  "$WAIT_ROSCORE; rosrun lste_core lste_cmd_vel_mux_node.py"

# 14: 持续健康检查（在控制器前启动，以便捕获 SA-PPO 启动失败）
tmux_new_window 14 "$WS" "health" \
  "$WAIT_ROSCORE; python3 $WS/scripts/tools/lste_health_audit.py"

# 15+: SA-PPO 与 teleop 始终运行，切换只改变 mux 输入。
SAPPO_SPEED="$SAPPO_SPEED" SAPPO_PYTHON="$SAPPO_PYTHON" \
  "$WS/scripts/lifecycle/switch_controller.sh" "$LSTE_CONTROLLER"
WINDOW_SUMMARY="17 个 lste 窗口 + 始终运行的 lste-teleop 会话"

echo ""
echo "============================================"
echo "  节点全部启动完毕 ($WINDOW_SUMMARY)"
echo "  控制器: $LSTE_CONTROLLER"
echo "  环境 session:  tmux attach -t lste-env"
echo "  节点 session:  tmux attach -t lste"
echo "  遥控 session:  teleop"
echo "============================================"

if [[ -t 0 && "${LSTE_NO_ATTACH:-0}" != "1" ]]; then
  tmux attach -t "$SESSION"
fi
