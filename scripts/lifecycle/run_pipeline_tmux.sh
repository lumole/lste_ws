#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1


# Auto-detect workspace root (can override with LSTE_WS env var)
if [ -z "${WS:-}" ]; then
  WS="${LSTE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
fi
LSTE_WS="$WS"
SESSION=${SESSION:-lste}
# 可选 YAML 配置：通过 PIPELINE_CONFIG 指定；存在时为未显式设置的变量提供默认值
PIPELINE_CONFIG=${PIPELINE_CONFIG:-$WS/scripts/config/pipeline_defaults.yaml}
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
DINO_MIN_INTERVAL=${DINO_MIN_INTERVAL:-${CFG_DINO_MIN_INTERVAL:-1.5}}
DINO_INTERVAL_PASS=${DINO_INTERVAL_PASS:-${CFG_DINO_INTERVAL_PASS:-1.5}}
DINO_INTERVAL_SUSPICIOUS=${DINO_INTERVAL_SUSPICIOUS:-${CFG_DINO_INTERVAL_SUSPICIOUS:-1.5}}
DINO_INTERVAL_LOCKED=${DINO_INTERVAL_LOCKED:-${CFG_DINO_INTERVAL_LOCKED:-1.5}}
DINO_INTERVAL_EXHAUSTED=${DINO_INTERVAL_EXHAUSTED:-${CFG_DINO_INTERVAL_EXHAUSTED:-3.0}}
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
ACCESS_TOPO_CONFIG=${ACCESS_TOPO_CONFIG:-$(_resolve "${CFG_ACCESS_TOPO_CONFIG:-src/lste_topo_access/topo_tree/cfgs/access_topo.yaml}")}
ACCESS_TOPO_CONFIG_PASS=${ACCESS_TOPO_CONFIG_PASS:-$(_resolve "${CFG_ACCESS_TOPO_CONFIG_PASS:-src/lste_topo_access/topo_tree/cfgs/access_topo_pass.yaml}")}
ACCESS_TOPO_CONFIG_SUS_C=${ACCESS_TOPO_CONFIG_SUS_C:-$(_resolve "${CFG_ACCESS_TOPO_CONFIG_SUS_C:-src/lste_topo_access/topo_tree/cfgs/access_topo_sus_c.yaml}")}
ACCESS_TOPO_TEST_NAME=${ACCESS_TOPO_TEST_NAME:-${CFG_ACCESS_TOPO_TEST_NAME:-default_test}}
ACCESS_TOPO_RUN_NAME=${ACCESS_TOPO_RUN_NAME:-${CFG_ACCESS_TOPO_RUN_NAME:-$ACCESS_TOPO_TEST_NAME}}
GP_FRONTIER_RVIZ=${GP_FRONTIER_RVIZ:-$(_resolve "${CFG_GP_FRONTIER_RVIZ:-src/lste_topo_access/launch/gp_frontier.rviz}")}
SUSPICIOUS_WINDOW=${SUSPICIOUS_WINDOW:-${CFG_SUSPICIOUS_WINDOW:-5}}
FOLLOW_LOCKED_DONE_TIME=${FOLLOW_LOCKED_DONE_TIME:-${CFG_FOLLOW_LOCKED_DONE_TIME:-4.0}}
FRONTIER_LOG=${FRONTIER_LOG:-${CFG_FRONTIER_LOG:-true}}
# The Gazebo click-coordinate plugin is part of the required operator UI.
GUI=true
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
  tmux new-window -t "=$SESSION:$index" -n "$title" -c "$dir" \
    "bash -lc 'WS=\"$WS\"; source \"$WS/scripts/config/pipeline_env.sh\"; set -e; $cmd'; echo; echo '[EXIT] $title'; exec bash"
}

# 如果 session 已存在则复用，避免重复启动
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  # 如果已有 session 的 WORLD 与当前需求不同，则先杀掉重新建，避免复用旧 world
  EXISTING_WORLD="$(tmux show-environment -t "=$SESSION" 2>/dev/null | awk -F= '/^LSTE_WORLD=/{print substr($0,length("LSTE_WORLD=")+1)}')"
  if [[ -n "$EXISTING_WORLD" && "$EXISTING_WORLD" != "$WORLD" ]]; then
    echo "[info] tmux session '$SESSION' exists with WORLD=$EXISTING_WORLD, restart with WORLD=$WORLD."
    tmux kill-session -t "=$SESSION"
  else
    echo "tmux session '$SESSION' 已存在，直接附着。" >&2
    exec tmux attach -t "=$SESSION"
  fi
fi

# 新建 session：tmux 默认会创建 window 0，所以直接把 window 0 用作 roscore
tmux new-session -d -s "$SESSION" -c "$WS" -n "roscore" \
  "bash -lc 'WS=\"$WS\"; source \"$WS/scripts/config/pipeline_env.sh\"; roscore'; echo; echo '[EXIT] roscore'; exec bash"
tmux set-environment -t "=$SESSION" LSTE_WORLD "$WORLD"
tmux set-option -t "=$SESSION:" remain-on-exit on

# helper: wait for roscore
WAIT_ROSCORE='until rostopic list >/dev/null 2>&1; do echo \"waiting for roscore...\"; sleep 1; done'

# 1: 仿真 + 机器人（等待 master 就绪）
tmux_new_window 1 "$WS" "world" \
  "$WAIT_ROSCORE; roslaunch lste_core lab_with_pro3.launch world_name:=$WORLD spawn_pro3:=$SPAWN_PRO3 gui:=$GUI"

# 2: 发布任务（latched，可随时替换 json/task_id）
tmux_new_window 2 "$WS" "task" \
  "$WAIT_ROSCORE; rosrun lste_core lste_task_node.py _json_path:=$TASK_JSON _task_id:=$TASK_ID"

# 3: 按 prompt 缓存决策启动 VLLM（MiniCPM）
tmux_new_window 3 "$WS/model/MiniCPM/test" "vllm" \
  "$WAIT_ROSCORE; echo \"Waiting for prompt cache decision...\"; \
   while true; do \
     decision=\$(rosparam get /lste_prompt_node/needs_vllm 2>/dev/null || true); \
     case \"\$decision\" in \
       true) echo \"Prompt cache miss; starting MiniCPM.\"; conda activate minicpm; exec bash start.sh ;; \
       false) echo \"Prompt cache hit; MiniCPM was not started.\"; exit 0 ;; \
     esac; \
     sleep 0.5; \
   done"

# 4: 全局目标（/lste/final_goal）
tmux_new_window 4 "$WS" "goal" \
  "$WAIT_ROSCORE; rosrun lste_topo_access lste_goal_manager.py _follow_locked_done_time:=$FOLLOW_LOCKED_DONE_TIME"

# 5: prompt 节点先查缓存，仅在 miss 时等待 VLLM
tmux_new_window 5 "$WS" "prompt" \
  "$WAIT_ROSCORE; conda activate minicpm; \
   rosparam set /lste_prompt_node/vllm_stop_command \"tmux kill-window -t =$SESSION:vllm\"; \
   rosrun lste_core lste_prompt_node.py \
     _vllm_base_url:=$VLLM_URL \
     _vllm_model_name:=$VLLM_MODEL \
     _cache_enabled:=$PROMPT_CACHE_ENABLED \
     _cache_dir:=$PROMPT_CACHE_DIR \
     _vllm_start_timeout:=$VLLM_START_TIMEOUT"

# 6: 独立检测节点：GroundingDINO 或 WeDetect，绝不在同一进程内混用。
tmux_new_window 6 "$WS" "detector" \
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
  "$WAIT_ROSCORE; roslaunch lste_core lste_det_vis.launch \
    tracking_enabled:=$DETECTION_VISUAL_TRACKING_ENABLED \
    display_sync_mode:=$DETECTION_DISPLAY_SYNC_MODE \
    display_history_size:=$DETECTION_DISPLAY_HISTORY_SIZE \
    tracking_mode:=$DETECTION_TRACKING_MODE detector_backend:=$DETECTOR_BACKEND"

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

if [[ -t 0 ]]; then
  tmux attach -t "=$SESSION"
fi
