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

# Use the same ROS master and workspace overlay in this launcher process as
# the tmux panes. This makes the direct `run_nodes_tmux.sh` path safe even when
# the caller did not source pipeline_env.sh in the interactive shell.
source "$WS/scripts/config/pipeline_env.sh"

if [[ "${LSTE_LIFECYCLE_LOCK_HELD:-0}" != "1" ]]; then
  if ! command -v flock >/dev/null 2>&1; then
    echo "[error] flock is required to serialize LSTE lifecycle commands." >&2
    exit 1
  fi
  LOCK_DIR="$WS/runtime/lifecycle"
  mkdir -p "$LOCK_DIR"
  exec 9>"$LOCK_DIR/lifecycle.lock"
  if ! flock -n 9; then
    echo "[error] Another LSTE lifecycle command is already starting or stopping the system." >&2
    exit 75
  fi
  export LSTE_LIFECYCLE_LOCK_HELD=1
fi

# Keep the lifecycle flock owned by this shell only; tmux sessions must not
# inherit fd 9 because they outlive the startup command.
tmux() { command tmux "$@" 9>&-; }
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
PRO3_SPAWN_X=${PRO3_SPAWN_X:-${CFG_PRO3_SPAWN_X:-0.0}}
PRO3_SPAWN_Y=${PRO3_SPAWN_Y:-${CFG_PRO3_SPAWN_Y:-0.0}}
PRO3_SPAWN_Z=${PRO3_SPAWN_Z:-${CFG_PRO3_SPAWN_Z:-0.0}}
PRO3_SPAWN_YAW=${PRO3_SPAWN_YAW:-${CFG_PRO3_SPAWN_YAW:-0.0}}
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
LEGACY_GP_FRONTIER_ENABLED=${LEGACY_GP_FRONTIER_ENABLED:-${CFG_LEGACY_GP_FRONTIER_ENABLED:-false}}
SUSPICIOUS_WINDOW=${SUSPICIOUS_WINDOW:-${CFG_SUSPICIOUS_WINDOW:-5}}
STATE_SUBTYPE_SWITCH_MIN_HITS=${STATE_SUBTYPE_SWITCH_MIN_HITS:-${CFG_STATE_SUBTYPE_SWITCH_MIN_HITS:-3}}
FOLLOW_LOCKED_DONE_TIME=${FOLLOW_LOCKED_DONE_TIME:-${CFG_FOLLOW_LOCKED_DONE_TIME:-4.0}}
TARGET_LOCK_TOTAL_MIN=${TARGET_LOCK_TOTAL_MIN:-${CFG_TARGET_LOCK_TOTAL_MIN:-0.60}}
TARGET_LOCK_SCORE_MIN=${TARGET_LOCK_SCORE_MIN:-${CFG_TARGET_LOCK_SCORE_MIN:-0.60}}
TARGET_LOCK_WINDOW=${TARGET_LOCK_WINDOW:-${CFG_TARGET_LOCK_WINDOW:-2}}
TARGET_LOCK_MIN_HITS=${TARGET_LOCK_MIN_HITS:-${CFG_TARGET_LOCK_MIN_HITS:-2}}
TARGET_LOCK_EXIT_TOTAL_MIN=${TARGET_LOCK_EXIT_TOTAL_MIN:-${CFG_TARGET_LOCK_EXIT_TOTAL_MIN:-0.60}}
TARGET_LOCK_EXIT_SCORE_MIN=${TARGET_LOCK_EXIT_SCORE_MIN:-${CFG_TARGET_LOCK_EXIT_SCORE_MIN:-0.60}}
TARGET_LOCK_EXIT_UNSTABLE_FRAMES=${TARGET_LOCK_EXIT_UNSTABLE_FRAMES:-${CFG_TARGET_LOCK_EXIT_UNSTABLE_FRAMES:-5}}
TARGET_DONE_MIN_BOX_WIDTH=${TARGET_DONE_MIN_BOX_WIDTH:-${CFG_TARGET_DONE_MIN_BOX_WIDTH:-0.06}}
TARGET_DONE_MIN_BOX_HEIGHT=${TARGET_DONE_MIN_BOX_HEIGHT:-${CFG_TARGET_DONE_MIN_BOX_HEIGHT:-0.06}}
TARGET_DONE_MIN_SCORE=${TARGET_DONE_MIN_SCORE:-${CFG_TARGET_DONE_MIN_SCORE:-0.40}}
TARGET_DONE_MIN_HOLD_TIME=${TARGET_DONE_MIN_HOLD_TIME:-${CFG_TARGET_DONE_MIN_HOLD_TIME:-0.30}}
TARGET_DONE_MIN_FRESH_HITS=${TARGET_DONE_MIN_FRESH_HITS:-${CFG_TARGET_DONE_MIN_FRESH_HITS:-3}}
TARGET_DONE_MAX_DETECTION_AGE=${TARGET_DONE_MAX_DETECTION_AGE:-${CFG_TARGET_DONE_MAX_DETECTION_AGE:-0.75}}
TARGET_FOLLOW_MIN_SCORE=${TARGET_FOLLOW_MIN_SCORE:-${CFG_TARGET_FOLLOW_MIN_SCORE:-0.20}}
TARGET_FOLLOW_MIN_BOX_SIZE=${TARGET_FOLLOW_MIN_BOX_SIZE:-${CFG_TARGET_FOLLOW_MIN_BOX_SIZE:-0.01}}
TARGET_FOLLOW_CANDIDATE_TIMEOUT=${TARGET_FOLLOW_CANDIDATE_TIMEOUT:-${CFG_TARGET_FOLLOW_CANDIDATE_TIMEOUT:-8.0}}
TARGET_FOLLOW_CONFIRM_HITS=${TARGET_FOLLOW_CONFIRM_HITS:-${CFG_TARGET_FOLLOW_CONFIRM_HITS:-2}}
TARGET_FOLLOW_CONFIRM_WINDOW=${TARGET_FOLLOW_CONFIRM_WINDOW:-${CFG_TARGET_FOLLOW_CONFIRM_WINDOW:-20.0}}
TARGET_FOLLOW_WEAK_CONFIRM_HITS=${TARGET_FOLLOW_WEAK_CONFIRM_HITS:-${CFG_TARGET_FOLLOW_WEAK_CONFIRM_HITS:-5}}
TARGET_FOLLOW_WEAK_MIN_AVERAGE_SCORE=${TARGET_FOLLOW_WEAK_MIN_AVERAGE_SCORE:-${CFG_TARGET_FOLLOW_WEAK_MIN_AVERAGE_SCORE:-0.24}}
TARGET_FOLLOW_SPATIAL_TOLERANCE=${TARGET_FOLLOW_SPATIAL_TOLERANCE:-${CFG_TARGET_FOLLOW_SPATIAL_TOLERANCE:-0.10}}
TARGET_OBSERVATION_HOLD=${TARGET_OBSERVATION_HOLD:-${CFG_TARGET_OBSERVATION_HOLD:-4.5}}
TARGET_DONE_REQUIRE_LOCKED=${TARGET_DONE_REQUIRE_LOCKED:-${CFG_TARGET_DONE_REQUIRE_LOCKED:-false}}
DETECTION_MAX_SOURCE_IMAGE_AGE=${DETECTION_MAX_SOURCE_IMAGE_AGE:-${CFG_DETECTION_MAX_SOURCE_IMAGE_AGE:-1.0}}
FOLLOW_GOAL_PUBLISH_PERIOD=${FOLLOW_GOAL_PUBLISH_PERIOD:-${CFG_FOLLOW_GOAL_PUBLISH_PERIOD:-0.50}}
FOLLOW_TARGET_STEP_DISTANCE=${FOLLOW_TARGET_STEP_DISTANCE:-${CFG_FOLLOW_TARGET_STEP_DISTANCE:-1.5}}
FOLLOW_TARGET_USE_SCAN_CLIP=${FOLLOW_TARGET_USE_SCAN_CLIP:-${CFG_FOLLOW_TARGET_USE_SCAN_CLIP:-false}}
TARGET_BBOX_ALPHA=${TARGET_BBOX_ALPHA:-${CFG_TARGET_BBOX_ALPHA:-0.35}}
TARGET_HEADING_ALPHA=${TARGET_HEADING_ALPHA:-${CFG_TARGET_HEADING_ALPHA:-0.30}}
TARGET_MAX_HEADING_STEP_DEG=${TARGET_MAX_HEADING_STEP_DEG:-${CFG_TARGET_MAX_HEADING_STEP_DEG:-25.0}}
TARGET_GOAL_REACHED_RADIUS=${TARGET_GOAL_REACHED_RADIUS:-${CFG_TARGET_GOAL_REACHED_RADIUS:-0.75}}
TARGET_GOAL_HEADING_UPDATE_THRESHOLD_DEG=${TARGET_GOAL_HEADING_UPDATE_THRESHOLD_DEG:-${CFG_TARGET_GOAL_HEADING_UPDATE_THRESHOLD_DEG:-35.0}}
TARGET_CACHE_MAX_ADVANCES=${TARGET_CACHE_MAX_ADVANCES:-${CFG_TARGET_CACHE_MAX_ADVANCES:-2}}
TARGET_REACQUIRE_DURATION=${TARGET_REACQUIRE_DURATION:-${CFG_TARGET_REACQUIRE_DURATION:-6.0}}
TARGET_REACQUIRE_DISTANCE=${TARGET_REACQUIRE_DISTANCE:-${CFG_TARGET_REACQUIRE_DISTANCE:-1.0}}
TARGET_REACQUIRE_MAX_ATTEMPTS=${TARGET_REACQUIRE_MAX_ATTEMPTS:-${CFG_TARGET_REACQUIRE_MAX_ATTEMPTS:-1}}
FOLLOW_TARGET_LOST_TIMEOUT=${FOLLOW_TARGET_LOST_TIMEOUT:-${CFG_FOLLOW_TARGET_LOST_TIMEOUT:-12.0}}
FOLLOW_CONTEXT_STEP_DISTANCE=${FOLLOW_CONTEXT_STEP_DISTANCE:-${CFG_FOLLOW_CONTEXT_STEP_DISTANCE:-1.5}}
CONTEXT_GOAL_REACHED_RADIUS=${CONTEXT_GOAL_REACHED_RADIUS:-${CFG_CONTEXT_GOAL_REACHED_RADIUS:-0.55}}
CONTEXT_GOAL_HEADING_UPDATE_THRESHOLD_DEG=${CONTEXT_GOAL_HEADING_UPDATE_THRESHOLD_DEG:-${CFG_CONTEXT_GOAL_HEADING_UPDATE_THRESHOLD_DEG:-35.0}}
CONTEXT_FOLLOW_COOLDOWN=${CONTEXT_FOLLOW_COOLDOWN:-${CFG_CONTEXT_FOLLOW_COOLDOWN:-30.0}}
CONTEXT_STATE_HOLD_TIME=${CONTEXT_STATE_HOLD_TIME:-${CFG_CONTEXT_STATE_HOLD_TIME:-4.0}}
GLOBAL_FRONTIER_ENABLED=${GLOBAL_FRONTIER_ENABLED:-${CFG_GLOBAL_FRONTIER_ENABLED:-true}}
GLOBAL_FRONTIER_TOPIC=${GLOBAL_FRONTIER_TOPIC:-${CFG_GLOBAL_FRONTIER_TOPIC:-/lste/global_frontier_goal}}
TEB_GOAL_TERMINAL_TOPIC=${TEB_GOAL_TERMINAL_TOPIC:-${CFG_TEB_GOAL_TERMINAL_TOPIC:-/lste/teb_goal_terminal}}
TEB_GOAL_FAILURE_TOPIC=${TEB_GOAL_FAILURE_TOPIC:-${CFG_TEB_GOAL_FAILURE_TOPIC:-/lste/teb_goal_failure}}
GOAL_INTENT_TOPIC=${GOAL_INTENT_TOPIC:-${CFG_GOAL_INTENT_TOPIC:-/lste/goal_intent}}
GLOBAL_FRONTIER_MAX_AGE=${GLOBAL_FRONTIER_MAX_AGE:-${CFG_GLOBAL_FRONTIER_MAX_AGE:-3.0}}
GLOBAL_FRONTIER_PERIOD=${GLOBAL_FRONTIER_PERIOD:-${CFG_GLOBAL_FRONTIER_PERIOD:-1.0}}
GLOBAL_FRONTIER_UPDATE_RADIUS=${GLOBAL_FRONTIER_UPDATE_RADIUS:-${CFG_GLOBAL_FRONTIER_UPDATE_RADIUS:-0.75}}
GLOBAL_FRONTIER_EARLY_HANDOFF_RADIUS=${GLOBAL_FRONTIER_EARLY_HANDOFF_RADIUS:-${CFG_GLOBAL_FRONTIER_EARLY_HANDOFF_RADIUS:-0.95}}
GLOBAL_FRONTIER_MIN_HOLD_TIME=${GLOBAL_FRONTIER_MIN_HOLD_TIME:-${CFG_GLOBAL_FRONTIER_MIN_HOLD_TIME:-2.5}}
GLOBAL_FRONTIER_JUMP_DISTANCE=${GLOBAL_FRONTIER_JUMP_DISTANCE:-${CFG_GLOBAL_FRONTIER_JUMP_DISTANCE:-2.0}}
GLOBAL_FRONTIER_JUMP_RELEASE_RADIUS=${GLOBAL_FRONTIER_JUMP_RELEASE_RADIUS:-${CFG_GLOBAL_FRONTIER_JUMP_RELEASE_RADIUS:-0.40}}
GLOBAL_FRONTIER_PREEMPT_CONTEXT=${GLOBAL_FRONTIER_PREEMPT_CONTEXT:-${CFG_GLOBAL_FRONTIER_PREEMPT_CONTEXT:-false}}
GLOBAL_FRONTIER_TERMINAL_HOLD_TIMEOUT=${GLOBAL_FRONTIER_TERMINAL_HOLD_TIMEOUT:-${CFG_GLOBAL_FRONTIER_TERMINAL_HOLD_TIMEOUT:-30.0}}
GLOBAL_FRONTIER_CLEARANCE=${GLOBAL_FRONTIER_CLEARANCE:-${CFG_GLOBAL_FRONTIER_CLEARANCE:-0.52}}
GLOBAL_FRONTIER_FALLBACK_CLEARANCE=${GLOBAL_FRONTIER_FALLBACK_CLEARANCE:-${CFG_GLOBAL_FRONTIER_FALLBACK_CLEARANCE:-0.30}}
GLOBAL_FRONTIER_APPROACH_DISTANCE=${GLOBAL_FRONTIER_APPROACH_DISTANCE:-${CFG_GLOBAL_FRONTIER_APPROACH_DISTANCE:-1.0}}
GLOBAL_FRONTIER_MIN_PATH_DISTANCE=${GLOBAL_FRONTIER_MIN_PATH_DISTANCE:-${CFG_GLOBAL_FRONTIER_MIN_PATH_DISTANCE:-1.2}}
GLOBAL_FRONTIER_LOOKAHEAD_DISTANCE=${GLOBAL_FRONTIER_LOOKAHEAD_DISTANCE:-${CFG_GLOBAL_FRONTIER_LOOKAHEAD_DISTANCE:-4.0}}
GLOBAL_FRONTIER_WAYPOINT_RELEASE_RADIUS=${GLOBAL_FRONTIER_WAYPOINT_RELEASE_RADIUS:-${CFG_GLOBAL_FRONTIER_WAYPOINT_RELEASE_RADIUS:-1.20}}
GLOBAL_FRONTIER_ACTIVE_TIMEOUT=${GLOBAL_FRONTIER_ACTIVE_TIMEOUT:-${CFG_GLOBAL_FRONTIER_ACTIVE_TIMEOUT:-45.0}}
GLOBAL_FRONTIER_STALL_TIMEOUT=${GLOBAL_FRONTIER_STALL_TIMEOUT:-${CFG_GLOBAL_FRONTIER_STALL_TIMEOUT:-12.0}}
GLOBAL_FRONTIER_UNREACHABLE_GRACE=${GLOBAL_FRONTIER_UNREACHABLE_GRACE:-${CFG_GLOBAL_FRONTIER_UNREACHABLE_GRACE:-20.0}}
GLOBAL_FRONTIER_REJECTED_TIMEOUT=${GLOBAL_FRONTIER_REJECTED_TIMEOUT:-${CFG_GLOBAL_FRONTIER_REJECTED_TIMEOUT:-180.0}}
GLOBAL_FRONTIER_COMPLETED_RADIUS=${GLOBAL_FRONTIER_COMPLETED_RADIUS:-${CFG_GLOBAL_FRONTIER_COMPLETED_RADIUS:-1.25}}
GLOBAL_FRONTIER_STRUCTURE_RADIUS_CELLS=${GLOBAL_FRONTIER_STRUCTURE_RADIUS_CELLS:-${CFG_GLOBAL_FRONTIER_STRUCTURE_RADIUS_CELLS:-10}}
GLOBAL_FRONTIER_STRUCTURE_WEIGHT=${GLOBAL_FRONTIER_STRUCTURE_WEIGHT:-${CFG_GLOBAL_FRONTIER_STRUCTURE_WEIGHT:-0.16}}
GLOBAL_FRONTIER_MIN_STRUCTURE_CELLS=${GLOBAL_FRONTIER_MIN_STRUCTURE_CELLS:-${CFG_GLOBAL_FRONTIER_MIN_STRUCTURE_CELLS:-3}}
GLOBAL_FRONTIER_HEADING_WEIGHT=${GLOBAL_FRONTIER_HEADING_WEIGHT:-${CFG_GLOBAL_FRONTIER_HEADING_WEIGHT:-2.5}}
GLOBAL_FRONTIER_HEADING_HARD_LIMIT_DEG=${GLOBAL_FRONTIER_HEADING_HARD_LIMIT_DEG:-${CFG_GLOBAL_FRONTIER_HEADING_HARD_LIMIT_DEG:-115.0}}
GLOBAL_FRONTIER_PLANNING_PERIOD=${GLOBAL_FRONTIER_PLANNING_PERIOD:-${CFG_GLOBAL_FRONTIER_PLANNING_PERIOD:-1.5}}
TARGET_ROUTE_VALIDATION=${TARGET_ROUTE_VALIDATION:-${CFG_TARGET_ROUTE_VALIDATION:-true}}
TARGET_ROUTE_VALIDATION_TIMEOUT=${TARGET_ROUTE_VALIDATION_TIMEOUT:-${CFG_TARGET_ROUTE_VALIDATION_TIMEOUT:-0.05}}
TARGET_ROUTE_VALIDATION_TOLERANCE=${TARGET_ROUTE_VALIDATION_TOLERANCE:-${CFG_TARGET_ROUTE_VALIDATION_TOLERANCE:-0.20}}
TARGET_ROUTE_VALIDATION_PERIOD=${TARGET_ROUTE_VALIDATION_PERIOD:-${CFG_TARGET_ROUTE_VALIDATION_PERIOD:-1.0}}
GOAL_DEBUG_LOG=${GOAL_DEBUG_LOG:-${CFG_GOAL_DEBUG_LOG:-false}}
GLOBAL_GOAL_SOURCE=${GLOBAL_GOAL_SOURCE:-${CFG_GLOBAL_GOAL_SOURCE:-brain}}
FIXED_GLOBAL_GOAL_X=${FIXED_GLOBAL_GOAL_X:-${CFG_FIXED_GLOBAL_GOAL_X:-0.0}}
FIXED_GLOBAL_GOAL_Y=${FIXED_GLOBAL_GOAL_Y:-${CFG_FIXED_GLOBAL_GOAL_Y:-0.0}}
FIXED_GLOBAL_GOAL_YAW=${FIXED_GLOBAL_GOAL_YAW:-${CFG_FIXED_GLOBAL_GOAL_YAW:-0.0}}
FIXED_GLOBAL_GOAL_PUBLISH_PERIOD=${FIXED_GLOBAL_GOAL_PUBLISH_PERIOD:-${CFG_FIXED_GLOBAL_GOAL_PUBLISH_PERIOD:-1.0}}
FIXED_GLOBAL_GOAL_ALLOW_CLICK_OVERRIDE=${FIXED_GLOBAL_GOAL_ALLOW_CLICK_OVERRIDE:-${CFG_FIXED_GLOBAL_GOAL_ALLOW_CLICK_OVERRIDE:-false}}
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
LSTE_CONTROLLER=${LSTE_CONTROLLER:-${CFG_LSTE_CONTROLLER:-teb}}
SAPPO_SPEED=${SAPPO_SPEED:-0.50}
SAPPO_PYTHON=${SAPPO_PYTHON:-$HOME/miniconda3/envs/rlenvs/bin/python}
SAPPO_CONTROLLER_MODE=${SAPPO_CONTROLLER_MODE:-${CFG_SAPPO_CONTROLLER_MODE:-rl_grid_guard}}
SAPPO_IN_PLACE_TURN_ANGULAR_THRESHOLD=${SAPPO_IN_PLACE_TURN_ANGULAR_THRESHOLD:-${CFG_SAPPO_IN_PLACE_TURN_ANGULAR_THRESHOLD:-0.12}}
SAPPO_BOUNDARY_TURN_LOCK_TIME=${SAPPO_BOUNDARY_TURN_LOCK_TIME:-${CFG_SAPPO_BOUNDARY_TURN_LOCK_TIME:-3.0}}
SAPPO_BOUNDARY_TURN_MAX_DURATION=${SAPPO_BOUNDARY_TURN_MAX_DURATION:-${CFG_SAPPO_BOUNDARY_TURN_MAX_DURATION:-2.8}}
SAPPO_BOUNDARY_TURN_MAX_ANGLE_DEG=${SAPPO_BOUNDARY_TURN_MAX_ANGLE_DEG:-${CFG_SAPPO_BOUNDARY_TURN_MAX_ANGLE_DEG:-112.0}}
SAPPO_BOUNDARY_TURN_MAX_ATTEMPTS=${SAPPO_BOUNDARY_TURN_MAX_ATTEMPTS:-${CFG_SAPPO_BOUNDARY_TURN_MAX_ATTEMPTS:-2}}
SAPPO_BOUNDARY_REENTRY_COOLDOWN=${SAPPO_BOUNDARY_REENTRY_COOLDOWN:-${CFG_SAPPO_BOUNDARY_REENTRY_COOLDOWN:-1.5}}
SAPPO_BOUNDARY_PROGRESS_TIMEOUT=${SAPPO_BOUNDARY_PROGRESS_TIMEOUT:-${CFG_SAPPO_BOUNDARY_PROGRESS_TIMEOUT:-10.0}}
SAPPO_BOUNDARY_PROGRESS_MARGIN=${SAPPO_BOUNDARY_PROGRESS_MARGIN:-${CFG_SAPPO_BOUNDARY_PROGRESS_MARGIN:-0.45}}
SAPPO_BOUNDARY_WALL_DISTANCE=${SAPPO_BOUNDARY_WALL_DISTANCE:-${CFG_SAPPO_BOUNDARY_WALL_DISTANCE:-0.72}}
SAPPO_BOUNDARY_WALL_KP=${SAPPO_BOUNDARY_WALL_KP:-${CFG_SAPPO_BOUNDARY_WALL_KP:-0.45}}
SAPPO_BOUNDARY_WALL_HEADING_KP=${SAPPO_BOUNDARY_WALL_HEADING_KP:-${CFG_SAPPO_BOUNDARY_WALL_HEADING_KP:-0.35}}
SAPPO_BOUNDARY_WALL_MAX_LINEAR=${SAPPO_BOUNDARY_WALL_MAX_LINEAR:-${CFG_SAPPO_BOUNDARY_WALL_MAX_LINEAR:-0.28}}
SAPPO_BOUNDARY_WALL_MAX_RANGE=${SAPPO_BOUNDARY_WALL_MAX_RANGE:-${CFG_SAPPO_BOUNDARY_WALL_MAX_RANGE:-1.80}}
SAPPO_BOUNDARY_WALL_FILTER_ALPHA=${SAPPO_BOUNDARY_WALL_FILTER_ALPHA:-${CFG_SAPPO_BOUNDARY_WALL_FILTER_ALPHA:-0.28}}
SAPPO_BOUNDARY_WALL_ANGULAR_STEP=${SAPPO_BOUNDARY_WALL_ANGULAR_STEP:-${CFG_SAPPO_BOUNDARY_WALL_ANGULAR_STEP:-0.08}}
SAPPO_BOUNDARY_WALL_ANGULAR_DEADBAND=${SAPPO_BOUNDARY_WALL_ANGULAR_DEADBAND:-${CFG_SAPPO_BOUNDARY_WALL_ANGULAR_DEADBAND:-0.035}}
SAPPO_BOUNDARY_GOAL_CANCEL_ANGLE_DEG=${SAPPO_BOUNDARY_GOAL_CANCEL_ANGLE_DEG:-${CFG_SAPPO_BOUNDARY_GOAL_CANCEL_ANGLE_DEG:-165.0}}
SAPPO_BOUNDARY_GOAL_CANCEL_DISTANCE=${SAPPO_BOUNDARY_GOAL_CANCEL_DISTANCE:-${CFG_SAPPO_BOUNDARY_GOAL_CANCEL_DISTANCE:-0.75}}
SAPPO_WAYPOINT_HOLD_RADIUS=${SAPPO_WAYPOINT_HOLD_RADIUS:-${CFG_SAPPO_WAYPOINT_HOLD_RADIUS:-0.40}}
SAPPO_WAYPOINT_SWITCH_DISTANCE=${SAPPO_WAYPOINT_SWITCH_DISTANCE:-${CFG_SAPPO_WAYPOINT_SWITCH_DISTANCE:-0.75}}
SAPPO_GRID_OBSTACLE_RADIUS=${SAPPO_GRID_OBSTACLE_RADIUS:-${CFG_SAPPO_GRID_OBSTACLE_RADIUS:-0.48}}
SAPPO_GRID_EDGE_CLEARANCE=${SAPPO_GRID_EDGE_CLEARANCE:-${CFG_SAPPO_GRID_EDGE_CLEARANCE:-0.42}}
SAPPO_REORIENT_OBSTACLE_CLEARANCE=${SAPPO_REORIENT_OBSTACLE_CLEARANCE:-${CFG_SAPPO_REORIENT_OBSTACLE_CLEARANCE:-0.72}}
SAPPO_GOAL_TOLERANCE=${SAPPO_GOAL_TOLERANCE:-${CFG_SAPPO_GOAL_TOLERANCE:-0.30}}
SAPPO_INTERMEDIATE_GOAL_TOLERANCE=${SAPPO_INTERMEDIATE_GOAL_TOLERANCE:-${CFG_SAPPO_INTERMEDIATE_GOAL_TOLERANCE:-0.45}}
SAPPO_MAX_LINEAR_ACTION_STEP=${SAPPO_MAX_LINEAR_ACTION_STEP:-${CFG_SAPPO_MAX_LINEAR_ACTION_STEP:-0.12}}
SAPPO_MAX_LINEAR_ACTION_DECEL=${SAPPO_MAX_LINEAR_ACTION_DECEL:-${CFG_SAPPO_MAX_LINEAR_ACTION_DECEL:-0.06}}
SAPPO_MAX_ANGULAR_ACTION_STEP=${SAPPO_MAX_ANGULAR_ACTION_STEP:-${CFG_SAPPO_MAX_ANGULAR_ACTION_STEP:-0.16}}
SAPPO_ANGULAR_SIGN_SWITCH_THRESHOLD=${SAPPO_ANGULAR_SIGN_SWITCH_THRESHOLD:-${CFG_SAPPO_ANGULAR_SIGN_SWITCH_THRESHOLD:-0.18}}
TEB_MAX_LINEAR_SPEED=${TEB_MAX_LINEAR_SPEED:-${CFG_TEB_MAX_LINEAR_SPEED:-0.50}}
TEB_FORWARD_ONLY=${TEB_FORWARD_ONLY:-${CFG_TEB_FORWARD_ONLY:-true}}
TEB_ANGULAR_SIGN_SWITCH_THRESHOLD=${TEB_ANGULAR_SIGN_SWITCH_THRESHOLD:-${CFG_TEB_ANGULAR_SIGN_SWITCH_THRESHOLD:-0.12}}
MUX_GOVERNOR_DECEL=${MUX_GOVERNOR_DECEL:-${CFG_MUX_GOVERNOR_DECEL:-0.60}}
MUX_GOVERNOR_MIN_GAP=${MUX_GOVERNOR_MIN_GAP:-${CFG_MUX_GOVERNOR_MIN_GAP:-0.45}}
MUX_GOVERNOR_MAX_SPEED=${MUX_GOVERNOR_MAX_SPEED:-${CFG_MUX_GOVERNOR_MAX_SPEED:-0.50}}
TEB_ANGULAR_DEADBAND=${TEB_ANGULAR_DEADBAND:-${CFG_TEB_ANGULAR_DEADBAND:-0.02}}
TEB_ACTION_HANDOFF_DISTANCE=${TEB_ACTION_HANDOFF_DISTANCE:-${CFG_TEB_ACTION_HANDOFF_DISTANCE:-1.00}}
TEB_ACTION_HANDOFF_RADIUS=${TEB_ACTION_HANDOFF_RADIUS:-${CFG_TEB_ACTION_HANDOFF_RADIUS:-0.30}}
TEB_ACTION_PROGRESS_TIMEOUT=${TEB_ACTION_PROGRESS_TIMEOUT:-${CFG_TEB_ACTION_PROGRESS_TIMEOUT:-12.0}}
TEB_ACTION_PROGRESS_EPSILON=${TEB_ACTION_PROGRESS_EPSILON:-${CFG_TEB_ACTION_PROGRESS_EPSILON:-0.12}}
TEB_ACTION_HANDOFF_MIN_INTERVAL=${TEB_ACTION_HANDOFF_MIN_INTERVAL:-${CFG_TEB_ACTION_HANDOFF_MIN_INTERVAL:-4.0}}
TEB_NAVFN_ALLOW_UNKNOWN=${TEB_NAVFN_ALLOW_UNKNOWN:-${CFG_TEB_NAVFN_ALLOW_UNKNOWN:-true}}
TEB_TARGET_EARLY_HANDOFF_DISTANCE=${TEB_TARGET_EARLY_HANDOFF_DISTANCE:-${CFG_TEB_TARGET_EARLY_HANDOFF_DISTANCE:-0.70}}
TEB_TARGET_EARLY_HANDOFF_MIN_DELTA=${TEB_TARGET_EARLY_HANDOFF_MIN_DELTA:-${CFG_TEB_TARGET_EARLY_HANDOFF_MIN_DELTA:-0.40}}
TEB_FRONTIER_REPLACEMENT_MAX_DELTA=${TEB_FRONTIER_REPLACEMENT_MAX_DELTA:-${CFG_TEB_FRONTIER_REPLACEMENT_MAX_DELTA:-1.00}}
TEB_ALLOW_ROUTE_CONTINUATION_REPLACEMENT=${TEB_ALLOW_ROUTE_CONTINUATION_REPLACEMENT:-${CFG_TEB_ALLOW_ROUTE_CONTINUATION_REPLACEMENT:-true}}
TEB_FRONTIER_EARLY_HANDOFF_MAX_HEADING_DEG=${TEB_FRONTIER_EARLY_HANDOFF_MAX_HEADING_DEG:-${CFG_TEB_FRONTIER_EARLY_HANDOFF_MAX_HEADING_DEG:-75.0}}
TEB_FRONTIER_STALE_RECOVERY_GRACE=${TEB_FRONTIER_STALE_RECOVERY_GRACE:-${CFG_TEB_FRONTIER_STALE_RECOVERY_GRACE:-4.0}}
TEB_FRONTIER_STALE_RECOVERY_MAX_DISTANCE=${TEB_FRONTIER_STALE_RECOVERY_MAX_DISTANCE:-${CFG_TEB_FRONTIER_STALE_RECOVERY_MAX_DISTANCE:-1.50}}
case "$LSTE_CONTROLLER" in
  teleop|sappo|teb) ;;
  *)
    echo "[error] LSTE_CONTROLLER must be 'teleop', 'sappo', or 'teb' (got '$LSTE_CONTROLLER')" >&2
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

echo "[nodes] Waiting for Gazebo spawn service..."
gazebo_ready=0
for _ in $(seq 1 120); do
  if rosservice list 2>/dev/null | grep -qx /gazebo/spawn_urdf_model; then
    gazebo_ready=1
    break
  fi
  sleep 1
done
if [[ "$gazebo_ready" != "1" ]]; then
  echo "[error] Gazebo spawn service did not become ready; leave lste-env running and retry." >&2
  exit 1
fi

# ---- 如果节点 session 已存在，先杀 ----
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "[nodes] 发现旧 session '$SESSION'，杀掉重建..."
  tmux kill-session -t "=$SESSION"
fi

# A pane's shell can outlive the tmux session briefly, and rospy children may
# still own their fixed names while the replacement windows are starting.
# Ask every node owned by this session to unregister, then wait for the master
# to observe the cleanup. Without this barrier a duplicate state/goal node can
# shut down the freshly started goal manager and leave the vehicle idle.
stale_nodes=()
while IFS= read -r node; do
  [[ -n "$node" ]] && stale_nodes+=("$node")
done < <(
  rosnode list 2>/dev/null | grep -E '^/(lste_|move_base$|rviz_lste_det_vis$|pro3/)' || true
)
if (( ${#stale_nodes[@]} > 0 )); then
  echo "[nodes] 请求旧 ROS 节点退出: ${stale_nodes[*]}"
  for node in "${stale_nodes[@]}"; do
    # A dead XML-RPC endpoint can make rosnode kill wait for seconds. Send
    # all shutdown requests concurrently and cap each one so hot restart does
    # not become serially proportional to the number of stale nodes.
    (timeout 1 rosnode kill "$node" >/dev/null 2>&1 || true) &
  done
  wait || true
fi
for _ in $(seq 1 30); do
  if ! rosnode list 2>/dev/null | grep -Eq '^/(lste_|move_base$|rviz_lste_det_vis$|pro3/)'; then
    break
  fi
  sleep 0.2
done
if rosnode list 2>/dev/null | grep -Eq '^/(lste_|move_base$|rviz_lste_det_vis$|pro3/)'; then
  echo "[nodes] 警告: 旧 ROS 节点仍在注销，继续启动前保留最后一次检查" >&2
fi
echo "[nodes] 清理旧 session 完毕"

# lste-env may keep roscore alive across node restarts, so remove the previous
# prompt decision before the gated vLLM window starts.
bash -lc "WS=\"$WS\"; source \"$WS/scripts/config/pipeline_env.sh\"; rosparam delete /lste_prompt_node/needs_vllm >/dev/null 2>&1 || true"

# ---- 新建节点 session ----
# In the split lifecycle, this session is the sole owner of the Pro3 support
# nodes (robot_description, robot_state_publisher and LiDAR frame alias). It
# may reuse a Gazebo model already present in the world, but it must still start
# that one support-node set for SLAM's base_footprint -> laser TF chain.
echo "[nodes] 创建 session '$SESSION'..."
tmux new-session -d -s "$SESSION" -c "$WS" -n "pro3" \
  "bash -lc 'WS=\"$WS\"; source \"$WS/scripts/config/pipeline_env.sh\"; set -e; \
until rostopic list >/dev/null 2>&1; do echo \"waiting for rocore...\"; sleep 1; done; \
if rosservice call /gazebo/get_world_properties 2>/dev/null | grep -q \"pro3\"; then \
  echo \"[pro3] Existing Gazebo model found; start its TF support nodes.\"; \
  roslaunch lste_core spawn_pro3.launch spawn_model:=false start_support_nodes:=true \
    x:=$PRO3_SPAWN_X y:=$PRO3_SPAWN_Y z:=$PRO3_SPAWN_Z yaw:=$PRO3_SPAWN_YAW; \
else \
  roslaunch lste_core spawn_pro3.launch spawn_model:=true start_support_nodes:=true \
    x:=$PRO3_SPAWN_X y:=$PRO3_SPAWN_Y z:=$PRO3_SPAWN_Z yaw:=$PRO3_SPAWN_YAW; \
fi; \
echo; echo \"[EXIT] pro3 spawn\"; exec bash'"
echo "[nodes] 创建后检查 sessions: $(tmux list-sessions 2>&1)"
tmux set-option -t "=$SESSION:" remain-on-exit on
echo "[nodes] pro3 spawned (window 0)"

echo "[nodes] Waiting for Pro3 odometry..."
pro3_ready=0
for _ in $(seq 1 120); do
  if rostopic list 2>/dev/null | grep -qx /pro3/wheel_odom; then
    pro3_ready=1
    break
  fi
  sleep 1
done
if [[ "$pro3_ready" != "1" ]]; then
  echo "[error] Pro3 did not publish /pro3/wheel_odom; stopping incomplete node session." >&2
  tmux capture-pane -pJ -t "=$SESSION:pro3" -S -80 >&2 || true
  tmux kill-session -t "=$SESSION" || true
  exit 1
fi

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
  "$WAIT_ROSCORE; rosrun lste_topo_access lste_goal_manager.py \
    _follow_locked_done_time:=$FOLLOW_LOCKED_DONE_TIME _debug_goal_log:=$GOAL_DEBUG_LOG \
    _target_done_min_box_width:=$TARGET_DONE_MIN_BOX_WIDTH \
    _target_done_min_box_height:=$TARGET_DONE_MIN_BOX_HEIGHT \
    _target_done_min_score:=$TARGET_DONE_MIN_SCORE \
    _target_done_min_hold_time:=$TARGET_DONE_MIN_HOLD_TIME \
    _target_done_min_fresh_hits:=$TARGET_DONE_MIN_FRESH_HITS \
    _target_done_max_detection_age:=$TARGET_DONE_MAX_DETECTION_AGE \
    _target_follow_min_score:=$TARGET_FOLLOW_MIN_SCORE \
    _target_follow_min_box_size:=$TARGET_FOLLOW_MIN_BOX_SIZE \
    _target_follow_candidate_timeout:=$TARGET_FOLLOW_CANDIDATE_TIMEOUT \
    _target_follow_confirm_hits:=$TARGET_FOLLOW_CONFIRM_HITS \
    _target_follow_confirm_window:=$TARGET_FOLLOW_CONFIRM_WINDOW \
    _target_follow_weak_confirm_hits:=$TARGET_FOLLOW_WEAK_CONFIRM_HITS \
    _target_follow_weak_min_average_score:=$TARGET_FOLLOW_WEAK_MIN_AVERAGE_SCORE \
    _target_follow_spatial_tolerance:=$TARGET_FOLLOW_SPATIAL_TOLERANCE \
    _target_observation_hold:=$TARGET_OBSERVATION_HOLD \
    _target_done_require_locked:=$TARGET_DONE_REQUIRE_LOCKED \
    _follow_goal_publish_period:=$FOLLOW_GOAL_PUBLISH_PERIOD \
    _follow_target_step_distance:=$FOLLOW_TARGET_STEP_DISTANCE \
    _follow_target_use_scan_clip:=$FOLLOW_TARGET_USE_SCAN_CLIP \
    _target_bbox_alpha:=$TARGET_BBOX_ALPHA \
    _target_heading_alpha:=$TARGET_HEADING_ALPHA \
    _target_max_heading_step_deg:=$TARGET_MAX_HEADING_STEP_DEG \
    _target_goal_reached_radius:=$TARGET_GOAL_REACHED_RADIUS \
    _target_goal_heading_update_threshold_deg:=$TARGET_GOAL_HEADING_UPDATE_THRESHOLD_DEG \
    _target_cache_max_advances:=$TARGET_CACHE_MAX_ADVANCES \
    _follow_target_lost_timeout:=$FOLLOW_TARGET_LOST_TIMEOUT \
    _target_reacquire_duration:=$TARGET_REACQUIRE_DURATION \
    _target_reacquire_distance:=$TARGET_REACQUIRE_DISTANCE \
    _target_reacquire_max_attempts:=$TARGET_REACQUIRE_MAX_ATTEMPTS \
    _follow_context_step_distance:=$FOLLOW_CONTEXT_STEP_DISTANCE \
    _context_goal_reached_radius:=$CONTEXT_GOAL_REACHED_RADIUS \
    _context_goal_heading_update_threshold_deg:=$CONTEXT_GOAL_HEADING_UPDATE_THRESHOLD_DEG \
    _context_follow_cooldown:=$CONTEXT_FOLLOW_COOLDOWN \
    _context_state_hold_time:=$CONTEXT_STATE_HOLD_TIME \
    _global_frontier_enabled:=$GLOBAL_FRONTIER_ENABLED \
    _global_frontier_topic:=$GLOBAL_FRONTIER_TOPIC \
    _teb_goal_terminal_topic:=$TEB_GOAL_TERMINAL_TOPIC \
    _teb_goal_failure_topic:=$TEB_GOAL_FAILURE_TOPIC \
    _goal_intent_topic:=$GOAL_INTENT_TOPIC \
    _global_frontier_max_age:=$GLOBAL_FRONTIER_MAX_AGE \
    _global_frontier_period:=$GLOBAL_FRONTIER_PERIOD \
    _global_frontier_update_radius:=$GLOBAL_FRONTIER_UPDATE_RADIUS \
    _global_frontier_early_handoff_radius:=$GLOBAL_FRONTIER_EARLY_HANDOFF_RADIUS \
    _global_frontier_min_hold_time:=$GLOBAL_FRONTIER_MIN_HOLD_TIME \
    _global_frontier_jump_distance:=$GLOBAL_FRONTIER_JUMP_DISTANCE \
    _global_frontier_jump_release_radius:=$GLOBAL_FRONTIER_JUMP_RELEASE_RADIUS \
    _global_frontier_preempt_context:=$GLOBAL_FRONTIER_PREEMPT_CONTEXT \
    _global_frontier_terminal_hold_timeout:=$GLOBAL_FRONTIER_TERMINAL_HOLD_TIMEOUT \
    _target_route_validation:=$TARGET_ROUTE_VALIDATION \
    _target_route_validation_timeout:=$TARGET_ROUTE_VALIDATION_TIMEOUT \
    _target_route_validation_tolerance:=$TARGET_ROUTE_VALIDATION_TOLERANCE \
    _target_route_validation_period:=$TARGET_ROUTE_VALIDATION_PERIOD \
    _controller_mode:=$LSTE_CONTROLLER \
    _global_goal_source:=$GLOBAL_GOAL_SOURCE \
    _fixed_goal_x:=$FIXED_GLOBAL_GOAL_X _fixed_goal_y:=$FIXED_GLOBAL_GOAL_Y \
    _fixed_goal_yaw:=$FIXED_GLOBAL_GOAL_YAW \
    _fixed_goal_publish_period:=$FIXED_GLOBAL_GOAL_PUBLISH_PERIOD \
    _fixed_goal_allow_click_override:=$FIXED_GLOBAL_GOAL_ALLOW_CLICK_OVERRIDE"

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
      _max_source_image_age:=$DETECTION_MAX_SOURCE_IMAGE_AGE \
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
  "$WAIT_ROSCORE; rosrun lste_core lste_state_node.py \
    _suspicious_window:=$SUSPICIOUS_WINDOW \
    _subtype_switch_min_hits:=$STATE_SUBTYPE_SWITCH_MIN_HITS \
    _lock_total_enter:=$TARGET_LOCK_TOTAL_MIN _lock_target_enter:=$TARGET_LOCK_SCORE_MIN \
    _lock_enter_frames:=$TARGET_LOCK_WINDOW _lock_enter_min_hits:=$TARGET_LOCK_MIN_HITS \
    _lock_total_exit:=$TARGET_LOCK_EXIT_TOTAL_MIN _lock_target_exit:=$TARGET_LOCK_EXIT_SCORE_MIN \
    _lock_exit_unstable_frames:=$TARGET_LOCK_EXIT_UNSTABLE_FRAMES"

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

# 10: Legacy GP frontier (optional; redundant for online-SLAM + TEB).
if [[ "${LEGACY_GP_FRONTIER_ENABLED,,}" == "true" || "$LEGACY_GP_FRONTIER_ENABLED" == "1" ]]; then
  tmux_new_window 10 "$WS" "gp_frontier" \
    "$WAIT_ROSCORE; conda activate vsgp; roslaunch lste_topo_access gp_frontier.launch \
      access_topo_config:=$ACCESS_TOPO_CONFIG \
      access_topo_config_pass:=$ACCESS_TOPO_CONFIG_PASS \
      access_topo_config_sus_c:=$ACCESS_TOPO_CONFIG_SUS_C \
      access_topo_test_name:=$ACCESS_TOPO_TEST_NAME \
      access_topo_run_name:=$ACCESS_TOPO_RUN_NAME \
      follow_locked_done_time:=$FOLLOW_LOCKED_DONE_TIME \
      frontier_log:=$FRONTIER_LOG \
      launch_goal_manager:=false"
else
  echo "[nodes] Legacy GP frontier disabled; online SLAM frontier remains active."
fi

# 11: Online SLAM + globally connected frontier route.  It never replaces the
# local controller or publishes /lste/final_goal itself.
if [[ "${GLOBAL_FRONTIER_ENABLED,,}" == "true" || "$GLOBAL_FRONTIER_ENABLED" == "1" ]]; then
  tmux_new_window 11 "$WS" "global_frontier" \
    "$WAIT_ROSCORE; roslaunch lste_topo_access online_slam_frontier.launch \
      goal_topic:=$GLOBAL_FRONTIER_TOPIC \
      clearance:=$GLOBAL_FRONTIER_CLEARANCE \
      fallback_clearance:=$GLOBAL_FRONTIER_FALLBACK_CLEARANCE \
      frontier_approach_distance:=$GLOBAL_FRONTIER_APPROACH_DISTANCE \
      min_path_distance:=$GLOBAL_FRONTIER_MIN_PATH_DISTANCE \
      lookahead_distance:=$GLOBAL_FRONTIER_LOOKAHEAD_DISTANCE \
      waypoint_release_radius:=$GLOBAL_FRONTIER_WAYPOINT_RELEASE_RADIUS \
      early_handoff_distance:=$GLOBAL_FRONTIER_EARLY_HANDOFF_RADIUS \
      active_timeout:=$GLOBAL_FRONTIER_ACTIVE_TIMEOUT \
      stall_timeout:=$GLOBAL_FRONTIER_STALL_TIMEOUT \
      unreachable_grace:=$GLOBAL_FRONTIER_UNREACHABLE_GRACE \
      rejected_timeout:=$GLOBAL_FRONTIER_REJECTED_TIMEOUT \
      completed_radius:=$GLOBAL_FRONTIER_COMPLETED_RADIUS \
      structure_radius_cells:=$GLOBAL_FRONTIER_STRUCTURE_RADIUS_CELLS \
      structure_weight:=$GLOBAL_FRONTIER_STRUCTURE_WEIGHT \
      min_structure_cells:=$GLOBAL_FRONTIER_MIN_STRUCTURE_CELLS \
      heading_weight:=$GLOBAL_FRONTIER_HEADING_WEIGHT \
      heading_hard_limit_deg:=$GLOBAL_FRONTIER_HEADING_HARD_LIMIT_DEG \
      planning_period:=$GLOBAL_FRONTIER_PLANNING_PERIOD"
fi

# 12: TEB navigation stack. It always runs on an isolated mux input so the
# controller can be selected hot without rebuilding or restarting the world.
tmux_new_window 12 "$WS" "teb_nav" \
  "$WAIT_ROSCORE; until rostopic list | grep -qx /pro3/rlscan; do sleep 1; done; \
   roslaunch lste_topo_access teb_navigation.launch \
     scan_topic:=/pro3/rlscan odom_topic:=/pro3/wheel_odom \
     terminal_topic:=$TEB_GOAL_TERMINAL_TOPIC \
     target_failure_topic:=$TEB_GOAL_FAILURE_TOPIC \
     intent_topic:=$GOAL_INTENT_TOPIC \
     max_linear_speed:=$TEB_MAX_LINEAR_SPEED controller_mode:=$LSTE_CONTROLLER \
     handoff_distance:=$TEB_ACTION_HANDOFF_DISTANCE \
     handoff_radius:=$TEB_ACTION_HANDOFF_RADIUS \
     progress_timeout:=$TEB_ACTION_PROGRESS_TIMEOUT \
     progress_epsilon:=$TEB_ACTION_PROGRESS_EPSILON \
     handoff_min_interval:=$TEB_ACTION_HANDOFF_MIN_INTERVAL \
     navfn_allow_unknown:=$TEB_NAVFN_ALLOW_UNKNOWN \
     target_early_handoff_distance:=$TEB_TARGET_EARLY_HANDOFF_DISTANCE \
     target_early_handoff_min_delta:=$TEB_TARGET_EARLY_HANDOFF_MIN_DELTA \
     frontier_replacement_max_delta:=$TEB_FRONTIER_REPLACEMENT_MAX_DELTA \
     allow_route_continuation_replacement:=$TEB_ALLOW_ROUTE_CONTINUATION_REPLACEMENT \
     frontier_early_handoff_max_heading_deg:=$TEB_FRONTIER_EARLY_HANDOFF_MAX_HEADING_DEG \
     frontier_stale_recovery_grace:=$TEB_FRONTIER_STALE_RECOVERY_GRACE \
     frontier_stale_recovery_max_distance:=$TEB_FRONTIER_STALE_RECOVERY_MAX_DISTANCE"

# 13: Legacy GP frontier RViz (only useful with the legacy GP node).
if [[ "${LEGACY_GP_FRONTIER_ENABLED,,}" == "true" || "$LEGACY_GP_FRONTIER_ENABLED" == "1" ]]; then
  tmux_new_window 13 "$WS" "rviz_frontier" \
    "$WAIT_ROSCORE; rviz -d $GP_FRONTIER_RVIZ"
fi

# 14: GUI 热切换请求处理（只切换 mux，不停止控制器进程）
tmux_new_window 14 "$WS" "controller_switch" \
  "$WAIT_ROSCORE; rosrun lste_core lste_controller_switch_node.py \
    _initial_mode:=$LSTE_CONTROLLER"

# 15: 唯一 /cmd_vel 发布者
tmux_new_window 15 "$WS" "cmd_vel_mux" \
  "$WAIT_ROSCORE; rosrun lste_core lste_cmd_vel_mux_node.py \
    _initial_mode:=$LSTE_CONTROLLER _teb_forward_only:=$TEB_FORWARD_ONLY \
    _teb_angular_sign_switch_threshold:=$TEB_ANGULAR_SIGN_SWITCH_THRESHOLD \
    _teb_angular_deadband:=$TEB_ANGULAR_DEADBAND \
    _scan_topic:=/pro3/rlscan \
    _governor_decel:=$MUX_GOVERNOR_DECEL \
    _governor_min_gap:=$MUX_GOVERNOR_MIN_GAP \
    _governor_max_speed:=$MUX_GOVERNOR_MAX_SPEED"

# 16: 持续健康检查（在控制器前启动，以便捕获控制器启动失败）
tmux_new_window 16 "$WS" "health" \
  "$WAIT_ROSCORE; python3 $WS/scripts/tools/lste_health_audit.py"

# 17: Structured navigation telemetry. This observer never publishes control,
# but writes one timestamped metrics log for this live run so goal changes,
# action preemptions, steering sign flips, stops, and lidar clearance can be
# compared after each experiment.
mkdir -p "$WS/runtime/navigation"
tmux_new_window 17 "$WS/runtime/navigation" "metrics" \
  "$WAIT_ROSCORE; rosrun lste_topo_access lste_navigation_metrics.py \
    _log_dir:=$WS/runtime/navigation/logs _retention_days:=15 \
    _teb_strong_angular_threshold:=$TEB_ANGULAR_SIGN_SWITCH_THRESHOLD"

# Controller selection is handled at the end by switch_controller.sh. TEB and
# teleop are persistent; SA-PPO is started lazily only when selected (or when
# SAPPO_BACKGROUND_ENABLED is explicitly true).
SAPPO_SPEED="$SAPPO_SPEED" SAPPO_PYTHON="$SAPPO_PYTHON" SAPPO_CONTROLLER_MODE="$SAPPO_CONTROLLER_MODE" SAPPO_GOAL_TOLERANCE="$SAPPO_GOAL_TOLERANCE" \
SAPPO_INTERMEDIATE_GOAL_TOLERANCE="$SAPPO_INTERMEDIATE_GOAL_TOLERANCE" SAPPO_ANGULAR_SIGN_SWITCH_THRESHOLD="$SAPPO_ANGULAR_SIGN_SWITCH_THRESHOLD" \
SAPPO_MAX_LINEAR_ACTION_STEP="$SAPPO_MAX_LINEAR_ACTION_STEP" SAPPO_MAX_LINEAR_ACTION_DECEL="$SAPPO_MAX_LINEAR_ACTION_DECEL" \
SAPPO_MAX_ANGULAR_ACTION_STEP="$SAPPO_MAX_ANGULAR_ACTION_STEP" \
SAPPO_IN_PLACE_TURN_ANGULAR_THRESHOLD="$SAPPO_IN_PLACE_TURN_ANGULAR_THRESHOLD" \
SAPPO_BOUNDARY_TURN_LOCK_TIME="$SAPPO_BOUNDARY_TURN_LOCK_TIME" SAPPO_BOUNDARY_TURN_MAX_DURATION="$SAPPO_BOUNDARY_TURN_MAX_DURATION" \
SAPPO_BOUNDARY_TURN_MAX_ANGLE_DEG="$SAPPO_BOUNDARY_TURN_MAX_ANGLE_DEG" SAPPO_BOUNDARY_TURN_MAX_ATTEMPTS="$SAPPO_BOUNDARY_TURN_MAX_ATTEMPTS" \
SAPPO_BOUNDARY_REENTRY_COOLDOWN="$SAPPO_BOUNDARY_REENTRY_COOLDOWN" \
SAPPO_BOUNDARY_PROGRESS_TIMEOUT="$SAPPO_BOUNDARY_PROGRESS_TIMEOUT" SAPPO_BOUNDARY_PROGRESS_MARGIN="$SAPPO_BOUNDARY_PROGRESS_MARGIN" \
SAPPO_BOUNDARY_WALL_DISTANCE="$SAPPO_BOUNDARY_WALL_DISTANCE" SAPPO_BOUNDARY_WALL_KP="$SAPPO_BOUNDARY_WALL_KP" \
SAPPO_BOUNDARY_WALL_HEADING_KP="$SAPPO_BOUNDARY_WALL_HEADING_KP" SAPPO_BOUNDARY_WALL_MAX_LINEAR="$SAPPO_BOUNDARY_WALL_MAX_LINEAR" \
SAPPO_BOUNDARY_WALL_MAX_RANGE="$SAPPO_BOUNDARY_WALL_MAX_RANGE" \
SAPPO_BOUNDARY_WALL_FILTER_ALPHA="$SAPPO_BOUNDARY_WALL_FILTER_ALPHA" SAPPO_BOUNDARY_WALL_ANGULAR_STEP="$SAPPO_BOUNDARY_WALL_ANGULAR_STEP" \
SAPPO_BOUNDARY_WALL_ANGULAR_DEADBAND="$SAPPO_BOUNDARY_WALL_ANGULAR_DEADBAND" \
SAPPO_BOUNDARY_GOAL_CANCEL_ANGLE_DEG="$SAPPO_BOUNDARY_GOAL_CANCEL_ANGLE_DEG" SAPPO_BOUNDARY_GOAL_CANCEL_DISTANCE="$SAPPO_BOUNDARY_GOAL_CANCEL_DISTANCE" \
SAPPO_WAYPOINT_HOLD_RADIUS="$SAPPO_WAYPOINT_HOLD_RADIUS" SAPPO_WAYPOINT_SWITCH_DISTANCE="$SAPPO_WAYPOINT_SWITCH_DISTANCE" \
SAPPO_GRID_OBSTACLE_RADIUS="$SAPPO_GRID_OBSTACLE_RADIUS" SAPPO_GRID_EDGE_CLEARANCE="$SAPPO_GRID_EDGE_CLEARANCE" \
SAPPO_REORIENT_OBSTACLE_CLEARANCE="$SAPPO_REORIENT_OBSTACLE_CLEARANCE" \
  "$WS/scripts/lifecycle/switch_controller.sh" "$LSTE_CONTROLLER"
if [[ "${LEGACY_GP_FRONTIER_ENABLED,,}" == "true" || "$LEGACY_GP_FRONTIER_ENABLED" == "1" ]]; then
  WINDOW_SUMMARY="20 个 lste 窗口（含 legacy GP、TEB 和 metrics）+ 始终运行的 lste-teleop 会话"
else
  WINDOW_SUMMARY="18 个 lste 窗口（TEB、在线 SLAM 和 metrics；legacy GP 已关闭）+ 始终运行的 lste-teleop 会话"
fi

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
