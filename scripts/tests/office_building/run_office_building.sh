#!/usr/bin/env bash
set -euo pipefail

# One-command entry point for the single complex office-building benchmark.
# It keeps all four deterministic world artifacts and never edits production
# defaults or enters tmux.

WS="${LSTE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
WORLD=""
MANIFEST="$WS/worlds/benchmark/office_building_v1_manifest.yaml"
CONFIG="$WS/scripts/tests/office_building/office_building_config.yaml"
GENERATOR="$WS/scripts/tests/office_building/generate_world.py"
VALIDATOR="$WS/scripts/tests/office_building/validate_office_building.py"
DYNAMIC="$WS/scripts/tests/office_building/dynamic_obstacle.py"
SUMMARY="$WS/scripts/tests/office_building/summarize_run.py"
PREFIX_LOGGER="$WS/scripts/tests/office_building/prefix_log.py"
LOG_ROOT="$WS/runtime/office_building_benchmark/logs"
CURRENT_RUN_LINK="$WS/runtime/office_building_benchmark/current"
LEVEL="${OFFICE_BUILDING_LEVEL:-level_2}"
PROFILE="${OFFICE_BUILDING_PROFILE:-primary}"

RUN_TIMESTAMP=""
RUN_DIR=""
LIFECYCLE_LOG=""

usage() {
  cat <<'EOF'
Usage: run_office_building.sh <command> [options]

Commands:
  generate [level]         Regenerate one deterministic level without touching others.
  validate [level]         Validate the world, manifest, and local models.
  start [level]            Stop an old LSTE run, then start the benchmark in background.
  stop                     Stop the complete LSTE system.
  status                   Show the benchmark tmux sessions and ROS nodes.
  summary [log]            Summarize a navigation metrics log (latest by default).

Levels: level_1, level_2, level_3, level_4.
The launcher does not attach to tmux. Use `tmux attach -t lste` only for diagnosis.
EOF
}

select_profile() {
  local profile="$1"
  python3 - "$MANIFEST" "$profile" <<'PY'
import sys
import yaml

manifest = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
profile = (manifest.get("robot_profiles", {}) or {}).get(sys.argv[2])
if not profile or len(profile.get("pose", [])) != 3:
    raise SystemExit("unknown robot profile: %s" % sys.argv[2])
for value in profile["pose"]:
    print(value)
PY
}

world_for_level() {
  local level="$1"
  python3 - "$MANIFEST" "$level" <<'PY'
import sys
import yaml

manifest = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
level = (manifest.get("levels", {}) or {}).get(sys.argv[2], {}) or {}
world = str(level.get("world") or "").strip()
if not world:
    raise SystemExit("manifest has no world for level: %s" % sys.argv[2])
print(world)
PY
}

validate_world() {
  local level="$1"
  python3 "$VALIDATOR" --manifest "$MANIFEST" --level "$level"
}

generate_world() {
  local level="${1:-$LEVEL}"
  WORLD="$WS/$(world_for_level "$level")"
  python3 "$GENERATOR" --output "$WORLD" --level "$level"
}

config_scalar() {
  local key="$1"
  python3 - "$CONFIG" "$key" <<'PY'
import sys
import yaml

data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
value = data.get(sys.argv[2], "")
print(value if value is not None else "")
PY
}

target_xy() {
  python3 - "$MANIFEST" <<'PY'
import sys
import yaml

data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
target = (data.get("targets", {}) or {}).get("primary", {}) or {}
xy = target.get("pose_xy", [])
if len(xy) != 2:
    raise SystemExit("manifest primary target pose_xy must contain two values")
print("%s,%s" % (xy[0], xy[1]))
PY
}

prepare_run_logs() {
  mkdir -p "$LOG_ROOT"
  while true; do
    RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="$LOG_ROOT/$RUN_TIMESTAMP"
    if mkdir "$RUN_DIR" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  LIFECYCLE_LOG="$RUN_DIR/${RUN_TIMESTAMP}_lifecycle.log"
  ln -sfn "$RUN_DIR" "$CURRENT_RUN_LINK"
  export OFFICE_BUILDING_RUN_TIMESTAMP="$RUN_TIMESTAMP"
  export OFFICE_BUILDING_LOG_DIR="$RUN_DIR"
  export NAVIGATION_LOG_DIR="$RUN_DIR"
  export NAVIGATION_RUN_DIRECTORY="$RUN_DIR"
  export NAVIGATION_RUN_TIMESTAMP="$RUN_TIMESTAMP"
}

lifecycle_log() {
  local event="$1"
  shift || true
  printf '%s [INFO] process=office_building_launcher event=%s run_timestamp=%s %s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event" "${RUN_TIMESTAMP:-unknown}" "$*" \
    >> "$LIFECYCLE_LOG"
}

prefix_captured_lines() {
  local process="$1"
  local log_file="$2"
  awk -v process="$process" '{printf "%s [INFO] process=%s event=stdout line=%s\n", strftime("%Y-%m-%dT%H:%M:%S%z"), process, $0; fflush()}' \
    >> "$log_file"
}

attach_pane_log() {
  local session="$1"
  local window="$2"
  local process="$3"
  local log_file="$RUN_DIR/${RUN_TIMESTAMP}_${process}.log"
  : > "$log_file"
  tmux capture-pane -pJ -t "=$session:$window" -S -1000 2>/dev/null \
    | prefix_captured_lines "$process" "$log_file" || true
  tmux pipe-pane -o -t "=$session:$window" \
    "python3 '$PREFIX_LOGGER' --process '$process' --output '$log_file'"
}

attach_process_logs() {
  local session window process
  while IFS='|' read -r session window process; do
    [[ -z "$session" ]] && continue
    if tmux has-session -t "=$session" 2>/dev/null && \
       tmux list-windows -t "=$session" -F '#{window_name}' 2>/dev/null | grep -qx "$window"; then
      attach_pane_log "$session" "$window" "$process"
    fi
  done <<'EOF'
lste-env|roscore|roscore
lste-env|world|gazebo_world
lste-env|gazebo_gui|gazebo_gui
lste|pro3|pro3_support
lste|task|task
lste|vllm|vllm
lste|goal|goal_manager
lste|prompt|prompt
lste|detector|detector
lste|score|score
lste|state|state
lste|vis|detector_visualization
lste|oc_srfc|oc_srfc
lste|global_frontier|global_frontier
lste|teb_nav|teb_navigation
lste|controller_switch|controller_switch
lste|cmd_vel_mux|cmd_vel_mux
lste|health|health_audit
lste|office_dynamic|dynamic_obstacle
lste|teleop|teleop
EOF
}

record_resolved_start() {
  local level="$1"
  local world_hash git_revision git_dirty target
  world_hash="$(sha256sum "$WORLD" | awk '{print $1}')"
  git_revision="$(git -C "$WS" rev-parse HEAD 2>/dev/null || echo unknown)"
  git_dirty=false
  if ! git -C "$WS" diff --quiet || ! git -C "$WS" diff --cached --quiet; then
    git_dirty=true
  fi
  target="$(target_xy)"
  lifecycle_log "run_start" \
    "scenario_id=office_building_v1 level=$level profile=$PROFILE controller=$(config_scalar LSTE_CONTROLLER) detector=$(config_scalar DETECTOR) speed=$(config_scalar TEB_MAX_LINEAR_SPEED) initial_pose=${pose[*]} target=$target task_id=yellow_cup world=$WORLD world_sha256=$world_hash git_revision=$git_revision git_dirty=$git_dirty"
}

start_benchmark() {
  local level="$1"
  generate_world "$level"
  validate_world "$level"

  mapfile -t pose < <(select_profile "$PROFILE")
  if (( ${#pose[@]} != 3 )); then
    echo "[error] invalid pose profile: $PROFILE" >&2
    exit 1
  fi

  echo "[office] stopping any previous LSTE run"
  "$WS/scripts/bin/stopall" || true

  prepare_run_logs
  record_resolved_start "$level"

  export PIPELINE_CONFIG="$CONFIG"
  export WORLD
  export TASK_JSON="$WS/model/Data_exchange/vlm_prompt/lab/yellow_cup.json"
  export TASK_ID=yellow_cup
  export PRO3_SPAWN_X="${pose[0]}"
  export PRO3_SPAWN_Y="${pose[1]}"
  export PRO3_SPAWN_Z=0.1
  export PRO3_SPAWN_YAW="${pose[2]}"
  export LSTE_NO_ATTACH=1
  export OFFICE_BUILDING_LEVEL="$level"

  echo "[office] starting scenario=office_building_v1 level=$level profile=$PROFILE"
  local launcher_log="$RUN_DIR/${RUN_TIMESTAMP}_launcher.log"
  if ! "$WS/scripts/lifecycle/run_all_tmux.sh" 2>&1 \
      | awk -v process=office_building_launcher '{printf "%s [INFO] process=%s event=stdout line=%s\\n", strftime("%Y-%m-%dT%H:%M:%S%z"), process, $0; fflush()}' \
      >> "$launcher_log"; then
    lifecycle_log "startup_failed" "level=$level profile=$PROFILE"
    attach_process_logs
    return 1
  fi

  if [[ "$level" == "level_4" ]]; then
    tmux new-window -d -t "=lste" -n "office_dynamic" -c "$WS" \
      "bash -lc 'source \"$WS/scripts/config/pipeline_env.sh\"; exec python3 \"$DYNAMIC\" --model benchmark_dynamic_obstacle --period 24 --x-low 8 --x-high 15.5 --y 11 --rate 10'"
    echo "[office] deterministic dynamic obstacle started in lste:office_dynamic"
  fi
  attach_process_logs
  lifecycle_log "process_ready" "level=$level profile=$PROFILE controller=$(config_scalar LSTE_CONTROLLER) log_dir=$RUN_DIR"
  echo "[office] benchmark started in background"
}

case "${1:-}" in
  generate)
    generate_world "${2:-$LEVEL}"
    ;;
  validate)
    validate_world "${2:-$LEVEL}"
    ;;
  start)
    start_benchmark "${2:-$LEVEL}"
    ;;
  stop)
    if [[ -L "$CURRENT_RUN_LINK" || -d "$CURRENT_RUN_LINK" ]]; then
      RUN_DIR="$(readlink -f "$CURRENT_RUN_LINK" 2>/dev/null || true)"
      if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
        RUN_TIMESTAMP="$(basename "$RUN_DIR")"
        LIFECYCLE_LOG="$RUN_DIR/${RUN_TIMESTAMP}_lifecycle.log"
        lifecycle_log "stop_requested"
      fi
    fi
    "$WS/scripts/bin/stopall"
    if [[ -n "${LIFECYCLE_LOG:-}" && -f "$LIFECYCLE_LOG" ]]; then
      lifecycle_log "stop_completed"
    fi
    ;;
  status)
    tmux list-sessions 2>/dev/null || true
    rosnode list 2>/dev/null | rg '/(lste_goal_manager|lste_global_frontier|lste_navigation_metrics|move_base)$' || true
    ;;
  summary)
    if [[ $# -ge 2 ]]; then
      exec python3 "$SUMMARY" "$2"
    fi
    exec python3 "$SUMMARY"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
