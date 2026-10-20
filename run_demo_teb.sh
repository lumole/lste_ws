#!/usr/bin/env bash
# One-click launcher for the no_pro3 Place/Portal demo with direct TEB autonomous drive.
#
# Usage:
#   ./run_demo_teb.sh            # start directly with TEB control (clean restart if already running)
#   ./run_demo_teb.sh start      # start
#   ./run_demo_teb.sh stop       # stop everything
#   ./run_demo_teb.sh restart    # restart
#   ./run_demo_teb.sh status     # show stack status
#
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_SCRIPT="$WS/temp/manual_teleop/demo.sh"

export CONTROLLER_MODE="teb"

ACTION="${1:-restart}"

echo "[run_demo_teb] Controller mode set to: $CONTROLLER_MODE"
echo "[run_demo_teb] Executing action: $ACTION"

exec "$DEMO_SCRIPT" "$ACTION" "${@:2}"
