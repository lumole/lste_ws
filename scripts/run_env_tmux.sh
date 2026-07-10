#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# ============================================
# 环境启动脚本：只启动 roscore + Gazebo 仿真
# 仿真加载慢且几乎不变，单独跑，可以长期保持不动
# ============================================

# ---- 路径解析（与 run_nodes_tmux.sh 完全一致） ----
if [ -z "${WS:-}" ]; then
  WS="${LSTE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
LSTE_WS="$WS"
SESSION=lste-env
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

WORLD=${WORLD:-$(_resolve "${CFG_WORLD:-worlds/place1.world}")}
GUI=${GUI:-${CFG_GUI:-true}}

if [[ ! -f "$WORLD" ]]; then
  echo "[error] WORLD file not found: $WORLD" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux 未安装，请先安装 tmux。" >&2
  exit 1
fi

# ---- Session 管理 ----
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  EXISTING_WORLD="$(tmux show-environment -t "=$SESSION" 2>/dev/null | awk -F= '/^LSTE_WORLD=/{print substr($0,length("LSTE_WORLD=")+1)}')"
  if [[ -n "$EXISTING_WORLD" && "$EXISTING_WORLD" != "$WORLD" ]]; then
    echo "[info] WORLD changed ($EXISTING_WORLD → $WORLD), restarting session."
    tmux kill-session -t "=$SESSION"
  else
    echo "[info] Session '$SESSION' 已存在 (WORLD=$WORLD), 直接附着。" >&2
    echo "  提示：按 Ctrl+B 再按 D 可分离 tmux，Ctrl+C 退出。"
    tmux attach -t "=$SESSION"; exit 0
  fi
fi

# ---- 新建 session ----
tmux new-session -d -s "$SESSION" -c "$WS" -n "roscore" \
  "bash -lc 'WS=\"$WS\"; source \"$WS/scripts/pipeline_env.sh\"; roscore'; echo; echo '[EXIT] roscore'; exec bash"
tmux set-environment -t "=$SESSION" LSTE_WORLD "$WORLD"
tmux set-option -t "=$SESSION:" remain-on-exit on
echo "[env] roscore started (window 0)"

WAIT_ROSCORE='until rostopic list >/dev/null 2>&1; do echo \"waiting for roscore...\"; sleep 1; done'

# Window 1: Gazebo 场景（不含 pro3，用 _no_pro3.world）
tmux new-window -t "=$SESSION:1" -n "world" -c "$WS" \
  "bash -lc 'WS=\"$WS\"; source \"$WS/scripts/pipeline_env.sh\"; \
$WAIT_ROSCORE; \
roslaunch lste_core lab_with_pro3.launch world_name:=$WORLD spawn_pro3:=false gui:=$GUI; \
echo \"[world] roslaunch exited, sleeping to keep window alive...\"; sleep 86400'; \
echo; echo '[EXIT] world'; exec bash"
echo "[env] Gazebo world started (window 1)"

echo ""
echo "============================================"
echo "  环境已启动（场景无小车）。运行以下命令："
echo "  ./scripts/run_nodes_tmux.sh  （启动小车 + 节点）"
echo "============================================"

if [[ -t 0 ]]; then
  tmux attach -t "=$SESSION"
fi
