---

# 🤖 LSTE Workspace README

## 🚀 Quick Start

> 仿真 world 文件已迁移到工作区根目录的 `worlds/` 目录，命令里统一用相对路径（如 `worlds/place1.world`）。

```bash
# 清理旧的 session (如果存在)
tmux kill-session -t lste 2>/dev/null || true
cd /home/zrz/lste_ws
./scripts/run_pipeline_tmux.sh
```

---
## 🛠️ Tools & Debugging

辅助工具与调试指令。

### 键盘控制与话题监控

```bash
# 操控车来移动（键盘操控）
rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/cmd_vel

# 强制给直行
rostopic pub -r 10 /cmd_vel geometry_msgs/Twist \
'{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'

# rviz看frontier可视化
rviz -d /home/zrz/lste_ws/src/lste_topo_access/launch/gp_frontier.rviz

# 能看到发布的 task
rostopic echo /lste/task

# 能看到发布的 prompts
rostopic echo /lste/prompts

# 可视化拓扑
TEST_NAME=topo_3.0_catch_mode bash /home/zrz/lste_ws/src/lste_topo_access/topo_tree/tools/visualize_access_topo.sh
```

### 消失点检测

**注意环境切换：** `vsgp`

```bash
conda activate vsgp
rosrun vanish_point_detection vanish_point_detection.py

```

---

## 🧬 Pipeline Steps (Manual Launch)

如果需要分步调试或手动运行，请按照以下顺序在不同的终端窗口中执行。

### 1. 启动仿真环境与任务节点

```bash
# 启动基础环境
roslaunch lste_core lab_with_pro3.launch \
  world_name:=/home/zrz/lste_ws/worlds/topo2.0_explore_mode/place3.world \
  spawn_pro3:=false

  
# 启动任务节点
rosrun lste_core lste_task_node.py _json_path:=/home/zrz/lste_ws/model/Data_exchange/vlm_prompt/lab/yellow_cup.json _task_id:=yellow_cup

```

### 2. 启动 VLM (MiniCPM)

**注意环境切换：** `minicpm`

```bash
conda activate minicpm
cd ~/lste_ws/model/MiniCPM/test/
bash start.sh 

```

### 3. 启动 Prompt 节点

**注意环境切换：** `minicpm`

```bash
conda activate minicpm
rosparam set /lste_prompt_node/vllm_stop_command "pkill -f 'vllm serve'"

rosrun lste_core lste_prompt_node.py \
  _vllm_base_url:=http://localhost:8000/v1 \
  _vllm_model_name:=/home/zrz/lste_ws/model/MiniCPM/OpenBMB/MiniCPM4-0___5B

```

### 4. 启动检测与评分 (Dino & Scoring)

**注意环境切换：** `dino`

```bash
conda activate dino
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

# 目标检测节点
rosrun lste_core lste_det_node.py

# 启动打分节点
rosrun lste_core lste_score_node.py \
    _w_target:=0.2 _w_env:=0.3 _w_ctx:=0.5 \
    _lambda_neg:=0.7 _pos_midpoint:=0.25 _pos_steepness:=6 \
    _neg_midpoint:=0.15 _neg_steepness:=12

```

### 5. 状态机与可视化

```bash
# 状态管理节点
rosrun lste_core lste_state_node.py

# 可视化 Launch
roslaunch lste_core lste_det_vis.launch

```

---
## 🗺️ Access Topo (Topological Exploration)

拓扑访问与探索相关的模块。

```bash
# 1. 启动仿真环境
roslaunch lste_core lab_with_pro3.launch \
  world_name:=worlds/room.world

# 2. 启动 GP Frontier (注意环境: vsgp)
conda activate vsgp
roslaunch lste_topo_access gp_frontier.launch

# 3. 启动表面投影
roslaunch lste_oc_srfc oc_srfc_proj.launch

# 4. 启动 Rviz 可视化
rviz -d /home/zrz/lste_ws/src/lste_topo_access/launch/gp_frontier.rviz

```
