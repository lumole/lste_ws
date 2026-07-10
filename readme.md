# 🤖 LSTE Workspace README

## 🚀 Quick Start

> 仿真 world 文件已迁移到工作区根目录的 `worlds/` 目录，命令里统一用相对路径（如 `worlds/place1.world`）。

```bash
# 设置工作区路径（或让脚本自动检测）
export LSTE_WS=$(pwd)   # 在 lste_ws 目录下执行

# 清理旧的 session (如果存在)
tmux kill-session -t lste 2>/dev/null || true
./scripts/run_pipeline_tmux.sh
```

---

## 📦 Environment Setup

本项目的 Python 环境以**系统 Python 3.8 + apt 包**为基底，conda 仅在特定节点（vsgp / dino / minicpm）中按需激活。以下记录完整的环境依赖及常见问题。

### 系统级 apt 包

```bash
sudo apt install -y python3-opencv python3-pykdl
```

### Gazebo 模型 mesh 路径

world 文件中的 mesh 相对路径多了一层 `..`（`../../../../` 应为 `../../../`），导致 Gazebo 找不到 mesh 文件，机器人不可见。已修正：

```bash
sed -i 's|../../../../src/tools/|../../../src/tools/|g' worlds/*/session_test/*.world
```

### 系统 Python (pip) 基础包

> 注意：如果本地安装了 miniconda/anaconda，确保 base 环境能访问系统 `/usr/lib/python3/dist-packages`，否则 apt 安装的包无法导入。

```bash
pip install numpy PyYAML rospkg matplotlib scipy
```

如果 `pip install opencv-python` 覆盖了 apt 版，卸载它即可回退：

```bash
pip uninstall -y opencv-python   # 使用 apt 版的 python3-opencv (4.2.0)
```

### Conda 环境

#### vsgp — GP Frontier (高斯过程探索)

| 包 | 版本 | 说明 |
|---|---|---|
| python | 3.7-3.9 | |
| numpy | 1.21-1.23 | 需保留 `np.float`，**不能 ≥1.24** |
| tensorflow | 2.11.0 | |
| tensorflow-probability | 0.19.0 | 必须匹配 TF 版本 |
| opencv-python | 4.8.x | |
| scipy | 1.7.x | |
| PyYAML | ✅ | `ros_numpy` 的 genpy 依赖 |
| rospkg | ✅ | ROS 包管理 |

```bash
conda create -n vsgp python=3.9 -y
conda activate vsgp
pip install numpy==1.23.5 tensorflow==2.11.0 tensorflow-probability==0.19.0
pip install opencv-python==4.8.1.78 scipy==1.7.3 PyYAML rospkg
```

#### dino — 目标检测 (Grounding DINO)

```bash
conda create -n dino python=3.9 -y
conda activate dino
pip install torch torchvision
# Grounding DINO 安装见其官方仓库
```

#### minicpm — VLM 推理 (MiniCPM)

```bash
conda create -n minicpm python=3.10 -y
# 安装 MiniCPM 依赖，见 model/MiniCPM 目录下的 requirements
```

### LibFFI 冲突修复

conda 自带的 `libffi.so.8` 与系统 `libffi.so.7` 冲突，会导致 apt 版 opencv / PyKDL 导入失败：
```
undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
```

已在 `scripts/pipeline_env.sh` 中通过 `LD_PRELOAD` 修复；手动运行时如果遇到同样问题：

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
```

### 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `No module named 'yaml'` | vsgp 环境缺 PyYAML | `pip install PyYAML` |
| `No module named 'cv2'` | base 环境缺 opencv | `sudo apt install python3-opencv` |
| `No module named 'PyKDL'` | 系统包未安装或 conda 屏蔽 | `sudo apt install python3-pykdl`，确保 `.pth` 引入 `/usr/lib/python3/dist-packages` |
| `No module named 'rospkg'` | vsgp 环境缺 rospkg | `pip install rospkg` |
| `module 'numpy' has no attribute 'float'` | numpy ≥ 1.24 移除了 np.float | 以 conda 创建独立 env 并降级到 `numpy<1.24` |
| `This version of TFP requires TensorFlow >= 2.18` | tensorflow-probability 与 TF 版本不匹配 | 对齐 origin 机器版本：TF 2.11 + TFP 0.19 |
| `libffi.so: undefined symbol` | conda libffi 与系统库冲突 | `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7` |
| 窗口 4 (goal) 被顶掉 | gp_frontier.launch 自带同名节点 | 正常现象，不影响功能 |

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
rviz -d src/lste_topo_access/launch/gp_frontier.rviz

# 能看到发布的 task
rostopic echo /lste/task

# 能看到发布的 prompts
rostopic echo /lste/prompts

# 可视化拓扑
TEST_NAME=topo_3.0_catch_mode bash src/lste_topo_access/topo_tree/tools/visualize_access_topo.sh
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
以下路径均相对于 `lste_ws` 工作区根目录，或使用 `$LSTE_WS` 环境变量。

### 1. 启动仿真环境与任务节点

```bash
# 启动基础环境
roslaunch lste_core lab_with_pro3.launch \
  world_name:=worlds/topo3.0_catch_mode/session_test/test3_0.world \
  spawn_pro3:=false

  
# 启动任务节点
rosrun lste_core lste_task_node.py _json_path:=model/Data_exchange/vlm_prompt/lab/yellow_cup.json _task_id:=yellow_cup

```

### 2. 启动 VLM (MiniCPM)

**注意环境切换：** `minicpm`

```bash
conda activate minicpm
cd model/MiniCPM/test/
bash start.sh 

```

### 3. 启动 Prompt 节点

**注意环境切换：** `minicpm`

```bash
conda activate minicpm
rosparam set /lste_prompt_node/vllm_stop_command "pkill -f 'vllm serve'"

rosrun lste_core lste_prompt_node.py \
  _vllm_base_url:=http://localhost:8000/v1 \
  _vllm_model_name:=model/MiniCPM/OpenBMB/MiniCPM4-0___5B

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
rviz -d src/lste_topo_access/launch/gp_frontier.rviz

```
