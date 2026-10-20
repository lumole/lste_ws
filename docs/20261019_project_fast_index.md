# LSTE 工程极速全景索引（自动扫描生成）
> 自动生成日期: 2026-10-19 | 生成工具: `scripts/tools/generate_project_index.py`

本文档由工程扫描程序全自动生成，汇聚了 Gazebo 模型库、仿真环境生成映射、ROS 核心包结构、数据契约与常用操作流程。Agent 与开发者可直接据此进行精准开发，**无需盲目搜索文件**。

## 目录
1. [快速操作指南 (SOP)](#1-快速操作指南-sop)
2. [仿真环境与生成器映射](#2-仿真环境与生成器映射)
3. [Gazebo 核心模型资产大字典](#3-gazebo-核心模型资产大字典)
4. [ROS 核心工作区与包架构](#4-ros-核心工作区与包架构)
5. [核心 ROS 话题契约 (AST 提取)](#5-核心-ros-话题契约-ast-提取)

---

## 1. 快速操作指南 (SOP)

### 1.1 仿真与导航栈启动
```bash
# 一键启动/重启 TEB 自主导航全栈 (默认 T-junction 世界)
./run_demo_teb.sh restart

# 指定自定义世界与出生点启动
WORLD="/home/dh/dh_ws/lste_ws/worlds/t_junction_office_no_pro3.world" \
SPAWN_X=6.0 SPAWN_Y=1.0 SPAWN_YAW=1.5708 DISPLAY=:0 ./run_demo_teb.sh restart

# 停止全栈
./run_demo_teb.sh stop
```

### 1.2 控制器模式切换
```bash
# 切换为 TEB 拓扑自主规划驱动
rostopic pub -1 /lste/controller_mode std_msgs/String "data: 'teb'"

# 切换为键盘/遥控 Teleop 手动驱动
rostopic pub -1 /lste/controller_mode std_msgs/String "data: 'teleop'"
```

### 1.3 实时状态诊断 (免 Grep 专用工具)
```bash
# 查看当前活跃导航目标 (WorkItem ID、坐标)、Place 进度与租约
python3 .agents/skills/lste-status-debug/scripts/inspect_demo_status.py
```

### 1.4 SDF 世界文件语法校验
```bash
gz sdf -k <path_to_world_file>
```

---

## 2. 仿真环境与生成器映射

| 仿真世界 (.world) | 源头生成脚本 (Generator) | 规范清单 (Manifest) | 包含模型数 |
|---|---|---|:---:|
| [`worlds/benchmark/office_building_v1_level_1_no_pro3.world`](worlds/benchmark/office_building_v1_level_1_no_pro3.world) | *(静态手写 / 外部导入)* | `worlds/benchmark/office_building_v1_manifest.yaml` | 4 |
| [`worlds/benchmark/office_building_v1_level_2_no_pro3.world`](worlds/benchmark/office_building_v1_level_2_no_pro3.world) | `scripts/tests/office_building/generate_world.py` | `worlds/benchmark/office_building_v1_manifest.yaml` | 43 |
| [`worlds/benchmark/office_building_v1_level_3_no_pro3.world`](worlds/benchmark/office_building_v1_level_3_no_pro3.world) | *(静态手写 / 外部导入)* | `worlds/benchmark/office_building_v1_manifest.yaml` | 46 |
| [`worlds/benchmark/office_building_v1_level_4_no_pro3.world`](worlds/benchmark/office_building_v1_level_4_no_pro3.world) | *(静态手写 / 外部导入)* | `worlds/benchmark/office_building_v1_manifest.yaml` | 51 |
| [`worlds/new.world`](worlds/new.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/t_junction_office_no_pro3.world`](worlds/t_junction_office_no_pro3.world) | `scripts/tests/generate_t_junction_world.py` | `worlds/t_junction_office_manifest.yaml` | 11 |
| [`worlds/targeted_navigation/portal_retry_single_door.world`](worlds/targeted_navigation/portal_retry_single_door.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/targeted_navigation/t_junction_small.world`](worlds/targeted_navigation/t_junction_small.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo1.0_start/ding_road.world`](worlds/topo1.0_start/ding_road.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo1.0_start/guai_road.world`](worlds/topo1.0_start/guai_road.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo1.0_start/labtest.world`](worlds/topo1.0_start/labtest.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo1.0_start/topo_test2.world`](worlds/topo1.0_start/topo_test2.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo2.0_explore_mode/place1.world`](worlds/topo2.0_explore_mode/place1.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo2.0_explore_mode/place2.world`](worlds/topo2.0_explore_mode/place2.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo2.0_explore_mode/place3.world`](worlds/topo2.0_explore_mode/place3.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo2.0_explore_mode/place4.world`](worlds/topo2.0_explore_mode/place4.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo3.0_catch_mode/catch01.world`](worlds/topo3.0_catch_mode/catch01.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo3.0_catch_mode/session_test/env01.world`](worlds/topo3.0_catch_mode/session_test/env01.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo3.0_catch_mode/session_test/env02.world`](worlds/topo3.0_catch_mode/session_test/env02.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo3.0_catch_mode/session_test/env03.world`](worlds/topo3.0_catch_mode/session_test/env03.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo3.0_catch_mode/session_test/env04.world`](worlds/topo3.0_catch_mode/session_test/env04.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo3.0_catch_mode/session_test/env04_no_pro3.world`](worlds/topo3.0_catch_mode/session_test/env04_no_pro3.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo3.0_catch_mode/session_test/t_junction_office_no_pro3.world`](worlds/topo3.0_catch_mode/session_test/t_junction_office_no_pro3.world) | `scripts/tests/generate_t_junction_world.py` | `worlds/t_junction_office_manifest.yaml` | 11 |
| [`worlds/topo3.0_catch_mode/session_test/test3_0.world`](worlds/topo3.0_catch_mode/session_test/test3_0.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo_test/lab_building.world`](worlds/topo_test/lab_building.world) | *(静态手写 / 外部导入)* | — | 1 |
| [`worlds/topo_test/room.world`](worlds/topo_test/room.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo_test/test.world`](worlds/topo_test/test.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo_test/topo_test.world`](worlds/topo_test/topo_test.world) | *(静态手写 / 外部导入)* | — | 0 |
| [`worlds/topo_test/yellowcup.world`](worlds/topo_test/yellowcup.world) | *(静态手写 / 外部导入)* | — | 1 |

> **修改世界黄金准则**：若该世界存在对应的源头生成脚本（如 `generate_t_junction_world.py`），**务必修改生成脚本并重新运行**，严禁直接手改生成产物，避免后续被覆盖！

---

## 3. Gazebo 核心模型资产大字典

以下为工作区代码与世界文件中实际引用的 Gazebo 模型资产实测参数：

| 模型名称 (URI) | 类别 | 静态 | 典型地面/桌面高度 (Z) | 默认缩放 (Scale) | 出现频次 | 典型用途 |
|---|---|:---:|---|---|:---:|---|
| `model://kitchen_chair` | Chairs & Seating | 是 | `0.00m` | `1 1 1` | 6 | kitchen_chair |
| `model://office_chair` | Chairs & Seating | 是 | `0.00m` | `1 1 1` | 30 | office_chair |
| `model://sofa_set_1` | Chairs & Seating | 是 | `0.00m` | `1 1 1` | 3 | sofa_set_1 |
| `model://sofa_set_2` | Chairs & Seating | 是 | `0.00m` | `1 1 1` | 3 | sofa_set_2 |
| `model://cafe_table` | Desks & Tables | 是 | `0.00m` | `1 1 1` | 6 | cafe_table |
| `model://coffee_table_1` | Desks & Tables | 是 | `0.00m` | `1 1 1` | 3 | coffee_table_1 |
| `model://desk_brown` | Desks & Tables | 是 | `0.00m` | `1 1 1` | 24 | desk_brown |
| `model://table_conference_2` | Desks & Tables | 是 | `0.00m` | `1 1 1` | 5 | table_conference_2 |
| `model://computer_case` | Electronics & Office | 是 | `0.00m` | `1 1 1` | 0 | computer_case |
| `model://file_1` | Electronics & Office | 是 | `0.00m` | `1 1 1` | 0 | file_1 |
| `model://file_2` | Electronics & Office | 是 | `0.00m` | `1 1 1` | 0 | file_2 |
| `model://keyboard` | Electronics & Office | 是 | `0.00m` | `1 1 1` | 0 | keyboard |
| `model://monitor_1` | Electronics & Office | 是 | `0.72m` | `1 1 1` | 12 | monitor_1 |
| `model://monitor_2` | Electronics & Office | 是 | `0.72m` | `1 1 1` | 15 | monitor_2 |
| `model://monitor_3` | Electronics & Office | 是 | `0.72m` | `1 1 1` | 3 | monitor_3 |
| `model://monitor_4` | Electronics & Office | 是 | `0.00m` | `1 1 1` | 0 | monitor_4 |
| `model://mouse` | Electronics & Office | 是 | `0.00m` | `1 1 1` | 0 | mouse |
| `model://door_3d` | Fixtures & Safety | 是 | `0.00m` | `1 1 1` | 0 | door_3d |
| `model://fire_hydrant` | Fixtures & Safety | 是 | `0.00m` | `1 1 1` | 6 | fire_hydrant |
| `model://indoor` | Fixtures & Safety | 是 | `0.00m` | `1 1 1` | 0 | indoor |
| `model://Floor` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 0 | Floor |
| `model://cafe` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 0 | cafe |
| `model://ground_plane` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 2 | ground_plane |
| `model://office_small` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 0 | office_small |
| `model://picture_frame_large_6` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 0 | picture_frame_large_6 |
| `model://plant_1_single` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 1 | plant_1_single |
| `model://sun` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 2 | sun |
| `model://tools` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 0 | tools |
| `model://turtlebot3_autorace` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 0 | turtlebot3_autorace |
| `model://turtlebot3_autorace_2020` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 0 | turtlebot3_autorace_2020 |
| `model://turtlebot3_plaza` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 0 | turtlebot3_plaza |
| `model://turtlebot3_square` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 0 | turtlebot3_square |
| `model://turtlebot3_world` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 0 | turtlebot3_world |
| `model://utility_cart` | Miscellaneous | 是 | `0.00m` | `1 1 1` | 1 | utility_cart |
| `model://bookshelf` | Storage & Cabinets | 是 | `0.00m` | `1 1 1` | 17 | bookshelf |
| `model://cabinet` | Storage & Cabinets | 是 | `0.00m` | `1 1 1` | 1 | cabinet |
| `model://cardboard_box` | Storage & Cabinets | 是 | `0.00m` | `1 1 1` | 2 | cardboard_box |
| `model://file_box_1` | Storage & Cabinets | 是 | `0.00m` | `1 1 1` | 0 | file_box_1 |
| `model://file_box_2` | Storage & Cabinets | 是 | `0.00m` | `1 1 1` | 0 | file_box_2 |
| `model://file_box_3` | Storage & Cabinets | 是 | `0.00m` | `1 1 1` | 0 | file_box_3 |
| `model://file_box_4` | Storage & Cabinets | 是 | `0.00m` | `1 1 1` | 0 | file_box_4 |
| `model://file_cabinet_large` | Storage & Cabinets | 是 | `0.00m` | `1 1 1` | 8 | file_cabinet_large |
| `model://campus` | Structures & Outdoor | 是 | `0.00m` | `1 1 1` | 0 | campus |
| `model://forest` | Structures & Outdoor | 是 | `0.00m` | `1 1 1` | 0 | forest |
| `model://garage` | Structures & Outdoor | 是 | `0.00m` | `1 1 1` | 0 | garage |
| `model://house_1` | Structures & Outdoor | 是 | `0.00m` | `1 1 1` | 0 | house_1 |
| `model://house_2` | Structures & Outdoor | 是 | `0.00m` | `1 1 1` | 0 | house_2 |
| `model://lab_building` | Structures & Outdoor | 是 | `0.00m` | `1 1 1` | 1 | lab_building |
| `model://tunnel` | Structures & Outdoor | 是 | `0.00m` | `1 1 1` | 0 | tunnel |
| `model://turtlebot3_house` | Structures & Outdoor | 是 | `0.00m` | `1 1 1` | 0 | turtlebot3_house |
| `model://willowgarage` | Structures & Outdoor | 是 | `0.00m` | `1 1 1` | 0 | willowgarage |
| `model://cup_blue` | Target / Cup | 是 | `0.95m, 1.02m` | `1 1 1` | 9 | cup_blue |
| `model://cup_green` | Target / Cup | 是 | `1.02m` | `1 1 1` | 3 | cup_green |
| `model://cup_paper` | Target / Cup | 是 | `0.95m` | `1 1 1` | 1 | cup_paper |
| `model://cup_yellow` | Target / Cup | 是 | `0.75m, 1.02m` | `1.500 1.500 1.500, 1.800 1.800 1.800` | 6 | cup_yellow |
| `model://plastic_cup` | Target / Cup | 是 | `0.95m` | `1 1 1` | 1 | plastic_cup |
| `model://yellowcup` | Target / Cup | 是 | `0.00m` | `1 1 1` | 1 | yellowcup |

> **高度经验值备忘**：
> - `desk_brown`（办公桌）表面高度约 **0.72m**，放置在其上的物品（如 `monitor_1`, `monitor_2`）设为 `z=0.72`；黄色目标杯 `cup_yellow` 设为 `z=0.75`。
> - `table_conference_2`（研讨长桌）表面高度约 **0.75m**。
> - 地面摆放物品（`fire_hydrant`, `bookshelf`, `file_cabinet_large` 等）一律设为 `z=0.00`。

---

## 4. ROS 核心工作区与包架构

| Package 名称 | 相对路径 | 启动文件 (Launch) | 脚本节点 (Nodes) | 功能描述 |
|---|---|:---:|:---:|---|
| `autolabor_description` | `src/tools/autolabor_description` | 2 个 | — | autolabor_description provides a complete 3D model... |
| `gazebo_click_point` | `src/gazebo_click_point` | — | — | Gazebo Classic GUI plugin that publishes left-clic... |
| `kinect_description` | `src/tools/kinect_description` | 1 个 | — | Description of Kinect V2 sensor on a stand.       ... |
| `lste_control` | `src/lste_control` | — | — | The lste_control package |
| `lste_core` | `src/lste_core` | 4 个 | 14 个 | The lste_core package |
| `lste_msgs` | `src/lste_msgs` | — | — | The lste_msgs package |
| `lste_oc_srfc` | `src/tools/lste_oc_srfc` | 1 个 | — | The lste_oc_srfc package |
| `lste_topo_access` | `src/lste_topo_access` | 3 个 | 267 个 | The lste_topo_access package |
| `lste_topo_env` | `src/lste_topo_env` | — | — | The lste_topo_env package |
| `ouster_description` | `src/tools/ouster_description` | 1 个 | — | URDF and meshes describing Ouster laser scanners. |
| `pja` | `src/tools/pja` | 97 个 | — | Local Planner |
| `pointcloud_to_laserscan` | `src/tools/pointcloud_to_laserscan` | 8 个 | — | Converts a 3D Point Cloud into a 2D laser scan. Th... |
| `vanish_point_detection` | `src/tools/vanish_point_detection` | — | 1 个 | The vanish_point_detection package |
| `velodyne_description` | `src/tools/velodyne_description` | 1 个 | — | URDF and meshes describing Velodyne laser scanners... |


## 5. 核心 ROS 话题契约 (AST 提取)

### 5.1 发布话题 (Publishers)
| 话题名称 | 发布节点源码文件 |
|---|---|
| `/cmd_vel` | `lste_cmd_vel_mux_node.py` |
| `/lste/access_topo/active_mode` | `goal_manager_ros_interfaces.py` |
| `/lste/access_topo/backtrack_goal` | `gp_frontier_bootstrap.py` |
| `/lste/access_topo/mode` | `gp_frontier_bootstrap.py` |
| `/lste/controller_mode` | `lste_controller_switch_node.py` |
| `/lste/detections` | `detection_node_common.py`, `lste_det_fake_node.py` |
| `/lste/experiment/hard_reset_ack` | `lste_cmd_vel_mux_node.py` |
| `/lste/final_goal` | `goal_manager_ros_interfaces.py` |
| `/lste/goal_diagnostic` | `goal_manager_ros_interfaces.py` |
| `/lste/gp_frontier` | `gp_frontier_bootstrap.py` |
| `/lste/gp_frontier_dir` | `gp_frontier_bootstrap.py` |
| `/lste/gp_frontiers` | `gp_frontier_bootstrap.py` |
| `/lste/perception_decision` | `detection_node_common.py` |
| `/lste/prompts` | `lste_prompt_node.py` |
| `/lste/scores` | `lste_score_node.py` |
| `/lste/state` | `lste_state_node.py` |
| `/lste/task` | `lste_task_node.py` |
| `/lste/task_done` | `goal_manager_ros_interfaces.py` |
| `/vanish_point` | `vanish_point_detection.py` |
| `change_flag` | `gp_frontier_bootstrap.py` |
| `closed_markers` | `gp_frontier_bootstrap.py` |
| `gp_nav_actul_xy_gls` | `gp_frontier_bootstrap.py` |
| `gp_nav_oc` | `gp_frontier_bootstrap.py` |
| `gp_nav_pts` | `gp_frontier_bootstrap.py` |
| `gp_nav_var` | `gp_frontier_bootstrap.py` |
| `gp_subgl_rviz` | `gp_frontier_bootstrap.py` |
| `gp_subgoal` | `gp_frontier_bootstrap.py` |
| `gp_subgoal_os` | `gp_frontier_bootstrap.py` |
| `interest_subgoal` | `gp_frontier_bootstrap.py` |

### 5.2 订阅话题 (Subscribers)
| 话题名称 | 订阅节点源码文件 |
|---|---|
| `/cmd_vel` | `debug_goal_vote.py`, `goal_manager_ros_interfaces.py`, `navigation_metrics_ros_interfaces.py` |
| `/gazebo/model_states` | `navigation_metrics_ros_interfaces.py` |
| `/kinect/hd/camera_info` | `goal_manager_ros_interfaces.py` |
| `/kinect/hd/image_color_rect` | `vanish_point_detection.py` |
| `/lste/access_topo/active_mode` | `gp_frontier_bootstrap.py` |
| `/lste/access_topo/backtrack_goal` | `goal_manager_ros_interfaces.py` |
| `/lste/access_topo/mode` | `goal_manager_ros_interfaces.py` |
| `/lste/benchmark/contact_states` | `navigation_metrics_ros_interfaces.py` |
| `/lste/cmd_vel/sappo` | `lste_cmd_vel_mux_node.py` |
| `/lste/cmd_vel/teb` | `lste_cmd_vel_mux_node.py`, `navigation_metrics_ros_interfaces.py` |
| `/lste/cmd_vel/teb_planner` | `navigation_metrics_ros_interfaces.py` |
| `/lste/cmd_vel/teleop` | `lste_cmd_vel_mux_node.py` |
| `/lste/cmd_vel_mux/status` | `navigation_metrics_ros_interfaces.py` |
| `/lste/controller_mode` | `lste_cmd_vel_mux_node.py`, `navigation_metrics_ros_interfaces.py` |
| `/lste/controller_select` | `lste_controller_switch_node.py` |
| `/lste/controller_toggle` | `lste_controller_switch_node.py` |
| `/lste/detections` | `goal_manager_ros_interfaces.py`, `lste_score_node.py`, `lste_state_node.py` |
| `/lste/experiment/hard_reset` | `lste_cmd_vel_mux_node.py` |
| `/lste/experiment/hard_reset_release` | `lste_cmd_vel_mux_node.py` |
| `/lste/experiment/trial_boundary` | `navigation_metrics_ros_interfaces.py` |
| `/lste/final_goal` | `debug_goal_vote.py`, `gp_frontier_bootstrap.py`, `lste_det_vis_node.py`, `navigation_metrics_ros_interfaces.py` |
| `/lste/global_frontier/status` | `navigation_metrics_ros_interfaces.py` |
| `/lste/goal_arbitration` | `navigation_metrics_ros_interfaces.py` |
| `/lste/goal_diagnostic` | `navigation_metrics_ros_interfaces.py` |
| `/lste/gp_frontiers` | `debug_goal_vote.py` |
| `/lste/mission_goal` | `navigation_metrics_ros_interfaces.py` |
| `/lste/navigation_hold` | `navigation_metrics_ros_interfaces.py` |
| `/lste/pause` | `lste_cmd_vel_mux_node.py` |
| `/lste/perception_decision` | `navigation_metrics_ros_interfaces.py` |
| `/lste/persistent_execution/plan_event` | `navigation_metrics_ros_interfaces.py` |
| `/lste/persistent_execution/planner_command_contract` | `navigation_metrics_ros_interfaces.py` |
| `/lste/prompts` | `detection_node_common.py` |
| `/lste/sappo_controller_status` | `navigation_metrics_ros_interfaces.py` |
| `/lste/scores` | `goal_manager_ros_interfaces.py`, `lste_state_node.py`, `navigation_metrics_ros_interfaces.py` |
| `/lste/state` | `detection_node_common.py`, `goal_manager_ros_interfaces.py`, `gp_frontier_bootstrap.py`, `navigation_metrics_ros_interfaces.py` |
| `/lste/task` | `detection_node_common.py`, `goal_manager_ros_interfaces.py`, `lste_det_fake_node.py`, `lste_det_vis_node.py` |
| `/lste/task_done` | `lste_cmd_vel_mux_node.py`, `lste_det_vis_node.py`, `navigation_metrics_ros_interfaces.py` |
| `/lste/teb_goal_bridge/status` | `navigation_metrics_ros_interfaces.py` |
| `/lste/teb_goal_failure` | `navigation_metrics_ros_interfaces.py` |
| `/lste/teb_goal_terminal` | `navigation_metrics_ros_interfaces.py` |
| `/lste/teb_goal_terminal_contract` | `navigation_metrics_ros_interfaces.py` |
| `/lste/teb_turn_supervisor/status` | `navigation_metrics_ros_interfaces.py` |
| `/map` | `goal_manager_ros_interfaces.py`, `navigation_metrics_ros_interfaces.py` |
| `/move_base/GlobalPlanner/plan` | `navigation_metrics_ros_interfaces.py` |
| `/move_base/NavfnROS/plan` | `navigation_metrics_ros_interfaces.py` |
| `/move_base/TebLocalPlannerROS/global_plan` | `navigation_metrics_ros_interfaces.py` |
| `/move_base/TebLocalPlannerROS/local_plan` | `navigation_metrics_ros_interfaces.py` |
| `/move_base/TebLocalPlannerROS/teb_feedback` | `lste_cmd_vel_mux_node.py`, `navigation_metrics_ros_interfaces.py`, `teb_turn_supervisor_ros.py` |
| `/move_base/feedback` | `navigation_metrics_ros_interfaces.py` |
| `/move_base/global_costmap/costmap` | `navigation_metrics_ros_interfaces.py` |
| `/move_base/goal` | `navigation_metrics_ros_interfaces.py` |
| `/move_base/local_costmap/costmap` | `navigation_metrics_ros_interfaces.py` |
| `/move_base/recovery_status` | `navigation_metrics_ros_interfaces.py` |
| `/move_base/status` | `navigation_metrics_ros_interfaces.py` |
| `/move_base_simple/goal` | `navigation_metrics_ros_interfaces.py` |
| `/pro3/rlscan` | `navigation_metrics_ros_interfaces.py` |
| `/pro3/wheel_odom` | `navigation_metrics_ros_interfaces.py` |
| `/rbt_pose` | `debug_goal_vote.py`, `goal_manager_ros_interfaces.py`, `lste_det_vis_node.py`, `navigation_metrics_ros_interfaces.py` |
| `/vanish_point` | `gp_frontier_bootstrap.py` |
| `change_flag` | `gp_frontier_bootstrap.py` |
| `rbt_pose` | `gp_frontier_bootstrap.py` |
| `sph_pcl` | `gp_frontier_bootstrap.py` |

