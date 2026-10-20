# Place / Portal / WorkItem 的可视化方案

## 1. 文档信息

- 日期：2026-10-16
- 工作区：`temp/manual_teleop/`（两房间演示 demo）
- 目标：记录"如何把 Place / DirectedPortal / ObservationWorkItem 画出来"的既有方案，作为下一步"可视化状态机"的参照基线。
- 入口：`temp/manual_teleop/place_portal_markers.py`（唯一生产可视化节点）。

## 2. 一句话结论

不新造数据源，也不在节点里重复实现几何。节点只订阅三样东西 —— 主管状态
JSON、SLAM 地图、激光 —— 然后把"可持久图"投影成两个输出：**3D 场景里的短 ID
标记**（`/lste/demo/place_portal_markers`）和**图例列图**（`/lste/demo/legend_image`）。
地图上永远不画句子，只画短 ID 和形状。

## 3. 数据来源（读，不写）

| 话题 | 类型 | 用途 |
|---|---|---|
| `/lste/global_frontier/status` | `std_msgs/String`(JSON) | 唯一的状态源：`work_items`、`portal_hypotheses`、`portal_transaction`、`place_progress`、`goal`、`event_graph`、`goal_context` |
| `/map` | `nav_msgs/OccupancyGrid` | 重算前沿弧（frontier arc） |
| `/pro3/rlscan` | `sensor_msgs/LaserScan` | 弧的传感器视界 `range_max` |

关键原则：**WorkItem 的弧不在状态 JSON 里**（status 只发 anchor/normal/state），
所以弧要在这里**用生产同款 helper 重算**，再按锚点距离匹配回已发布的 WorkItem。
这样画出来的弧是真实候选支撑，而不是一份可能漂移的再实现。

## 4. 两个输出，故意分开

- `/lste/demo/place_portal_markers`（`MarkerArray`，frame=`odom`）：只有短 ID
  （`P1`/`W3`/`T1`/`G`）和形状（球、柱、箭头），统一字号。地图上不画任何句子。
- `/lste/demo/legend_image`（`sensor_msgs/Image`，PIL 渲染，`rgb8`）：图例列，
  每行一项、颜色与标记一致。用 PIL 是因为 RViz 自带的 OGRE 字体没有 CJK 字形，
  PIL 用系统 NotoSansCJK / wqy-zenhei 补齐中文。

## 5. 帧与坐标

持久图存在 odom/物理坐标里，而不是会移动的 SLAM map。所以**所有标记都盖
`odom` 帧**；只有重算的 frontier 弧留在 `map` 帧（RViz 自己会做 map→odom）。

## 6. 状态 → 颜色映射

| 语义 | 颜色 |
|---|---|
| 未解决 WorkItem | 蓝 `(0.20,0.45,1.00)` |
| 正在执行的 WorkItem | 黄 `(1.00,0.95,0.20)` |
| 已解决 WorkItem | 灰 `(0.60,0.60,0.60)` |
| 失败 ViewpointAttempt | 红 `(1.00,0.20,0.20)` |
| Place（观察中 / 已覆盖） | 橙 / 绿 |
| Portal 边 / 活动传送门事务 | 青绿 / 红橙 |
| 目标点 | 绿 |
| 未被 WorkItem 认领的弧 | 暗灰 |

## 7. 关键机制（照抄即可复用）

1. **WorkItem 球 + 法向箭头 + 短 ID**：球画在 `anchor_xy`，箭头画 `normal_xy`，
   `W3` 文本标在锚点上方。失败的 attempt 在 `viewpoint_xy` 画小红球。
2. **Place 锚点是持久记忆**（模块级 `PLACE_ANCHORS` / `PLACE_ARRIVAL_ANCHORS`）：
   Place 在机器人离开后仍要画出来，否则"新 P2 是否存在"这个问题永远看不清。
   锚点优先级 = 到达点（`portal_place_entered` 的 `goal`）> 它拥有的 WorkItem 质心 >
   传送门视点（标 `Pv`）。
3. **frontier 弧 = 生产 helper 重算 + 距离匹配**：
   `global_frontier_grid.frontier_mask` 取未知/已知自由边界，再用
   `global_frontier_work_item_components.observation_supports` 算 8 连通支撑，
   最后用 `GLOBAL_FRONTIER_COMPLETED_RADIUS = 1.25 m` 的同一半径按锚点距离匹配回
   WorkItem。匹配不上的画成暗灰（那是别的房间 / 被 Place scoping 丢弃的弧）。
4. **文本描边**：RViz `TEXT_VIEW_FACING` 没有描边，节点用 8 个黑色偏移副本来模拟。
5. **图例面板宽度必须等于设备像素**：PIL 渲染宽度要匹配 RViz Image 面板的实际
   像素宽（默认左停靠 344 px）。渲染过宽会让 RViz 缩图、字变小变糊。两个值都是
   ROS 参数（`~legend_width` / `~legend_font_size`），不用改文件。

## 8. 运行方式

`demo.sh start` 里已经起了一个 tmux 窗口 `markers`，跑
`place_portal_markers.py _legend_width:=344 _legend_font_size:=19`，并等待
`/lste/demo/legend_image` 出现后才算就绪。RViz 用 `place_portal.rviz` 把
`place_portal_markers` 和 `legend_image` 两个显示项装好。

同一个启动脚本还会起 `pausepanel` 窗口，运行
`temp/manual_teleop/pause_panel.py`。点击 `PAUSE ALL` 时，面板先关闭执行器
输出和导航保持，再暂停 Gazebo，并逐个 `SIGSTOP` 当前图中的 ROS 节点进程；
RViz、Gazebo 界面、ROS master 和面板本身保留，便于查看静态图和状态。点击
`RESUME ALL` 才会恢复进程、仿真和导航。暂停期间只有面板的控制话题继续发送，
其余节点不处理或发布新的话题消息。

## 9. 已知坑（下次做新面板直接避）

- frame 写错会整批标记消失：持久图在 odom，弧在 map。
- legend 宽了 RViz 缩图：改参数而不是改文件。
- status 不带弧几何：别去 status 里找 arc，要么重算要么改 reporting。
- CJK 文字必须走 PIL（OGRE 字体无字形），所以纯 3D 文本标记只用短 ID/ASCII。
