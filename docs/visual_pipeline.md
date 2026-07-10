# 视觉链路性能与框跟随

更新日期：2026-07-10

提交记录：`perf(vision): smooth detection visualization and Gazebo rendering`

## 目标

提高 Gazebo/RViz 画面流畅度，并让检测框在 GroundingDINO 两次推理之间持续跟随图像。保持现有启动方式不变：

```bash
./scripts/run_env_tmux.sh
./scripts/run_nodes_tmux.sh
```

## 性能策略

- Kinect Gazebo 相机更新率为 30 Hz。
- 场景关闭阴影，避免不必要的 Gazebo 渲染负担。
- `GAZEBO_MODEL_DATABASE_URI` 默认指向工作区的本地空模型数据库，避免 Gazebo Classic 访问已退役的公网模型数据库而阻塞启动。
- GroundingDINO 最小推理间隔为 3 秒。DINO 与 Gazebo、RViz 共用 GPU；限制推理频率为渲染留下稳定时间，不改变相机的 30 Hz 图像发布。

可用环境变量临时覆盖 DINO 间隔：

```bash
DINO_MIN_INTERVAL=2.0 ./scripts/run_nodes_tmux.sh
```

## 检测框策略

`lste_det_vis_node.py` 在每次 DINO 结果到达时，在对应历史图像上初始化 OpenCV CSRT 跟踪器，再重放缓冲帧追到当前画面。

- 跟踪计算在 1/4 分辨率、每 3 帧执行一次，显示图像仍按相机回调发布。
- CSRT 跟随框的位置和尺度。
- 对远处小物体或水平转动时 CSRT 丢失/卡住的情形，使用相邻图像的稀疏 Lucas-Kanade 光流估计全局二维画面平移，作为短时纯图像兜底。该过程不读取车辆位姿、`/rbt_pose` 或 TF。
- 框实际触及图像边界并持续向外运动时，会逐步显示为被裁剪的部分框；接近完全出画或持续外移约 1.2 秒后移除。若画面运动反向，退出过程取消。
- 光流和 CSRT 都不能继续支持时，框会移除；下一次 DINO 结果重新初始化跟踪。

## 关键参数

这些参数均为 `lste_det_vis_node` 的 ROS 私有参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `tracking_scale` | `0.25` | 跟踪使用的图像缩放比例 |
| `tracking_update_stride` | `3` | 每多少个相机帧更新一次跟踪 |
| `max_tracking_age` | `4.0` | 单次 DINO 结果允许被跟踪的最长秒数 |
| `tracking_flow_max_lost_frames` | `30` | CSRT 丢失时允许光流兜底的更新次数 |
| `tracking_edge_exit_frames` | `12` | 框触边向外移动后最多保留的更新次数 |

## 限制

CSRT 和光流输出的是轴对齐矩形。旋转、剧烈运动模糊、画面缺少纹理或物体被遮挡时，短时跟踪可能失效；DINO 的下一次推理会重新校正框的位置和大小。
