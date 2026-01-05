#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goal Manager
- 唯一对外发布 /lste/final_goal (PoseStamped, frame_id=odom)
- 依据 state/subtype + frontier/检测/激光裁剪生成全局目标
- 各状态有独立更新周期；state 变化可立即打断更新
"""

import math
from typing import Optional, Tuple, List

import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose2D, PoseStamped, PointStamped, Vector3Stamped, TransformStamped, Twist
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from std_msgs.msg import UInt8
from image_geometry import PinholeCameraModel
import tf2_ros
from tf.transformations import quaternion_matrix, quaternion_from_euler

from lste_msgs.msg import LsteState, LsteDetections, LsteDetection, LsteTask, LsteFrontiers


STATE_PASS = 0
STATE_SUSPICIOUS = 1
STATE_LOCKED = 2
# STATE_EXHAUSTED = 3  # 当前忽略


class GoalManager:
    def __init__(self):
        rospy.init_node("lste_goal_manager")

        # --- 参数 ---
        gp = rospy.get_param
        self.pass_period = float(gp("~pass_period", 5.0))
        self.sus_c_period = float(gp("~sus_c_period", 3.0))
        self.locked_period = float(gp("~locked_period", 2.0))
        self.sus_a_period = float(gp("~sus_a_period", 2.0))
        self.sus_b_period = float(gp("~sus_b_period", 2.0))
        self.forward_dist = float(gp("~forward_dist", 10.0))
        self.goal_dist_det = float(gp("~goal_dist_det", 5.0))  # 无深度：检测方向前推距离
        self.frontier_topic = gp("~frontier_topic", "/lste/gp_frontier_dir")
        self.frontiers_topic = gp("~frontiers_topic", "/lste/gp_frontiers")
        self.depth_topic = gp("~depth_topic", "/kinect/hd/image_depth_rect")
        self.image_topic = gp("~image_topic", "/kinect/hd/image_color_rect")
        self.use_depth = bool(gp("~use_depth", False))  # 默认关闭深度，当前环境无 depth
        # 激光裁剪相关
        self.scan_topic = gp("~scan_topic", "/pro3/rl_scan")
        self.scan_frame = gp("~scan_frame", "")  # 留空则使用 scan.header.frame_id
        self.safety_margin = float(gp("~safety_margin", 0.6))
        self.scan_window_bins = int(gp("~scan_window_bins", 2))
        # 前进优先/投票相关参数
        self.front_sigma = math.radians(float(gp("~front_sigma_deg", 25.0)))
        self.area_power = float(gp("~area_power", 1.0))
        self.v_min = float(gp("~v_min", 0.05))
        self.v_scale = float(gp("~v_scale", 0.3))
        self.front_deg = math.radians(float(gp("~front_deg", 60.0)))
        self.back_deg = math.radians(float(gp("~back_deg", 60.0)))
        self.w_front_max = float(gp("~w_front_max", 1.5))
        self.w_back_min = float(gp("~w_back_min", 0.2))
        self.forward_only = bool(gp("~forward_only", True))
        self.v_gate = float(gp("~v_gate", 0.08))
        self.min_allowed_score = float(gp("~min_allowed_score", 0.05))
        self.switch_margin = float(gp("~switch_margin", 0.15))

        # --- 状态缓存 ---
        self.current_state = STATE_PASS
        self.current_subtype = ""
        self.last_state = None
        self.last_goal: Optional[PoseStamped] = None
        self.next_update_time = 0.0
        self.start_pose: Optional[Pose2D] = None

        self.latest_pose: Optional[Pose2D] = None
        self.latest_state_msg: Optional[LsteState] = None
        self.latest_task: Optional[LsteTask] = None
        self.latest_dets: Optional[LsteDetections] = None
        self.latest_frontiers: Optional[LsteFrontiers] = None
        self.latest_frontier: Optional[Vector3Stamped] = None
        self.latest_scan: Optional[LaserScan] = None
        self.latest_cmd_vel: Optional[Twist] = None
        self.headings: Optional[List[float]] = None
        self.last_dir_idx: Optional[int] = None
        # Access-Topo 覆盖
        self.access_mode = 0
        self.access_backtrack_goal: Optional[PoseStamped] = None

        # 深度/相机
        self.bridge = CvBridge()
        self.depth_image: Optional[Image] = None
        self.depth_stamp: Optional[rospy.Time] = None
        self.camera_info: Optional[CameraInfo] = None
        self.camera_frame: Optional[str] = None
        self.camera_model = PinholeCameraModel()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 发布
        self.pub_goal = rospy.Publisher("/lste/final_goal", PoseStamped, queue_size=1)

        # 订阅
        self.sub_state = rospy.Subscriber("/lste/state", LsteState, self.on_state, queue_size=1)
        self.sub_dets = rospy.Subscriber("/lste/detections", LsteDetections, self.on_dets, queue_size=1)
        self.sub_task = rospy.Subscriber("/lste/task", LsteTask, self.on_task, queue_size=1)
        self.sub_pose = rospy.Subscriber("/rbt_pose", Pose2D, self.on_pose, queue_size=1)
        self.sub_cam_info = rospy.Subscriber("/kinect/hd/camera_info", CameraInfo, self.on_cam_info, queue_size=1)
        if self.use_depth:
            self.sub_depth = rospy.Subscriber(self.depth_topic, Image, self.on_depth, queue_size=1)
        self.sub_frontier = rospy.Subscriber(self.frontier_topic, Vector3Stamped, self.on_frontier, queue_size=1)
        self.sub_frontiers = rospy.Subscriber(self.frontiers_topic, LsteFrontiers, self.on_frontiers, queue_size=1)
        self.sub_scan = rospy.Subscriber(self.scan_topic, LaserScan, self.on_scan, queue_size=1)
        self.sub_cmd_vel = rospy.Subscriber("/cmd_vel", Twist, self.on_cmd_vel, queue_size=1)
        self.sub_access_mode = rospy.Subscriber("/lste/access_topo/mode", UInt8, self.on_access_mode, queue_size=1)
        self.sub_access_goal = rospy.Subscriber("/lste/access_topo/backtrack_goal", PoseStamped, self.on_access_goal, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(0.2), self.on_timer)  # 5Hz
        rospy.loginfo("Goal Manager started: publishes /lste/final_goal")

    # -------------------- Callbacks --------------------
    def on_state(self, msg: LsteState):
        prev_state = self.current_state
        prev_subtype = self.current_subtype
        self.latest_state_msg = msg
        self.current_state = msg.state
        self.current_subtype = msg.subtype or ""
        if self.last_state is None or prev_state != msg.state or prev_subtype != self.current_subtype:
            # 立即打断更新
            self.next_update_time = 0.0
        self.last_state = msg.state

    def on_dets(self, msg: LsteDetections):
        self.latest_dets = msg

    def on_task(self, msg: LsteTask):
        self.latest_task = msg

    def on_pose(self, msg: Pose2D):
        self.latest_pose = msg
        if self.start_pose is None:
            self.start_pose = msg
            self.init_headings(msg)
        elif self.headings is None:
            self.init_headings(msg)

    def on_cam_info(self, msg: CameraInfo):
        try:
            self.camera_model.fromCameraInfo(msg)
            self.camera_info = msg
            self.camera_frame = msg.header.frame_id or self.camera_frame
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Failed to load camera info: %s", exc)

    def on_depth(self, msg: Image):
        self.depth_image = msg
        self.depth_stamp = msg.header.stamp

    def on_frontier(self, msg: Vector3Stamped):
        self.latest_frontier = msg

    def on_frontiers(self, msg: LsteFrontiers):
        self.latest_frontiers = msg

    def on_scan(self, msg: LaserScan):
        self.latest_scan = msg

    def on_cmd_vel(self, msg: Twist):
        self.latest_cmd_vel = msg

    def on_access_mode(self, msg: UInt8):
        try:
            self.access_mode = int(msg.data)
        except Exception:
            self.access_mode = 0

    def on_access_goal(self, msg: PoseStamped):
        self.access_backtrack_goal = msg

    # -------------------- Timer --------------------
    def on_timer(self, _event):
        now = rospy.Time.now().to_sec()
        if self.latest_pose is None:
            return

        # 初始 goal：起步正前方 10m（立即发布一次）
        if self.last_goal is None and self.start_pose is not None:
            goal = self.build_goal_from_pose(self.start_pose, self.forward_dist)
            if goal:
                self.publish_goal(goal)
                self.next_update_time = now + self.pass_period
            return

        period = self.state_period()
        if now < self.next_update_time and not self.force_update_due_state():
            return

        goal = self.compute_goal()
        if goal is None:
            return  # 保持原 goal，不发布

        self.publish_goal(goal)
        self.next_update_time = now + period

    def force_update_due_state(self) -> bool:
        if self.latest_state_msg is None:
            return False
        if self.last_goal is None:
            return True
        # 如果 state/subtype 发生变化则强制更新（在 on_state 已经 next_update_time=0）
        return False

    def init_headings(self, pose: Pose2D):
        base = wrap_angle(pose.theta)
        self.headings = [
            base,
            wrap_angle(base + math.pi * 0.5),
            wrap_angle(base + math.pi),
            wrap_angle(base + math.pi * 1.5),
        ]
        self.last_dir_idx = 0

    # -------------------- Goal computation --------------------
    def state_period(self) -> float:
        if self.current_state == STATE_LOCKED:
            return self.locked_period
        if self.current_state == STATE_SUSPICIOUS:
            if self.current_subtype == "Sus-C":
                return self.sus_c_period
            if self.current_subtype == "Sus-A":
                return self.sus_a_period
            if self.current_subtype == "Sus-B":
                return self.sus_b_period
        return self.pass_period

    def compute_goal(self) -> Optional[PoseStamped]:
        # Access-Topo 回退模式优先
        if self.access_mode in (1, 2) and self.access_backtrack_goal is not None:
            return self.access_backtrack_goal
        if self.current_state == STATE_LOCKED:
            return self.goal_from_target()
        if self.current_state == STATE_SUSPICIOUS:
            if self.current_subtype == "Sus-A":
                return self.goal_from_target()
            if self.current_subtype == "Sus-B":
                return self.goal_from_ctx_mid()
            if self.current_subtype == "Sus-C":
                return self.goal_from_frontiers_prior()
            # 未知 subtype：保持现状
            return None
        # PASS 默认
        return self.goal_from_frontiers_prior()

    def goal_from_target(self) -> Optional[PoseStamped]:
        det = self.pick_best_target()
        if det is None:
            rospy.logwarn_throttle(5.0, "GoalManager: no target det available for target-based goal")
            return None
        heading_world = self.det_heading_world(det)
        if heading_world is None:
            return None
        dist = self.clip_distance(heading_world, self.goal_dist_det)
        if dist is None:
            return None
        x = self.latest_pose.x + dist * math.cos(heading_world)
        y = self.latest_pose.y + dist * math.sin(heading_world)
        return self.make_goal_pose((x, y, 0.0), heading_world)

    def goal_from_ctx_mid(self) -> Optional[PoseStamped]:
        ctx_dets = self.pick_ctx_dets()
        if len(ctx_dets) == 0:
            rospy.logwarn_throttle(5.0, "GoalManager: no ctx detections for Sus-B")
            return None
        dirs = []
        for d in ctx_dets[:2]:
            h = self.det_heading_world(d)
            if h is not None:
                dirs.append(h)
        if len(dirs) == 0:
            return None
        if len(dirs) == 1:
            heading_world = dirs[0]
        else:
            v = np.array([math.cos(dirs[0]) + math.cos(dirs[1]), math.sin(dirs[0]) + math.sin(dirs[1])])
            heading_world = math.atan2(v[1], v[0])
        dist = self.clip_distance(heading_world, self.goal_dist_det)
        if dist is None:
            return None
        x = self.latest_pose.x + dist * math.cos(heading_world)
        y = self.latest_pose.y + dist * math.sin(heading_world)
        return self.make_goal_pose((x, y, 0.0), heading_world)

    def goal_from_frontier(self, period_mode: str) -> Optional[PoseStamped]:
        if self.latest_frontier is None:
            rospy.logwarn_throttle(5.0, "GoalManager: frontier info not available")
            return None
        theta_rel = float(self.latest_frontier.vector.x)
        heading_world = float(self.latest_pose.theta) + theta_rel
        dist = self.clip_distance(heading_world, self.forward_dist)
        if dist is None:
            return None
        x = self.latest_pose.x + dist * math.cos(heading_world)
        y = self.latest_pose.y + dist * math.sin(heading_world)
        return self.make_goal_pose((x, y, 0.0), heading_world)

    def goal_from_frontiers_prior(self) -> Optional[PoseStamped]:
        """PASS/Sus-C：使用全量 frontier + 运动先验投影到 4 固定方向后选取目标。"""
        if self.headings is None:
            if self.start_pose:
                self.init_headings(self.start_pose)
            elif self.latest_pose:
                self.init_headings(self.latest_pose)
        if self.headings is None:
            rospy.logwarn_throttle(5.0, "GoalManager: headings not initialized")
            return None
        if self.latest_pose is None:
            return None
        msg = self.latest_frontiers
        if msg is None or len(msg.theta_rel) == 0:
            rospy.logwarn_throttle(5.0, "GoalManager: gp_frontiers not available")
            return None

        thetas_rel = list(msg.theta_rel)
        areas_raw = list(msg.area) if msg.area else [1.0] * len(thetas_rel)
        if len(areas_raw) < len(thetas_rel):
            areas_raw += [1.0] * (len(thetas_rel) - len(areas_raw))
        areas = [max(0.0, float(a)) for a in areas_raw[: len(thetas_rel)]]

        total_area = sum(areas)
        if total_area <= 0:
            return None
        # 运动方向（世界系）
        dir_move = None
        prior_strength = 0.0
        if self.latest_cmd_vel is not None:
            vx = float(self.latest_cmd_vel.linear.x)
            speed = abs(vx)
            if speed >= self.v_min and self.latest_pose is not None:
                dir_move = wrap_angle(self.latest_pose.theta if vx >= 0 else self.latest_pose.theta + math.pi)
                if self.v_scale > 1e-3:
                    prior_strength = min(1.0, max(0.0, (speed - self.v_min) / self.v_scale))
                else:
                    prior_strength = 1.0

        scores = [0.0, 0.0, 0.0, 0.0]
        sigma = self.front_sigma if self.front_sigma > 1e-3 else 0.35
        front_thr = self.front_deg
        back_thr = math.pi - self.back_deg
        base_heading = float(self.latest_pose.theta)

        for theta_rel, area in zip(thetas_rel, areas):
            heading_world = wrap_angle(base_heading + float(theta_rel))
            a_norm = area / total_area
            weight = math.pow(a_norm, self.area_power)
            # 运动先验权重（投影前）
            if dir_move is not None:
                delta = abs(wrap_angle(heading_world - dir_move))
                if delta <= front_thr:
                    w_move = 1.0 + prior_strength * (self.w_front_max - 1.0)
                elif delta >= back_thr:
                    w_move = 1.0 - prior_strength * (1.0 - self.w_back_min)
                else:
                    w_move = 1.0
            else:
                w_move = 1.0
            vote = weight * w_move
            for idx, Hk in enumerate(self.headings):
                diff = abs(wrap_angle(heading_world - Hk))
                proj = math.exp(-0.5 * (diff / sigma) ** 2)
                scores[idx] += vote * proj

        # 前进时限制后半平面（可选）
        allow_mask = [True] * 4
        if self.forward_only and dir_move is not None and abs(float(self.latest_cmd_vel.linear.x)) > self.v_gate:
            for idx, Hk in enumerate(self.headings):
                if math.cos(wrap_angle(Hk - base_heading)) < 0:
                    allow_mask[idx] = False
            if not any(allow_mask) and any(s > self.min_allowed_score for s in scores):
                allow_mask = [True if s > self.min_allowed_score else False for s in scores]

        best_idx = None
        best_val = -1.0
        second_val = -1.0
        for idx, s in enumerate(scores):
            if not allow_mask[idx]:
                continue
            if s > best_val:
                second_val = best_val
                best_val = s
                best_idx = idx
            elif s > second_val:
                second_val = s
        if best_idx is None:
            return None

        # 滞回：优势不够则保持上次方向
        if self.last_dir_idx is not None and best_idx != self.last_dir_idx:
            margin = second_val * (1.0 + self.switch_margin)
            if best_val < margin:
                best_idx = self.last_dir_idx
                best_val = scores[best_idx]
        self.last_dir_idx = best_idx

        heading_world = self.headings[best_idx]
        dist = self.clip_distance(heading_world, self.forward_dist)
        if dist is None:
            return None
        x = self.latest_pose.x + dist * math.cos(heading_world)
        y = self.latest_pose.y + dist * math.sin(heading_world)
        return self.make_goal_pose((x, y, 0.0), heading_world)

    # -------------------- Helpers --------------------
    def make_goal_pose(self, xyz: Tuple[float, float, float], yaw: Optional[float] = None) -> PoseStamped:
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "odom"
        goal.pose.position.x = float(xyz[0])
        goal.pose.position.y = float(xyz[1])
        goal.pose.position.z = float(xyz[2])
        # 始终填充合法四元数，避免下游解析 yaw 失败
        if yaw is None:
            goal.pose.orientation.w = 1.0
        else:
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(yaw))
            goal.pose.orientation.x = qx
            goal.pose.orientation.y = qy
            goal.pose.orientation.z = qz
            goal.pose.orientation.w = qw
        return goal

    def publish_goal(self, goal: PoseStamped):
        self.last_goal = goal
        self.pub_goal.publish(goal)
        rospy.loginfo_throttle(2.0, "Publish /lste/final_goal: x=%.2f y=%.2f state=%s",
                               goal.pose.position.x, goal.pose.position.y, self.current_state)

    def build_goal_from_pose(self, pose: Pose2D, dist: float) -> PoseStamped:
        x = pose.x + dist * math.cos(pose.theta)
        y = pose.y + dist * math.sin(pose.theta)
        return self.make_goal_pose((x, y, 0.0), pose.theta)

    def pick_best_target(self) -> Optional[LsteDetection]:
        if self.latest_dets is None or len(self.latest_dets.target_dets) == 0:
            return None
        dets = sorted(self.latest_dets.target_dets, key=lambda d: float(d.score), reverse=True)
        return dets[0]

    def pick_ctx_dets(self) -> List[LsteDetection]:
        if self.latest_dets is None:
            return []
        terms = []
        if self.latest_task:
            for key in ("ctx_left", "ctx_right"):
                val = getattr(self.latest_task, key, "")
                if val:
                    s = str(val).strip().lower()
                    if s and s != "none":
                        terms.append(s)
        if not terms:
            return []
        hits: List[LsteDetection] = []
        # env_dets + target_dets 都查一遍
        all_dets = list(self.latest_dets.env_dets) + list(self.latest_dets.target_dets)
        for d in all_dets:
            label = (d.label or "").lower()
            if any(t in label for t in terms):
                hits.append(d)
        hits.sort(key=lambda d: float(d.score), reverse=True)
        return hits

    # -------------------- Projection --------------------
    def det_heading_world(self, det: LsteDetection) -> Optional[float]:
        """无深度：返回检测射线在 odom 下的朝向（弧度）。"""
        if self.camera_info is None or self.camera_frame is None:
            rospy.logwarn_throttle(5.0, "GoalManager: camera info not ready")
            return None
        w = int(self.camera_info.width) if self.camera_info.width else 0
        h = int(self.camera_info.height) if self.camera_info.height else 0
        if w <= 0 or h <= 0:
            return None
        u = int(float(det.cx) * w)
        v = int(float(det.cy) * h)
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))
        ray = self.camera_model.projectPixelTo3dRay((u, v))  # optical frame direction
        tfm = self.lookup_transform("odom", self.camera_frame, rospy.Time(0))
        if tfm is None:
            return None
        rot = tfm.transform.rotation
        mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        dir_cam = np.array([ray[0], ray[1], ray[2]], dtype=float)
        dir_odom = mat[:3, :3].dot(dir_cam)
        heading_world = math.atan2(dir_odom[1], dir_odom[0])
        return heading_world

    def clip_distance(self, heading_world: float, desired: float) -> Optional[float]:
        """
        用 LaserScan 沿给定方向裁剪距离，返回裁剪后的距离。
        - heading_world: 方向（odom 下弧度）
        - desired: 希望的前推距离
        """
        if self.latest_scan is None:
            return desired

        scan = self.latest_scan
        scan_frame = self.scan_frame or scan.header.frame_id
        if not scan_frame:
            return desired

        yaw_scan = self.frame_yaw(scan_frame)
        if yaw_scan is None:
            return desired

        theta_scan = wrap_angle(heading_world - yaw_scan)
        ang_min = scan.angle_min
        ang_inc = scan.angle_increment if scan.angle_increment != 0 else 1e-6
        idx = int(round((theta_scan - ang_min) / ang_inc))
        idx = max(0, min(len(scan.ranges) - 1, idx))

        win = self.scan_window_bins
        idx_min = max(0, idx - win)
        idx_max = min(len(scan.ranges) - 1, idx + win)
        r_min = None
        for i in range(idx_min, idx_max + 1):
            r = scan.ranges[i]
            if math.isfinite(r) and r > 0.05:
                r_min = r if r_min is None else min(r_min, r)
        if r_min is None:
            return desired

        clipped = min(desired, r_min - self.safety_margin)
        if clipped <= 0.2:
            rospy.logwarn_throttle(2.0, "GoalManager: clipped dist too small (%.2f)", clipped)
            return None
        return clipped

    def transform_point(self, pt: PointStamped, target_frame: str) -> Optional[PointStamped]:
        try:
            tfm = self.tf_buffer.lookup_transform(
                target_frame,
                pt.header.frame_id,
                pt.header.stamp,
                rospy.Duration(0.05),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(5.0, "GoalManager: TF lookup failed: %s", exc)
            return None

        trans = tfm.transform.translation
        rot = tfm.transform.rotation
        mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        mat[0, 3] = trans.x
        mat[1, 3] = trans.y
        mat[2, 3] = trans.z
        vec = np.array([pt.point.x, pt.point.y, pt.point.z, 1.0], dtype=float)
        out = mat.dot(vec)

        out_pt = PointStamped()
        out_pt.header.frame_id = target_frame
        out_pt.header.stamp = tfm.header.stamp if tfm.header.stamp else rospy.Time.now()
        out_pt.point.x = float(out[0])
        out_pt.point.y = float(out[1])
        out_pt.point.z = float(out[2])
        return out_pt

    def lookup_transform(self, target: str, source: str, stamp: rospy.Time) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                target,
                source,
                stamp,
                rospy.Duration(0.05),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(5.0, "GoalManager: TF lookup failed: %s", exc)
            return None

    def frame_yaw(self, frame: str) -> Optional[float]:
        """返回 frame 在 odom 下的 yaw。"""
        tfm = self.lookup_transform("odom", frame, rospy.Time(0))
        if tfm is None:
            return None
        rot = tfm.transform.rotation
        mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        yaw = math.atan2(mat[1, 0], mat[0, 0])
        return yaw


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def main():
    GoalManager()
    rospy.spin()


if __name__ == "__main__":
    main()
