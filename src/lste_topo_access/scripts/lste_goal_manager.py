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
from geometry_msgs.msg import Pose2D, PoseStamped, PointStamped, Vector3Stamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from image_geometry import PinholeCameraModel
import tf2_ros
from tf.transformations import quaternion_matrix, quaternion_from_euler

from lste_msgs.msg import LsteState, LsteDetections, LsteDetection, LsteTask


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
        self.depth_topic = gp("~depth_topic", "/kinect/hd/image_depth_rect")
        self.image_topic = gp("~image_topic", "/kinect/hd/image_color_rect")
        self.use_depth = bool(gp("~use_depth", False))  # 默认关闭深度，当前环境无 depth
        # 激光裁剪相关
        self.scan_topic = gp("~scan_topic", "/pro3/rl_scan")
        self.scan_frame = gp("~scan_frame", "")  # 留空则使用 scan.header.frame_id
        self.safety_margin = float(gp("~safety_margin", 0.6))
        self.scan_window_bins = int(gp("~scan_window_bins", 2))

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
        self.latest_frontier: Optional[Vector3Stamped] = None
        self.latest_scan: Optional[LaserScan] = None

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
        self.sub_scan = rospy.Subscriber(self.scan_topic, LaserScan, self.on_scan, queue_size=1)

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

    def on_scan(self, msg: LaserScan):
        self.latest_scan = msg

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
        if self.current_state == STATE_LOCKED:
            return self.goal_from_target()
        if self.current_state == STATE_SUSPICIOUS:
            if self.current_subtype == "Sus-A":
                return self.goal_from_target()
            if self.current_subtype == "Sus-B":
                return self.goal_from_ctx_mid()
            if self.current_subtype == "Sus-C":
                return self.goal_from_frontier(period_mode="sus_c")
            # 未知 subtype：保持现状
            return None
        # PASS 默认
        return self.goal_from_frontier(period_mode="pass")

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
