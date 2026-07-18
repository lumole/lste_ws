#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时检测可视化节点（高级风格版）：
- 订阅 /lste/detections + /lste/scores + /lste/task 和相机图像
- 按照原 offline 脚本的配色绘制 target / env / ctx 框
- 左上角用条状图展示 S_target / S_env / S_ctx / S_total
"""

import math
import copy
import threading
import time
from collections import deque
from typing import Iterable, Tuple, Optional, List

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D, PointStamped, PoseStamped
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  # 注册 Point/PointStamped 的 TF 转换
from image_geometry import PinholeCameraModel

from std_msgs.msg import Bool
from lste_msgs.msg import LsteDetections, LsteDetection, LsteScores, LsteTask, LsteState


class DetectionVisualizer:
    def __init__(self):
        rospy.init_node("lste_det_vis_node")

        self.bridge = CvBridge()
        self.latest_image = None  # type: Image
        self.latest_detections = None  # type: LsteDetections
        self.display_detections = None  # type: LsteDetections
        self.latest_scores = None  # type: LsteScores
        self.latest_task = None  # type: LsteTask
        self.latest_state = None  # type: LsteState
        self.task_done = False

        self.image_topic = rospy.get_param("~image_topic", "/kinect/hd/image_color_rect")
        self.detections_topic = rospy.get_param("~detections_topic", "/lste/detections")
        self.output_topic = rospy.get_param("~output_topic", "/lste/det_vis_image")
        self.scores_topic = rospy.get_param("~scores_topic", "/lste/scores")
        self.state_topic = rospy.get_param("~state_topic", "/lste/state")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "")
        self.base_frame_id = rospy.get_param("~base_frame_id", "base_footprint")
        self.draw_labels = bool(rospy.get_param("~draw_labels", True))
        self.font_scale = float(rospy.get_param("~font_scale", 0.65))
        self.line_thickness = int(rospy.get_param("~line_thickness", 2))
        self.camera_fov_deg = float(rospy.get_param("~camera_fov_deg", 60.0))
        self.tracking_enabled = bool(rospy.get_param("~tracking_enabled", True))
        self.tracking_scale = float(rospy.get_param("~tracking_scale", 0.25))
        self.tracking_history_size = int(rospy.get_param("~tracking_history_size", 90))
        self.max_tracking_lag = float(rospy.get_param("~max_tracking_lag", 3.0))
        # Leave a small grace window beyond the DINO refresh cadence so a
        # scheduler delay does not create a visible one-frame flicker.
        self.max_tracking_age = float(rospy.get_param("~max_tracking_age", 4.0))
        self.tracking_update_stride = max(1, int(rospy.get_param("~tracking_update_stride", 3)))
        # CSRT may clamp a lost target to the last image edge.  Continue its
        # last outward motion briefly so the box leaves the frame naturally.
        self.tracking_edge_touch_pixels = float(rospy.get_param("~tracking_edge_touch_pixels", 2.0))
        self.tracking_edge_exit_frames = max(1, int(rospy.get_param("~tracking_edge_exit_frames", 12)))
        self.tracking_edge_exit_speed_scale = float(
            rospy.get_param("~tracking_edge_exit_speed_scale", 0.65)
        )
        self.tracking_edge_min_visible_fraction = float(
            rospy.get_param("~tracking_edge_min_visible_fraction", 0.05)
        )
        # Pure image-motion fallback for small/distant objects.  It does not
        # use robot pose or any TF information.
        self.tracking_flow_enabled = bool(rospy.get_param("~tracking_flow_enabled", True))
        self.tracking_flow_min_points = max(4, int(rospy.get_param("~tracking_flow_min_points", 12)))
        self.tracking_flow_max_lost_frames = max(
            1, int(rospy.get_param("~tracking_flow_max_lost_frames", 30))
        )
        self._frame_history = deque(maxlen=max(1, self.tracking_history_size))
        self._trackers = []
        self._tracking_started_at = None
        self._tracking_frame_count = 0
        self._previous_tracking_image = None
        self._tracking_lock = threading.Lock()

        if not self.camera_info_topic:
            # 简单猜测 camera_info 话题：把最后一级替换成 camera_info
            if "/" in self.image_topic:
                prefix = self.image_topic.rsplit("/", 1)[0]
                self.camera_info_topic = f"{prefix}/camera_info"
            else:
                self.camera_info_topic = "/camera/camera_info"

        # 输出图像发布器（先创建，防止回调早于属性初始化触发）
        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=1)

        # 调色板（BGR），与离线 8B-05B.py 保持一致风格
        self.palette = {
            "target": {"edge": (48, 72, 255), "fill": (48, 72, 255)},
            "env": {"edge": (0, 204, 255)},
            "ctx": {"edge": (72, 201, 176)},
            "neg": {"edge": (140, 140, 140)},
        }

        # 简化的全局目标可视化：仅依赖全局目标点 + 机器人位姿（odom 坐标系）
        self.global_goal_point = None  # type: Optional[Tuple[float, float, float]]
        self.global_goal_frame = "odom"
        self.robot_pose = None  # type: Optional[Pose2D]
        self.camera_model = PinholeCameraModel()
        self.has_camera_info = False
        self.camera_frame_id = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.on_image, queue_size=1)
        self.sub_detections = rospy.Subscriber(
            self.detections_topic, LsteDetections, self.on_detections, queue_size=1
        )
        self.sub_scores = rospy.Subscriber(self.scores_topic, LsteScores, self.on_scores, queue_size=1)
        self.sub_task = rospy.Subscriber("/lste/task", LsteTask, self.on_task, queue_size=1)
        self.sub_state = rospy.Subscriber(self.state_topic, LsteState, self.on_state, queue_size=1)
        self.sub_task_done = rospy.Subscriber("/lste/task_done", Bool, self.on_task_done, queue_size=1)
        self.sub_robot_pose = rospy.Subscriber("/rbt_pose", Pose2D, self.on_robot_pose, queue_size=1)
        self.sub_camera_info = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.on_camera_info, queue_size=1)
        # 只订阅 /lste/final_goal 作为全局目标
        self.sub_global_goal_pose = rospy.Subscriber("/lste/final_goal", PoseStamped, self.on_global_goal_pose, queue_size=1)

        rospy.loginfo(
            "lste_det_vis_node started. image_topic=%s detections_topic=%s scores_topic=%s output_topic=%s",
            self.image_topic,
            self.detections_topic,
            self.scores_topic,
            self.output_topic,
        )

    # ----------------- Callbacks -----------------
    def on_image(self, msg: Image):
        self.latest_image = msg
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("cv_bridge failed to convert Image: %s", exc)
            return

        if cv_img is None or cv_img.size == 0:
            return

        with self._tracking_lock:
            tracking_image = self._tracking_image(cv_img)
            self._frame_history.append((self._stamp_ns(msg.header), tracking_image))
            self._tracking_frame_count += 1
            if self._tracking_frame_count % self.tracking_update_stride == 0:
                self._update_trackers(tracking_image, cv_img.shape[1], cv_img.shape[0])
        self.try_publish(cv_img)

    def on_detections(self, msg: LsteDetections):
        self.latest_detections = msg
        with self._tracking_lock:
            self._start_trackers(msg)

    def on_scores(self, msg: LsteScores):
        self.latest_scores = msg

    def on_task(self, msg: LsteTask):
        self.latest_task = msg
        # 新任务默认清除 task_done 标志
        self.task_done = False

    def on_state(self, msg: LsteState):
        self.latest_state = msg

    def on_task_done(self, msg: Bool):
        try:
            self.task_done = bool(msg.data)
        except Exception:
            self.task_done = False

    def on_camera_info(self, msg: CameraInfo):
        try:
            self.camera_model.fromCameraInfo(msg)
            self.has_camera_info = True
            self.camera_frame_id = msg.header.frame_id or self.camera_frame_id
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Failed to load camera info: %s", exc)
            self.has_camera_info = False

    def on_robot_pose(self, msg: Pose2D):
        self.robot_pose = msg

    def on_global_goal_pose(self, msg: PoseStamped):
        self.global_goal_point = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )
        self.global_goal_frame = msg.header.frame_id or "odom"

    # ----------------- Helpers -----------------
    def try_publish(self, cv_img=None):
        # 只要有图像，就先转发图像；如果有检测/score，就叠加可视化
        if self.latest_image is None:
            return

        if cv_img is None:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="bgr8")
            except CvBridgeError as exc:
                rospy.logwarn("cv_bridge failed to convert Image: %s", exc)
                return

        with self._tracking_lock:
            detections = copy.deepcopy(self.display_detections or self.latest_detections)
        annotated = self.draw_detections(cv_img.copy(), detections, self.latest_scores, self.latest_state)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("cv_bridge failed to convert cv image back to ROS Image: %s", exc)
            return

        out_msg.header = self.latest_image.header
        # 兜底：如果由于某些原因 pub 还没初始化，懒加载一个
        if not hasattr(self, "pub"):
            self.pub = rospy.Publisher(self.output_topic, Image, queue_size=1)
        self.pub.publish(out_msg)

    # ----------------- Lightweight box tracking -----------------
    @staticmethod
    def _stamp_ns(header) -> int:
        try:
            return int(header.stamp.to_nsec())
        except Exception:
            return 0

    def _tracking_image(self, image):
        if self.tracking_scale >= 0.999:
            return image
        return cv2.resize(image, None, fx=self.tracking_scale, fy=self.tracking_scale,
                          interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _make_tracker():
        return cv2.TrackerCSRT_create()

    def _start_trackers(self, detections: LsteDetections):
        """Initialize CSRT on the source frame and replay buffered frames to now."""
        self.display_detections = copy.deepcopy(detections)
        self._trackers = []
        self._tracking_started_at = time.monotonic()
        if not self.tracking_enabled or not self._frame_history:
            return

        stamp_ns = self._stamp_ns(detections.header)
        if not stamp_ns:
            return
        history = list(self._frame_history)
        frame_index = min(range(len(history)), key=lambda i: abs(history[i][0] - stamp_ns))
        matched_stamp, source_frame = history[frame_index]
        if abs(matched_stamp - stamp_ns) > int(self.max_tracking_lag * 1e9):
            rospy.logwarn_throttle(2.0, "Detection image is no longer in the tracker history")
            return

        self._previous_tracking_image = source_frame.copy()

        height, width = source_frame.shape[:2]
        for kind in ("target_dets", "env_dets"):
            for det in getattr(self.display_detections, kind):
                x = max(0.0, (float(det.cx) - 0.5 * float(det.w)) * width)
                y = max(0.0, (float(det.cy) - 0.5 * float(det.h)) * height)
                w = max(2.0, float(det.w) * width)
                h = max(2.0, float(det.h) * height)
                w = min(w, width - x)
                h = min(h, height - y)
                if w <= 1.0 or h <= 1.0:
                    continue
                try:
                    tracker = self._make_tracker()
                    if tracker.init(source_frame, (x, y, w, h)):
                        self._trackers.append({
                            "tracker": tracker,
                            "det": det,
                            "edge_exit_count": 0,
                            "edge_exit_direction": None,
                            "edge_exit_offset": (0.0, 0.0),
                            "edge_exit_speed": 0.0,
                            "last_center": (x + 0.5 * w, y + 0.5 * h),
                            "last_bbox": (x, y, w, h),
                            "flow_lost_count": 0,
                        })
                except cv2.error as exc:
                    rospy.logwarn_throttle(2.0, "Failed to initialize CSRT tracker: %s", exc)

        rospy.loginfo("Initialized %d CSRT trackers for the latest detection", len(self._trackers))

        # GroundingDINO returns after processing its source frame. Replay the
        # buffered intermediate frames before drawing on the next live frame.
        replay_frames = history[frame_index + 1::self.tracking_update_stride]
        if history[-1:] and (not replay_frames or replay_frames[-1][0] != history[-1][0]):
            replay_frames.append(history[-1])
        for _, frame in replay_frames:
            self._update_trackers(frame, width, height)

    def _update_trackers(self, tracking_image, full_width: int, full_height: int):
        if (
            self._tracking_started_at is not None
            and time.monotonic() - self._tracking_started_at > self.max_tracking_age
        ):
            self._clear_display_tracks()
            self._trackers = []
            return
        if not self._trackers:
            self._previous_tracking_image = tracking_image.copy()
            return
        height, width = tracking_image.shape[:2]
        image_flow = self._global_image_flow(self._previous_tracking_image, tracking_image)
        self._previous_tracking_image = tracking_image.copy()
        alive = []
        expired = set()
        for track in self._trackers:
            tracker = track["tracker"]
            det = track["det"]
            try:
                ok, bbox = tracker.update(tracking_image)
            except cv2.error:
                ok = False
            last_bbox = track["last_bbox"]
            x, y, w, h = last_bbox
            if ok:
                candidate = tuple(float(v) for v in bbox)
                tracker_motion = (
                    candidate[0] + 0.5 * candidate[2] - track["last_center"][0],
                    candidate[1] + 0.5 * candidate[3] - track["last_center"][1],
                )
                # CSRT can report success while frozen on the old patch during
                # a fast pan. Prefer image motion only in that stalled case.
                if self._flow_explains_stalled_tracker(image_flow, tracker_motion):
                    x, y, w, h = self._apply_image_flow(last_bbox, image_flow)
                else:
                    x, y, w, h = candidate
                    track["flow_lost_count"] = 0
            elif image_flow is not None and track["flow_lost_count"] < self.tracking_flow_max_lost_frames:
                x, y, w, h = self._apply_image_flow(last_bbox, image_flow)
                track["flow_lost_count"] += 1
                rospy.logdebug_throttle(1.0, "CSRT lost; using image-flow box fallback")
            else:
                expired.add(id(det))
                continue

            center = (x + 0.5 * w, y + 0.5 * h)
            last_x, last_y = track["last_center"]
            motion = (center[0] - last_x, center[1] - last_y)
            direction = self._outward_edge_direction(x, y, w, h, width, height, motion)
            active_direction = track["edge_exit_direction"]

            # Start only after the tracker reaches the real image edge while
            # moving outward.  This deliberately does not use a percentage
            # margin: a box may be partly outside the image before it exits.
            if direction is not None:
                if active_direction != direction:
                    track["edge_exit_direction"] = direction
                    track["edge_exit_count"] = 0
                    track["edge_exit_offset"] = (0.0, 0.0)
                    track["edge_exit_speed"] = self._outward_speed(direction, motion)
                active_direction = direction
            elif active_direction is not None and self._moves_back_into_frame(active_direction, motion):
                # A reversal means the target has re-entered the image; use
                # the live CSRT result again instead of continuing an exit.
                track["edge_exit_direction"] = None
                track["edge_exit_count"] = 0
                track["edge_exit_offset"] = (0.0, 0.0)
                track["edge_exit_speed"] = 0.0
                active_direction = None

            draw_x, draw_y = x, y
            if active_direction is not None:
                speed = max(
                    1.0,
                    track["edge_exit_speed"] * 0.8,
                    self._outward_speed(active_direction, motion),
                )
                track["edge_exit_speed"] = speed
                offset_x, offset_y = track["edge_exit_offset"]
                step_x, step_y = self._edge_step(active_direction, speed)
                offset_x += step_x
                offset_y += step_y
                track["edge_exit_offset"] = (offset_x, offset_y)
                track["edge_exit_count"] += 1
                draw_x = x + offset_x
                draw_y = y + offset_y
            track["last_center"] = center
            track["last_bbox"] = (x, y, w, h)

            if (
                active_direction is not None
                and (
                    self._visible_fraction(draw_x, draw_y, w, h, width, height)
                    <= self.tracking_edge_min_visible_fraction
                    or track["edge_exit_count"] >= self.tracking_edge_exit_frames
                )
            ):
                expired.add(id(det))
                continue
            det.cx = float((draw_x + 0.5 * w) / width)
            det.cy = float((draw_y + 0.5 * h) / height)
            det.w = float(w / width)
            det.h = float(h / height)
            alive.append(track)
        self._trackers = alive
        if expired:
            self._remove_display_tracks(expired)

    def _global_image_flow(self, previous, current):
        """Estimate global 2D image motion from sparse optical flow."""
        if not self.tracking_flow_enabled or previous is None or current is None:
            return None
        if previous.shape[:2] != current.shape[:2]:
            return None
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=120, qualityLevel=0.01, minDistance=6, blockSize=7
        )
        if points is None or len(points) < self.tracking_flow_min_points:
            return None
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None, winSize=(21, 21), maxLevel=3
        )
        if next_points is None or status is None:
            return None
        valid = status.reshape(-1).astype(bool)
        if error is not None:
            valid &= error.reshape(-1) < 25.0
        vectors = (next_points.reshape(-1, 2) - points.reshape(-1, 2))[valid]
        if len(vectors) < self.tracking_flow_min_points:
            return None
        median = np.median(vectors, axis=0)
        deviations = np.linalg.norm(vectors - median, axis=1)
        inliers = vectors[deviations <= max(0.5, 2.5 * np.median(deviations))]
        if len(inliers) < self.tracking_flow_min_points:
            return None
        return tuple(float(value) for value in np.median(inliers, axis=0))

    @staticmethod
    def _apply_image_flow(bbox, image_flow):
        x, y, w, h = bbox
        dx, dy = image_flow
        return x + dx, y + dy, w, h

    @staticmethod
    def _flow_explains_stalled_tracker(image_flow, tracker_motion):
        if image_flow is None:
            return False
        flow_size = math.hypot(*image_flow)
        tracker_size = math.hypot(*tracker_motion)
        return flow_size >= 0.75 and tracker_size <= 0.25

    def _outward_edge_direction(self, x, y, w, h, width, height, motion):
        """Return the touched edge only when the tracked centre moves outward."""
        dx, dy = motion
        touch = self.tracking_edge_touch_pixels
        candidates = []
        if x <= touch and dx < -0.2:
            candidates.append((abs(dx), "left"))
        if x + w >= width - touch and dx > 0.2:
            candidates.append((abs(dx), "right"))
        if y <= touch and dy < -0.2:
            candidates.append((abs(dy), "top"))
        if y + h >= height - touch and dy > 0.2:
            candidates.append((abs(dy), "bottom"))
        return max(candidates)[1] if candidates else None

    def _outward_speed(self, direction, motion):
        dx, dy = motion
        component = {
            "left": -dx,
            "right": dx,
            "top": -dy,
            "bottom": dy,
        }[direction]
        return max(0.0, component * self.tracking_edge_exit_speed_scale)

    @staticmethod
    def _edge_step(direction, speed):
        return {
            "left": (-speed, 0.0),
            "right": (speed, 0.0),
            "top": (0.0, -speed),
            "bottom": (0.0, speed),
        }[direction]

    @staticmethod
    def _moves_back_into_frame(direction, motion):
        dx, dy = motion
        return {
            "left": dx > 0.2,
            "right": dx < -0.2,
            "top": dy > 0.2,
            "bottom": dy < -0.2,
        }[direction]

    @staticmethod
    def _visible_fraction(x, y, w, h, width, height):
        if w <= 0.0 or h <= 0.0:
            return 0.0
        visible_w = max(0.0, min(x + w, width) - max(x, 0.0))
        visible_h = max(0.0, min(y + h, height) - max(y, 0.0))
        return (visible_w * visible_h) / (w * h)

    def _clear_display_tracks(self):
        if self.display_detections is None:
            return
        del self.display_detections.target_dets[:]
        del self.display_detections.env_dets[:]

    def _remove_display_tracks(self, expired_ids):
        if self.display_detections is None:
            return
        self.display_detections.target_dets[:] = [
            det for det in self.display_detections.target_dets if id(det) not in expired_ids
        ]
        self.display_detections.env_dets[:] = [
            det for det in self.display_detections.env_dets if id(det) not in expired_ids
        ]

    def draw_detections(
        self,
        image,
        detections: Optional[LsteDetections],
        scores: Optional[LsteScores] = None,
        state: Optional[LsteState] = None,
    ):
        if image is None:
            return image

        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return image

        # 没有检测结果时，仅转发原图
        if detections is None:
            if scores is not None:
                self._draw_scores(image, scores, state)
            return image

        # ==== 1) 基于 task 语义构造 ctx / neg 关键词 ====
        ctx_terms_lower: List[str] = []
        neg_terms_lower: List[str] = []
        if self.latest_task is not None:
            for key in ("ctx_left", "ctx_right"):
                val = getattr(self.latest_task, key, "")
                if val:
                    s = str(val).strip().lower()
                    if s and s != "none":
                        ctx_terms_lower.append(s)
            for term in (self.latest_task.obj_negative_clues or []):
                s = str(term).strip().lower()
                if s:
                    neg_terms_lower.append(s)

        # ==== 2) 叠加 target 区域的半透明填充 ====
        target_boxes_px = []
        for det in detections.target_dets:
            x1, y1, x2, y2 = self._norm_box_to_pixels(det, width, height)
            target_boxes_px.append((x1, y1, x2, y2))

        if target_boxes_px:
            overlay = image.copy()
            fill_color = self.palette["target"]["fill"]["edge"] if isinstance(
                self.palette["target"].get("fill"), dict
            ) else self.palette["target"]["fill"]
            # 这里兼容上面 fill 写法，实际上 fill 就是一个 BGR tuple
            if isinstance(fill_color, tuple):
                for (x1, y1, x2, y2) in target_boxes_px:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
                cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

        # ==== 3) 绘制 target / env / ctx 框 ====
        for det in detections.target_dets:
            label_low = (det.label or "").lower()
            is_ctx = any(term in label_low for term in ctx_terms_lower) if ctx_terms_lower else False
            color_key = "ctx" if is_ctx else "target"
            color = self.palette[color_key]["edge"]
            self._draw_box(image, det, width, height, color)

        for det in detections.env_dets:
            label_low = (det.label or "").lower()
            is_ctx = any(term in label_low for term in ctx_terms_lower) if ctx_terms_lower else False
            is_neg = any(term in label_low for term in neg_terms_lower) if neg_terms_lower else False
            if is_ctx:
                color_key = "ctx"
            elif is_neg:
                color_key = "neg"
            else:
                color_key = "env"
            color = self.palette[color_key]["edge"]
            self._draw_box(image, det, width, height, color)

        if scores is not None:
            self._draw_scores(image, scores, state)

        try:
            self._draw_global_goal_indicator(image)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Draw global goal failed: %s", exc)

        return image

    def _draw_box(self, image, det: LsteDetection, width: int, height: int, color: Tuple[int, int, int]):
        x1, y1, x2, y2 = self._norm_box_to_pixels(det, width, height)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_thickness)
        if self.draw_labels:
            label = self._build_label(det)
            if label:
                self._draw_label(image, label, (x1, y1 - 4), color)

    def _norm_box_to_pixels(self, det: LsteDetection, width: int, height: int) -> Tuple[int, int, int, int]:
        cx = float(det.cx) * width
        cy = float(det.cy) * height
        w = max(1.0, float(det.w) * width)
        h = max(1.0, float(det.h) * height)

        x1 = int(max(0, min(width - 1, math.floor(cx - 0.5 * w))))
        y1 = int(max(0, min(height - 1, math.floor(cy - 0.5 * h))))
        x2 = int(max(0, min(width - 1, math.ceil(cx + 0.5 * w))))
        y2 = int(max(0, min(height - 1, math.ceil(cy + 0.5 * h))))

        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        return x1, y1, x2, y2

    def _build_label(self, det: LsteDetection) -> str:
        label = (det.label or "").strip()
        if det.score:
            try:
                label = f"{label} {float(det.score):.2f}" if label else f"{float(det.score):.2f}"
            except (TypeError, ValueError):
                pass
        return label

    def _draw_label(self, image, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]):
        if not text:
            return
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = self.font_scale
        thickness = max(2, self.line_thickness)
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = origin
        x = max(0, min(image.shape[1] - text_size[0], x))
        y = max(text_size[1], min(image.shape[0] - baseline, y))

        cv2.rectangle(
            image,
            (x, y - text_size[1] - baseline),
            (x + text_size[0], y + baseline),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(image, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)

    def _draw_scores(self, image, scores: LsteScores, state: Optional[LsteState] = None):
        # 高级风格条状图：上方深色横幅 + 四条渐变色 bar
        h, w = image.shape[:2]
        panel_margin = 10
        panel_width = min(w - 2 * panel_margin, max(330, int(w * 0.36)))
        panel_height = min(h - 2 * panel_margin, max(145, int(h * 0.22)))
        x0 = panel_margin
        y0 = panel_margin
        x1 = x0 + panel_width
        y1 = y0 + panel_height

        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (12, 12, 12), -1)
        cv2.rectangle(overlay, (x0, y1 - 1), (x1, y1), (45, 45, 45), 1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

        # 标题：S_total 数值 + 状态
        status = "SEARCHING"
        status_color = (50, 70, 220)  # blue
        subtype_text = ""
        if self.task_done:
            status = "DONE"
            status_color = (0, 200, 0)
        elif state is not None:
            if state.state == 1:
                status_color = (0, 165, 255)  # orange
                raw_subtype = (state.subtype or "").strip()
                suffix = ""
                if raw_subtype:
                    low = raw_subtype.lower()
                    if low.startswith("sus-"):
                        suffix = low.split("-", 1)[1].strip().upper()
                    else:
                        suffix = raw_subtype.strip().upper()
                if suffix not in ("A", "B", "C"):
                    suffix = "C"  # 兜底显示 SUSPICIOUS-C
                status = f"SUSPICIOUS-{suffix}"
            elif state.state == 2:
                status = "LOCKED"
                status_color = (72, 210, 170)  # green
            elif state.state == 3:
                status = "EXHAUSTED"
                status_color = (140, 140, 140)  # gray
            else:
                status = "PASS"
                status_color = (50, 70, 220)
        else:
            if scores.detected:
                status = "LOCKED"
                status_color = (72, 210, 170)
        if not subtype_text and state is not None and state.state != 1 and state.subtype:
            subtype_text = state.subtype
        font = cv2.FONT_HERSHEY_DUPLEX
        title = f"S_total {scores.s_total:.2f}"
        cv2.putText(
            image,
            title,
            (x0 + 12, y0 + 30),
            font,
            0.76,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        status_size, _ = cv2.getTextSize(status, font, 0.64, 2)
        status_x = max(x0 + 12, x1 - 12 - status_size[0])
        cv2.putText(
            image,
            status,
            (status_x, y0 + 30),
            font,
            0.64,
            status_color,
            2,
            cv2.LINE_AA,
        )
        if subtype_text:
            subtype_size, _ = cv2.getTextSize(subtype_text, font, 0.56, 2)
            subtype_x = max(x0 + 12, x1 - 12 - subtype_size[0])
            cv2.putText(
                image,
                subtype_text,
                (subtype_x, y0 + 54),
                font,
                0.56,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )

        # 条状图参数
        bar_left = x0 + 14
        bar_top = y0 + (72 if subtype_text else 52)
        bar_width = panel_width - 28
        bar_height = max(9, int(h * 0.014))
        bar_gap = max(13, int(h * 0.018))

        def norm01(value: float, vmin: float, vmax: float) -> float:
            if value <= vmin:
                return 0.0
            if value >= vmax:
                return 1.0
            return float((value - vmin) / (vmax - vmin + 1e-9))

        metrics = [
            ("S_target", scores.s_target, 0.0, 1.0, self.palette["target"]["edge"]),
            ("S_env", scores.s_env, -1.0, 1.0, self.palette["env"]["edge"]),
            ("S_ctx", scores.s_ctx, -1.0, 1.0, self.palette["ctx"]["edge"]),
        ]

        for idx, (name, val, vmin, vmax, color) in enumerate(metrics):
            y_bar_top = bar_top + idx * (bar_height + bar_gap)
            y_bar_bottom = y_bar_top + bar_height

            # 背景条
            cv2.rectangle(
                image,
                (bar_left, y_bar_top),
                (bar_left + bar_width, y_bar_bottom),
                (35, 35, 35),
                -1,
            )

            # 前景条（值）
            ratio = norm01(float(val), vmin, vmax)
            bar_len = int(bar_width * ratio)
            if bar_len > 0:
                color_fg = tuple(int(0.75 * c + 0.25 * 255) for c in color)
                cv2.rectangle(
                    image,
                    (bar_left, y_bar_top),
                    (bar_left + bar_len, y_bar_bottom),
                    color_fg,
                    -1,
                )

            # 文本标签与数值
            label_text = f"{name}"
            value_text = f"{val:+.2f}" if vmin < 0.0 else f"{val:.2f}"
            cv2.putText(
                image,
                label_text,
                (bar_left, y_bar_top - 3),
                font,
                0.58,
                (210, 210, 210),
                2,
                cv2.LINE_AA,
            )
            value_size, _ = cv2.getTextSize(value_text, font, 0.58, 2)
            cv2.putText(
                image,
                value_text,
                (bar_left + bar_width - value_size[0], y_bar_top + bar_height - 2),
                font,
                0.58,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

    def _draw_goal_hint(self, image, text: str):
        h, w = image.shape[:2]
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.68
        thickness = 2
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        margin = 10
        x = max(margin, w - text_size[0] - margin)
        y = h - margin
        cv2.rectangle(
            image,
            (x - 6, y - text_size[1] - baseline - 4),
            (x + text_size[0] + 6, y + baseline + 4),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(image, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def _draw_global_goal_indicator(self, image):
        """显示 /lste/final_goal 的 (x,y)，并尝试将该点投影到相机图像上绘制固定大小的绿色标记。"""
        if self.global_goal_point is None:
            self._draw_goal_hint(image, "Global goal not received")
            return

        gx, gy, gz = self.global_goal_point
        text = f"Global goal (odom): x={gx:.2f}, y={gy:.2f}"
        self._draw_goal_hint(image, text)

        # 仅在有内参时尝试投影
        if not self.has_camera_info:
            return

        def draw_at_uv(u, v):
            h, w = image.shape[:2]
            u_int = int(round(u))
            v_int = int(round(v))
            if not (0 <= u_int < w and 0 <= v_int < h):
                rospy.logwarn_throttle(5.0, "Global goal projection outside image: (%.1f, %.1f)", u, v)
                return
            center = (u_int, v_int)
            cv2.circle(image, center, 12, (0, 255, 0), 3)
            cv2.circle(image, center, 6, (0, 255, 0), -1)
            cv2.putText(
                image,
                "GLOBAL",
                (center[0] - 25, max(12, center[1] - 14)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.58,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # 优先用 TF 直接从 global_goal_frame 变到相机光学帧
        if self.camera_frame_id:
            goal_pt = PointStamped()
            goal_pt.header.frame_id = self.global_goal_frame or "odom"
            goal_pt.header.stamp = rospy.Time(0)
            goal_pt.point.x = gx
            goal_pt.point.y = gy
            goal_pt.point.z = 0.0
            try:
                goal_cam = self.tf_buffer.transform(goal_pt, self.camera_frame_id, rospy.Duration(0.05))
                Z = goal_cam.point.z
                if Z > 1e-3:
                    try:
                        u, v = self.camera_model.project3dToPixel(
                            (goal_cam.point.x, goal_cam.point.y, goal_cam.point.z)
                        )
                        draw_at_uv(u, v)
                        return
                    except Exception as exc:
                        rospy.logwarn_throttle(5.0, "Project global goal failed: %s", exc)
                else:
                    rospy.logwarn_throttle(5.0, "Global goal behind camera (z=%.3f)", Z)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
                rospy.logwarn_throttle(5.0, "TF lookup for global goal failed: %s", exc)

    def spin(self):
        rospy.spin()


def main():
    try:
        DetectionVisualizer().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
