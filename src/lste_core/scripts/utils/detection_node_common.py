#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS plumbing shared by the independently implemented detector nodes."""

import json
import threading
import time
from typing import Iterable, Sequence

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from lste_msgs.msg import LsteDetection, LsteDetections, LstePrompts, LsteState, LsteTask
from utils import detector as det_utils


STATE_PASS = 0
STATE_SUSPICIOUS = 1
STATE_LOCKED = 2
STATE_EXHAUSTED = 3


def task_to_task_parsed(msg: LsteTask) -> dict:
    if msg.raw_json:
        try:
            data = json.loads(msg.raw_json)
            return data.get("task_parsed", data)
        except Exception as exc:
            rospy.logwarn("Failed to parse raw_json in LsteTask: %s", exc)
    return {
        "target": {"name": msg.target_name, "attributes": list(msg.target_attributes)},
        "env": {
            "env_type_prior": list(msg.env_type_prior),
            "related_structures": list(msg.env_related_structures),
        },
        "obj_related": {
            "key_objects": list(msg.obj_key_objects),
            "negative_clues": list(msg.obj_negative_clues),
        },
        "target_ctx": {"left": msg.ctx_left, "right": msg.ctx_right},
    }


class DetectionNodeBase:
    """Common ROS lifecycle; model-specific inference lives in each node file."""

    detector_name = "detector"

    def __init__(self):
        rospy.init_node("lste_det_node")
        self.interval_pass = float(rospy.get_param("~interval_pass", 1.5))
        self.interval_suspicious = float(rospy.get_param("~interval_suspicious", 1.5))
        self.interval_locked = float(rospy.get_param("~interval_locked", 1.5))
        self.interval_exhausted = float(rospy.get_param("~interval_exhausted", 3.0))
        self.min_inference_interval = float(rospy.get_param("~min_inference_interval", 1.5))
        # A model can finish its initial load after Gazebo has already advanced
        # several seconds.  Publishing that queued image would turn an old
        # camera ray into a new navigation command, so bound source-frame age
        # independently of inference cadence.  Set <= 0 only for offline use.
        self.max_source_image_age = float(rospy.get_param("~max_source_image_age", 1.0))
        self.current_task = None
        self.task_parsed = None
        self.current_prompts = None
        self.current_state = STATE_PASS
        self.last_det_time = 0.0
        self._infer_lock = threading.Lock()
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_topic = rospy.get_param("~image_topic", "/kinect/hd/image_color_rect")
        self.sub_task = rospy.Subscriber("/lste/task", LsteTask, self.on_task, queue_size=1)
        self.sub_prompts = rospy.Subscriber("/lste/prompts", LstePrompts, self.on_prompts, queue_size=1)
        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.on_image, queue_size=1)
        self.sub_state = rospy.Subscriber("/lste/state", LsteState, self.on_state, queue_size=1)
        self.pub = rospy.Publisher("/lste/detections", LsteDetections, queue_size=5)

    def on_task(self, msg: LsteTask):
        self.current_task = msg
        self.task_parsed = task_to_task_parsed(msg)
        rospy.loginfo("Received /lste/task task_id=%s", msg.task_id)

    def on_prompts(self, msg: LstePrompts):
        self.current_prompts = msg
        rospy.loginfo("Received /lste/prompts for task_id=%s", msg.task_id)

    def on_state(self, msg: LsteState):
        self.current_state = int(msg.state)

    def on_image(self, msg: Image):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if image is None or getattr(image, "size", 0) == 0:
                raise ValueError("empty image")
            self.latest_image = (msg.header, image)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "Failed to convert Image: %s", exc)
            self.latest_image = (msg.header, None)

    def publish_empty_dets(self, header=None, reason: str = ""):
        self._publish(header, "", [], [], [], [], [])
        if reason:
            rospy.logwarn_throttle(1.0, "Publish empty detections (%s)", reason)

    def _publish(self, header, prompt_a, prompt_b_terms, target_boxes, target_scores, target_labels,
                 env_boxes, env_scores=None, env_labels=None):
        msg = LsteDetections()
        msg.header = header if isinstance(header, Header) else Header(stamp=rospy.Time.now())
        if not msg.header.stamp:
            msg.header.stamp = rospy.Time.now()
        msg.task_id = self.current_task.task_id if self.current_task else ""
        msg.prompt_a = prompt_a
        msg.prompt_b_terms = list(prompt_b_terms)
        self._append_detections(msg.target_dets, target_boxes, target_scores, target_labels)
        self._append_detections(msg.env_dets, env_boxes, env_scores or [], env_labels or [])
        self.pub.publish(msg)
        rospy.loginfo_throttle(
            5.0, "Published /lste/detections via %s: %d target boxes, %d env boxes (task_id=%s)",
            self.detector_name, len(msg.target_dets), len(msg.env_dets), msg.task_id,
        )

    @staticmethod
    def _append_detections(destination, boxes, scores, labels):
        for index, box in enumerate(boxes):
            det = LsteDetection()
            det.label = str(labels[index]) if index < len(labels) else ""
            det.score = float(scores[index]) if index < len(scores) else 0.0
            try:
                det.cx, det.cy, det.w, det.h = [float(value) for value in box]
            except Exception:
                det.cx = det.cy = det.w = det.h = 0.0
            destination.append(det)

    def current_interval(self) -> float:
        interval = {
            STATE_PASS: self.interval_pass,
            STATE_SUSPICIOUS: self.interval_suspicious,
            STATE_LOCKED: self.interval_locked,
            STATE_EXHAUSTED: self.interval_exhausted,
        }.get(self.current_state, self.interval_pass)
        return max(self.min_inference_interval, interval)

    def ready(self) -> bool:
        return (
            self.current_task is not None
            and self.task_parsed is not None
            and self.current_prompts is not None
            and self.latest_image is not None
            and (not self.current_prompts.task_id or self.current_prompts.task_id == self.current_task.task_id)
        )

    def should_detect(self) -> bool:
        now = time.time()
        if now - self.last_det_time < self.current_interval():
            return False
        self.last_det_time = now
        return True

    @staticmethod
    def _as_list(values) -> list:
        if values is None:
            return []
        try:
            return values.cpu().tolist()
        except AttributeError:
            return values.tolist() if hasattr(values, "tolist") else list(values)

    def _postprocess(self, image_source, target_boxes, target_scores, target_labels,
                     env_boxes, env_scores, env_labels):
        required_colors = det_utils.extract_color_terms(
            (self.task_parsed.get("target") or {}).get("attributes", [])
        )
        if required_colors and target_boxes is not None and len(target_boxes) > 0:
            target_boxes, target_scores, target_labels, _, _ = det_utils.filter_boxes_by_color(
                image_source, target_boxes, target_scores, target_labels, required_colors, ratio_threshold=0.02
            )
        if target_boxes is not None and len(target_boxes) > 1:
            target_boxes, target_scores, target_labels = det_utils.nms_iou(
                target_boxes, target_scores, target_labels, threshold=0.9
            )
        if target_boxes is not None and len(target_boxes) > 0:
            target_boxes, target_scores, target_labels = det_utils.validate_color_by_phrase(
                image_source, target_boxes, target_scores, target_labels, ratio_threshold=0.02,
                blur_ksize=3, dilate_iter=1,
            )
            target_boxes, target_scores, target_labels = det_utils.keep_top_confidence_detection(
                target_boxes, target_scores, target_labels
            )
        if env_boxes is not None and target_boxes is not None and len(env_boxes) > 0 and len(target_boxes) > 0:
            env_boxes, env_scores, env_labels = det_utils.filter_boxes_by_overlap(
                env_boxes, env_scores, env_labels, target_boxes, threshold=0.5
            )
        if env_boxes is not None and len(env_boxes) > 0:
            env_boxes, env_scores, env_labels = det_utils.validate_color_by_phrase(
                image_source, env_boxes, env_scores, env_labels, ratio_threshold=0.02,
                blur_ksize=3, dilate_iter=1,
            )
        if env_boxes is not None and len(env_boxes) > 1:
            env_boxes, env_scores, env_labels = det_utils.nms_iou(
                env_boxes, env_scores, env_labels, threshold=0.9
            )
        return target_boxes, target_scores, target_labels, env_boxes, env_scores, env_labels

    def run_detection(self):
        if not self._infer_lock.acquire(blocking=False):
            rospy.logwarn_throttle(1.0, "Skip frame: inference busy")
            return
        try:
            if self.latest_image is None:
                return
            header, image_bgr = self.latest_image
            if image_bgr is None:
                self.publish_empty_dets(header, "empty image frame")
                return
            prompt_a = self.current_prompts.prompt_a
            prompt_b_terms = list(self.current_prompts.prompt_b_terms)
            if not prompt_a or not prompt_b_terms:
                self.publish_empty_dets(header, "empty prompt")
                return
            try:
                result = self.infer(image_bgr, prompt_a, prompt_b_terms)
                result = self._postprocess(*result)
            except Exception as exc:
                rospy.logerr_throttle(1.0, "%s exception: %s", self.detector_name, exc)
                self.publish_empty_dets(header, "exception: %s" % type(exc).__name__)
                return
            stamp = header.stamp.to_sec() if isinstance(header, Header) else 0.0
            age = rospy.Time.now().to_sec() - stamp if stamp > 0.0 else 0.0
            if self.max_source_image_age > 0.0 and age > self.max_source_image_age:
                rospy.logwarn_throttle(
                    1.0,
                    "Discard %s result from stale image: age=%.2fs limit=%.2fs",
                    self.detector_name,
                    age,
                    self.max_source_image_age,
                )
                return
            target_boxes, target_scores, target_labels, env_boxes, env_scores, env_labels = result
            self._publish(
                header, prompt_a, prompt_b_terms,
                self._as_list(target_boxes), self._as_list(target_scores), [str(v) for v in target_labels],
                self._as_list(env_boxes), self._as_list(env_scores), [str(v) for v in env_labels],
            )
        finally:
            self._infer_lock.release()

    def spin(self):
        rate = rospy.Rate(max(0.1, float(rospy.get_param("~spin_hz", 10.0))))
        while not rospy.is_shutdown():
            if self.ready() and self.should_detect():
                self.run_detection()
            rate.sleep()

    def infer(self, image_bgr, prompt_a: str, prompt_b_terms: Sequence[str]):
        raise NotImplementedError
