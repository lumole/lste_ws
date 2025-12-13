#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lste_det_node
- 常驻节点：订阅 /lste/task + /lste/prompts + /camera/color/image_raw (+ /lste/state 可选)
- 复用 /home/zrz/Desktop/LSTE/Data_exchange/8B-05B.py 的 DINO/后处理逻辑
- 每次触发检测：prompt_A 走目标检测，prompt_B_list 走环境/上下文检测
- 输出 /lste/detections（LsteDetections）
"""

import os
import tempfile
import time

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from lste_msgs.msg import LsteDetections, LsteDetection, LstePrompts, LsteState, LsteTask
from utils import detector as det_utils


STATE_PASS = 0
STATE_SUSPICIOUS = 1
STATE_LOCKED = 2
STATE_EXHAUSTED = 3


def task_to_task_parsed(msg: LsteTask) -> dict:
    """把 LsteTask 转成 8B-05B.py 期待的 task_parsed 结构。"""
    if msg.raw_json:
        import json

        try:
            data = json.loads(msg.raw_json)
            return data.get("task_parsed", data)
        except Exception as e:
            rospy.logwarn("Failed to parse raw_json in LsteTask: %s", e)

    return {
        "target": {
            "name": msg.target_name,
            "attributes": list(msg.target_attributes),
        },
        "env": {
            "env_type_prior": list(msg.env_type_prior),
            "related_structures": list(msg.env_related_structures),
        },
        "obj_related": {
            "key_objects": list(msg.obj_key_objects),
            "negative_clues": list(msg.obj_negative_clues),
        },
        "target_ctx": {
            "left": msg.ctx_left,
            "right": msg.ctx_right,
        },
    }


class DetNode:
    def __init__(self):
        rospy.init_node("lste_det_node")

        # 环境检查：建议在 conda env=dino 下运行
        self.expected_env = rospy.get_param("~expected_conda_env", "dino")
        current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        if self.expected_env and current_env != self.expected_env:
            rospy.logwarn(
                "当前 CONDA_DEFAULT_ENV=%s，建议在环境 '%s' 下运行以确保 GroundingDINO 依赖正确。",
                current_env,
                self.expected_env,
            )

        # 参数：DINO 路径与阈值
        default_cfg = str(det_utils.DEFAULT_DINO_CONFIG_PATH)
        default_weights = str(det_utils.DEFAULT_DINO_WEIGHTS_PATH)
        self.config_path = rospy.get_param("~dino_config_path", default_cfg)
        self.weights_path = rospy.get_param("~dino_weights_path", default_weights)
        self.box_threshold = float(rospy.get_param("~box_threshold", 0.35))
        self.text_threshold = float(rospy.get_param("~text_threshold", 0.25))

        # 频率控制：不同 state 对应的最小检测间隔（秒）
        self.interval_pass = float(rospy.get_param("~interval_pass", 2.0))
        self.interval_suspicious = float(rospy.get_param("~interval_suspicious", 0.5))
        self.interval_locked = float(rospy.get_param("~interval_locked", 0.3))
        self.interval_exhausted = float(rospy.get_param("~interval_exhausted", 3.0))

        # 模型加载
        rospy.loginfo("Loading GroundingDINO model...")
        self.model = det_utils.load_model(self.config_path, self.weights_path)
        rospy.loginfo("GroundingDINO model loaded.")

        # 状态
        self.current_task = None
        self.task_parsed = None
        self.current_prompts = None
        self.current_state = STATE_PASS
        self.last_det_time = 0.0

        self.bridge = CvBridge()
        self.latest_image = None  # (header, cv_image_bgr)
        self.tmp_image_path = os.path.join(tempfile.gettempdir(), "lste_det_node_latest.png")

        # ROS I/O
        self.sub_task = rospy.Subscriber("/lste/task", LsteTask, self.on_task, queue_size=1)
        self.sub_prompts = rospy.Subscriber("/lste/prompts", LstePrompts, self.on_prompts, queue_size=1)
        self.sub_image = rospy.Subscriber("/camera/color/image_raw", Image, self.on_image, queue_size=1)
        self.sub_state = rospy.Subscriber("/lste/state", LsteState, self.on_state, queue_size=1)

        self.pub = rospy.Publisher("/lste/detections", LsteDetections, queue_size=5)

    # ----------------- Callbacks -----------------
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
            cv_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = (msg.header, cv_image_bgr)
        except Exception as e:
            rospy.logwarn("Failed to convert Image: %s", e)

    # ----------------- Helpers -----------------
    def current_interval(self) -> float:
        if self.current_state == STATE_PASS:
            return self.interval_pass
        if self.current_state == STATE_SUSPICIOUS:
            return self.interval_suspicious
        if self.current_state == STATE_LOCKED:
            return self.interval_locked
        if self.current_state == STATE_EXHAUSTED:
            return self.interval_exhausted
        return self.interval_pass

    def ready(self) -> bool:
        if self.current_task is None or self.task_parsed is None:
            return False
        if self.current_prompts is None:
            return False
        if self.latest_image is None:
            return False
        if self.current_prompts.task_id and self.current_prompts.task_id != self.current_task.task_id:
            return False
        return True

    def should_detect(self) -> bool:
        now = time.time()
        if now - self.last_det_time >= self.current_interval():
            self.last_det_time = now
            return True
        return False

    def prepare_image(self):
        """把最新 ROS Image 保存到临时文件，再用 det_core.load_image 读取。"""
        if self.latest_image is None:
            return None, None, None
        header, cv_bgr = self.latest_image
        try:
            # GroundingDINO 的 load_image 用 BGR 路径读入即可
            os.makedirs(os.path.dirname(self.tmp_image_path), exist_ok=True)

            import cv2

            cv2.imwrite(self.tmp_image_path, cv_bgr)
            image_source, image = det_utils.load_image(self.tmp_image_path)
            return header, image_source, image
        except Exception as e:
            rospy.logwarn("Failed to prepare image for DINO: %s", e)
            return None, None, None

    # ----------------- Core detection -----------------
    def run_detection(self):
        header, image_source, image = self.prepare_image()
        if image is None:
            return

        prompt_a = self.current_prompts.prompt_a
        prompt_b_terms = list(self.current_prompts.prompt_b_terms)
        env_caption = " . ".join(prompt_b_terms).strip()
        if not prompt_a:
            rospy.logwarn("Empty prompt_a, skip detection.")
            return
        if not env_caption:
            rospy.logwarn("Empty prompt_b_terms, skip detection.")
            return

        required_color_terms = det_utils.extract_color_terms(
            (self.task_parsed.get("target") or {}).get("attributes", [])
        )

        # === TARGET (prompt_A) ===
        target_boxes, target_logits, target_phrases = det_utils.run_grounding_dino_with_caption(
            model=self.model,
            image_source=image_source,
            image=image,
            caption=prompt_a,
            run_label="TARGET",
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            output_path=None,
        )

        color_filtered = False
        removed_color_phrases = []
        if required_color_terms and len(target_boxes) > 0:
            target_boxes, target_logits, target_phrases, color_filtered, removed_color_phrases = (
                det_utils.filter_boxes_by_color(
                    image_source,
                    target_boxes,
                    target_logits,
                    target_phrases,
                    required_color_terms,
                    ratio_threshold=0.02,
                )
            )
        if len(target_boxes) > 1:
            target_boxes, target_logits, target_phrases = det_utils.nms_iou(
                target_boxes, target_logits, target_phrases, threshold=0.9
            )
        if len(target_boxes) > 0:
            target_boxes, target_logits, target_phrases = det_utils.validate_color_by_phrase(
                image_source, target_boxes, target_logits, target_phrases, ratio_threshold=0.02, blur_ksize=3, dilate_iter=1
            )
            target_boxes, target_logits, target_phrases = det_utils.keep_top_confidence_detection(
                target_boxes, target_logits, target_phrases
            )

        # === ENV (prompt_B) ===
        env_boxes, env_logits, env_phrases = det_utils.run_grounding_dino_with_caption(
            model=self.model,
            image_source=image_source,
            image=image,
            caption=env_caption,
            run_label="ENV",
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            output_path=None,
        )

        if len(env_boxes) > 0 and len(target_boxes) > 0:
            env_boxes, env_logits, env_phrases = det_utils.filter_boxes_by_overlap(
                env_boxes, env_logits, env_phrases, target_boxes, threshold=0.5
            )
        if len(env_boxes) > 0:
            env_boxes, env_logits, env_phrases = det_utils.validate_color_by_phrase(
                image_source, env_boxes, env_logits, env_phrases, ratio_threshold=0.02, blur_ksize=3, dilate_iter=1
            )
        if len(env_boxes) > 1:
            env_boxes, env_logits, env_phrases = det_utils.nms_iou(
                env_boxes, env_logits, env_phrases, threshold=0.9
            )

        # === Publish /lste/detections ===
        msg = LsteDetections()
        if isinstance(header, Header):
            msg.header = header
        else:
            msg.header = Header()
            msg.header.stamp = rospy.Time.now()
        msg.task_id = self.current_task.task_id
        msg.prompt_a = prompt_a
        msg.prompt_b_terms = prompt_b_terms

        def to_float_list(tensor_like):
            try:
                return tensor_like.cpu().tolist()
            except Exception:
                try:
                    return tensor_like.tolist()
                except Exception:
                    return list(tensor_like)

        target_boxes_list = to_float_list(target_boxes) if target_boxes is not None else []
        target_logits_list = to_float_list(target_logits) if target_logits is not None else []
        target_phrases_list = [str(p) for p in target_phrases] if target_phrases is not None else []

        env_boxes_list = to_float_list(env_boxes) if env_boxes is not None else []
        env_logits_list = to_float_list(env_logits) if env_logits is not None else []
        env_phrases_list = [str(p) for p in env_phrases] if env_phrases is not None else []

        for idx, box in enumerate(target_boxes_list):
            det = LsteDetection()
            det.label = target_phrases_list[idx] if idx < len(target_phrases_list) else ""
            det.score = float(target_logits_list[idx]) if idx < len(target_logits_list) else 0.0
            try:
                det.cx, det.cy, det.w, det.h = [float(x) for x in box]
            except Exception:
                det.cx = det.cy = det.w = det.h = 0.0
            msg.target_dets.append(det)

        for idx, box in enumerate(env_boxes_list):
            det = LsteDetection()
            det.label = env_phrases_list[idx] if idx < len(env_phrases_list) else ""
            det.score = float(env_logits_list[idx]) if idx < len(env_logits_list) else 0.0
            try:
                det.cx, det.cy, det.w, det.h = [float(x) for x in box]
            except Exception:
                det.cx = det.cy = det.w = det.h = 0.0
            msg.env_dets.append(det)

        self.pub.publish(msg)
        rospy.loginfo(
            "Published /lste/detections: %d target boxes, %d env boxes (task_id=%s)",
            len(msg.target_dets),
            len(msg.env_dets),
            msg.task_id,
        )

    # ----------------- Spin Loop -----------------
    def spin(self):
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            if self.ready() and self.should_detect():
                try:
                    self.run_detection()
                except Exception as e:
                    rospy.logerr("Detection failed: %s", e)
            rate.sleep()


if __name__ == "__main__":
    DetNode().spin()
