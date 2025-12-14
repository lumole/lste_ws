#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时检测可视化节点（高级风格版）：
- 订阅 /lste/detections + /lste/scores + /lste/task 和相机图像
- 按照原 offline 脚本的配色绘制 target / env / ctx 框
- 左上角用条状图展示 S_target / S_env / S_ctx / S_total
"""

import math
from typing import Iterable, Tuple, Optional, List

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from lste_msgs.msg import LsteDetections, LsteDetection, LsteScores, LsteTask


class DetectionVisualizer:
    def __init__(self):
        rospy.init_node("lste_det_vis_node")

        self.bridge = CvBridge()
        self.latest_image = None  # type: Image
        self.latest_detections = None  # type: LsteDetections
        self.latest_scores = None  # type: LsteScores
        self.latest_task = None  # type: LsteTask

        self.image_topic = rospy.get_param("~image_topic", "/kinect/hd/image_color_rect")
        self.detections_topic = rospy.get_param("~detections_topic", "/lste/detections")
        self.output_topic = rospy.get_param("~output_topic", "/lste/det_vis_image")
        self.scores_topic = rospy.get_param("~scores_topic", "/lste/scores")
        self.draw_labels = bool(rospy.get_param("~draw_labels", True))
        self.font_scale = float(rospy.get_param("~font_scale", 0.5))
        self.line_thickness = int(rospy.get_param("~line_thickness", 2))

        # 调色板（BGR），与离线 8B-05B.py 保持一致风格
        self.palette = {
            "target": {"edge": (48, 72, 255), "fill": (48, 72, 255)},
            "env": {"edge": (0, 204, 255)},
            "ctx": {"edge": (72, 201, 176)},
            "neg": {"edge": (140, 140, 140)},
        }

        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.on_image, queue_size=1)
        self.sub_detections = rospy.Subscriber(
            self.detections_topic, LsteDetections, self.on_detections, queue_size=1
        )
        self.sub_scores = rospy.Subscriber(self.scores_topic, LsteScores, self.on_scores, queue_size=1)
        self.sub_task = rospy.Subscriber("/lste/task", LsteTask, self.on_task, queue_size=1)
        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=1)

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
        self.try_publish()

    def on_detections(self, msg: LsteDetections):
        self.latest_detections = msg
        self.try_publish()

    def on_scores(self, msg: LsteScores):
        self.latest_scores = msg

    def on_task(self, msg: LsteTask):
        self.latest_task = msg

    # ----------------- Helpers -----------------
    def try_publish(self):
        # 只要有图像，就先转发图像；如果有检测/score，就叠加可视化
        if self.latest_image is None:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("cv_bridge failed to convert Image: %s", exc)
            return

        annotated = self.draw_detections(cv_img.copy(), self.latest_detections, self.latest_scores)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("cv_bridge failed to convert cv image back to ROS Image: %s", exc)
            return

        out_msg.header = self.latest_image.header
        self.pub.publish(out_msg)

    def draw_detections(self, image, detections: Optional[LsteDetections], scores: Optional[LsteScores] = None):
        if image is None:
            return image

        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return image

        # 没有检测结果时，仅转发原图
        if detections is None:
            if scores is not None:
                self._draw_scores(image, scores)
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
            self._draw_scores(image, scores)

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
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self.font_scale
        thickness = max(1, self.line_thickness - 1)
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

    def _draw_scores(self, image, scores: LsteScores):
        # 高级风格条状图：上方深色横幅 + 四条渐变色 bar
        h, w = image.shape[:2]
        panel_margin = 10
        panel_width = int(w * 0.32)
        panel_height = int(h * 0.18)
        x0 = panel_margin
        y0 = panel_margin
        x1 = x0 + panel_width
        y1 = y0 + panel_height

        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (12, 12, 12), -1)
        cv2.rectangle(overlay, (x0, y1 - 1), (x1, y1), (45, 45, 45), 1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

        # 标题：S_total 数值 + 状态
        status = "LOCKED" if scores.detected else "SEARCHING"
        status_color = (72, 210, 170) if scores.detected else (50, 70, 220)
        title = f"S_total {scores.s_total:.2f}"
        cv2.putText(
            image,
            title,
            (x0 + 12, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            status,
            (x1 - 120, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # 条状图参数
        bar_left = x0 + 14
        bar_top = y0 + 40
        bar_width = panel_width - 28
        bar_height = max(6, int(h * 0.012))
        bar_gap = max(6, int(h * 0.008))

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
                (bar_left, y_bar_top - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                value_text,
                (bar_left + bar_width - 70, y_bar_top + bar_height - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

    def spin(self):
        rospy.spin()


def main():
    try:
        DetectionVisualizer().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
