#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时检测可视化节点：
- 订阅 /lste/detections 和相机图像
- 在图像上绘制 target / env 的检测框与标签
- 发布新的 Image 话题，便于在 RViz 中直接查看检测结果
"""

import math
from typing import Iterable, Tuple

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from lste_msgs.msg import LsteDetections, LsteDetection


class DetectionVisualizer:
    def __init__(self):
        rospy.init_node("lste_det_vis_node")

        self.bridge = CvBridge()
        self.latest_image = None  # type: Image
        self.latest_detections = None  # type: LsteDetections

        self.image_topic = rospy.get_param("~image_topic", "/kinect/hd/image_color_rect")
        self.detections_topic = rospy.get_param("~detections_topic", "/lste/detections")
        self.output_topic = rospy.get_param("~output_topic", "/lste/det_vis_image")
        self.draw_labels = bool(rospy.get_param("~draw_labels", True))
        self.font_scale = float(rospy.get_param("~font_scale", 0.5))
        self.line_thickness = int(rospy.get_param("~line_thickness", 2))

        # BGR 颜色
        self.target_color = (0, 0, 255)  # red
        self.env_color = (0, 200, 0)  # green

        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.on_image, queue_size=1)
        self.sub_detections = rospy.Subscriber(
            self.detections_topic, LsteDetections, self.on_detections, queue_size=1
        )
        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=1)

        rospy.loginfo(
            "lste_det_vis_node started. image_topic=%s detections_topic=%s output_topic=%s",
            self.image_topic,
            self.detections_topic,
            self.output_topic,
        )

    # ----------------- Callbacks -----------------
    def on_image(self, msg: Image):
        self.latest_image = msg
        self.try_publish()

    def on_detections(self, msg: LsteDetections):
        self.latest_detections = msg
        self.try_publish()

    # ----------------- Helpers -----------------
    def try_publish(self):
        if self.latest_image is None or self.latest_detections is None:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("cv_bridge failed to convert Image: %s", exc)
            return

        annotated = self.draw_detections(cv_img.copy(), self.latest_detections)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("cv_bridge failed to convert cv image back to ROS Image: %s", exc)
            return

        out_msg.header = self.latest_image.header
        self.pub.publish(out_msg)

    def draw_detections(self, image, detections: LsteDetections):
        if image is None or detections is None:
            return image

        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return image

        for det in detections.target_dets:
            self._draw_box(image, det, width, height, self.target_color)
        for det in detections.env_dets:
            self._draw_box(image, det, width, height, self.env_color)

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

    def spin(self):
        rospy.spin()


def main():
    try:
        DetectionVisualizer().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
