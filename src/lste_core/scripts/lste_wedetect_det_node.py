#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WeDetect-only LSTE detection node.

This node is independent from GroundingDINO and keeps the same ROS output
contract so it can be selected without changing the rest of LSTE.
"""

import os

import cv2
import rospy
import torch

from utils.detection_node_common import DetectionNodeBase
from utils.wedetect_backend import (
    DEFAULT_CHECKPOINT,
    DEFAULT_LANGUAGE_MODEL,
    DEFAULT_SOURCE_DIR,
    DEFAULT_TRT_CACHE,
    DEFAULT_VISION_ONNX,
    WeDetectBackend,
    WeDetectUnavailable,
)


class WeDetectDetNode(DetectionNodeBase):
    detector_name = "wedetect"

    def __init__(self):
        super().__init__()
        expected_env = rospy.get_param("~expected_conda_env", "dino")
        current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        if expected_env and current_env != expected_env:
            rospy.logwarn("Current CONDA_DEFAULT_ENV=%s; expected %s", current_env, expected_env)
        label_map = rospy.get_param("~wedetect_label_map", {})
        if not isinstance(label_map, dict):
            raise ValueError("~wedetect_label_map must map English labels to Chinese labels")
        variant = rospy.get_param("~wedetect_variant", "base")
        rospy.loginfo("Loading WeDetect-%s model with cached text embeddings...", variant)
        try:
            self.model = WeDetectBackend(
                source_dir=rospy.get_param("~wedetect_source_dir", str(DEFAULT_SOURCE_DIR)),
                variant=variant,
                checkpoint=rospy.get_param("~wedetect_checkpoint", str(DEFAULT_CHECKPOINT)),
                language_model=rospy.get_param("~wedetect_language_model", str(DEFAULT_LANGUAGE_MODEL)),
                score_threshold=float(rospy.get_param("~wedetect_score_threshold", 0.20)),
                nms_iou=float(rospy.get_param("~wedetect_nms_iou", 0.70)),
                pre_nms_topk=int(rospy.get_param("~wedetect_pre_nms_topk", 3000)),
                max_detections=int(rospy.get_param("~wedetect_max_detections", 100)),
                use_fp16=bool(rospy.get_param("~wedetect_use_fp16", True)),
                label_map=label_map,
                runtime=rospy.get_param("~wedetect_runtime", "tensorrt"),
                vision_onnx=rospy.get_param("~wedetect_vision_onnx", str(DEFAULT_VISION_ONNX)),
                trt_engine_cache_path=rospy.get_param("~wedetect_trt_engine_cache", str(DEFAULT_TRT_CACHE)),
                trt_max_classes=int(rospy.get_param("~wedetect_trt_max_classes", 16)),
            )
        except WeDetectUnavailable as exc:
            rospy.logfatal("WeDetect initialization failed: %s", exc)
            raise
        rospy.loginfo("WeDetect-%s model loaded.", self.model.variant)

    def infer(self, image_bgr, prompt_a, prompt_b_terms):
        image_source = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, scores, labels = self.model.detect(image_source, [prompt_a] + list(prompt_b_terms))
        target_boxes, target_scores, target_labels = [], [], []
        env_boxes, env_scores, env_labels = [], [], []
        env_set = set(prompt_b_terms)
        for box, score, label in zip(boxes, scores, labels):
            if label == prompt_a:
                target_boxes.append(box)
                target_scores.append(score)
                target_labels.append(label)
            elif label in env_set:
                env_boxes.append(box)
                env_scores.append(score)
                env_labels.append(label)
        return (
            image_source,
            torch.tensor(target_boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(target_scores, dtype=torch.float32),
            target_labels,
            torch.tensor(env_boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(env_scores, dtype=torch.float32),
            env_labels,
        )


if __name__ == "__main__":
    WeDetectDetNode().spin()
