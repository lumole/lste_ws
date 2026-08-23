#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WeDetect-Base adapter that preserves the LSTE detection message contract."""

import importlib.util
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _workspace_root() -> Path:
    return Path(os.environ.get("LSTE_WS", Path(__file__).resolve().parents[4]))


DEFAULT_SOURCE_DIR = _workspace_root() / "model" / "WeDetect"
DEFAULT_CHECKPOINT = DEFAULT_SOURCE_DIR / "checkpoints" / "wedetect_base.pth"
DEFAULT_LANGUAGE_MODEL = DEFAULT_SOURCE_DIR / "xlm-roberta-base"
DEFAULT_VISION_ONNX = DEFAULT_SOURCE_DIR / "deploy" / "onnx_models" / "wedetect_base_vision.onnx"
DEFAULT_TRT_CACHE = DEFAULT_SOURCE_DIR / "deploy" / "trt_cache"


class WeDetectUnavailable(RuntimeError):
    """Raised when the optional local WeDetect installation is incomplete."""


class WeDetectBackend:
    """One-image WeDetect inference with cached text embeddings.

    The official deployment module implements the vision tower in plain PyTorch,
    so this adapter does not require MMCV or MMDetection in the ROS environment.
    """

    # WeDetect is trained primarily with Chinese category names.  The ROS API
    # continues to use its original English labels; only the text tower input is
    # translated.
    DEFAULT_LABEL_MAP = {
        "blue mug": "蓝色杯子",
        "blue cup": "蓝色杯子",
        "yellow mug": "黄色杯子",
        "yellow cup": "黄色杯子",
        "mug": "杯子",
        "cup": "杯子",
        "desk": "书桌",
        "chair": "椅子",
        "monitor": "电脑显示器",
        "computer monitor": "电脑显示器",
        "file folder": "文件夹",
        "door": "门",
        "fire hydrant": "消防栓",
        "fire extinguisher": "灭火器",
    }

    def __init__(
        self,
        source_dir: str = str(DEFAULT_SOURCE_DIR),
        variant: str = "base",
        checkpoint: str = str(DEFAULT_CHECKPOINT),
        language_model: str = str(DEFAULT_LANGUAGE_MODEL),
        score_threshold: float = 0.20,
        nms_iou: float = 0.70,
        pre_nms_topk: int = 3000,
        max_detections: int = 100,
        use_fp16: bool = True,
        label_map: Dict[str, str] = None,
        runtime: str = "tensorrt",
        vision_onnx: str = str(DEFAULT_VISION_ONNX),
        trt_engine_cache_path: str = str(DEFAULT_TRT_CACHE),
        trt_max_classes: int = 16,
    ) -> None:
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.variant = str(variant).strip().lower()
        if self.variant not in ("tiny", "base", "large"):
            raise WeDetectUnavailable("WeDetect variant must be 'tiny', 'base', or 'large'.")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.language_model = Path(language_model).expanduser().resolve()
        self.score_threshold = float(score_threshold)
        self.nms_iou = float(nms_iou)
        self.pre_nms_topk = int(pre_nms_topk)
        self.max_detections = int(max_detections)
        self.use_fp16 = bool(use_fp16)
        self.runtime = str(runtime).strip().lower()
        if self.runtime == "trt":
            self.runtime = "tensorrt"
        if self.runtime not in ("pytorch", "tensorrt"):
            raise WeDetectUnavailable("WeDetect runtime must be 'pytorch' or 'tensorrt'.")
        self.vision_onnx = Path(vision_onnx).expanduser().resolve()
        self.trt_engine_cache_path = Path(trt_engine_cache_path).expanduser().resolve()
        self.trt_max_classes = max(1, int(trt_max_classes))

        if not torch.cuda.is_available():
            raise WeDetectUnavailable("WeDetect is configured for CUDA, but CUDA is unavailable.")
        if not self.source_dir.is_dir():
            raise WeDetectUnavailable(
                "WeDetect source is missing at %s. Run scripts/models/setup_wedetect.sh." % self.source_dir
            )
        if not self.checkpoint.is_file():
            raise WeDetectUnavailable(
                "WeDetect checkpoint is missing at %s." % self.checkpoint
            )
        if not self.language_model.is_dir():
            raise WeDetectUnavailable("WeDetect language model files are missing at %s." % self.language_model)

        deployment = self._load_official_deployment_module()
        self._torch = torch
        self._deployment = deployment
        self.device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Each model variant has a fixed square input. Let cuDNN retain the
        # fastest convolution kernels and keep convolution tensors NHWC-like.
        torch.backends.cudnn.benchmark = True

        self.language_encoder = deployment.XLMRobertaLanguageBackbone(
            str(self.language_model), str(self.checkpoint)
        ).to(self.device).eval()
        self.model = None
        self._vision_session = None
        self.img_size = {"tiny": (640, 640), "base": (640, 640), "large": (1280, 1280)}[self.variant]
        if self.runtime == "pytorch":
            self.model = deployment.SimpleYOLOWorldDetector(
                self.variant,
                score_thr=self.score_threshold,
                nms_iou=self.nms_iou,
                pre_nms_topk=self.pre_nms_topk,
                post_nms_topk=self.max_detections,
            )
            deployment.load_vision_checkpoint(self.model, str(self.checkpoint))
            self.model = self.model.to(self.device, memory_format=torch.channels_last).eval()
            self.img_size = self.model.img_size
        else:
            self._vision_session = self._create_tensorrt_session()

        self.label_map = {key.lower(): value for key, value in self.DEFAULT_LABEL_MAP.items()}
        self.label_map.update({str(key).lower(): str(value) for key, value in (label_map or {}).items()})
        self._cached_signature: Tuple[Tuple[str, str], ...] = ()
        self._cached_embeddings = None
        self._cached_embeddings_np = None
        self._cached_labels: List[str] = []

    def _create_tensorrt_session(self):
        if not self.vision_onnx.is_file():
            raise WeDetectUnavailable(
                "WeDetect TensorRT vision ONNX is missing: %s. Run the ONNX export first." % self.vision_onnx
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise WeDetectUnavailable(
                "onnxruntime-gpu is required for WeDetect TensorRT runtime."
            ) from exc
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            raise WeDetectUnavailable("onnxruntime-gpu was installed without TensorRT support.")

        self.trt_engine_cache_path.mkdir(parents=True, exist_ok=True)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(
            str(self.vision_onnx),
            sess_options=options,
            providers=[
                ("TensorrtExecutionProvider", {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(self.trt_engine_cache_path),
                }),
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ],
        )
        if not session.get_providers() or session.get_providers()[0] != "TensorrtExecutionProvider":
            raise WeDetectUnavailable(
                "TensorRT vision session fell back from TensorrtExecutionProvider; "
                "ensure TensorRT libraries are on LD_LIBRARY_PATH."
            )
        return session

    def _load_official_deployment_module(self):
        module_path = self.source_dir / "deploy" / "test_coco_pytorch.py"
        if not module_path.is_file():
            raise WeDetectUnavailable("Official WeDetect deployment module is missing: %s" % module_path)
        spec = importlib.util.spec_from_file_location("lste_wedetect_deployment", module_path)
        if spec is None or spec.loader is None:
            raise WeDetectUnavailable("Cannot load WeDetect deployment module: %s" % module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _to_chinese(self, label: str) -> str:
        normalized = " ".join(str(label).lower().split())
        return self.label_map.get(normalized, label)

    def _set_classes(self, labels: Iterable[str]) -> None:
        canonical = []
        seen = set()
        for label in labels:
            label = " ".join(str(label).split())
            if label and label not in seen:
                canonical.append(label)
                seen.add(label)
        signature = tuple((label, self._to_chinese(label)) for label in canonical)
        if signature == self._cached_signature:
            return
        if not signature:
            self._cached_signature = ()
            self._cached_embeddings = None
            self._cached_embeddings_np = None
            self._cached_labels = []
            return
        prompts = [chinese for _, chinese in signature]
        if self.runtime == "tensorrt" and len(prompts) > self.trt_max_classes:
            raise WeDetectUnavailable(
                "WeDetect has %d prompt classes, exceeding fixed TensorRT capacity %d. "
                "Increase ~wedetect_trt_max_classes and rebuild its engine."
                % (len(prompts), self.trt_max_classes)
            )
        with torch.inference_mode():
            embeddings = self.language_encoder(prompts)
            self._cached_embeddings = F.normalize(embeddings, dim=-1).to(self.device)
            if self.runtime == "tensorrt":
                # Keep the TensorRT input shape stable across tasks.  Only the
                # active columns are considered during score filtering below.
                self._cached_embeddings_np = np.zeros(
                    (1, self.trt_max_classes, self._cached_embeddings.shape[-1]), dtype=np.float32
                )
                self._cached_embeddings_np[0, :len(prompts)] = self._cached_embeddings.float().cpu().numpy()
            else:
                self._cached_embeddings_np = self._cached_embeddings.float().cpu().numpy()[None, ...]
        self._cached_signature = signature
        self._cached_labels = [label for label, _ in signature]

    def _filter_tensorrt_outputs(self, bboxes: np.ndarray, scores: np.ndarray):
        """Apply the official WeDetect score filter and class-aware NMS on CUDA."""
        # TensorRT returns host NumPy arrays.  Moving the full score grid back
        # to CPU for sorting/NMS costs more than the visual TensorRT forward;
        # keep the 8400 x K reduction on CUDA and copy only final detections.
        boxes = torch.from_numpy(bboxes[0]).to(self.device, non_blocking=True).float()
        score_tensor = torch.from_numpy(scores[0, :, :len(self._cached_labels)]).to(
            self.device, non_blocking=True
        ).float()
        kept_scores, labels, keep_indices, _ = self._deployment.filter_scores_and_topk(
            score_tensor, self.score_threshold, self.pre_nms_topk
        )
        boxes = boxes[keep_indices]
        kept = self._deployment.torchvision.ops.batched_nms(
            boxes.float(), kept_scores.float(), labels, self.nms_iou
        )[:self.max_detections]
        return boxes[kept], kept_scores[kept], labels[kept]

    def detect(
        self, image: Union[str, np.ndarray], labels: Sequence[str]
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        """Return normalized boxes for an RGB array or a local image path."""
        self._set_classes(labels)
        if self._cached_embeddings is None:
            return [], [], []

        if isinstance(image, str):
            with Image.open(image) as loaded:
                source = loaded.convert("RGB").copy()
        else:
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("WeDetect expects an HxWx3 RGB numpy array or an image path.")
            source = Image.fromarray(image, mode="RGB")
        width, height = source.size
        resized, ratio, offset = self._deployment.letterbox(source, self.img_size)
        if self.runtime == "tensorrt":
            inputs = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
            output_boxes, output_scores = self._vision_session.run(
                ["bboxes", "scores"],
                {"image": inputs, "txt_feats": self._cached_embeddings_np},
            )
            boxes, scores, indices = self._filter_tensorrt_outputs(output_boxes, output_scores)
        else:
            inputs = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            inputs = inputs.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
                    img_feats = self.model.backbone(inputs.to(self.device, non_blocking=True))
                    img_feats = self.model.neck(img_feats)
                    result = self.model.head_predict(img_feats, self._cached_embeddings)[0]
            boxes, scores, indices = result["bboxes"], result["scores"], result["labels"]

        # The official model head returns letterboxed xyxy pixels. Convert at
        # this boundary so the remaining LSTE post-processing stays detector-
        # agnostic, without writing every ROS frame to disk.
        boxes, scores, indices = boxes.cpu(), scores.cpu(), indices.cpu()
        normalized_boxes: List[List[float]] = []
        normalized_scores: List[float] = []
        output_labels: List[str] = []
        for box, score, index in zip(boxes.tolist(), scores.tolist(), indices.tolist()):
            x1, y1, x2, y2 = box
            x1, x2 = x1 - offset[0], x2 - offset[0]
            y1, y2 = y1 - offset[1], y2 - offset[1]
            x1, x2 = x1 / ratio, x2 / ratio
            y1, y2 = y1 / ratio, y2 / ratio
            x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
            y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1 or not (0 <= int(index) < len(self._cached_labels)):
                continue
            normalized_boxes.append([
                ((x1 + x2) * 0.5) / width,
                ((y1 + y2) * 0.5) / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            ])
            normalized_scores.append(float(score))
            output_labels.append(self._cached_labels[int(index)])
        return normalized_boxes, normalized_scores, output_labels
