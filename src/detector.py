"""YOLOv8-based object detector module.

Provides a high-level wrapper around Ultralytics YOLOv8 for real-time
person/vehicle detection in surveillance video streams. Supports dynamic
device selection, FP16 inference, and ONNX backend switching.

Typical usage:
    detector = ObjectDetector(config)
    detections = detector.detect(frame)
"""

import time

import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO


class Detection:
    """Single detection result container.

    Attributes:
        bbox:       [x1, y1, x2, y2] bounding box coordinates (pixels).
        confidence: Detection confidence score in [0, 1].
        class_id:   COCO class index.
        class_name: Human-readable class label.
    """

    __slots__ = ("bbox", "confidence", "class_id", "class_name")

    def __init__(
        self,
        bbox: np.ndarray,
        confidence: float,
        class_id: int,
        class_name: str,
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    def __repr__(self) -> str:
        return (
            f"Detection(cls={self.class_name}, conf={self.confidence:.2f}, "
            f"bbox={self.bbox.astype(int).tolist()})"
        )

    @property
    def area(self) -> float:
        """Bounding box area in pixels²."""
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return float(max(w, 0) * max(h, 0))

    @property
    def center(self) -> tuple[float, float]:
        """Center (cx, cy) of the bounding box."""
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        return (float(cx), float(cy))


class ObjectDetector:
    """YOLOv8 object detection wrapper with latency tracking.

    Args:
        config: Dictionary containing detector configuration. Expected keys:
            - model (str):          Path to YOLOv8 weights (.pt or .onnx).
            - device (str):         Device string ("auto", "cuda:0", "cpu").
            - conf_threshold (float): Minimum detection confidence.
            - iou_threshold (float):  NMS IoU overlap threshold.
            - classes (list[int]):    COCO class indices to detect.
            - img_size (int):         Input image dimension.
            - half (bool):            Enable FP16 inference.
            - max_det (int):          Max detections per frame.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = self._resolve_device(config.get("device", "auto"))
        self.conf_threshold = config.get("conf_threshold", 0.25)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.classes = config.get("classes", [0])  # default: person only
        self.img_size = config.get("img_size", 640)
        self.half = config.get("half", True) and self.device != "cpu"
        self.max_det = config.get("max_det", 100)

        # Latency tracking
        self._latencies: list[float] = []

        # Load model
        model_path = config.get("model", "yolov8s.pt")
        logger.info(f"Loading YOLOv8 model: {model_path} on {self.device}")
        self.model = YOLO(model_path)

        # Warm up the model
        self._warmup()
        logger.success(
            f"Detector ready | device={self.device} | "
            f"half={self.half} | classes={self.classes}"
        )

    # ── Public API ─────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as a NumPy array (H, W, 3).

        Returns:
            List of Detection objects for the frame.
        """
        t0 = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.img_size,
            half=self.half,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )

        detections = self._parse_results(results)
        elapsed = time.perf_counter() - t0
        self._latencies.append(elapsed)

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Run detection on a batch of frames.

        Args:
            frames: List of BGR images.

        Returns:
            List of detection lists, one per frame.
        """
        t0 = time.perf_counter()

        results = self.model.predict(
            source=frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.img_size,
            half=self.half,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )

        batch_detections = [self._parse_results([r]) for r in results]
        elapsed = time.perf_counter() - t0
        self._latencies.append(elapsed / max(len(frames), 1))

        return batch_detections

    @property
    def avg_latency_ms(self) -> float:
        """Average detection latency in milliseconds."""
        if not self._latencies:
            return 0.0
        return float(np.mean(self._latencies[-100:])) * 1000

    @property
    def fps(self) -> float:
        """Current detection throughput in frames per second."""
        lat = self.avg_latency_ms
        return 1000.0 / lat if lat > 0 else 0.0

    def reset_latency(self) -> None:
        """Clear accumulated latency measurements."""
        self._latencies.clear()

    # ── Private helpers ────────────────────────────────────

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' to the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                resolved = "cuda:0"
            else:
                resolved = "cpu"
            logger.info(f"Auto-detected device: {resolved}")
            return resolved
        return device

    def _warmup(self, n: int = 3) -> None:
        """Run a few dummy inferences to warm up the model and CUDA kernels."""
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for _ in range(n):
            self.model.predict(
                source=dummy,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
            )
        logger.debug(f"Model warmed up with {n} dummy inferences")

    def _parse_results(self, results) -> list[Detection]:
        """Convert Ultralytics results to our Detection objects."""
        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
                class_name = result.names.get(cls_id, f"class_{cls_id}")
                detections.append(
                    Detection(
                        bbox=bbox,
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=class_name,
                    )
                )

        return detections
