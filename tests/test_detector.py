"""Unit tests for the ObjectDetector module."""

import numpy as np


class TestDetection:
    """Tests for the Detection dataclass."""

    def test_detection_creation(self):
        from src.detector import Detection

        det = Detection(
            bbox=np.array([100, 200, 300, 400]),
            confidence=0.85,
            class_id=0,
            class_name="person",
        )

        assert det.confidence == 0.85
        assert det.class_id == 0
        assert det.class_name == "person"

    def test_detection_area(self):
        from src.detector import Detection

        det = Detection(
            bbox=np.array([0, 0, 100, 200]),
            confidence=0.9,
            class_id=0,
            class_name="person",
        )

        assert det.area == 20000  # 100 * 200

    def test_detection_center(self):
        from src.detector import Detection

        det = Detection(
            bbox=np.array([100, 200, 300, 400]),
            confidence=0.9,
            class_id=0,
            class_name="person",
        )

        cx, cy = det.center
        assert cx == 200.0
        assert cy == 300.0

    def test_detection_repr(self):
        from src.detector import Detection

        det = Detection(
            bbox=np.array([10.5, 20.5, 110.5, 220.5]),
            confidence=0.95,
            class_id=0,
            class_name="person",
        )

        repr_str = repr(det)
        assert "person" in repr_str
        assert "0.95" in repr_str

    def test_zero_area_detection(self):
        from src.detector import Detection

        det = Detection(
            bbox=np.array([100, 200, 100, 200]),  # Zero-size box
            confidence=0.5,
            class_id=0,
            class_name="person",
        )

        assert det.area == 0


class TestObjectDetector:
    """Tests for the ObjectDetector class (integration-level)."""

    def test_detector_init_cpu(self):
        """Test detector can initialize on CPU."""
        from src.detector import ObjectDetector

        config = {
            "model": "yolov8n.pt",
            "device": "cpu",
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "classes": [0],
            "img_size": 320,
            "half": False,
            "max_det": 100,
        }

        detector = ObjectDetector(config)
        assert detector.device == "cpu"
        assert detector.half is False

    def test_detector_detect_empty(self):
        """Test detection on a blank frame returns valid results."""
        from src.detector import ObjectDetector

        config = {
            "model": "yolov8n.pt",
            "device": "cpu",
            "img_size": 320,
            "half": False,
        }

        detector = ObjectDetector(config)
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert isinstance(detections, list)

    def test_detector_latency_tracking(self):
        """Test that latency tracking works after detection."""
        from src.detector import ObjectDetector

        config = {
            "model": "yolov8n.pt",
            "device": "cpu",
            "img_size": 320,
            "half": False,
        }

        detector = ObjectDetector(config)
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        detector.detect(frame)

        assert detector.avg_latency_ms > 0
        assert detector.fps > 0

    def test_detector_reset_latency(self):
        """Test latency reset."""
        from src.detector import ObjectDetector

        config = {
            "model": "yolov8n.pt",
            "device": "cpu",
            "img_size": 320,
            "half": False,
        }

        detector = ObjectDetector(config)
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        detector.detect(frame)

        detector.reset_latency()
        assert detector.avg_latency_ms == 0.0
