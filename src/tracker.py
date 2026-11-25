"""ByteTrack multi-object tracker module.

Wraps the Ultralytics-native ByteTrack implementation for real-time
multi-object tracking. Maintains track histories, manages track lifecycles,
and provides trajectory data for downstream anomaly analysis.

Typical usage:
    tracker = MultiObjectTracker(config)
    tracks  = tracker.update(frame, detections)
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from ultralytics import YOLO


@dataclass
class Track:
    """Represents a single tracked object across frames.

    Attributes:
        track_id:    Unique integer identifier for this track.
        bbox:        Current [x1, y1, x2, y2] bounding box.
        confidence:  Latest detection confidence.
        class_id:    COCO class index.
        class_name:  Human-readable class label.
        frame_count: Number of frames this track has been active.
        trajectory:  History of (cx, cy) center positions.
        is_active:   Whether the track is currently matched to a detection.
    """

    track_id: int
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    frame_count: int = 1
    trajectory: list = field(default_factory=list)
    is_active: bool = True

    def __post_init__(self):
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        if not self.trajectory:
            self.trajectory = [(float(cx), float(cy))]

    @property
    def center(self) -> tuple[float, float]:
        """Current center (cx, cy)."""
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        return (float(cx), float(cy))

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def velocity(self) -> tuple[float, float]:
        """Estimate velocity from last two trajectory points."""
        if len(self.trajectory) < 2:
            return (0.0, 0.0)
        dx = self.trajectory[-1][0] - self.trajectory[-2][0]
        dy = self.trajectory[-1][1] - self.trajectory[-2][1]
        return (dx, dy)

    @property
    def speed(self) -> float:
        """Magnitude of velocity vector (pixels/frame)."""
        vx, vy = self.velocity
        return float(np.sqrt(vx**2 + vy**2))


class MultiObjectTracker:
    """ByteTrack-based multi-object tracker with trajectory management.

    Uses the Ultralytics YOLO model.track() API which integrates ByteTrack
    natively, then wraps the results into our Track dataclass for downstream
    consumption.

    Args:
        config: Dictionary with tracker configuration. Expected keys:
            - track_buffer (int):       Frames to keep lost tracks.
            - match_threshold (float):  IoU matching threshold.
            - min_track_length (int):   Minimum frames for valid track.
        detector_config: Detector configuration for the YOLO model.
    """

    def __init__(self, config: dict, detector_config: dict):
        self.config = config
        self.detector_config = detector_config
        self.track_buffer = config.get("track_buffer", 30)
        self.match_threshold = config.get("match_threshold", 0.8)
        self.min_track_length = config.get("min_track_length", 5)
        self.tracker_config = config.get("config", "configs/bytetrack.yaml")

        # Active tracks indexed by track_id
        self._tracks: dict[int, Track] = {}
        self._track_history: dict[int, list[np.ndarray]] = defaultdict(list)
        self._frame_count = 0

        # Latency tracking
        self._latencies: list[float] = []

        # Load model for tracking (same model, but we call .track())
        model_path = detector_config.get("model", "yolov8s.pt")
        self.model = YOLO(model_path)

        logger.info(
            f"Tracker initialized | buffer={self.track_buffer} | "
            f"match_thresh={self.match_threshold}"
        )

    # ── Public API ─────────────────────────────────────────

    def update(self, frame: np.ndarray) -> list[Track]:
        """Run detection + tracking on a single frame.

        Args:
            frame: BGR image as a NumPy array (H, W, 3).

        Returns:
            List of active Track objects in the current frame.
        """
        t0 = time.perf_counter()
        self._frame_count += 1

        # Run YOLO with integrated ByteTrack
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.detector_config.get("conf_threshold", 0.25),
            iou=self.detector_config.get("iou_threshold", 0.45),
            classes=self.detector_config.get("classes", [0]),
            imgsz=self.detector_config.get("img_size", 640),
            half=self.detector_config.get("half", True),
            max_det=self.detector_config.get("max_det", 100),
            device=self.detector_config.get("device", "cpu"),
            verbose=False,
        )

        # Parse tracking results
        current_tracks = self._parse_tracking_results(results)

        elapsed = time.perf_counter() - t0
        self._latencies.append(elapsed)

        return current_tracks

    def get_track(self, track_id: int) -> Track | None:
        """Retrieve a specific track by ID."""
        return self._tracks.get(track_id)

    def get_active_tracks(self) -> list[Track]:
        """Return all currently active tracks."""
        return [t for t in self._tracks.values() if t.is_active]

    def get_mature_tracks(self) -> list[Track]:
        """Return tracks that have existed for at least min_track_length frames."""
        return [
            t
            for t in self._tracks.values()
            if t.is_active and t.frame_count >= self.min_track_length
        ]

    def get_track_clip_frames(
        self,
        track_id: int,
        clip_length: int = 16,
    ) -> list[np.ndarray] | None:
        """Get the bounding box history for a track (for VideoMAE input).

        Returns None if the track doesn't have enough frames.
        """
        history = self._track_history.get(track_id)
        if history is None or len(history) < clip_length:
            return None
        return list(history[-clip_length:])

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return float(np.mean(self._latencies[-100:])) * 1000

    @property
    def num_active_tracks(self) -> int:
        return len([t for t in self._tracks.values() if t.is_active])

    @property
    def total_tracks(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """Reset all tracking state."""
        self._tracks.clear()
        self._track_history.clear()
        self._latencies.clear()
        self._frame_count = 0
        # Reload model to reset tracker state
        model_path = self.detector_config.get("model", "yolov8s.pt")
        self.model = YOLO(model_path)
        logger.info("Tracker reset")

    # ── Private helpers ────────────────────────────────────

    def _parse_tracking_results(self, results) -> list[Track]:
        """Convert Ultralytics tracking results to Track objects."""
        current_track_ids = set()
        current_tracks = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            # Check if tracking IDs are available
            if result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for bbox, conf, cls_id, tid in zip(boxes, confs, cls_ids, track_ids):
                tid = int(tid)
                current_track_ids.add(tid)
                class_name = result.names.get(cls_id, f"class_{cls_id}")

                if tid in self._tracks:
                    # Update existing track
                    track = self._tracks[tid]
                    track.bbox = bbox
                    track.confidence = float(conf)
                    track.frame_count += 1
                    track.is_active = True
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    track.trajectory.append((float(cx), float(cy)))

                    # Keep trajectory bounded
                    max_trail = 300
                    if len(track.trajectory) > max_trail:
                        track.trajectory = track.trajectory[-max_trail:]
                else:
                    # Create new track
                    track = Track(
                        track_id=tid,
                        bbox=bbox,
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=class_name,
                    )
                    self._tracks[tid] = track

                current_tracks.append(track)

        # Mark tracks not seen in this frame as inactive
        for tid, track in self._tracks.items():
            if tid not in current_track_ids:
                track.is_active = False

        return current_tracks
