"""Video I/O module for multi-source video capture.

Provides a unified interface for reading video from:
  - Local webcam (device index)
  - RTSP/RTMP streams
  - Video files (MP4, AVI, etc.)

Supports asynchronous frame reading with a threaded buffer to prevent
frame drops in real-time scenarios.

Typical usage:
    source = VideoSource("rtsp://camera.local/stream1", buffer_size=64)
    source.start()
    while source.is_running:
        frame = source.read()
        if frame is not None:
            process(frame)
    source.stop()
"""

import os
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


class VideoSource:
    """Asynchronous video capture with threaded frame buffer.

    Reads frames in a background thread to prevent blocking the main
    inference pipeline, ensuring maximum throughput on live streams.

    Args:
        source:      Video source — int (webcam), str (RTSP URL or file path).
        width:       Desired frame width (may not be honored by all sources).
        height:      Desired frame height.
        fps:         Desired capture FPS.
        buffer_size: Maximum frames to buffer in memory.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        buffer_size: int = 64,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size

        self._cap: cv2.VideoCapture | None = None
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._frame_count = 0
        self._start_time = 0.0

        # Source metadata
        self._actual_fps = 0.0
        self._actual_width = 0
        self._actual_height = 0
        self._total_frames = 0  # 0 for live streams
        self._source_type = self._classify_source(source)

    # ── Public API ─────────────────────────────────────────

    def start(self) -> "VideoSource":
        """Open the video source and begin buffering frames.

        Returns:
            self (for method chaining).

        Raises:
            RuntimeError: If the video source cannot be opened.
        """
        if self._source_type == "webcam" and os.name == "nt":
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Cannot open video source: {self.source}")

        # Set properties (best-effort for webcams)
        if self._source_type == "webcam":
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Read actual properties
        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._actual_fps = self._cap.get(cv2.CAP_PROP_FPS) or self.fps
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self._source_type == "webcam" and (
            self._actual_width != self.width or self._actual_height != self.height
        ):
            logger.warning(
                f"Webcam requested {self.width}x{self.height}, but opened at "
                f"{self._actual_width}x{self._actual_height}"
            )

        logger.info(
            f"Video source opened | type={self._source_type} | "
            f"resolution={self._actual_width}x{self._actual_height} | "
            f"fps={self._actual_fps:.1f}"
        )

        # Start background thread for async reading
        self._running = True
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return self

    def read(self) -> np.ndarray | None:
        """Read the latest frame from the buffer.

        Returns the most recent frame (drops intermediate frames for
        real-time processing). Returns None if the buffer is empty.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1]

    def read_sequential(self) -> np.ndarray | None:
        """Read the oldest unprocessed frame (FIFO order).

        Use this for offline/evaluation mode where every frame matters.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer.popleft()

    def stop(self) -> None:
        """Stop the capture thread and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._cap is not None:
            self._cap.release()
        logger.info("Video source stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._actual_width, self._actual_height)

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def has_buffered_frames(self) -> bool:
        with self._lock:
            return bool(self._buffer)

    @property
    def elapsed_time(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def measured_fps(self) -> float:
        """Measured input FPS based on actual frames read."""
        elapsed = self.elapsed_time
        if elapsed <= 0:
            return 0.0
        return self._frame_count / elapsed

    @property
    def duration(self) -> float:
        """Total video duration in seconds (0 for live streams)."""
        if self._total_frames > 0 and self._actual_fps > 0:
            return self._total_frames / self._actual_fps
        return 0.0

    @property
    def progress(self) -> float:
        """Current playback progress (0-1) for video files."""
        if self._total_frames > 0:
            return min(self._frame_count / self._total_frames, 1.0)
        return 0.0

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    # ── Private helpers ────────────────────────────────────

    def _capture_loop(self) -> None:
        """Background thread: continuously read frames into the buffer."""
        while self._running:
            if self._source_type == "file":
                while self._running:
                    with self._lock:
                        buffer_full = len(self._buffer) >= self.buffer_size
                    if not buffer_full:
                        break
                    time.sleep(0.001)

            if self._cap is None or not self._cap.isOpened():
                self._running = False
                break

            ret, frame = self._cap.read()
            if not ret:
                if self._source_type == "file":
                    logger.info("End of video file reached")
                    self._running = False
                else:
                    logger.warning("Frame read failed, retrying...")
                    time.sleep(0.01)
                continue

            with self._lock:
                self._buffer.append(frame)
                self._frame_count += 1

    def _classify_source(self, source: int | str) -> str:
        """Classify the video source type."""
        if isinstance(source, int):
            return "webcam"
        source_str = str(source).lower()
        if source_str.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            return "stream"
        if Path(source_str).suffix in (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"):
            return "file"
        return "unknown"


class VideoWriter:
    """Simple video writer for saving annotated output.

    Args:
        output_path: Path to the output video file.
        fps:         Output video FPS.
        width:       Frame width.
        height:      Frame height.
        codec:       FourCC codec string (default: mp4v).
    """

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        width: int = 1280,
        height: int = 720,
        codec: str = "mp4v",
    ):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self._frame_count = 0
        logger.info(f"Video writer opened: {output_path} ({width}x{height} @ {fps}fps)")

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame."""
        self.writer.write(frame)
        self._frame_count += 1

    def release(self) -> None:
        """Release the writer."""
        self.writer.release()
        logger.info(f"Video saved: {self.output_path} ({self._frame_count} frames)")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    @property
    def frame_count(self) -> int:
        return self._frame_count
