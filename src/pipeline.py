"""End-to-end real-time video understanding pipeline.

Orchestrates object detection → tracking → temporal action classification
pipeline, managing frame flow, temporal buffers, visualization overlays,
and output recording.

Architecture:
    VideoSource → YOLOv8 (Detect) → ByteTrack (Track) → VideoMAE (Actions)
                                                              ↓
                                                Tracked frame evidence → Output

Typical usage:
    python -m src.pipeline --source video.mp4 --config configs/default.yaml

    # Or programmatically:
    pipeline = AnomalyDetectionPipeline(config)
    pipeline.run(source="video.mp4")
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from src.anomaly_classifier import VideoMAEActionClassifier
from src.heatmap import (
    draw_detections,
    draw_fps_overlay,
)
from src.tracker import MultiObjectTracker
from src.utils import FPSCounter, LatencyTracker, load_config, setup_logging
from src.video_io import VideoSource, VideoWriter


class AnomalyDetectionPipeline:
    """Real-time multi-object tracking and temporal action pipeline.

    Combines YOLOv8 object detection, ByteTrack multi-object tracking,
    and VideoMAE Kinetics-400 action classification.

    Action labels are not anomaly probabilities; anomaly claims require
    a separately trained and validated anomaly model.

    Args:
        config: Full pipeline configuration dictionary.
    """

    def __init__(self, config: dict):
        self.config = config

        # Setup logging
        log_level = config.get("pipeline", {}).get("log_level", "INFO")
        setup_logging(level=log_level)

        # Initialize components
        logger.info("=" * 60)
        logger.info("Initializing Anomaly Detection Pipeline")
        logger.info("=" * 60)

        self.tracker = MultiObjectTracker(
            config=config.get("tracker", {}),
            detector_config=config.get("detector", {}),
        )

        self.classifier = VideoMAEActionClassifier(config.get("anomaly", {}))

        # State
        self._action_predictions: dict[int, tuple[str, float]] = {}
        self._frame_buffers: dict[int, list[np.ndarray]] = defaultdict(list)
        self._fps_counter = FPSCounter(smoothing=0.05)
        self._latency_tracker = LatencyTracker()
        self._frame_idx = 0

        # Configuration
        self.clip_length = config.get("anomaly", {}).get("clip_length", 16)
        self.stride = config.get("anomaly", {}).get("stride", 4)
        self.show_display = config.get("pipeline", {}).get("show_display", True)
        self.save_output = config.get("pipeline", {}).get("save_output", True)
        self.output_path = config.get("pipeline", {}).get("output_path", "outputs/")
        self.viz_config = config.get("visualization", {})

        logger.success("Pipeline initialized successfully")

    # ── Public API ─────────────────────────────────────────

    def run(self, source: int | str | None = None) -> dict:
        """Run the full anomaly detection pipeline.

        Args:
            source: Video source override (webcam index, RTSP URL, or file path).
                    If None, uses the source from config.

        Returns:
            Dictionary with pipeline statistics and alert log.
        """
        if source is None:
            source = self.config.get("video", {}).get("source", 0)

        video_config = self.config.get("video", {})
        video_source = VideoSource(
            source=source,
            width=video_config.get("width", 1280),
            height=video_config.get("height", 720),
            fps=video_config.get("fps", 30),
            buffer_size=video_config.get("buffer_size", 64),
        )

        # Output writer
        writer = None
        if self.save_output:
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            output_file = str(Path(self.output_path) / "output.mp4")

        logger.info(f"Starting pipeline | source={source}")

        try:
            video_source.start()

            if self.save_output:
                w, h = video_source.resolution
                writer = VideoWriter(
                    output_path=output_file,
                    fps=video_source.actual_fps,
                    width=w,
                    height=h,
                )

            # Wait briefly for buffer to fill
            time.sleep(0.5)

            while (
                video_source.is_running
                or (
                    video_source.source_type == "file"
                    and video_source.has_buffered_frames
                )
            ):
                if video_source.source_type == "file":
                    frame = video_source.read_sequential()
                else:
                    frame = video_source.read()
                if frame is None:
                    time.sleep(0.001)
                    continue

                # Process frame
                annotated = self.process_frame(frame)

                # Display
                if self.show_display:
                    cv2.imshow("Anomaly Detection Pipeline", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # q or ESC
                        logger.info("User requested stop")
                        break

                # Save
                if writer is not None:
                    writer.write(annotated)

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        finally:
            video_source.stop()
            if writer is not None:
                writer.release()
            if self.show_display:
                cv2.destroyAllWindows()

        # Generate summary
        stats = self.generate_stats()
        logger.info(f"Pipeline finished | {stats['total_frames']} frames processed")
        return stats

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the full pipeline.

        Steps:
          1. Run YOLOv8 + ByteTrack detection and tracking.
          2. Buffer tracked object crops for temporal analysis.
          3. Run VideoMAE action classification on mature track clips.
          4. Generate visualizations and overlays.

        Args:
            frame: BGR input frame.

        Returns:
            Annotated frame with all overlays.
        """
        self._frame_idx += 1
        self._fps_counter.tick()

        # ── Stage 1+2: Detection + Tracking ────────────────
        t0 = time.perf_counter()
        tracks = self.tracker.update(frame)
        track_ms = (time.perf_counter() - t0) * 1000
        self._latency_tracker.record("detection+tracking", track_ms)

        # ── Buffer frames for tracked objects ──────────────
        for track in tracks:
            tid = track.track_id
            x1, y1, x2, y2 = track.bbox.astype(int)

            # Clamp to frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 - x1 > 10 and y2 - y1 > 10:
                crop = frame[y1:y2, x1:x2].copy()
                self._frame_buffers[tid].append(crop)

                # Keep buffer bounded
                max_buffer = self.clip_length * 2
                if len(self._frame_buffers[tid]) > max_buffer:
                    self._frame_buffers[tid] = self._frame_buffers[tid][-max_buffer:]

        # ── Stage 3: Temporal Action Classification ────────
        classifier_ms = 0.0
        if self._frame_idx % self.stride == 0:
            t1 = time.perf_counter()
            self._run_action_classification(tracks)
            classifier_ms = (time.perf_counter() - t1) * 1000
            self._latency_tracker.record("classification", classifier_ms)

        # ── Visualization ──────────────────────────────────
        annotated = frame.copy()

        # Detection boxes, tracks, and temporal action predictions
        annotated = draw_detections(
            annotated,
            tracks,
            action_predictions=self._action_predictions,
            config=self.viz_config,
        )

        # FPS overlay
        if self.viz_config.get("show_fps", True):
            annotated = draw_fps_overlay(
                annotated,
                fps=self._fps_counter.fps,
                detector_ms=track_ms,
                tracker_ms=0,  # Combined with detector
                classifier_ms=classifier_ms,
                num_tracks=self.tracker.num_active_tracks,
            )

        return annotated

    # ── Private helpers ────────────────────────────────────

    def _run_action_classification(self, tracks: list) -> None:
        """Classify actions for tracks with sufficient frame history."""
        for track in tracks:
            tid = track.track_id
            buffer = self._frame_buffers.get(tid, [])

            if len(buffer) < self.clip_length:
                continue

            # Only classify if track is mature enough
            if track.frame_count < self.tracker.min_track_length:
                continue

            # Get the latest clip
            clip = buffer[-self.clip_length:]

            try:
                result = self.classifier.classify_clip(clip)
                self._action_predictions[tid] = (
                    result.predicted_action,
                    result.confidence,
                )
            except Exception:
                logger.exception(f"Action classification failed for track {tid}")
                raise

    def get_live_stats(self) -> dict:
        """Return lightweight counters for interactive dashboard updates."""
        return {
            "total_frames": self._frame_idx,
            "avg_fps": self._fps_counter.fps,
            "active_tracks": self.tracker.num_active_tracks,
            "action_predictions": len(self._action_predictions),
        }

    def generate_stats(self) -> dict:
        """Generate pipeline run statistics."""
        latency_summary = self._latency_tracker.summary()

        stats = {
            "total_frames": self._frame_idx,
            "avg_fps": self._fps_counter.fps,
            "total_tracks": self.tracker.total_tracks,
            "action_predictions": [
                {
                    "track_id": track_id,
                    "action": action,
                    "confidence": confidence,
                }
                for track_id, (action, confidence) in self._action_predictions.items()
            ],
            "latency": latency_summary,
        }

        # Save stats to JSON
        stats_path = Path(self.output_path) / "pipeline_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(f"Stats saved to {stats_path}")
        return stats


# ── CLI Entry Point ────────────────────────────────────────

def main():
    """Command-line entry point for the anomaly detection pipeline."""
    parser = argparse.ArgumentParser(
        description="Real-Time Video Anomaly Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on webcam
  python -m src.pipeline --source 0

  # Run on video file
  python -m src.pipeline --source surveillance.mp4

  # Run on RTSP stream
  python -m src.pipeline --source rtsp://camera.local/stream

  # Run with custom config
  python -m src.pipeline --config configs/default.yaml --source video.mp4

  # Run without display (headless)
  python -m src.pipeline --source video.mp4 --no-display
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (0), RTSP URL, or file path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode (no OpenCV window)",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        default=True,
        help="Save annotated output video",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/",
        help="Directory for output files",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        default=True,
        help="Display FPS counter overlay",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.no_display:
        config.setdefault("pipeline", {})["show_display"] = False
    config.setdefault("pipeline", {})["save_output"] = args.save_output
    config.setdefault("pipeline", {})["output_path"] = args.output_path
    config.setdefault("visualization", {})["show_fps"] = args.show_fps

    # Parse source (convert "0" to int for webcam)
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    # Run pipeline
    pipeline = AnomalyDetectionPipeline(config)
    stats = pipeline.run(source=source)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total Frames:  {stats['total_frames']}")
    print(f"  Average FPS:   {stats['avg_fps']:.1f}")
    print(f"  Total Tracks:  {stats['total_tracks']}")
    print(f"  Action Predictions: {len(stats['action_predictions'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
