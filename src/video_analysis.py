"""Sampled, evidence-based video analysis for the dashboard."""

from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np


DEFAULT_SEMANTIC_MODEL = "openai/clip-vit-base-patch32"


def build_sample_indices(
    total_frames: int,
    fps: float,
    sample_interval_seconds: float,
    max_samples: int = 120,
) -> list[int]:
    """Return bounded frame indices spanning the full video."""
    if total_frames <= 0:
        raise ValueError("The video contains no readable frames.")
    if fps <= 0 or sample_interval_seconds <= 0 or max_samples <= 0:
        raise ValueError("FPS, sample interval, and sample limit must be positive.")

    step = max(1, round(fps * sample_interval_seconds))
    indices = list(range(0, total_frames, step))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)

    if len(indices) > max_samples:
        positions = np.linspace(0, len(indices) - 1, max_samples, dtype=int)
        indices = [indices[position] for position in positions]

    return indices


def count_nearby_person_pairs(
    centers: list[tuple[float, float]],
    distance_threshold: float,
) -> int:
    """Count person-center pairs within a pixel-distance threshold."""
    if distance_threshold <= 0:
        return 0

    return sum(
        np.hypot(
            centers[left][0] - centers[right][0],
            centers[left][1] - centers[right][1],
        )
        <= distance_threshold
        for left in range(len(centers))
        for right in range(left + 1, len(centers))
    )


def infer_scene_context(object_counts: dict[str, int]) -> str:
    """Infer a broad scene type from detected objects, not a physical address."""
    road_objects = {
        name: count
        for name, count in object_counts.items()
        if name in {"bicycle", "bus", "car", "motorcycle", "truck"}
        and count > 0
    }
    if road_objects:
        plural_names = {
            "bicycle": "bicycles",
            "bus": "buses",
            "car": "cars",
            "motorcycle": "motorcycles",
            "truck": "trucks",
        }
        evidence = ", ".join(
            f"{count} {name if count == 1 else plural_names[name]}"
            for name, count in sorted(road_objects.items())
        )
        return (
            f"Likely road or traffic setting (visual cues: {evidence}). "
            "This is not a GPS location or verified address."
        )

    indoor_objects = {
        name: count
        for name, count in object_counts.items()
        if name
        in {
            "bed",
            "dining table",
            "microwave",
            "oven",
            "refrigerator",
            "sofa",
            "toilet",
            "tv",
        }
        and count > 0
    }
    if indoor_objects:
        evidence = ", ".join(sorted(indoor_objects))
        return (
            f"Likely indoor setting (visual cues: {evidence}). "
            "This is not a GPS location or verified address."
        )

    return (
        "Scene type could not be inferred from detected objects; "
        "no GPS location or address is available."
    )


class ClipPromptMatcher:
    """Rank user-provided event descriptions against frames with CLIP."""

    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_MODEL,
        device: str = "auto",
    ):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def score(self, frame: np.ndarray, prompts: list[str]) -> dict[str, float]:
        from PIL import Image
        import torch

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(
            text=prompts,
            images=Image.fromarray(rgb_frame),
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits_per_image[0]
            scores = torch.softmax(logits, dim=0).cpu().tolist()

        return {prompt: float(score) for prompt, score in zip(prompts, scores)}


class VideoAnalyzer:
    """Analyze sampled evidence and render a full-length annotated MP4."""

    def __init__(
        self,
        config: dict,
        use_semantic_model: bool = False,
        semantic_model_name: str = DEFAULT_SEMANTIC_MODEL,
        detector=None,
        semantic_matcher=None,
    ):
        from ultralytics import YOLO

        self.config = config
        project_dir = Path(__file__).resolve().parent.parent
        detector_config = config.get("detector", {})
        model_path = Path(detector_config.get("model", "yolov8s.pt"))
        if not model_path.is_absolute():
            model_path = project_dir / model_path

        self.device = detector_config.get("device", "auto")
        if self.device == "auto":
            import torch

            self.device = "0" if torch.cuda.is_available() else "cpu"

        if detector is not None:
            self.detector = detector
        else:
            if not model_path.is_file():
                raise FileNotFoundError(f"Detector weights not found: {model_path}")
            self.detector = YOLO(str(model_path))

        tracker_config = config.get("tracker", {}).get(
            "config", "configs/bytetrack.yaml"
        )
        self.tracker_config = Path(tracker_config)
        if not self.tracker_config.is_absolute():
            self.tracker_config = project_dir / self.tracker_config

        self.detector_config = detector_config
        self.semantic_matcher = semantic_matcher
        if use_semantic_model and self.semantic_matcher is None:
            self.semantic_matcher = ClipPromptMatcher(
                model_name=semantic_model_name,
                device=detector_config.get("device", "auto"),
            )

    def _reset_tracking_state(self) -> None:
        predictor = getattr(self.detector, "predictor", None)
        if predictor is None:
            return

        trackers = getattr(predictor, "trackers", [])
        for tracker in trackers:
            tracker.reset()
        predictor.vid_path = [None] * len(trackers)

    def _track_frame(self, frame: np.ndarray):
        return self.detector.track(
            source=frame,
            persist=True,
            tracker=str(self.tracker_config),
            conf=self.detector_config.get("conf_threshold", 0.25),
            iou=self.detector_config.get("iou_threshold", 0.45),
            classes=self.detector_config.get("classes"),
            imgsz=self.detector_config.get("img_size", 640),
            max_det=self.detector_config.get("max_det", 100),
            device=self.device,
            verbose=False,
        )[0]

    def _create_annotated_video(
        self,
        video_path: Path,
        fps: float,
        width: int,
        height: int,
        progress_callback=None,
    ) -> tuple[bytes, int, int]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not reopen video file: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._reset_tracking_state()

        with tempfile.TemporaryDirectory(prefix="video-analysis-") as directory:
            output_path = Path(directory) / "annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.CAP_MSMF,
                fourcc,
                fps,
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (width, height),
                )
            if not writer.isOpened():
                writer.release()
                capture.release()
                raise RuntimeError(
                    "Could not initialize an H.264 MP4 encoder. "
                    "Install a compatible system video encoder and retry."
                )

            processed_frames = 0
            unique_track_ids = set()
            progress_interval = max(1, total_frames // 100)
            last_progress_frame = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break

                    timestamp = processed_frames / fps
                    result = self._track_frame(frame)
                    annotated = result.plot()
                    cv2.putText(
                        annotated,
                        f"{timestamp:.2f}s",
                        (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    writer.write(annotated)

                    boxes = result.boxes
                    if boxes is not None and boxes.id is not None:
                        unique_track_ids.update(
                            boxes.id.cpu().numpy().astype(int).tolist()
                        )

                    processed_frames += 1
                    if progress_callback is not None and (
                        processed_frames % progress_interval == 0
                        or processed_frames == total_frames
                    ):
                        progress_callback(
                            processed_frames,
                            total_frames,
                            timestamp,
                        )
                        last_progress_frame = processed_frames
            finally:
                writer.release()
                capture.release()

            if processed_frames == 0:
                raise RuntimeError("No video frames could be annotated.")
            if (
                progress_callback is not None
                and last_progress_frame != processed_frames
            ):
                progress_callback(
                    processed_frames,
                    total_frames,
                    (processed_frames - 1) / fps,
                )
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("The annotated MP4 could not be written.")

            return (
                output_path.read_bytes(),
                processed_frames,
                len(unique_track_ids),
            )

    def analyze(
        self,
        video_path: str | Path,
        sample_interval_seconds: float = 1.0,
        semantic_prompts: list[str] | None = None,
        max_samples: int = 120,
        progress_callback=None,
    ) -> dict:
        """Return sampled measurements and a per-frame annotated video."""
        video_path = Path(video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video file: {video_path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            indices = build_sample_indices(
                total_frames,
                fps,
                sample_interval_seconds,
                max_samples,
            )
            prompts = [
                prompt.strip()
                for prompt in semantic_prompts or []
                if prompt.strip()
            ]
            score_history: dict[str, list[float]] = defaultdict(list)
            rows = []
            preview_frames = []
            previous_centers: dict[int, tuple[float, float, float]] = {}
            unique_track_ids = set()
            frame_diagonal = float(np.hypot(width, height))
            self._reset_tracking_state()
            preview_positions = set(
                np.linspace(
                    0,
                    len(indices) - 1,
                    min(8, len(indices)),
                    dtype=int,
                ).tolist()
            )

            for position, frame_index in enumerate(indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue

                timestamp = frame_index / fps
                result = self._track_frame(frame)

                boxes = result.boxes
                class_counts = Counter()
                person_centers = []
                current_centers = {}
                object_names = []
                confidences = []
                tracked_count = 0
                if boxes is not None and len(boxes):
                    coordinates = boxes.xyxy.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    confidences = boxes.conf.cpu().numpy().astype(float).tolist()
                    track_ids = (
                        boxes.id.cpu().numpy().astype(int).tolist()
                        if boxes.id is not None
                        else [-1] * len(class_ids)
                    )

                    for coordinates_for_box, class_id, confidence, track_id in zip(
                        coordinates,
                        class_ids,
                        confidences,
                        track_ids,
                    ):
                        class_name = result.names.get(
                            int(class_id),
                            f"class_{class_id}",
                        )
                        class_counts[class_name] += 1
                        object_names.append(class_name)
                        x1, y1, x2, y2 = coordinates_for_box
                        center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        if class_name == "person":
                            person_centers.append(center)
                        if track_id >= 0:
                            tracked_count += 1
                            unique_track_ids.add(track_id)
                            current_centers[track_id] = center

                speeds = []
                for track_id, center in current_centers.items():
                    if track_id in previous_centers:
                        previous_x, previous_y, previous_time = previous_centers[
                            track_id
                        ]
                        elapsed = max(timestamp - previous_time, 1e-6)
                        speeds.append(
                            float(np.hypot(center[0] - previous_x, center[1] - previous_y))
                            / elapsed
                        )
                    previous_centers[track_id] = (
                        center[0],
                        center[1],
                        timestamp,
                    )

                prompt_scores = {}
                top_prompt = ""
                top_prompt_score = None
                if prompts and self.semantic_matcher is not None:
                    raw_scores = self.semantic_matcher.score(frame, prompts)
                    for prompt in prompts:
                        score_history[prompt].append(raw_scores[prompt])
                        prompt_scores[prompt] = float(
                            np.mean(score_history[prompt][-3:])
                        )
                    top_prompt = max(prompt_scores, key=prompt_scores.get)
                    top_prompt_score = prompt_scores[top_prompt]

                row = {
                    "frame": int(frame_index),
                    "timestamp_s": round(timestamp, 3),
                    "object_count": len(object_names),
                    "tracked_objects": tracked_count,
                    "unique_tracks_seen": len(unique_track_ids),
                    "people": class_counts.get("person", 0),
                    "nearby_person_pairs": count_nearby_person_pairs(
                        person_centers,
                        frame_diagonal * 0.06,
                    ),
                    "max_track_speed_px_s": round(max(speeds, default=0.0), 2),
                    "mean_detection_confidence": round(
                        float(np.mean(confidences)) if confidences else 0.0,
                        3,
                    ),
                    "object_counts": dict(class_counts),
                    "object_classes": ", ".join(
                        f"{name}: {count}"
                        for name, count in sorted(class_counts.items())
                    ),
                    "top_prompt_match": top_prompt,
                    "prompt_match_score": (
                        round(top_prompt_score, 4)
                        if top_prompt_score is not None
                        else None
                    ),
                    "prompt_scores": prompt_scores,
                }
                rows.append(row)

                annotated = result.plot()
                cv2.putText(
                    annotated,
                    f"{timestamp:.2f}s",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if top_prompt:
                    cv2.putText(
                        annotated,
                        f"Prompt match: {top_prompt[:60]}",
                        (12, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                if position in preview_positions:
                    preview_frames.append(
                        {
                            "frame": frame_index,
                            "timestamp_s": timestamp,
                            "image": annotated,
                        }
                    )

            sampled_class_totals = Counter()
            for row in rows:
                sampled_class_totals.update(row["object_counts"])
            scene_context = infer_scene_context(sampled_class_totals)
            (
                annotated_video,
                processed_frames,
                full_video_unique_tracks,
            ) = self._create_annotated_video(
                video_path,
                fps,
                width,
                height,
                progress_callback=progress_callback,
            )

            duration_seconds = total_frames / fps
            return {
                "video_name": video_path.name,
                "metadata": {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "total_frames": total_frames,
                    "duration_seconds": duration_seconds,
                    "processed_frames": processed_frames,
                    "sampled_frames": len(rows),
                    "sampling_interval_seconds": sample_interval_seconds,
                },
                "rows": rows,
                "preview_frames": preview_frames,
                "annotated_video": annotated_video,
                "analysis_version": 2,
                "scene_context": scene_context,
                "unique_tracks": full_video_unique_tracks,
                "generated_at": time.time(),
            }
        finally:
            capture.release()
