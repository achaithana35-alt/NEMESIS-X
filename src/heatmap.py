"""Anomaly heatmap and visualization overlays.

Generates visual explanations for detected anomalies, including:
  - Grad-CAM style saliency maps on anomalous regions.
  - Color-coded bounding boxes (green=normal, red=anomalous).
  - Anomaly score bars and tracking trajectories.
  - FPS and system status overlays.

These visualizations are critical for:
  1. Explainability — stakeholders can verify WHY the system flagged something.
  2. Debugging — engineers can visually inspect model attention.
  3. Demo quality — impressive real-time visualization for portfolio/interviews.
"""

import cv2
import numpy as np


def draw_detections(
    frame: np.ndarray,
    tracks: list,
    anomaly_scores: dict[int, float] | None = None,
    config: dict | None = None,
) -> np.ndarray:
    """Draw detection boxes, track IDs, and anomaly scores on a frame.

    Args:
        frame:          BGR image to annotate (modified in place).
        tracks:         List of Track objects from the tracker.
        anomaly_scores: Optional dict mapping track_id → anomaly score.
        config:         Visualization config dict.

    Returns:
        Annotated frame.
    """
    if config is None:
        config = {}

    normal_color = tuple(config.get("normal_color", [0, 255, 0]))
    anomaly_color = tuple(config.get("anomaly_color", [0, 0, 255]))
    anomaly_threshold = config.get("anomaly_threshold", 0.65)
    show_scores = config.get("show_scores", True)
    show_tracks = config.get("show_tracks", True)
    trail_length = config.get("trail_length", 30)

    annotated = frame.copy()

    for track in tracks:
        tid = track.track_id
        x1, y1, x2, y2 = track.bbox.astype(int)

        # Determine color based on anomaly score
        score = 0.0
        if anomaly_scores and tid in anomaly_scores:
            score = anomaly_scores[tid]

        if score >= anomaly_threshold:
            color = anomaly_color
            thickness = 3
        else:
            # Interpolate color from green to red based on score
            t = min(score / anomaly_threshold, 1.0)
            color = _interpolate_color(normal_color, anomaly_color, t)
            thickness = 2

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        label = f"ID:{tid} {track.class_name}"
        if show_scores and anomaly_scores and tid in anomaly_scores:
            label += f" | Anomaly: {score:.2f}"

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 5, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw anomaly score bar
        if show_scores and anomaly_scores and tid in anomaly_scores:
            _draw_score_bar(annotated, x1, y2, x2 - x1, score, anomaly_threshold)

        # Draw tracking trail
        if show_tracks and hasattr(track, "trajectory"):
            _draw_trail(annotated, track.trajectory, trail_length, color)

    return annotated


def generate_anomaly_heatmap(
    frame: np.ndarray,
    tracks: list,
    anomaly_scores: dict[int, float],
    alpha: float = 0.4,
) -> np.ndarray:
    """Generate a Grad-CAM style anomaly heatmap overlay.

    Creates a spatial heatmap where anomalous track regions glow red,
    providing visual explainability for detected anomalies.

    Args:
        frame:          BGR image.
        tracks:         List of Track objects.
        anomaly_scores: Dict mapping track_id → anomaly score.
        alpha:          Heatmap overlay transparency.

    Returns:
        Frame with heatmap overlay.
    """
    h, w = frame.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for track in tracks:
        tid = track.track_id
        if tid not in anomaly_scores:
            continue

        score = anomaly_scores[tid]
        x1, y1, x2, y2 = track.bbox.astype(int)

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Create a Gaussian blob centered on the bounding box
        roi_h = y2 - y1
        roi_w = x2 - x1
        cy, cx = roi_h // 2, roi_w // 2
        sigma_x = roi_w / 3
        sigma_y = roi_h / 3

        y_grid, x_grid = np.ogrid[:roi_h, :roi_w]
        gaussian = np.exp(
            -((x_grid - cx) ** 2 / (2 * sigma_x**2) +
              (y_grid - cy) ** 2 / (2 * sigma_y**2))
        )

        # Scale by anomaly score
        heatmap[y1:y2, x1:x2] = np.maximum(
            heatmap[y1:y2, x1:x2],
            gaussian.astype(np.float32) * score,
        )

    # Normalize and apply colormap
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)

    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay only where there's heat
    mask = heatmap > 10
    overlay = frame.copy()
    if mask.any():
        overlay[mask] = cv2.addWeighted(
            frame[mask], 1 - alpha, colored_heatmap[mask], alpha, 0
        )

    return overlay


def draw_fps_overlay(
    frame: np.ndarray,
    fps: float,
    detector_ms: float = 0.0,
    tracker_ms: float = 0.0,
    classifier_ms: float = 0.0,
    num_tracks: int = 0,
) -> np.ndarray:
    """Draw FPS counter and system metrics overlay.

    Args:
        frame:         BGR image.
        fps:           Current pipeline FPS.
        detector_ms:   Detector latency in ms.
        tracker_ms:    Tracker latency in ms.
        classifier_ms: Classifier latency in ms.
        num_tracks:    Number of active tracks.

    Returns:
        Frame with metrics overlay.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Semi-transparent background panel
    panel_h = 130
    panel_w = 280
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    # FPS (large, prominent)
    fps_color = (0, 255, 0) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        fps_color,
        2,
        cv2.LINE_AA,
    )

    # Component latencies
    metrics = [
        f"Detector:   {detector_ms:.1f}ms",
        f"Tracker:    {tracker_ms:.1f}ms",
        f"Classifier: {classifier_ms:.1f}ms",
        f"Tracks:     {num_tracks}",
    ]

    for i, text in enumerate(metrics):
        cv2.putText(
            annotated,
            text,
            (20, 75 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_alert_banner(
    frame: np.ndarray,
    message: str,
    severity: str = "warning",
) -> np.ndarray:
    """Draw an alert banner at the top of the frame.

    Args:
        frame:    BGR image.
        message:  Alert text.
        severity: "warning" (yellow) or "critical" (red).

    Returns:
        Frame with alert banner.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Banner colors
    if severity == "critical":
        bg_color = (0, 0, 200)
        text_color = (255, 255, 255)
        prefix = "⚠ ANOMALY DETECTED"
    else:
        bg_color = (0, 165, 255)
        text_color = (0, 0, 0)
        prefix = "⚡ ALERT"

    banner_h = 40
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), bg_color, -1)
    cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

    full_text = f"{prefix}: {message}"
    cv2.putText(
        annotated,
        full_text,
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
        cv2.LINE_AA,
    )

    return annotated


# ── Private helpers ────────────────────────────────────────

def _interpolate_color(
    color_a: tuple, color_b: tuple, t: float
) -> tuple[int, int, int]:
    """Linearly interpolate between two BGR colors."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(color_a, color_b))


def _draw_score_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    score: float,
    threshold: float,
) -> None:
    """Draw a small anomaly score bar below a bounding box."""
    bar_h = 6
    bar_y = y + 4

    # Background
    cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + bar_h), (60, 60, 60), -1)

    # Filled portion
    fill_w = int(width * score)
    if score >= threshold:
        fill_color = (0, 0, 255)  # Red
    else:
        fill_color = (0, 200, 0)  # Green
    cv2.rectangle(frame, (x, bar_y), (x + fill_w, bar_y + bar_h), fill_color, -1)

    # Threshold marker
    thresh_x = x + int(width * threshold)
    cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + bar_h), (255, 255, 255), 1)


def _draw_trail(
    frame: np.ndarray,
    trajectory: list[tuple[float, float]],
    max_length: int,
    color: tuple,
) -> None:
    """Draw a fading trajectory trail."""
    points = trajectory[-max_length:]
    if len(points) < 2:
        return

    for i in range(1, len(points)):
        alpha = i / len(points)  # Fade in
        pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
        pt2 = (int(points[i][0]), int(points[i][1]))

        thickness = max(1, int(alpha * 3))
        faded_color = tuple(int(c * alpha) for c in color)

        cv2.line(frame, pt1, pt2, faded_color, thickness, cv2.LINE_AA)
