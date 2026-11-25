"""Utility functions and helpers.

Provides reusable utilities for the anomaly detection pipeline:
  - FPS counter with exponential moving average
  - Configuration loading and merging
  - Logging setup
  - Timer context manager
  - Color palette utilities
"""

import time
import yaml
from pathlib import Path
from contextlib import contextmanager

import numpy as np
from loguru import logger


# ── FPS Counter ────────────────────────────────────────────

class FPSCounter:
    """Exponential moving average FPS counter.

    Provides a smooth, jitter-free FPS measurement by applying an EMA
    filter to frame-to-frame durations.

    Args:
        smoothing: EMA smoothing factor (0-1). Higher = more responsive.
    """

    def __init__(self, smoothing: float = 0.1):
        self.smoothing = smoothing
        self._prev_time = time.perf_counter()
        self._avg_duration = 1.0 / 30  # Initialize at 30 FPS
        self._frame_count = 0

    def tick(self) -> float:
        """Record a frame and return the current smoothed FPS."""
        now = time.perf_counter()
        duration = now - self._prev_time
        self._prev_time = now
        self._frame_count += 1

        if self._frame_count == 1:
            self._avg_duration = duration
        else:
            self._avg_duration = (
                self.smoothing * duration + (1 - self.smoothing) * self._avg_duration
            )

        return self.fps

    @property
    def fps(self) -> float:
        """Current smoothed FPS."""
        if self._avg_duration <= 0:
            return 0.0
        return 1.0 / self._avg_duration

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def reset(self) -> None:
        self._prev_time = time.perf_counter()
        self._avg_duration = 1.0 / 30
        self._frame_count = 0


# ── Configuration ──────────────────────────────────────────

def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config or {}


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge two configuration dictionaries.

    Values in `override` take precedence over `base`.
    """
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


# ── Timer ──────────────────────────────────────────────────

@contextmanager
def timer(name: str = "Block"):
    """Context manager for timing code blocks.

    Usage:
        with timer("Detection"):
            result = model.detect(frame)
        # Logs: "Detection took 12.3ms"
    """
    t0 = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug(f"{name} took {elapsed:.1f}ms")


class LatencyTracker:
    """Tracks component-wise latencies with rolling statistics.

    Args:
        window_size: Number of recent measurements to keep.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._measurements: dict[str, list[float]] = {}

    def record(self, component: str, latency_ms: float) -> None:
        """Record a latency measurement."""
        if component not in self._measurements:
            self._measurements[component] = []
        measurements = self._measurements[component]
        measurements.append(latency_ms)
        if len(measurements) > self.window_size:
            self._measurements[component] = measurements[-self.window_size:]

    def get_avg(self, component: str) -> float:
        """Get average latency for a component."""
        measurements = self._measurements.get(component, [])
        return float(np.mean(measurements)) if measurements else 0.0

    def get_p95(self, component: str) -> float:
        """Get P95 latency for a component."""
        measurements = self._measurements.get(component, [])
        return float(np.percentile(measurements, 95)) if measurements else 0.0

    def get_p99(self, component: str) -> float:
        """Get P99 latency for a component."""
        measurements = self._measurements.get(component, [])
        return float(np.percentile(measurements, 99)) if measurements else 0.0

    def summary(self) -> dict[str, dict[str, float]]:
        """Get full latency summary for all components."""
        return {
            comp: {
                "avg_ms": self.get_avg(comp),
                "p95_ms": self.get_p95(comp),
                "p99_ms": self.get_p99(comp),
                "count": len(self._measurements.get(comp, [])),
            }
            for comp in self._measurements
        }


# ── Logging ────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure loguru logging.

    Args:
        level:    Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file.
    """
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_file:
        logger.add(
            log_file,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
        )

    logger.info(f"Logging initialized at level={level}")


# ── Color Palette ──────────────────────────────────────────

# Distinct colors for tracking visualization (BGR)
TRACK_COLORS = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 255),   # Orange
    (255, 128, 0),   # Azure
    (0, 128, 255),   # Orange
    (128, 255, 0),   # Spring Green
]


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Get a consistent color for a track ID."""
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]
