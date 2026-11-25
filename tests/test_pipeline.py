"""Unit tests for the pipeline utilities and helper modules."""

import time

import numpy as np


class TestFPSCounter:
    """Tests for the FPS counter utility."""

    def test_fps_counter_initial(self):
        from src.utils import FPSCounter

        counter = FPSCounter()
        assert counter.frame_count == 0

    def test_fps_counter_tick(self):
        from src.utils import FPSCounter

        counter = FPSCounter(smoothing=1.0)  # No smoothing for test
        time.sleep(0.01)
        fps = counter.tick()

        assert fps > 0
        assert counter.frame_count == 1

    def test_fps_counter_reset(self):
        from src.utils import FPSCounter

        counter = FPSCounter()
        counter.tick()
        counter.tick()

        counter.reset()
        assert counter.frame_count == 0


class TestLatencyTracker:
    """Tests for the LatencyTracker utility."""

    def test_record_and_avg(self):
        from src.utils import LatencyTracker

        tracker = LatencyTracker()
        tracker.record("detector", 10.0)
        tracker.record("detector", 20.0)
        tracker.record("detector", 30.0)

        assert tracker.get_avg("detector") == 20.0

    def test_percentiles(self):
        from src.utils import LatencyTracker

        tracker = LatencyTracker()
        for i in range(100):
            tracker.record("test", float(i))

        p95 = tracker.get_p95("test")
        assert p95 >= 94.0

    def test_unknown_component(self):
        from src.utils import LatencyTracker

        tracker = LatencyTracker()
        assert tracker.get_avg("unknown") == 0.0

    def test_summary(self):
        from src.utils import LatencyTracker

        tracker = LatencyTracker()
        tracker.record("a", 10.0)
        tracker.record("b", 20.0)

        summary = tracker.summary()
        assert "a" in summary
        assert "b" in summary
        assert summary["a"]["avg_ms"] == 10.0


class TestConfigLoading:
    """Tests for configuration loading utilities."""

    def test_load_missing_config(self):
        from src.utils import load_config

        config = load_config("nonexistent.yaml")
        assert config == {}

    def test_merge_configs(self):
        from src.utils import merge_configs

        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10}, "e": 5}

        merged = merge_configs(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 10
        assert merged["b"]["d"] == 3
        assert merged["e"] == 5


class TestTrackColors:
    """Tests for track color utility."""

    def test_get_track_color(self):
        from src.utils import get_track_color

        color = get_track_color(0)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    def test_color_consistency(self):
        from src.utils import get_track_color

        # Same ID should always return same color
        assert get_track_color(5) == get_track_color(5)

    def test_color_wrapping(self):
        from src.utils import get_track_color, TRACK_COLORS

        n = len(TRACK_COLORS)
        assert get_track_color(0) == get_track_color(n)


class TestHeatmapVisualization:
    """Tests for the heatmap module."""

    def test_draw_detections_empty(self):
        from src.heatmap import draw_detections

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_detections(frame, tracks=[], anomaly_scores={})

        assert result.shape == frame.shape

    def test_generate_heatmap_empty(self):
        from src.heatmap import generate_anomaly_heatmap

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = generate_anomaly_heatmap(frame, tracks=[], anomaly_scores={})

        assert result.shape == frame.shape

    def test_draw_fps_overlay(self):
        from src.heatmap import draw_fps_overlay

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_fps_overlay(frame, fps=30.0, detector_ms=5.0)

        assert result.shape == frame.shape

    def test_draw_alert_banner(self):
        from src.heatmap import draw_alert_banner

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_alert_banner(frame, "Test alert", severity="critical")

        assert result.shape == frame.shape


class TestComputeMetrics:
    """Tests for the metrics computation module."""

    def test_roc_auc_perfect(self):
        from evaluation.compute_metrics import compute_roc_auc

        gt = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        auc_val, fpr, tpr, thresholds = compute_roc_auc(gt, pred)
        assert auc_val == 1.0

    def test_roc_auc_random(self):
        from evaluation.compute_metrics import compute_roc_auc

        np.random.seed(42)
        gt = np.random.randint(0, 2, 1000)
        pred = np.random.rand(1000)

        auc_val, _, _, _ = compute_roc_auc(gt, pred)
        assert 0.4 < auc_val < 0.6  # Should be near 0.5

    def test_compute_all_metrics(self):
        from evaluation.compute_metrics import compute_all_metrics

        gt = np.array([0, 0, 1, 1, 0, 1])
        pred = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7])

        metrics = compute_all_metrics(gt, pred)

        assert "roc_auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

    def test_eer_computation(self):
        from evaluation.compute_metrics import compute_eer

        gt = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        eer, threshold = compute_eer(gt, pred)
        assert 0 <= eer <= 1
