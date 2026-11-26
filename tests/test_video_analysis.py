"""Tests for sampled video analysis helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.video_analysis import (
    VideoAnalyzer,
    build_sample_indices,
    count_nearby_person_pairs,
    infer_scene_context,
)


def test_build_sample_indices_covers_video_end():
    indices = build_sample_indices(
        total_frames=101,
        fps=10,
        sample_interval_seconds=2,
    )

    assert indices == [0, 20, 40, 60, 80, 100]


def test_build_sample_indices_respects_sample_limit():
    indices = build_sample_indices(
        total_frames=1000,
        fps=30,
        sample_interval_seconds=0.1,
        max_samples=8,
    )

    assert len(indices) == 8
    assert indices[0] == 0
    assert indices[-1] == 999


@pytest.mark.parametrize(
    ("centers", "threshold", "expected"),
    [
        ([(0, 0), (3, 4), (30, 30)], 5, 1),
        ([(0, 0), (30, 30)], 5, 0),
        ([(0, 0), (1, 1)], 0, 0),
    ],
)
def test_count_nearby_person_pairs(centers, threshold, expected):
    assert count_nearby_person_pairs(centers, threshold) == expected


def test_reset_tracking_state_clears_persistent_tracker():
    tracker = Mock()
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.detector = SimpleNamespace(
        predictor=SimpleNamespace(trackers=[tracker], vid_path=["previous-video"])
    )

    analyzer._reset_tracking_state()

    tracker.reset.assert_called_once_with()
    assert analyzer.detector.predictor.vid_path == [None]


def test_infer_scene_context_identifies_road_scene_from_vehicle_detections():
    context = infer_scene_context({"bus": 2, "car": 3, "person": 4})

    assert "road or traffic setting" in context
    assert "2 buses" in context
    assert "3 cars" in context
    assert "not a GPS location" in context


def test_infer_scene_context_does_not_guess_when_no_scene_cues_exist():
    context = infer_scene_context({"person": 3})

    assert "could not be inferred" in context


@pytest.mark.parametrize(
    ("total_frames", "fps", "interval", "max_samples"),
    [
        (0, 30, 1, 120),
        (100, 0, 1, 120),
        (100, 30, 0, 120),
        (100, 30, 1, 0),
    ],
)
def test_build_sample_indices_rejects_invalid_input(
    total_frames,
    fps,
    interval,
    max_samples,
):
    with pytest.raises(ValueError):
        build_sample_indices(total_frames, fps, interval, max_samples)
