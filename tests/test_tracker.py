"""Unit tests for the MultiObjectTracker module."""

import numpy as np


class TestTrack:
    """Tests for the Track dataclass."""

    def test_track_creation(self):
        from src.tracker import Track

        track = Track(
            track_id=1,
            bbox=np.array([100, 200, 300, 400]),
            confidence=0.9,
            class_id=0,
            class_name="person",
        )

        assert track.track_id == 1
        assert track.confidence == 0.9
        assert track.is_active is True
        assert track.frame_count == 1

    def test_track_center(self):
        from src.tracker import Track

        track = Track(
            track_id=1,
            bbox=np.array([100, 200, 300, 400]),
            confidence=0.9,
            class_id=0,
            class_name="person",
        )

        cx, cy = track.center
        assert cx == 200.0
        assert cy == 300.0

    def test_track_dimensions(self):
        from src.tracker import Track

        track = Track(
            track_id=1,
            bbox=np.array([100, 200, 350, 500]),
            confidence=0.9,
            class_id=0,
            class_name="person",
        )

        assert track.width == 250
        assert track.height == 300
        assert track.area == 75000

    def test_track_velocity_single_point(self):
        from src.tracker import Track

        track = Track(
            track_id=1,
            bbox=np.array([100, 200, 300, 400]),
            confidence=0.9,
            class_id=0,
            class_name="person",
        )

        vx, vy = track.velocity
        assert vx == 0.0
        assert vy == 0.0

    def test_track_velocity_multi_point(self):
        from src.tracker import Track

        track = Track(
            track_id=1,
            bbox=np.array([100, 200, 300, 400]),
            confidence=0.9,
            class_id=0,
            class_name="person",
            trajectory=[(100, 200), (105, 210), (110, 220)],
        )

        vx, vy = track.velocity
        assert vx == 5.0
        assert vy == 10.0

    def test_track_speed(self):
        from src.tracker import Track

        track = Track(
            track_id=1,
            bbox=np.array([100, 200, 300, 400]),
            confidence=0.9,
            class_id=0,
            class_name="person",
            trajectory=[(0, 0), (3, 4)],
        )

        assert track.speed == 5.0  # 3-4-5 triangle
