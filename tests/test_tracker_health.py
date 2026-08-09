"""Tracker liveness reporting.

A capture producer that dies keeps ``latest()`` returning its final frame, so a
frozen robot is indistinguishable from a motionless operator. These tests pin the
API that makes the difference observable.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from tracker.base import TrackerHealth, health_of
from tracker.mock_tracker import MockTracker


class _LegacyTracker:
    """A tracker written before health() existed."""

    def latest(self) -> None:
        return None


def test_health_of_falls_back_for_trackers_without_the_api() -> None:
    hp = health_of(_LegacyTracker())
    assert hp.alive and hp.running
    assert hp.frame_age_s == 0.0  # unknown age must not read as stale


def test_is_stale_compares_against_threshold() -> None:
    assert TrackerHealth(running=True, alive=True, frame_age_s=0.6).is_stale(0.5)
    assert not TrackerHealth(running=True, alive=True, frame_age_s=0.4).is_stale(0.5)


def test_describe_distinguishes_the_failure_modes() -> None:
    assert "DEAD" in TrackerHealth(True, False, math.inf, error="boom").describe()
    assert TrackerHealth(False, False, 0.0).describe() == "STOPPED"
    assert TrackerHealth(True, True, math.inf).describe() == "NO FRAMES YET"
    assert "ms" in TrackerHealth(True, True, 0.02).describe()


def test_mock_tracker_reports_no_frames_before_any_are_consumed(
    tpose_jsonl: Path,
) -> None:
    tracker = MockTracker(jsonl_path=tpose_jsonl, loop=False)
    assert not tracker.health().running          # not started yet
    tracker.start()
    hp = tracker.health()
    assert hp.running and hp.alive
    assert not math.isfinite(hp.frame_age_s)     # nothing consumed yet
    tracker.stop()


def test_mock_tracker_frame_age_tracks_consumption(tpose_jsonl: Path) -> None:
    tracker = MockTracker(jsonl_path=tpose_jsonl, loop=False)
    tracker.start()
    try:
        stream = tracker.stream()
        next(stream)
        assert tracker.health().frame_age_s < 0.5
        # Stop consuming: the age must grow, which is exactly the signal a caller
        # needs to tell "held still" from "stopped feeding us".
        time.sleep(0.15)
        aged = tracker.health().frame_age_s
        assert aged >= 0.1
        assert tracker.health().is_stale(0.05)
    finally:
        tracker.stop()
    assert not tracker.health().running
