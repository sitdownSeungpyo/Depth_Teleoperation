"""Unit tests for TemporalMedianFilter — lone-frame 3D spike rejection."""

from __future__ import annotations

import numpy as np

from tracker.depth_lift import TemporalMedianFilter


def test_lone_spike_rejected() -> None:
    f = TemporalMedianFilter(window=3)
    good = np.array([0.1, -0.2, 1.0])
    f({"w": good})
    f({"w": good})
    # one-frame outlier (e.g. background-bleed wrist) -> median ignores it
    out = f({"w": np.array([0.1, -0.2, 3.5])})
    np.testing.assert_allclose(out["w"], good, atol=1e-9)
    # back to good
    np.testing.assert_allclose(f({"w": good})["w"], good, atol=1e-9)


def test_genuine_motion_tracks_with_small_lag() -> None:
    f = TemporalMedianFilter(window=3)
    out = None
    for z in np.linspace(1.0, 0.5, 12):  # arm moving steadily toward camera
        out = f({"w": np.array([0.0, 0.0, z])})
    # follows the trend (median of last 3 monotone samples = middle one)
    assert out is not None
    assert 0.5 <= out["w"][2] <= 0.6


def test_zero_keypoint_passthrough_and_history_reset() -> None:
    f = TemporalMedianFilter(window=3)
    good = np.array([0.1, -0.2, 1.0])
    f({"w": good})
    f({"w": good})
    # rejected frame (zero) passes through untouched
    out = f({"w": np.zeros(3)})
    np.testing.assert_allclose(out["w"], np.zeros(3))
    # after recovery, it does NOT median against the stale pre-dropout value
    far = np.array([0.1, -0.2, 2.0])
    np.testing.assert_allclose(f({"w": far})["w"], far, atol=1e-9)


def test_window_one_is_passthrough() -> None:
    f = TemporalMedianFilter(window=1)
    for z in (1.0, 3.5, 1.0):
        out = f({"w": np.array([0.0, 0.0, z])})
        np.testing.assert_allclose(out["w"], [0.0, 0.0, z], atol=1e-9)
