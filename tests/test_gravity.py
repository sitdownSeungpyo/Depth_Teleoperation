"""Unit tests for IMU gravity estimation (tracker.gravity) — no camera needed.

Accel samples are given already in the optical frame (+x right, +y down,
+z forward), the same convention the tracker feeds after applying the
accel→color extrinsics. A stationary accel points UP (specific force = −g), so a
level camera reads ≈ (0, −9.8, 0) and up = (0, −1, 0).
"""

from __future__ import annotations

import numpy as np

from tracker.gravity import _G, GravityEstimator, tilt_degrees


def test_level_camera_gives_minus_y_up() -> None:
    est = GravityEstimator(lpf_alpha=0.5, warmup_frames=3)
    out = None
    for _ in range(5):
        out = est.update(np.array([0.0, -_G, 0.0]))
    assert out is not None
    np.testing.assert_allclose(out, [0.0, -1.0, 0.0], atol=1e-6)
    assert est.ready


def test_warmup_returns_none_until_enough_samples() -> None:
    est = GravityEstimator(lpf_alpha=1.0, warmup_frames=4)
    a = np.array([0.0, -_G, 0.0])
    assert est.update(a) is None
    assert est.update(a) is None
    assert est.update(a) is None
    assert est.update(a) is not None  # 4th accepted sample -> ready


def test_tilted_camera_tracks_gravity_direction() -> None:
    # Camera pitched so gravity has a +z component; up should mirror it.
    est = GravityEstimator(lpf_alpha=1.0, warmup_frames=1)
    g = np.array([0.0, -_G * np.cos(np.radians(20)), -_G * np.sin(np.radians(20))])
    up = est.update(g)
    assert up is not None
    expected = np.array([0.0, -np.cos(np.radians(20)), -np.sin(np.radians(20))])
    np.testing.assert_allclose(up, expected, atol=1e-6)
    assert tilt_degrees(up, [0.0, -1.0, 0.0]) == 20.0 or abs(
        tilt_degrees(up, [0.0, -1.0, 0.0]) - 20.0) < 1e-4


def test_moving_camera_samples_rejected() -> None:
    # A large transient acceleration (|a| far from g) must not move the estimate.
    est = GravityEstimator(lpf_alpha=0.5, warmup_frames=1, norm_tol=0.3)
    good = np.array([0.0, -_G, 0.0])
    for _ in range(3):
        est.update(good)
    locked = est.up.copy()
    moving = np.array([8.0, -_G, 7.0])  # |a| ~ 14.5 (ratio ~0.48 > 0.3) -> rejected
    out = est.update(moving)
    np.testing.assert_allclose(out, locked, atol=1e-9)
    accepted, rejected = est.stats
    assert rejected >= 1


def test_zero_sample_held() -> None:
    est = GravityEstimator(lpf_alpha=0.5, warmup_frames=1)
    est.update(np.array([0.0, -_G, 0.0]))
    held = est.up.copy()
    out = est.update(np.zeros(3))  # dropout
    np.testing.assert_allclose(out, held, atol=1e-9)


def test_low_pass_smooths_noise() -> None:
    # Noisy direction should converge near true up with heavy smoothing, and the
    # smoothed vector is always unit length.
    rng = np.random.default_rng(0)
    est = GravityEstimator(lpf_alpha=0.05, warmup_frames=5)
    out = None
    for _ in range(200):
        noise = rng.normal(0, 0.4, 3)
        out = est.update(np.array([0.0, -_G, 0.0]) + noise)
    assert out is not None
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-9
    assert tilt_degrees(out, [0.0, -1.0, 0.0]) < 8.0


def test_axis_sign_flips_up() -> None:
    est = GravityEstimator(lpf_alpha=1.0, warmup_frames=1, axis_sign=-1.0)
    out = est.update(np.array([0.0, -_G, 0.0]))
    np.testing.assert_allclose(out, [0.0, 1.0, 0.0], atol=1e-6)
