"""Unit tests for SegmentConsistencyGate — the frontal-reach z-jump fix.

The gate must: (1) pass a frontal reach untouched (length constant, only the
wrist's z/direction change), (2) HOLD the last good direction when a bad depth
read balloons the forearm length (the "arm flips backward then returns" bug),
and (3) recover when a genuinely new length persists for confirm_frames.
"""

from __future__ import annotations

import numpy as np

from tracker.depth_lift import SegmentConsistencyGate

SEG = (("right_elbow", "right_wrist"),)
ELBOW = np.array([0.0, 0.0, 1.0])
FOREARM = 0.30  # constant true forearm length (m)


def _kp(wrist: np.ndarray) -> dict[str, np.ndarray]:
    return {"right_elbow": ELBOW.copy(), "right_wrist": wrist.copy()}


def _frontal_wrist(t: float) -> np.ndarray:
    """Forearm of constant length swinging toward the camera (z decreases) as t
    goes 0→1; only direction changes, length stays FOREARM."""
    ang = np.radians(80 * t)  # 0 = straight down-ish, grows toward camera
    d = np.array([0.0, -np.cos(ang), -np.sin(ang)])
    return ELBOW + FOREARM * d


def test_frontal_reach_passes_through() -> None:
    gate = SegmentConsistencyGate(SEG, min_history=3, ratio_tol=0.35)
    for i in range(30):
        w = _frontal_wrist(i / 29.0)
        out = gate(_kp(w))
        np.testing.assert_allclose(out["right_wrist"], w, atol=1e-9)
    assert gate.rejected == 0  # constant length -> never gated


def test_background_spike_is_held() -> None:
    gate = SegmentConsistencyGate(SEG, min_history=5, ratio_tol=0.35)
    good = ELBOW + FOREARM * np.array([0.0, -0.2, -0.98])
    for _ in range(8):
        gate(_kp(good))
    last_good = gate(_kp(good))["right_wrist"].copy()
    # Background bleed: wrist z jumps far back -> forearm length ~3x -> reject+hold.
    bad = ELBOW + np.array([0.0, -0.2, -0.98]) * (FOREARM * 3.0)
    out = gate(_kp(bad))
    np.testing.assert_allclose(out["right_wrist"], last_good, atol=1e-9)
    assert gate.rejected >= 1
    # When depth recovers, it tracks again.
    out2 = gate(_kp(good))
    np.testing.assert_allclose(out2["right_wrist"], good, atol=1e-9)


def test_lone_spike_then_recovers_to_truth_not_spike() -> None:
    gate = SegmentConsistencyGate(SEG, min_history=5, ratio_tol=0.35, confirm_frames=3)
    good = ELBOW + FOREARM * np.array([0.0, -0.3, -0.95])
    for _ in range(8):
        gate(_kp(good))
    bad = ELBOW + np.array([0.0, -0.3, -0.95]) * (FOREARM * 2.5)
    held = gate(_kp(bad))["right_wrist"]
    # held at last good, NOT at the spike
    assert float(np.linalg.norm(held - ELBOW)) < FOREARM * 1.5


def test_genuine_scale_change_recovers_after_confirm() -> None:
    # Operator steps closer: forearm genuinely measures longer and STAYS there.
    gate = SegmentConsistencyGate(SEG, min_history=5, ratio_tol=0.30, confirm_frames=3)
    short = ELBOW + 0.25 * np.array([0.0, -0.4, -0.92])
    for _ in range(8):
        gate(_kp(short))
    longer = ELBOW + 0.40 * np.array([0.0, -0.4, -0.92])  # +60% length, persistent
    outs = [gate(_kp(longer))["right_wrist"] for _ in range(5)]
    # First couple frames held (near old length), then confirmed -> follows new.
    assert float(np.linalg.norm(outs[0] - ELBOW)) < 0.34   # held short-ish
    np.testing.assert_allclose(outs[-1], longer, atol=1e-9)  # recovered


def test_zero_keypoint_skipped() -> None:
    gate = SegmentConsistencyGate(SEG, min_history=2)
    out = gate({"right_elbow": ELBOW.copy(), "right_wrist": np.zeros(3)})
    np.testing.assert_allclose(out["right_wrist"], np.zeros(3))  # untouched
