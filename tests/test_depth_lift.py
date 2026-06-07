"""Camera-free unit tests for robust depth lifting (tracker/depth_lift.py)."""

from __future__ import annotations

import numpy as np

from tracker.depth_lift import (
    BoneLengthStabilizer,
    RobustDepthLifter,
    foreground_depth,
)

SCALE = 0.001  # raw uint16 units (mm) -> metres
MAXM = 4.0


def _uniform(depth_mm: int, shape=(48, 48)) -> np.ndarray:
    return np.full(shape, depth_mm, dtype=np.uint16)


def test_foreground_rejects_background_bleed() -> None:
    # A window straddling a limb silhouette: ~40% body (1.0 m), ~60% background
    # (3.0 m). Plain median lands on the background; foreground picks the body.
    win = np.zeros((7, 7), dtype=np.uint16)
    flat = win.ravel()
    flat[:20] = 1000   # 20 px body
    flat[20:] = 3000   # 29 px background
    win = flat.reshape(7, 7)
    valid = win[win > 0].astype(float) * SCALE
    assert np.median(valid) == 3.0  # plain median = background (the bug)
    fg = foreground_depth(win, 3, 3, window=7, foreground_percentile=40.0,
                          depth_scale=SCALE, depth_max_m=MAXM)
    assert abs(fg - 1.0) < 1e-6  # foreground recovers the body surface


def test_foreground_empty_returns_zero() -> None:
    assert foreground_depth(np.zeros((7, 7), np.uint16), 3, 3, 7, 40.0, SCALE, MAXM) == 0.0


def test_lifter_hole_fill_then_drop() -> None:
    lift = RobustDepthLifter(SCALE, MAXM, max_stale_frames=2)
    lift.begin_frame()
    assert lift.lift("w", _uniform(1000), 24, 24) == 1.0          # f1: valid
    lift.begin_frame()
    assert lift.lift("w", _uniform(0), 24, 24) == 1.0             # f2: hole -> reuse
    lift.begin_frame()
    assert lift.lift("w", _uniform(0), 24, 24) == 1.0             # f3: still fresh
    lift.begin_frame()
    assert lift.lift("w", _uniform(0), 24, 24) is None            # f4: too stale -> drop


def test_lifter_rejects_single_frame_spike() -> None:
    lift = RobustDepthLifter(SCALE, MAXM, max_jump_m=0.25)
    lift.begin_frame(); assert lift.lift("w", _uniform(1000), 24, 24) == 1.0
    lift.begin_frame(); assert lift.lift("w", _uniform(1000), 24, 24) == 1.0
    lift.begin_frame()
    # 1.0 -> 2.5 m is a 1.5 m jump; lone spike must be rejected (hold prev).
    assert lift.lift("w", _uniform(2500), 24, 24) == 1.0
    lift.begin_frame()
    assert lift.lift("w", _uniform(1000), 24, 24) == 1.0


def test_lifter_commits_sustained_jump() -> None:
    lift = RobustDepthLifter(SCALE, MAXM, max_jump_m=0.25)
    lift.begin_frame(); assert lift.lift("w", _uniform(1000), 24, 24) == 1.0
    lift.begin_frame()
    assert lift.lift("w", _uniform(1500), 24, 24) == 1.0   # jump -> pending, hold
    lift.begin_frame()
    assert lift.lift("w", _uniform(1500), 24, 24) == 1.5   # confirmed -> commit


def test_bone_stabilizer_reduces_length_variance() -> None:
    stab = BoneLengthStabilizer([("s", "e")], history=60)
    shoulder = np.array([0.0, 0.0, 2.0])  # metres from camera (never the origin)
    rng_lengths = [1.0, 1.3, 0.7, 1.2, 0.8, 1.1, 0.9, 1.25, 0.75, 1.05]
    in_lens, out_lens = [], []
    for L in rng_lengths:
        kp = {"s": shoulder, "e": np.array([L, 0.0, 0.0])}
        out = stab({k: v.copy() for k, v in kp.items()})
        in_lens.append(L)
        out_lens.append(float(np.linalg.norm(out["e"] - out["s"])))
    assert np.var(out_lens) < np.var(in_lens)  # jitter suppressed


def test_bone_stabilizer_ratio_clamp() -> None:
    stab = BoneLengthStabilizer([("s", "e")], ratio_min=0.5, ratio_max=2.0)
    shoulder = np.array([0.0, 0.0, 2.0])
    for L in [1.0, 1.0, 1.0]:  # establish ref ~1.0
        stab({"s": shoulder, "e": shoulder + np.array([L, 0.0, 0.0])})
    # Degenerate tiny measurement; ref/measured would be huge -> clamp to 2x.
    out = stab({"s": shoulder, "e": shoulder + np.array([0.01, 0.0, 0.0])})
    assert float(np.linalg.norm(out["e"] - out["s"])) <= 0.01 * 2.0 + 1e-9


def test_bone_stabilizer_chain_connectivity() -> None:
    stab = BoneLengthStabilizer([("s", "e"), ("e", "w")])
    kp = {
        "s": np.array([0.0, 0.0, 2.0]),
        "e": np.array([1.0, 0.0, 2.0]),
        "w": np.array([1.0, 1.0, 2.0]),
    }
    out = stab({k: v.copy() for k, v in kp.items()})
    # wrist is placed relative to the *stabilized* elbow (chain stays connected).
    ew = float(np.linalg.norm(out["w"] - out["e"]))
    assert ew > 0.0 and np.isfinite(ew)
