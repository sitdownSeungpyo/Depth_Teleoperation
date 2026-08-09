from __future__ import annotations

import logging
import math

import pytest

from core.filter import (
    FilterAndLimiter,
    JointLimiterConfig,
    JointLimits,
    OneEuroFilter,
    OneEuroParams,
)


def test_one_euro_smooths_step() -> None:
    f = OneEuroFilter(OneEuroParams(min_cutoff=1.0, beta=0.0))
    out = []
    t = 0.0
    for _ in range(10):
        out.append(f.update(0.0, t))
        t += 1 / 30.0
    for _ in range(40):
        out.append(f.update(1.0, t))
        t += 1 / 30.0
    # The first step output should be < the raw target (lag), and outputs should
    # rise monotonically toward 1.0.
    assert out[10] < 1.0
    assert out[-1] > out[10]
    assert out[-1] <= 1.0 + 1e-9


def _limiter_cfg() -> JointLimiterConfig:
    return JointLimiterConfig(
        limits={
            "r_shoulder_pitch": JointLimits(soft_min=-1.0, soft_max=1.0, max_velocity=2.0),
            "r_elbow": JointLimits(soft_min=0.0, soft_max=2.0, max_velocity=2.0),
        },
        velocity_violation_factor=5.0,
    )


def test_out_of_limit_clamped_exactly() -> None:
    fl = FilterAndLimiter(one_euro=OneEuroParams(min_cutoff=1e6), limiter=_limiter_cfg())
    cmd = fl({"r_shoulder_pitch": 5.0, "r_elbow": 0.5}, timestamp=0.0, source_frame_ts=0.0)
    assert cmd.positions["r_shoulder_pitch"] == pytest.approx(1.0)


def test_nan_holds_last_command(caplog: pytest.LogCaptureFixture) -> None:
    fl = FilterAndLimiter(one_euro=OneEuroParams(), limiter=_limiter_cfg())
    cmd = fl({"r_shoulder_pitch": 0.5, "r_elbow": 0.5}, timestamp=0.0, source_frame_ts=0.0)
    with caplog.at_level(logging.WARNING):
        cmd2 = fl(
            {"r_shoulder_pitch": math.nan, "r_elbow": 0.5},
            timestamp=1 / 30.0,
            source_frame_ts=1 / 30.0,
        )
    # Same positions held, but re-stamped: returning the stored command object
    # would freeze the timestamp the publisher interpolates on (and alias it).
    assert cmd2 is not cmd
    assert cmd2.positions == cmd.positions
    assert cmd2.timestamp == pytest.approx(1 / 30.0)
    assert any("non-finite" in rec.message for rec in caplog.records)


def test_velocity_clamp_per_step() -> None:
    fl = FilterAndLimiter(one_euro=OneEuroParams(min_cutoff=1e6), limiter=_limiter_cfg())
    fl({"r_shoulder_pitch": 0.0, "r_elbow": 0.5}, timestamp=0.0, source_frame_ts=0.0)
    # Target between per-step max (0.0667) and 5x violation (0.333) — clamp, don't reject.
    cmd = fl({"r_shoulder_pitch": 0.2, "r_elbow": 0.5}, timestamp=1 / 30.0, source_frame_ts=1 / 30.0)
    # max_velocity=2.0, dt~0.0333 → max delta ~0.0667.
    assert cmd.positions["r_shoulder_pitch"] <= 0.0 + 2.0 / 30.0 + 1e-9


def test_velocity_violation_holds(caplog: pytest.LogCaptureFixture) -> None:
    fl = FilterAndLimiter(one_euro=OneEuroParams(min_cutoff=1e6), limiter=_limiter_cfg())
    fl({"r_shoulder_pitch": 0.0, "r_elbow": 0.5}, timestamp=0.0, source_frame_ts=0.0)
    with caplog.at_level(logging.WARNING):
        # 5x the max velocity * dt is ~0.333; jump 1.0 in one step is well over that.
        cmd = fl(
            {"r_shoulder_pitch": 1.0, "r_elbow": 0.5},
            timestamp=1 / 30.0,
            source_frame_ts=1 / 30.0,
        )
    assert any("velocity violation" in rec.message for rec in caplog.records)
    assert cmd.positions["r_shoulder_pitch"] == pytest.approx(0.0)


def test_velocity_violation_keeps_filter_state_aligned_with_output() -> None:
    """A rejected sample must not desync the OneEuro from what we emitted.

    Skipping the filter update left its internal x_prev at the pre-rejection
    value, so the first accepted sample afterwards smoothed from a baseline the
    joint had never been commanded to and pulled the output backwards.
    """
    limiter = _limiter_cfg()
    fl = FilterAndLimiter(one_euro=OneEuroParams(min_cutoff=0.5), limiter=limiter)
    dt = 1 / 30.0
    # Walk the joint up gently to 0.5 so filter state and output agree.
    t = 0.0
    for target in (0.0, 0.02, 0.04, 0.06):
        fl({"r_shoulder_pitch": target, "r_elbow": 0.5}, timestamp=t, source_frame_ts=t)
        t += dt
    held = fl({"r_shoulder_pitch": 5.0, "r_elbow": 0.5}, timestamp=t, source_frame_ts=t)
    t += dt
    rejected_at = held.positions["r_shoulder_pitch"]
    # Next frame resumes right where we were holding: the output must continue
    # forward from the held value, not snap back toward a stale internal state.
    resumed = fl(
        {"r_shoulder_pitch": rejected_at + 0.02, "r_elbow": 0.5},
        timestamp=t, source_frame_ts=t,
    )
    assert resumed.positions["r_shoulder_pitch"] >= rejected_at - 1e-9


def test_missing_joint_is_held_not_dropped() -> None:
    """The retargeter omits a whole arm when its geometry is degenerate. If the
    command's joint set shrank with it, the recovery frame would find no previous
    value — skipping BOTH the ±π unwrap and the velocity clamp for that joint."""
    fl = FilterAndLimiter(one_euro=OneEuroParams(min_cutoff=1e6), limiter=_limiter_cfg())
    fl({"r_shoulder_pitch": 0.5, "r_elbow": 0.5}, timestamp=0.0, source_frame_ts=0.0)
    # r_shoulder_pitch absent this frame (arm failed).
    cmd = fl({"r_elbow": 0.5}, timestamp=1 / 30.0, source_frame_ts=1 / 30.0)
    assert set(cmd.positions) == {"r_shoulder_pitch", "r_elbow"}
    assert cmd.positions["r_shoulder_pitch"] == pytest.approx(0.5)


def test_recovery_after_a_missing_joint_is_still_velocity_clamped() -> None:
    fl = FilterAndLimiter(one_euro=OneEuroParams(min_cutoff=1e6), limiter=_limiter_cfg())
    fl({"r_shoulder_pitch": 0.0, "r_elbow": 0.5}, timestamp=0.0, source_frame_ts=0.0)
    fl({"r_elbow": 0.5}, timestamp=1 / 30.0, source_frame_ts=1 / 30.0)  # arm dropped
    # Arm returns with a big jump; the clamp must still apply because the held
    # value kept a previous position on record.
    cmd = fl({"r_shoulder_pitch": 1.0, "r_elbow": 0.5}, timestamp=2 / 30.0, source_frame_ts=2 / 30.0)
    assert cmd.positions["r_shoulder_pitch"] < 0.5
