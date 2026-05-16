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
    assert cmd2 is cmd
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
