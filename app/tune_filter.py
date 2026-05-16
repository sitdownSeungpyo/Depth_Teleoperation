"""Iterate One Euro filter parameters on a recorded JSONL and report jitter / lag.

Workflow: record a real session with `app.record`, then sweep filter parameters here
to find a (min_cutoff, beta) pair that suppresses jitter while keeping lag low.

    python -m app.tune_filter --config .\\config\\default.yaml \\
        --replay .\\recordings\\session.jsonl \\
        --joints r_elbow,r_shoulder_pitch,r_shoulder_roll \\
        --params 0.5,0.0 1.0,0.05 1.5,0.1

For each (min_cutoff, beta), prints:
- jitter_rms_rad: standard deviation of (filtered[t] - filtered[t-1]) — lower = smoother.
- lag_rad: max |filtered[t] - raw[t]| during ramp regions — lower = more responsive.

Recommendation: pick the lowest jitter_rms that still keeps lag below ~0.05 rad.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.aligner import AlignmentError, align_to_torso
from core.filter import (
    FilterAndLimiter,
    JointLimits,
    JointLimiterConfig,
    OneEuroParams,
)
from core.retarget import (
    Calibration,
    CalibrationCollector,
    RobotGeometry,
    SingularConfigurationError,
    retarget_full_upper_body,
)
from core.types import SkeletonFrame


def _load_recording(path: Path) -> list[SkeletonFrame]:
    frames: list[SkeletonFrame] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            frames.append(
                SkeletonFrame(
                    timestamp=float(d["timestamp"]),
                    keypoints={k: np.asarray(v, dtype=np.float64) for k, v in d["keypoints"].items()},
                    confidence={k: float(v) for k, v in d["confidence"].items()},
                )
            )
    return frames


def _retarget_all(frames: list[SkeletonFrame], cfg: dict[str, Any]) -> tuple[list[float], dict[str, list[float]]]:
    robot_cfg = cfg["retarget"]["robot"]
    robot = RobotGeometry(
        upper_arm_length=float(robot_cfg["upper_arm_length"]),
        lower_arm_length=float(robot_cfg["lower_arm_length"]),
        shoulder_offset=tuple(robot_cfg["shoulder_offset"]),
    )
    target = int(cfg["retarget"].get("calibration_frames", 30))
    fallback = float(cfg["retarget"]["fixed_arm_length"])
    collector = CalibrationCollector(target_frames=target)
    calibration: Calibration | None = None

    timestamps: list[float] = []
    raw_per_joint: dict[str, list[float]] = {}

    for frame in frames:
        try:
            aligned = align_to_torso(frame)
        except AlignmentError:
            continue
        if calibration is None:
            collector.push(aligned)
            if collector.ready():
                try:
                    calibration = collector.finalise()
                except SingularConfigurationError:
                    calibration = Calibration(operator_arm_length=fallback)
            else:
                continue
        try:
            angles = retarget_full_upper_body(aligned, robot, calibration)
        except SingularConfigurationError:
            continue
        timestamps.append(frame.timestamp)
        for j, v in angles.items():
            raw_per_joint.setdefault(j, []).append(v)
    return timestamps, raw_per_joint


def _filter_trace(
    timestamps: list[float],
    raw_per_joint: dict[str, list[float]],
    cfg: dict[str, Any],
    one_euro: OneEuroParams,
) -> dict[str, list[float]]:
    f_cfg = cfg["filter"]
    factor = float(f_cfg["joint_limits_factor"])
    max_v = float(f_cfg["max_velocity_rad_s"])
    limits = {
        joint: JointLimits(soft_min=lo * factor, soft_max=hi * factor, max_velocity=max_v)
        for joint, (lo, hi) in f_cfg["mechanical_limits"].items()
    }
    fl = FilterAndLimiter(
        one_euro=one_euro,
        limiter=JointLimiterConfig(
            limits=limits,
            velocity_violation_factor=float(f_cfg.get("velocity_violation_factor", 5.0)),
        ),
    )

    filtered: dict[str, list[float]] = {j: [] for j in raw_per_joint}
    for i, ts in enumerate(timestamps):
        positions = {j: raw_per_joint[j][i] for j in raw_per_joint}
        cmd = fl(positions, timestamp=ts, source_frame_ts=ts)
        for j in filtered:
            filtered[j].append(cmd.positions.get(j, positions[j]))
    return filtered


def _stats(raw: list[float], filt: list[float]) -> tuple[float, float]:
    arr_filt = np.asarray(filt)
    diff = np.diff(arr_filt)
    jitter = float(np.std(diff)) if diff.size else 0.0
    lag = float(np.max(np.abs(np.asarray(raw) - arr_filt))) if raw else 0.0
    return jitter, lag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--joints",
        type=str,
        default="r_elbow,r_shoulder_pitch,r_shoulder_roll",
        help="comma-separated joint names",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=["0.5,0.0", "1.0,0.05", "1.5,0.1", "2.0,0.2"],
        help="space-separated (min_cutoff,beta) pairs",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    frames = _load_recording(args.replay)
    print(f"loaded {len(frames)} frames from {args.replay}")
    timestamps, raw_per_joint = _retarget_all(frames, cfg)
    print(f"retargeted {len(timestamps)} frames into joint trajectories")

    joint_names = [j.strip() for j in args.joints.split(",") if j.strip()]
    print(f"\n{'min_cutoff':>10} {'beta':>6} | " + " | ".join(f"{j[:18]:>18}" for j in joint_names))
    print("-" * (20 + len(joint_names) * 21))
    for spec in args.params:
        try:
            mc, beta = (float(x) for x in spec.split(","))
        except ValueError:
            print(f"  invalid param spec: {spec!r}")
            continue
        one_euro = OneEuroParams(min_cutoff=mc, beta=beta)
        filtered = _filter_trace(timestamps, raw_per_joint, cfg, one_euro)
        cells = []
        for j in joint_names:
            if j not in filtered:
                cells.append(f"{'(missing)':>18}")
                continue
            jitter, lag = _stats(raw_per_joint[j], filtered[j])
            cells.append(f"j={jitter*1000:5.1f}mr l={lag*1000:5.1f}mr")
        print(f"{mc:>10.2f} {beta:>6.2f} | " + " | ".join(f"{c:>18}" for c in cells))

    print(
        "\njitter_rms (mr=milliradians of frame-to-frame change in filtered output) — lower is smoother."
    )
    print(
        "lag (mr=worst-case |raw - filtered|) — lower is more responsive."
    )


if __name__ == "__main__":
    main()
