"""IMU gravity verification tool (Step 6) — confirm the measured up vector on a
real D435i BEFORE enabling enable_imu in config.

Standalone: opens its own minimal RealSense pipeline (color + accel only, no pose
model) and runs the SAME path the tracker uses — accel sample → accel→color
extrinsics rotation → tracker.gravity.GravityEstimator. Prints the smoothed up
vector (color optical frame), the tilt vs the assumed level-up, the raw accel
magnitude, and accepted/rejected counts.

What to check:
  * Hold the camera level → up ≈ (0, -1, 0), tilt ≈ 0°.
  * Tilt the camera forward/down → the up vector's z/y change accordingly and
    tilt grows. If up points the OPPOSITE way (≈ (0, +1, 0) when level), set
    tracker.realsense.imu.axis_sign: -1.0 in config.
  * |accel| should sit near 9.81 when the camera is still.

Usage:
    python -m app.debug_imu --config .\\config\\ubp.yaml --duration 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=None, help="자동 종료(초)")
    args = parser.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 not installed", file=sys.stderr)
        return 2

    from core.config import load_config
    from tracker.gravity import GravityEstimator, tilt_degrees

    cfg = load_config(args.config)
    rs_cfg = cfg["tracker"]["realsense"]
    imu_cfg = rs_cfg.get("imu", {}) or {}
    fixed_up = rs_cfg.get("gravity_up")
    level_up = np.asarray(fixed_up, dtype=np.float64) if fixed_up else np.array([0.0, -1.0, 0.0])

    est = GravityEstimator(
        lpf_alpha=float(imu_cfg.get("lpf_alpha", 0.02)),
        warmup_frames=int(imu_cfg.get("warmup_frames", 10)),
        norm_tol=float(imu_cfg.get("norm_tol", 0.30)),
        axis_sign=float(imu_cfg.get("axis_sign", 1.0)),
    )

    color_res = tuple(rs_cfg["color_resolution"])
    fps = int(rs_cfg["fps"])
    pipe = rs.pipeline()
    conf = rs.config()
    accel_fps = int(imu_cfg.get("accel_fps", 200))  # D435i: 100/200/400 only
    conf.enable_stream(rs.stream.color, color_res[0], color_res[1], rs.format.bgr8, fps)
    conf.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_fps)
    try:
        profile = pipe.start(conf)
    except Exception as exc:  # noqa: BLE001
        print(f"failed to start pipeline with accel stream: {exc}", file=sys.stderr)
        print("→ D435i IMU(Motion Module) 가 안 잡힙니다. 연결/펌웨어 확인.", file=sys.stderr)
        return 2

    try:
        accel_profile = profile.get_stream(rs.stream.accel)
        color_profile = profile.get_stream(rs.stream.color)
        extr = accel_profile.get_extrinsics_to(color_profile)
        R = np.array(extr.rotation, dtype=np.float64).reshape(3, 3, order="F")
        print(f"accel->color rotation:\n{np.round(R, 3)}", flush=True)
        print(f"assumed level-up (config gravity_up) = {level_up.tolist()}", flush=True)
        print("hold still / tilt the camera to verify. Ctrl+C to quit.\n", flush=True)

        start = time.perf_counter()
        last_print = 0.0
        while True:
            now = time.perf_counter()
            if args.duration is not None and now - start > args.duration:
                break
            frames = pipe.wait_for_frames(timeout_ms=1000)
            accel = frames.first_or_default(rs.stream.accel)
            if not accel:
                continue
            m = accel.as_motion_frame().get_motion_data()
            a_sensor = np.array([m.x, m.y, m.z], dtype=np.float64)
            a_optical = R @ a_sensor
            est.update(a_optical)  # result read below via est.up
            if now - last_print >= 0.5:
                last_print = now
                acc, rej = est.stats
                cur = est.up
                if cur is None:
                    print(f"|accel|={np.linalg.norm(a_sensor):5.2f}  warming up... (acc={acc} rej={rej})",
                          flush=True)
                else:
                    tilt = tilt_degrees(cur, level_up)
                    ready = "READY" if est.ready else "warmup"
                    print(
                        f"up=[{cur[0]:+.3f} {cur[1]:+.3f} {cur[2]:+.3f}]  "
                        f"tilt={tilt:5.1f}deg  |accel|={np.linalg.norm(a_sensor):5.2f}  "
                        f"{ready}  acc={acc} rej={rej}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
