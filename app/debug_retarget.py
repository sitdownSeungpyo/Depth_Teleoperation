"""실시간 retargeter 디버깅 — filter / safety / publisher 우회.

라이브 카메라에서 retargeter가 실제로 출력하는 raw 각도를 콘솔에 출력한다.
elbow나 다른 관절이 "안 움직이는" 것 같을 때 원인 파악용.

출력 예:
    [0.10s] r_elbow=  0.05  l_elbow=  0.02 | r_sh_pitch= -0.10 r_sh_roll=  1.45 | conf=0.62 align=OK
    [0.13s] r_elbow=  1.42  l_elbow=  0.05 | r_sh_pitch= -0.08 r_sh_roll=  1.48 | conf=0.65 align=OK

confidence 별 표시:
    "align=NOFRAME" — aligner 실패 (필수 keypoint 누락)
    "align=ZEROKP"  — elbow/wrist가 zero-vector (depth/visibility 거부)
    "align=SING"    — singular geometry
    "align=OK"      — 정상

사용법:
    python -m app.debug_retarget --config .\\config\\ubp.yaml --duration 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.aligner import AlignmentError, align_to_torso
from core.filter import KeypointSmoother, OneEuroParams
from core.retarget import (
    Calibration,
    CalibrationCollector,
    RobotGeometry,
    SingularConfigurationError,
    retarget_full_upper_body,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("./config/ubp.yaml"))
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="KeypointSmoother 우회 (raw retargeter 출력만)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from app.main import _build_hand_backend
    from tracker.realsense_tracker import RealSenseTracker

    rs_cfg = cfg["tracker"]["realsense"]
    tracker: Any = RealSenseTracker(
        color_resolution=tuple(rs_cfg["color_resolution"]),
        depth_resolution=tuple(rs_cfg["depth_resolution"]),
        fps=int(rs_cfg["fps"]),
        depth_max_m=float(rs_cfg["depth_max_m"]),
        min_visibility=float(cfg["tracker"]["pose"]["min_visibility"]),
        model_asset_path=rs_cfg.get("model_asset_path"),
        use_world_landmarks=bool(rs_cfg.get("use_world_landmarks", False)),
        hand_backend=_build_hand_backend(rs_cfg),
    )

    robot_cfg = cfg["retarget"]["robot"]
    robot = RobotGeometry(
        upper_arm_length=float(robot_cfg["upper_arm_length"]),
        lower_arm_length=float(robot_cfg["lower_arm_length"]),
        shoulder_offset=tuple(robot_cfg["shoulder_offset"]),
    )
    target_calib = int(cfg["retarget"].get("calibration_frames", 30))
    fallback = float(cfg["retarget"]["fixed_arm_length"])
    collector = CalibrationCollector(target_frames=target_calib)
    calibration: Calibration | None = None

    # KeypointSmoother — config의 filter.keypoint_smoother 섹션과 동일 파라미터.
    kp_cfg = cfg.get("filter", {}).get("keypoint_smoother", {})
    smoother = None if args.no_smooth else KeypointSmoother(
        OneEuroParams(
            min_cutoff=float(kp_cfg.get("min_cutoff", 0.5)),
            beta=float(kp_cfg.get("beta", 0.005)),
            d_cutoff=float(kp_cfg.get("d_cutoff", 1.0)),
        )
    )
    smooth_note = "(SMOOTHED)" if smoother is not None else "(RAW, no smoother)"
    print(f"keypoint smoother: {smooth_note}", flush=True)

    # gravity_up: config 의 tracker.realsense.gravity_up. None 이면 기존 body-relative.
    gravity_up_cfg = cfg.get("tracker", {}).get("realsense", {}).get("gravity_up")
    gravity_up: np.ndarray | None = (
        np.asarray(gravity_up_cfg, dtype=np.float64) if gravity_up_cfg else None
    )
    align_note = (
        f"gravity-aligned (v={gravity_up.tolist()})"
        if gravity_up is not None
        else "body-relative (v=head-mid_hip)"
    )
    print(f"aligner mode: {align_note}", flush=True)

    print("loading camera + MediaPipe (~10s)...", flush=True)
    tracker.start()
    print("recording (live)", flush=True)
    print(f"{'time':>7}  {'r_elb':>7}  {'l_elb':>7} | {'r_sp':>7}  {'r_sr':>7} | "
          f"{'r_shy':>7}  {'r_wy':>7}  {'r_wp':>7} | conf  status")
    start = time.perf_counter()
    last_log = start
    axis_dump_done = False
    try:
        for frame in tracker.stream():
            now = time.perf_counter()
            if now - start > args.duration:
                break

            # 첫 frame에서 head/hip y 방향 진단 — MediaPipe +y convention 결정.
            if not axis_dump_done:
                head = frame.keypoints.get("head")
                lh = frame.keypoints.get("left_hip")
                rh = frame.keypoints.get("right_hip")
                if head is not None and lh is not None and rh is not None:
                    mid_hip_y = (float(lh[1]) + float(rh[1])) / 2
                    head_y = float(head[1])
                    direction = "+y=UP" if head_y > mid_hip_y else "+y=DOWN"
                    print(
                        f"[AXIS] head.y={head_y:+.3f}, mid_hip.y={mid_hip_y:+.3f} → "
                        f"MediaPipe coord {direction} "
                        f"→ gravity_up = (0, {'+' if direction=='+y=UP' else '-'}1, 0)",
                        flush=True,
                    )
                    axis_dump_done = True

            if now - last_log < 0.1:  # 10 Hz print
                continue
            last_log = now

            conf = frame.mean_confidence()
            # 각 keypoint의 raw camera-frame Z 좌표 보존 (depth)
            r_elb_kp = frame.keypoints.get("right_elbow")
            r_elb_z = f"{r_elb_kp[2]:+.3f}" if r_elb_kp is not None else "  N/A   "
            r_elb_conf = frame.confidence.get("right_elbow", 0.0)

            if smoother is not None:
                frame = smoother.smooth(frame)
            try:
                aligned = align_to_torso(frame, gravity_up=gravity_up)
            except AlignmentError:
                print(f"[{now-start:6.2f}s]   ---      --- |   ---      --- | "
                      f"{r_elb_z:>9} | {conf:.2f} align=NOFRAME (r_elb_conf={r_elb_conf:.2f})",
                      flush=True)
                continue

            if calibration is None:
                collector.push(aligned)
                if collector.ready():
                    try:
                        calibration = collector.finalise()
                    except SingularConfigurationError:
                        calibration = Calibration(operator_arm_length=fallback)
                    print(f"[{now-start:6.2f}s] CALIBRATED operator_arm_length="
                          f"{calibration.operator_arm_length:.3f} m")
                else:
                    continue

            try:
                angles = retarget_full_upper_body(aligned, robot, calibration)
                status = "OK"
            except SingularConfigurationError as exc:
                msg = str(exc)
                if "zero vector" in msg:
                    status = "ZEROKP"
                else:
                    status = "SING"
                print(f"[{now-start:6.2f}s]   ---      --- |   ---      --- | "
                      f"{r_elb_z:>9} | {conf:.2f} align=OK retarget={status} ({msg})",
                      flush=True)
                continue

            r_elb = angles.get("r_elbow", 0.0)
            l_elb = angles.get("l_elbow", 0.0)
            r_sp = angles.get("r_shoulder_pitch", 0.0)
            r_sr = angles.get("r_shoulder_roll", 0.0)
            r_shy = angles.get("r_shoulder_yaw", 0.0)
            r_wy = angles.get("r_wrist_yaw", 0.0)
            r_wp = angles.get("r_wrist_pitch", 0.0)
            hand_ok = "hand" if "right_hand_middle_mcp" in frame.keypoints else "----"
            print(
                f"[{now-start:6.2f}s] {r_elb:+7.3f}  {l_elb:+7.3f} | "
                f"{r_sp:+7.3f}  {r_sr:+7.3f} | "
                f"{r_shy:+7.3f}  {r_wy:+7.3f}  {r_wp:+7.3f} | {conf:.2f} {status}/{hand_ok}",
                flush=True,
            )
    finally:
        tracker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
