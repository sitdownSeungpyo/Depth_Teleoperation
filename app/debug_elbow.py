"""elbow 추종 진단 — 왜 robot elbow 가 안 움직이는지 데이터로 가른다.

retargeter 의 elbow 각도(theta_5)는 elbow 키포인트를 직접 쓰지 않고
shoulder→wrist 직선거리 c 와 캘리브레이션 팔길이 L 로만 law-of-cosines 계산한다
(core/retarget.py). 따라서 elbow 가 안 움직이는 원인은 둘 중 하나:

  (A) c/L 이 항상 ≈1 → c 가 a+b 로 clamp → theta_5≈0 고정 (L 캘리브레이션 과소/과대)
  (B) 키포인트로 잰 *실제* 팔꿈치 각도는 변하는데 theta_5 만 안 변함 → 공식/clamp 문제

이 스크립트는 매 프레임 다음을 같이 출력/기록한다:
  L       = calibration.operator_arm_length
  c       = |wrist - shoulder|            (theta_5 의 유일한 입력)
  c/L     = clamp 여부 판단용 (>=1 이면 straight 로 clamp)
  flexKP  = 키포인트 직접 측정 팔꿈치 굴곡각 (upper·forearm), 0=편 상태
  th5     = law-of-cosines elbow (현재 robot 에 나가는 raw 값)
  elbOut  = rest-offset 뺀 최종 r_elbow 명령

사용법:
    python -m app.debug_elbow --config .\\config\\ubp.yaml --duration 20
    python -m app.debug_elbow --config .\\config\\ubp.yaml --dump .\\recordings\\elbow.csv

권장 동작(기록 중): 캘리브 끝나면 → 오른팔 앞으로 쭉 펴기 → 팔꿈치 90도 굽히기 →
다시 펴기 2~3회 반복. 그 csv 를 보면 (A)인지 (B)인지 확정된다.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.aligner import AlignmentError, align_to_torso, resolve_gravity_up
from core.filter import KeypointSmoother, OneEuroParams
from core.retarget import (
    Calibration,
    CalibrationCollector,
    RobotGeometry,
    SingularConfigurationError,
    retarget_full_upper_body,
)


def _flex_from_keypoints(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """키포인트로 직접 잰 팔꿈치 굴곡각(rad). 0=쭉 편 상태, 증가=굽힘."""
    upper = elbow - shoulder
    fore = wrist - elbow
    nu, nf = float(np.linalg.norm(upper)), float(np.linalg.norm(fore))
    if nu < 1e-6 or nf < 1e-6:
        return float("nan")
    cosang = float(np.dot(upper, fore) / (nu * nf))
    return float(np.arccos(np.clip(cosang, -1.0, 1.0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dump", type=Path, default=None, help="per-frame CSV 경로")
    parser.add_argument("--side", choices=("right", "left"), default="right")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from core.config import load_config
    cfg = load_config(args.config)
    from app.main import _build_tracker

    tracker: Any = _build_tracker(cfg, "realsense", replay=None)

    kp_cfg = cfg["filter"].get("keypoint_smoother", {})
    smoother = KeypointSmoother(OneEuroParams(
        min_cutoff=float(kp_cfg.get("min_cutoff", 0.5)),
        beta=float(kp_cfg.get("beta", 0.005)),
        d_cutoff=float(kp_cfg.get("d_cutoff", 1.0)),
    ))
    gravity_up_cfg = cfg.get("tracker", {}).get("realsense", {}).get("gravity_up")
    gravity_up = np.asarray(gravity_up_cfg, dtype=np.float64) if gravity_up_cfg else None
    decouple = bool(cfg.get("retarget", {}).get("decouple_pitch_elbow", False))

    robot_cfg = cfg["retarget"]["robot"]
    robot = RobotGeometry(
        upper_arm_length=float(robot_cfg["upper_arm_length"]),
        lower_arm_length=float(robot_cfg["lower_arm_length"]),
        shoulder_offset=tuple(robot_cfg["shoulder_offset"]),
    )
    fallback = float(cfg["retarget"]["fixed_arm_length"])
    collector = CalibrationCollector(target_frames=int(cfg["retarget"].get("calibration_frames", 30)))
    calibration: Calibration | None = None
    rest_cfg = cfg["retarget"].get("rest_offsets") or {}

    side = args.side
    prefix = "r" if side == "right" else "l"

    dump = None
    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        dump = args.dump.open("w", encoding="utf-8")
        dump.write("t,status,conf,L,c,c_over_L,flexKP_deg,th5_deg,elbOut_deg,sp_deg,"
                   "c_raw_over_L,flexKP_raw_deg,th5_raw_deg\n")

    print("body_backend loading (~10s for hmr2)...", flush=True)
    tracker.start()
    print(f"{'t':>6} {'st':>5} {'conf':>5} | {'c/L':>5} {'flexKP':>7} {'th5':>6} {'elbOut':>7} | "
          f"{'c/L_RAW':>7} {'flxRAW':>7} {'th5RAW':>7} (deg)  [RAW=평활 전]", flush=True)

    start = time.perf_counter()
    last_ts = 0.0
    last_log = start
    try:
        while True:
            now = time.perf_counter()
            if now - start > args.duration:
                break
            frame = tracker.latest()
            if frame is None or frame.timestamp == last_ts:
                time.sleep(0.002)
                continue
            last_ts = frame.timestamp

            sframe = smoother.smooth(frame)
            try:
                aligned = align_to_torso(sframe, gravity_up=resolve_gravity_up(tracker, gravity_up))
            except AlignmentError:
                continue

            if calibration is None:
                collector.push(aligned)
                if collector.ready():
                    try:
                        calibration = collector.finalise(robot=robot, decouple_pitch_elbow=decouple)
                        if rest_cfg:
                            calibration.rest_offsets = {
                                **calibration.rest_offsets,
                                **{k: float(v) for k, v in rest_cfg.items()},
                            }
                        print(f"[CALIBRATED] operator_arm_length L={calibration.operator_arm_length:.3f} m  "
                              f"rest_offsets={ {k: round(v,3) for k,v in calibration.rest_offsets.items()} }",
                              flush=True)
                    except SingularConfigurationError:
                        calibration = Calibration(operator_arm_length=fallback)
                        print(f"[CALIB fallback] L={fallback:.3f}", flush=True)
                continue

            L = calibration.operator_arm_length

            # L bound as a default arg: the closure is redefined every loop
            # iteration, so capturing the loop variable by reference would make
            # it read whatever L holds at *call* time (ruff B023).
            def _geom(af: Any, L: float = L) -> tuple[float, float, float]:
                """주어진 aligned frame 에서 (c, flexKP[rad], th5[rad]) 반환. 실패 시 nan."""
                s = af.keypoints.get(f"{side}_shoulder")
                e = af.keypoints.get(f"{side}_elbow")
                w = af.keypoints.get(f"{side}_wrist")
                if s is None or e is None or w is None or \
                        float(np.linalg.norm(e)) < 1e-6 or float(np.linalg.norm(w)) < 1e-6:
                    return float("nan"), float("nan"), float("nan")
                cc = float(np.linalg.norm(w - s))
                fx = _flex_from_keypoints(s, e, w)
                a = L / 2.0
                c_cl = min(max(cc, 0.0), 2 * a)
                cos_inner = (2 * a * a - c_cl * c_cl) / (2 * a * a)
                t5 = math.pi - math.acos(max(-1.0, min(1.0, cos_inner)))
                return cc, fx, t5

            status = "OK"
            conf = frame.mean_confidence()
            c, flex, th5 = _geom(aligned)
            if math.isnan(c):
                status = "ZEROKP"
            c_over_L = c / max(L, 1e-6) if not math.isnan(c) else float("nan")

            # RAW (평활 전) — 같은 프레임을 smoother 거치지 않고 align.
            try:
                aligned_raw = align_to_torso(frame, gravity_up=resolve_gravity_up(tracker, gravity_up))
                c_raw, flex_raw, th5_raw = _geom(aligned_raw)
            except AlignmentError:
                c_raw = flex_raw = th5_raw = float("nan")
            c_raw_over_L = c_raw / max(L, 1e-6) if not math.isnan(c_raw) else float("nan")

            try:
                angles = retarget_full_upper_body(aligned, robot, calibration, decouple_pitch_elbow=decouple)
                elb_out = angles.get(f"{prefix}_elbow", float("nan"))
                sp_out = angles.get(f"{prefix}_shoulder_pitch", float("nan"))
            except SingularConfigurationError:
                status = "SING"
                elb_out = float("nan")
                sp_out = float("nan")

            if now - last_log >= 0.1:
                last_log = now
                def d(x: float) -> float:
                    return math.degrees(x) if not math.isnan(x) else float("nan")
                print(f"{now-start:6.1f} {status:>5} {conf:5.2f} | "
                      f"{c_over_L:5.2f} {d(flex):7.1f} {d(th5):6.1f} {d(elb_out):7.1f} | "
                      f"{c_raw_over_L:7.2f} {d(flex_raw):7.1f} {d(th5_raw):7.1f}", flush=True)
            if dump is not None and not math.isnan(c):
                def _deg(x: float) -> float:
                    return math.degrees(x) if not math.isnan(x) else float("nan")
                dump.write(f"{now-start:.3f},{status},{conf:.3f},{L:.4f},{c:.4f},{c_over_L:.4f},"
                           f"{_deg(flex):.2f},{_deg(th5):.2f},{_deg(elb_out):.2f},{_deg(sp_out):.2f},"
                           f"{c_raw_over_L:.4f},{_deg(flex_raw):.2f},{_deg(th5_raw):.2f}\n")
    finally:
        tracker.stop()
        if dump is not None:
            dump.close()
            print(f"dumped → {args.dump}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
