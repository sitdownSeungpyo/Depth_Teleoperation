"""Live teleop simulation — camera estimation → IK → MuJoCo robot.

Two IK modes (``--ik``):
  numeric  (default) — core.robot_model + core.numik: task-space damped-least-
            squares IK on the robot model, driving elbow + wrist to targets
            placed at the robot's own link lengths along the operator's
            upper-arm/forearm directions. Matches the recent literature
            (Ryan 2025 / Jiang 2025 joint+Cartesian blend / OmniDP / MIRROR).
            Tracks bent arms accurately and needs NO arm-length calibration
            (direction-based, auto link-scaling). Robot model + frame come from
            config.robot — drop in a real URDF/MJCF there.
  analytic — legacy closed-form Yi-2012 IK (core.retarget). Exact for straight
            arms, ~25 deg approx when bent (paper's 3-DOF limitation).

Alongside the MuJoCo robot window, a separate OpenCV camera window shows the
operator's RealSense color image with the estimated upper-body skeleton drawn
on it (the "pipe" overlay) — so you can watch what the camera estimates and
what the robot does at the same time. Disable it with --no-camera.

Usage:
    python -m app.sim_teleop --config .\\config\\ubp.yaml                 # numeric
    python -m app.sim_teleop --config .\\config\\ubp.yaml --ik analytic
    python -m app.sim_teleop --config .\\config\\ubp.yaml --model .\\models\\my_robot.urdf
    python -m app.sim_teleop --config .\\config\\ubp.yaml --no-camera     # robot only
MuJoCo viewer: drag rotate / right-drag pan / wheel zoom / close to quit.
Camera window: 'q' or ESC to quit, 's' to save a snapshot.
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

log = logging.getLogger(__name__)
_SIDES = ("right", "left")


def _valid(kp: dict[str, Any], name: str) -> np.ndarray | None:
    v = kp.get(name)
    if v is None or float(np.linalg.norm(v)) < 1e-6:
        return None
    return np.asarray(v, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ik", choices=("numeric", "analytic"), default="numeric")
    parser.add_argument("--model", type=Path, default=None, help="robot model 경로 override")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--no-camera", action="store_true",
                        help="카메라 스켈레톤 창 끄기(로봇 창만)")
    parser.add_argument("--no-filter", action="store_true",
                        help="관절 각도 OneEuro 필터 끄기(진동 비교용; 기본은 켬)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    try:
        import mujoco
        from mujoco import viewer as mj_viewer
    except ImportError:
        print("mujoco not installed; pip install mujoco", file=sys.stderr)
        return 2

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from app.main import _build_filter, _build_tracker

    tracker: Any = _build_tracker(cfg, "realsense", replay=None)
    kp_cfg = cfg["filter"].get("keypoint_smoother", {})
    smoother = KeypointSmoother(OneEuroParams(
        min_cutoff=float(kp_cfg.get("min_cutoff", 0.5)),
        beta=float(kp_cfg.get("beta", 0.005)),
        d_cutoff=float(kp_cfg.get("d_cutoff", 1.0)),
    ))
    gravity_up_cfg = cfg.get("tracker", {}).get("realsense", {}).get("gravity_up")
    gravity_up = np.asarray(gravity_up_cfg, dtype=np.float64) if gravity_up_cfg else None

    # ---- build the chosen IK driver ----
    drive_numeric = args.ik == "numeric"
    if drive_numeric:
        from core.robot_model import RobotModel
        robot_cfg = dict(cfg["robot"])
        if args.model is not None:
            robot_cfg["model_path"] = str(args.model)
        rm = RobotModel(robot_cfg)
        model, data = rm.model, rm.data
        # Joint-space OneEuro (+ velocity/position clamp) on the IK output — the
        # numeric path wrote raw IK angles straight to qpos before, so per-frame
        # IK noise showed up as visible jitter. Map canonical joint name -> qpos
        # address so we can write the FILTERED angles back. (--no-filter to skip.)
        num_filt = None if args.no_filter else _build_filter(cfg)
        num_qadr: dict[str, int] = {}
        for arm in rm.arms.values():
            for n, a in zip(arm.ik.joint_names, arm.ik.qadr):
                num_qadr[n[:-6] if n.endswith("_joint") else n] = int(a)
        print(f"IK=numeric  model={Path(robot_cfg['model_path']).name}  "
              f"filter={'off' if args.no_filter else 'one_euro'}  "
              f"link_lengths={ {s: tuple(round(x,3) for x in v) for s,v in rm.link_lengths().items()} }",
              flush=True)
    else:
        from core.retarget import (
            Calibration, CalibrationCollector, RobotGeometry,
            SingularConfigurationError, retarget_full_upper_body,
        )
        model_path = args.model or Path(cfg["robot"]["model_path"])
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        rc = cfg["retarget"]["robot"]
        ageo = RobotGeometry(float(rc["upper_arm_length"]), float(rc["lower_arm_length"]),
                             tuple(rc["shoulder_offset"]))
        decouple = bool(cfg["retarget"].get("decouple_pitch_elbow", False))
        gain = dict(cfg["retarget"].get("output_gain", {}))
        collector = CalibrationCollector(target_frames=int(cfg["retarget"].get("calibration_frames", 30)))
        calibration = None
        fallback = float(cfg["retarget"]["fixed_arm_length"])
        qadr = {}
        for j in range(model.njnt):
            jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jn:
                qadr[jn[:-6] if jn.endswith("_joint") else jn] = int(model.jnt_qposadr[j])
        print(f"IK=analytic  model={Path(model_path).name}", flush=True)

    backend = (cfg["tracker"]["realsense"].get("body_backend") or "mediapipe").lower()
    print(f"body_backend={backend}  loading camera + model ...", flush=True)

    # ---- optional camera skeleton window (reuses viz_teleop's drawing) ----
    show_camera = not args.no_camera
    cam_win = "imitation_upper camera (skeleton)  [q/ESC quit, s snapshot]"
    snap_dir = Path("./recordings")
    if show_camera:
        try:
            import cv2  # noqa: F401

            from app.viz_teleop import _draw_camera_panel, _draw_detection, _hud_lines
            snap_dir.mkdir(parents=True, exist_ok=True)
        except ImportError as exc:
            print(f"camera window disabled (opencv import failed: {exc})", file=sys.stderr)
            show_camera = False

    tracker.start()

    last_ts = 0.0
    start = time.perf_counter()
    last_log = start

    def pump_camera(frame: Any, now: float) -> bool:
        """Draw + show the camera skeleton window. Returns False if the user
        asked to quit (q/ESC). Independent of IK/alignment so it keeps showing
        live video and the estimated skeleton even when no pose is usable."""
        color = tracker.latest_color()
        if color is None:
            return True
        color = color.copy()
        intr = tracker.intrinsics_params
        det = tracker.latest_detection()
        detected = _draw_detection(color, det)
        if frame is not None:
            _draw_camera_panel(color, frame, intr)  # reproject skeleton ("pipe")
        conf = frame.mean_confidence() if frame is not None else 0.0
        _hud_lines(color, [
            (f"{backend}  ik={args.ik}", (0, 220, 0) if detected else (0, 165, 255)),
            (f"conf {conf:.2f}", (255, 255, 255)),
        ])
        h, w = color.shape[:2]
        banner, bcol = (("BODY DETECTED", (0, 200, 0)) if detected
                        else ("NO BODY - stand 1.5-2.5m, full torso in frame", (0, 0, 255)))
        cv2.rectangle(color, (0, h - 28), (w, h), (0, 0, 0), -1)
        cv2.putText(color, banner, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, bcol, 2, cv2.LINE_AA)
        cv2.imshow(cam_win, color)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord("s"):
            out = snap_dir / f"sim_teleop_{int(now)}.png"
            cv2.imwrite(str(out), color)
            print(f"saved {out}", flush=True)
        return True

    try:
        with mj_viewer.launch_passive(model, data) as v:
            v.cam.lookat = np.array([0.1, 0.0, 1.1])
            v.cam.distance = 2.2
            v.cam.azimuth = 150.0
            v.cam.elevation = -10.0
            while v.is_running():
                now = time.perf_counter()
                if args.duration is not None and now - start > args.duration:
                    break
                frame = tracker.latest()

                # Camera skeleton window — pump every iteration (all branches),
                # so live video + estimated skeleton stay smooth and 'q' quits.
                if show_camera and not pump_camera(frame, now):
                    break

                if frame is None or frame.timestamp == last_ts:
                    time.sleep(0.003)
                    v.sync()
                    continue
                last_ts = frame.timestamp

                try:
                    aligned = align_to_torso(smoother.smooth(frame), gravity_up=gravity_up)
                except AlignmentError:
                    v.sync()
                    continue
                kp = aligned.keypoints

                if drive_numeric:
                    raw: dict[str, float] = {}
                    for side in _SIDES:
                        sh = _valid(kp, f"{side}_shoulder")
                        el = _valid(kp, f"{side}_elbow")
                        wr = _valid(kp, f"{side}_wrist")
                        if sh is not None and el is not None and wr is not None:
                            sol = rm.solve_arm(side, sh, el, wr)  # warm-started, mutates data
                            if sol:
                                for n, val in sol.items():
                                    raw[n[:-6] if n.endswith("_joint") else n] = val
                    if num_filt is not None and raw:
                        # Smooth the IK angles, then write the filtered values back
                        # to qpos so the displayed/commanded pose is the smooth one
                        # (and the next frame's IK warm-starts from it).
                        cmd = num_filt(raw, timestamp=now, source_frame_ts=frame.timestamp)
                        for cname, val in cmd.positions.items():
                            a = num_qadr.get(cname)
                            if a is not None:
                                data.qpos[a] = float(val)
                    mujoco.mj_forward(model, data)
                else:
                    if calibration is None:
                        collector.push(aligned)
                        if collector.ready():
                            try:
                                calibration = collector.finalise(robot=ageo, decouple_pitch_elbow=decouple)
                            except SingularConfigurationError:
                                calibration = Calibration(operator_arm_length=fallback)
                            print("[CALIBRATED]", flush=True)
                        else:
                            v.sync(); continue
                    try:
                        tgt = retarget_full_upper_body(aligned, ageo, calibration, decouple_pitch_elbow=decouple)
                    except SingularConfigurationError:
                        v.sync(); continue
                    if gain:
                        tgt = {j: val * gain.get(j, 1.0) for j, val in tgt.items()}
                    for name, val in tgt.items():
                        a = qadr.get(name)
                        if a is not None:
                            data.qpos[a] = float(val)
                    mujoco.mj_forward(model, data)

                v.sync()
                if now - last_log >= 1.0:
                    last_log = now
                    js = rm.joint_qpos if drive_numeric else {k: data.qpos[qadr[k]] for k in qadr}
                    print(
                        f"r_sp={js.get('r_shoulder_pitch', 0):+.2f} "
                        f"r_sr={js.get('r_shoulder_roll', 0):+.2f} "
                        f"r_sy={js.get('r_shoulder_yaw', 0):+.2f} "
                        f"r_elb={js.get('r_elbow', 0):+.2f}  conf={frame.mean_confidence():.2f}",
                        flush=True,
                    )
    finally:
        tracker.stop()
        if show_camera:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
