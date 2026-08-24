"""Phase 1 entrypoint — wires tracker, aligner, IK, filter, safety, publisher.

Two IK modes (``--ik``, default from ``main.ik`` in config/runtime.yaml):
  numeric  (default) — task-space damped-least-squares IK on the robot model
            (core.robot_model + core.numik). Drives elbow AND wrist to targets
            placed at the robot's own link lengths along the operator's
            upper-arm/forearm directions. Tracks bent arms accurately and needs
            no arm-length calibration.
  analytic — closed-form Yi-2012 IK (core.retarget) with auto rest-pose
            calibration. Exact for straight arms, ~25 deg off when bent.

The numeric solver used to live only in ``app/sim_teleop.py``, which has no
safety layer and no publisher, while this file had the safety layer but only the
analytic solver — so there was no way to run accurate IK *and* the dead-man,
E-stop, watchdog and loss-ramp at the same time. That is what the ``--ik`` switch
here fixes.

Usage (PowerShell):
    .\\scripts\\run.ps1 --config .\\config\\default.yaml --tracker mock --publisher mock \\
        --replay .\\tests\\fixtures\\arm_circle.jsonl
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.aligner import AlignedFrame, AlignmentError, align_to_torso, resolve_gravity_up
from core.filter import (
    FilterAndLimiter,
    JointLimiterConfig,
    JointLimits,
    KeypointSmoother,
    OneEuroParams,
)
from core.retarget import (
    Calibration,
    CalibrationCollector,
    RobotGeometry,
    SingularConfigurationError,
    retarget_full_upper_body,
)
from core.safety import PynputHotkey, SafetyConfig, SafetyLayer
from core.types import SkeletonFrame
from publisher.mock_publisher import MockPublisher
from publisher.udp_publisher import UdpPublisherSkeleton
from tracker.base import health_of
from tracker.body_backend import BodyBackend, MediaPipeBodyBackend
from tracker.hand_backend import HandBackend, MediaPipeHandBackend
from tracker.mock_tracker import MockTracker

log = logging.getLogger(__name__)

_SIDES = ("right", "left")


def _canonical(joint_name: str) -> str:
    """Model joint name -> command key ('r_elbow_joint' -> 'r_elbow')."""
    return joint_name[:-6] if joint_name.endswith("_joint") else joint_name


def _load_config(path: Path) -> dict[str, Any]:
    # Resolves the include: list in config/ubp.yaml and deep-merges the
    # purpose-split files (tracker/retarget/robot/filter/joint_limit/...).
    from core.config import load_config

    return load_config(path)


def _build_publisher(cfg: dict[str, Any], publisher_kind: str) -> Any:
    pub_cfg = cfg.get("publisher", {})
    rate_hz = int(pub_cfg.get("rate_hz", 100))
    if publisher_kind == "mock":
        log_path = pub_cfg.get("log_path")
        return MockPublisher(rate_hz=rate_hz, log_path=Path(log_path) if log_path else None)
    if publisher_kind == "udp":
        udp = pub_cfg.get("udp", {})
        return UdpPublisherSkeleton(
            host=str(udp.get("host", "127.0.0.1")),
            port=int(udp.get("port", 9000)),
            rate_hz=rate_hz,
        )
    if publisher_kind == "pybullet":
        from publisher.pybullet_publisher import PyBulletPublisher

        pb_cfg = pub_cfg.get("pybullet", {})
        return PyBulletPublisher(
            urdf_path=Path(pb_cfg.get("urdf_path", "./models/ubp.urdf")),
            rate_hz=rate_hz,
            gui=bool(pb_cfg.get("gui", True)),
            fixed_base=bool(pb_cfg.get("fixed_base", True)),
            init_pose=tuple(pb_cfg.get("init_pose", [0.0, 0.0, 0.0])),
        )
    if publisher_kind == "mujoco":
        from publisher.mujoco_publisher import MuJoCoPublisher

        mj_cfg = pub_cfg.get("mujoco", {})
        return MuJoCoPublisher(
            model_path=Path(mj_cfg.get("model_path", "./models/ubp.xml")),
            rate_hz=rate_hz,
            gui=bool(mj_cfg.get("gui", True)),
        )
    if publisher_kind == "dynamixel":
        from publisher.dynamixel_publisher import (
            DynamixelPublisher,
            build_servos_from_config,
        )

        dxl_cfg = pub_cfg.get("dynamixel", {})
        return DynamixelPublisher(
            port=str(dxl_cfg["port"]),
            baud=int(dxl_cfg.get("baud", 1_000_000)),
            servos=build_servos_from_config(dxl_cfg["servos"]),
            rate_hz=rate_hz,
            profile_velocity=dxl_cfg.get("profile_velocity"),
            profile_acceleration=dxl_cfg.get("profile_acceleration"),
            torque_off_on_stop=bool(dxl_cfg.get("torque_off_on_stop", False)),
        )
    raise ValueError(f"unknown publisher: {publisher_kind}")


def _build_body_backend(rs_cfg: dict[str, Any], pose_cfg: dict[str, Any]) -> BodyBackend:
    """Build the body-pose backend selected by ``tracker.realsense.body_backend``.

    Options: ``mediapipe`` (default) or ``hmr2`` (4D-Humans / HMR2.0,
    SMPL-prior, GPU; better under upper-body occlusion). HMR2 wraps a
    MediaPipe helper internally for the body bbox + wrist image coords.
    """
    backend_name = (rs_cfg.get("body_backend") or "mediapipe").lower()
    model_path = rs_cfg.get("model_asset_path")
    if not model_path:
        raise ValueError("tracker.realsense.model_asset_path is required")
    min_visibility = float(pose_cfg.get("min_visibility", 0.5))
    use_world = bool(rs_cfg.get("use_world_landmarks", False))
    depth_max_m = float(rs_cfg.get("depth_max_m", 4.0))

    mp_backend = MediaPipeBodyBackend(
        model_asset_path=str(model_path),
        min_visibility=min_visibility,
        use_world_landmarks=use_world,
        depth_max_m=depth_max_m,
    )
    if backend_name == "mediapipe":
        return mp_backend
    if backend_name == "hmr2":
        try:
            from tracker.hmr2_body_backend import Hmr2BodyBackend
        except ImportError as exc:
            log.warning("HMR2 backend unavailable (%s); falling back to MediaPipe", exc)
            return mp_backend
        return Hmr2BodyBackend(
            mediapipe_helper=mp_backend,
            device=str(rs_cfg.get("hmr2_device", "cuda")),
            checkpoint_path=rs_cfg.get("hmr2_checkpoint_path"),
            bbox_padding=float(rs_cfg.get("hmr2_bbox_padding", 0.15)),
        )
    if backend_name == "rtmpose":
        try:
            from tracker.rtmpose_body_backend import RTMPoseBodyBackend
        except ImportError as exc:
            log.warning("RTMPose backend unavailable (%s); falling back to MediaPipe", exc)
            return mp_backend
        # RTMPose is 2D-only -> always the depth-deproject path (use_world_landmarks
        # is ignored here; it only applies to the MediaPipe fallback).
        return RTMPoseBodyBackend(
            mode=str(rs_cfg.get("rtmpose_mode", "balanced")),
            device=str(rs_cfg.get("rtmpose_device", "cuda")),
            onnx_backend=str(rs_cfg.get("rtmpose_onnx_backend", "onnxruntime")),
            min_visibility=min_visibility,
            depth_max_m=depth_max_m,
            depth_lift=rs_cfg.get("depth_lift"),
        )
    raise ValueError(f"unknown body_backend: {backend_name!r}")


def _build_hand_backend(rs_cfg: dict[str, Any]) -> HandBackend | None:
    """Build the hand backend selected by ``tracker.realsense.hand_backend``.

    Options: ``mediapipe`` (default, CPU, 7.5 MB model) or ``hamer`` (GPU,
    requires PyTorch + HaMeR + MANO; see scripts/install_hamer.ps1). When the
    backend is ``null`` or its model is missing, returns None and the
    retargeter falls back to 0 for sh_yaw/w_yaw/w_pitch.
    """
    backend_name = (rs_cfg.get("hand_backend") or "mediapipe").lower()
    if backend_name in ("none", "off", "null"):
        return None
    if backend_name == "hamer":
        try:
            from tracker.hamer_hand_backend import HamerHandBackend
        except ImportError as exc:
            log.warning("HaMeR backend unavailable (%s); skipping hand tracking", exc)
            return None
        return HamerHandBackend(
            device=str(rs_cfg.get("hamer_device", "cuda")),
            checkpoint_path=rs_cfg.get("hamer_checkpoint_path"),
            mano_dir=rs_cfg.get("mano_dir"),
        )
    if backend_name == "mediapipe":
        model_path = rs_cfg.get("hand_model_asset_path")
        if not model_path:
            return None
        return MediaPipeHandBackend(
            model_asset_path=str(model_path),
            min_confidence=float(rs_cfg.get("hand_min_confidence", 0.5)),
        )
    raise ValueError(f"unknown hand_backend: {backend_name!r}")


def _build_tracker(cfg: dict[str, Any], tracker_kind: str, replay: Path | None) -> Any:
    if tracker_kind == "mock":
        path = replay or Path(cfg.get("tracker", {}).get("mock", {}).get("jsonl_path") or "")
        if not path or not path.exists():
            raise FileNotFoundError(f"mock tracker requires --replay <jsonl>; got {path!r}")
        return MockTracker(jsonl_path=path, loop=bool(cfg["tracker"]["mock"].get("loop", True)))
    if tracker_kind == "realsense":
        from tracker.realsense_tracker import RealSenseTracker

        rs_cfg = cfg["tracker"]["realsense"]
        body_backend = _build_body_backend(rs_cfg, cfg["tracker"]["pose"])
        hand_backend = _build_hand_backend(rs_cfg)
        imu_cfg = rs_cfg.get("imu", {}) or {}
        return RealSenseTracker(
            color_resolution=tuple(rs_cfg["color_resolution"]),
            depth_resolution=tuple(rs_cfg["depth_resolution"]),
            fps=int(rs_cfg["fps"]),
            depth_max_m=float(rs_cfg["depth_max_m"]),
            body_backend=body_backend,
            hand_backend=hand_backend,
            enable_imu=bool(rs_cfg.get("enable_imu", False)),
            accel_fps=int(imu_cfg.get("accel_fps", 200)),
            gravity_lpf_alpha=float(imu_cfg.get("lpf_alpha", 0.02)),
            gravity_warmup_frames=int(imu_cfg.get("warmup_frames", 10)),
            gravity_norm_tol=float(imu_cfg.get("norm_tol", 0.30)),
            gravity_axis_sign=float(imu_cfg.get("axis_sign", 1.0)),
        )
    raise ValueError(f"unknown tracker: {tracker_kind}")


def _build_filter(cfg: dict[str, Any]) -> FilterAndLimiter:
    f_cfg = cfg["filter"]
    one_euro = OneEuroParams(
        min_cutoff=float(f_cfg["one_euro"]["min_cutoff"]),
        beta=float(f_cfg["one_euro"]["beta"]),
        d_cutoff=float(f_cfg["one_euro"].get("d_cutoff", 1.0)),
    )
    factor = float(f_cfg["joint_limits_factor"])
    max_v = float(f_cfg["max_velocity_rad_s"])
    limits = {
        joint: JointLimits(soft_min=lo * factor, soft_max=hi * factor, max_velocity=max_v)
        for joint, (lo, hi) in f_cfg["mechanical_limits"].items()
    }
    return FilterAndLimiter(
        one_euro=one_euro,
        limiter=JointLimiterConfig(
            limits=limits,
            velocity_violation_factor=float(f_cfg.get("velocity_violation_factor", 5.0)),
        ),
    )


def _build_safety(
    cfg: dict[str, Any],
    publisher: Any,
    loop_dt: float,
    require_deadman: bool,
    enable_hotkeys: bool,
) -> SafetyLayer:
    """Build the safety layer.

    ``require_deadman`` and ``enable_hotkeys`` are deliberately separate. They
    used to be one flag, which meant running without the dead-man also silently
    disabled the E-stop key — the one control you never want to lose.
    """
    s_cfg = cfg["safety"]
    safety_cfg = SafetyConfig(
        deadman_key=str(s_cfg["deadman"]["key"]),
        estop_key=str(s_cfg["estop_key"]),
        reset_key=str(s_cfg.get("reset_key", "r")),
        require_deadman=require_deadman,
        confidence_threshold=float(s_cfg["confidence_threshold"]),
        loss_grace_period_s=float(s_cfg["loss_grace_period_s"]),
        ramp_to_safe_s=float(s_cfg["ramp_to_safe_s"]),
        watchdog_factor=int(s_cfg["watchdog_factor"]),
        cycle_dt_s=loop_dt,
        watchdog_timeout_s=(
            float(s_cfg["watchdog_timeout_s"])
            if s_cfg.get("watchdog_timeout_s") is not None
            else None
        ),
        safe_pose=dict(s_cfg["safe_pose"]),
    )
    hotkey = (
        PynputHotkey(
            [safety_cfg.deadman_key, safety_cfg.estop_key, safety_cfg.reset_key],
            # A dead-man we cannot read is no dead-man at all: fail to start
            # rather than run a robot with no keyboard stop.
            strict=require_deadman,
        )
        if enable_hotkeys
        else None
    )
    return SafetyLayer(publisher=publisher, config=safety_cfg, hotkey=hotkey)


def _valid_kp(keypoints: dict[str, Any], name: str) -> np.ndarray | None:
    """Keypoint as float64, or None when absent / rejected (zero-vector)."""
    v = keypoints.get(name)
    if v is None or float(np.linalg.norm(v)) < 1e-6:
        return None
    return np.asarray(v, dtype=np.float64)


def _solve_numeric_arms(
    robot_model: Any,
    aligned: AlignedFrame,
    min_arm_confidence: float,
    held_arms: dict[str, int],
) -> dict[str, float]:
    """Run the task-space IK for both arms; return canonical joint targets.

    An arm whose worst keypoint score is below ``min_arm_confidence`` is skipped
    entirely, so the downstream limiter holds its previous command instead of
    tracking a plausible-but-wrong estimate.
    """
    keypoints = aligned.keypoints
    out: dict[str, float] = {}
    for side in _SIDES:
        if aligned.arm_confidence(side) < min_arm_confidence:
            held_arms[side] += 1
            continue
        shoulder = _valid_kp(keypoints, f"{side}_shoulder")
        elbow = _valid_kp(keypoints, f"{side}_elbow")
        wrist = _valid_kp(keypoints, f"{side}_wrist")
        if shoulder is None or elbow is None or wrist is None:
            continue
        solution = robot_model.solve_arm(side, shoulder, elbow, wrist)
        if not solution:
            continue
        for joint, value in solution.items():
            out[_canonical(joint)] = float(value)
        # The numeric IK drives shoulder pitch/roll/yaw + elbow only; wrist twist
        # and pitch come from the hand backend, which is an analytic-path input.
        # Pin them for arms that solved so the command's joint set stays constant
        # frame to frame — same reason retarget_full_upper_body does it.
        prefix = "r" if side == "right" else "l"
        out.setdefault(f"{prefix}_wrist_yaw", 0.0)
        out.setdefault(f"{prefix}_wrist_pitch", 0.0)
    if out:
        out.setdefault("neck_yaw", 0.0)
        out.setdefault("head_pitch", 0.0)
    return out


class _FramePump:
    """Consume ``tracker.stream()`` on its own thread; keep only the newest frame.

    The control loop used to *be* the stream consumer (``for frame in
    tracker.stream()``), so it stopped iterating whenever the tracker stopped
    producing — including the common, harmless case of the operator briefly
    leaving the shot. A loop that is not iterating cannot feed the safety layer,
    so a detector miss read as a stall and latched an E-stop. Pumping the stream
    separately lets the control loop run at its own rate regardless of what the
    tracker is doing, and lets the loop decide whether a gap is a camera fault or
    a tracking fault.
    """

    def __init__(self, tracker: Any) -> None:
        self._tracker = tracker
        self._lock = threading.Lock()
        self._latest: SkeletonFrame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ended = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="frame-pump", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for frame in self._tracker.stream():
                if self._stop.is_set():
                    break
                with self._lock:
                    self._latest = frame
        except Exception:  # noqa: BLE001
            log.exception("frame pump stopped on an unexpected error")
        finally:
            self._ended = True

    def latest(self) -> SkeletonFrame | None:
        with self._lock:
            return self._latest

    @property
    def ended(self) -> bool:
        """True once the tracker's stream returned — replay finished, or producer dead."""
        return self._ended

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), p))


def run(
    config_path: Path,
    tracker_kind: str,
    publisher_kind: str,
    replay: Path | None = None,
    duration_s: float | None = None,
    require_deadman: bool = False,
    ik: str | None = None,
    model_override: Path | None = None,
    enable_hotkeys: bool | None = None,
) -> dict[str, float]:
    cfg = _load_config(config_path)
    main_cfg = cfg.get("main", {})
    loop_rate = float(main_cfg.get("loop_rate_hz", 60))
    loop_dt = 1.0 / loop_rate
    log_every = int(main_cfg.get("latency_log_every", 30))
    p95_budget = float(main_cfg.get("latency_p95_budget_s", 0.1))
    camera_stale_s = float(main_cfg.get("camera_stale_s", 0.5))
    camera_stall_factor = float(main_cfg.get("camera_stall_factor", 4.0))
    camera_startup_grace_s = float(main_cfg.get("camera_startup_grace_s", 10.0))
    detection_gap_s = float(main_cfg.get("detection_gap_s", 0.3))

    ik_mode = (ik or str(main_cfg.get("ik", "numeric"))).lower()
    if ik_mode not in ("numeric", "analytic"):
        raise ValueError(f"unknown ik mode: {ik_mode!r} (expected 'numeric' or 'analytic')")
    use_numeric = ik_mode == "numeric"

    # A real robot must not move without a dead-man. Everything else defaults to
    # the caller's choice; this one is not negotiable by omission.
    if publisher_kind == "dynamixel" and not require_deadman:
        log.warning(
            "publisher=dynamixel: forcing the dead-man on — a real robot must not move "
            "without one. Pass --deadman explicitly to silence this."
        )
        require_deadman = True
    if enable_hotkeys is None:
        # Hotkeys need a live operator at the keyboard. A headless replay has none,
        # and installing a global keyboard listener there is pure downside.
        enable_hotkeys = tracker_kind == "realsense" or publisher_kind == "dynamixel"

    tracker = _build_tracker(cfg, tracker_kind, replay)
    publisher = _build_publisher(cfg, publisher_kind)
    filt = _build_filter(cfg)
    safety = _build_safety(cfg, publisher, loop_dt, require_deadman, enable_hotkeys)
    # After an E-stop release the robot sits where it was frozen, not where the
    # limiter last commanded. Re-baseline so the first new command is clamped
    # against reality instead of a stale pre-fault value.
    safety.on_estop_reset = filt.reset_to

    # Keypoint smoothing — IK 직선팔 singularity 노이즈 억제. retarget 전에 적용.
    kp_cfg = cfg["filter"].get("keypoint_smoother", {})
    kp_smoother = KeypointSmoother(
        OneEuroParams(
            min_cutoff=float(kp_cfg.get("min_cutoff", 0.5)),
            beta=float(kp_cfg.get("beta", 0.005)),
            d_cutoff=float(kp_cfg.get("d_cutoff", 1.0)),
        )
    )

    # Per-joint motion scaling — analytic path only. The numeric solver returns a
    # kinematically consistent pose on the robot model; scaling its joints would
    # break that consistency, which is why sim_teleop never applied gain either.
    output_gain: dict[str, float] = dict(cfg["retarget"].get("output_gain", {}))

    # Gravity-aligned aligner — 운영자 토르소 tilt 의 영향을 제거.
    gravity_up_cfg = cfg.get("tracker", {}).get("realsense", {}).get("gravity_up")
    gravity_up: np.ndarray | None = (
        np.asarray(gravity_up_cfg, dtype=np.float64) if gravity_up_cfg else None
    )

    # Decoupled shoulder_pitch ↔ elbow (Yi 2012 식 3에서 `-theta_5` 제거).
    decouple_pitch_elbow = bool(cfg.get("retarget", {}).get("decouple_pitch_elbow", False))

    # Per-arm confidence gate — skip an arm whose worst keypoint score is too low
    # so the filter holds its last command instead of tracking a bad estimate.
    min_arm_confidence = float(
        cfg.get("tracker", {}).get("pose", {}).get("min_arm_confidence", 0.0)
    )
    held_arms = {"right": 0, "left": 0}

    # ---- IK driver ----
    robot_model: Any = None
    numeric_qadr: dict[str, int] = {}
    collector: CalibrationCollector | None = None
    calibration: Calibration | None = None
    robot_geometry: RobotGeometry | None = None
    fallback_arm_length = float(cfg["retarget"]["fixed_arm_length"])

    if use_numeric:
        from core.robot_model import RobotModel

        robot_cfg = dict(cfg["robot"])
        if model_override is not None:
            robot_cfg["model_path"] = str(model_override)
        robot_model = RobotModel(robot_cfg)
        for arm in robot_model.arms.values():
            for name, addr in zip(arm.ik.joint_names, arm.ik.qadr, strict=True):
                numeric_qadr[_canonical(name)] = int(addr)
        log.info(
            "IK=numeric model=%s link_lengths=%s",
            Path(robot_cfg["model_path"]).name,
            {s: tuple(round(x, 3) for x in v)
             for s, v in robot_model.link_lengths().items()},
        )
    else:
        rc = cfg["retarget"]["robot"]
        robot_geometry = RobotGeometry(
            upper_arm_length=float(rc["upper_arm_length"]),
            lower_arm_length=float(rc["lower_arm_length"]),
            shoulder_offset=tuple(rc["shoulder_offset"]),
        )
        target_calib_frames = int(cfg["retarget"].get("calibration_frames", 30))
        collector = CalibrationCollector(target_frames=target_calib_frames)
        log.info(
            "IK=analytic. [CALIBRATION] 양팔 자연스럽게 내리고 어깨 편안히 (~%.1fs hold). "
            "자동으로 rest pose 캡쳐 후 robot zero 로 매핑됩니다.",
            target_calib_frames / loop_rate,
        )

    latencies: list[float] = []
    start = time.perf_counter()
    # The numeric path needs no calibration, so its throughput window opens at t0.
    post_calib_start: float | None = start if use_numeric else None
    frame_count = 0
    last_ts = 0.0
    last_cmd: Any = None
    camera_stale_warned = False
    detection_gap_active = False
    last_pose_wall = start

    def no_usable_pose(now: float, reason: str) -> None:
        """Handle a tick that produced no command the robot can follow.

        Every way of failing to get a pose lands here — no new frame, a torso
        basis the aligner could not build, both arms held on low confidence — and
        they all mean the same thing to the robot, so they get the same response:
        stay alive until the gap is long enough to matter, then let the
        confidence ramp take the arms to the safe pose.

        Routing them together also fixes two things seen on hardware. An aligner
        rejection used to call note_alive() forever, so an operator who was
        present but whose shoulders were not being resolved never triggered the
        ramp at all. And it logged once per frame: a 25 s run with nobody in shot
        produced 464 identical warnings, which is how a real message gets missed.
        """
        nonlocal detection_gap_active
        if now - last_pose_wall > detection_gap_s and last_cmd is not None:
            if not detection_gap_active:
                detection_gap_active = True
                log.warning(
                    "no usable pose for >%.2fs (%s) while the camera is healthy; "
                    "engaging the tracking-loss policy",
                    detection_gap_s, reason,
                )
            # Zero confidence drives the grace period and the ramp to the safe
            # pose. That is the designed response to losing the operator; the
            # stall watchdog is for losing the camera.
            safety.update(last_cmd, mean_confidence=0.0)
        else:
            safety.note_alive()

    tracker.start()
    pump = _FramePump(tracker)
    try:
        safety.start()
        pump.start()

        while True:
            now = time.perf_counter()
            if duration_s is not None and now - start > duration_s:
                break

            health = health_of(tracker)
            if health.error is not None:
                log.error("tracker capture failed: %s", health.error)
                safety.trigger_estop("tracker capture failure")
                break
            if pump.ended:
                log.info("frame stream ended (replay finished or producer stopped)")
                break

            # ---- camera liveness (a real fault) --------------------------
            if not math.isfinite(health.frame_age_s):
                # No frame at all yet — normal while the camera and models warm up.
                if now - start > camera_startup_grace_s:
                    log.error(
                        "no camera frame within %.1fs of start; giving up",
                        camera_startup_grace_s,
                    )
                    safety.trigger_estop("camera never delivered a frame")
                    break
                safety.note_alive()
                time.sleep(loop_dt)
                continue
            # Size the stall threshold against the cadence the pipeline actually
            # achieves. Capture and detection share a thread, so a fixed 0.5 s
            # limit falsely trips whenever the detector is slower than that —
            # which a cold CUDA warm-up on the first inference always is.
            stall_limit = health.stall_limit_s(camera_stale_s, camera_stall_factor)
            warming_up = (
                not math.isfinite(health.capture_period_s)
                and now - start <= camera_startup_grace_s
            )
            if not warming_up and health.frame_age_s > stall_limit:
                # Deliberately do NOT call note_alive() here: the camera really is
                # gone, and withholding the liveness stamp is what lets the safety
                # watchdog E-stop us.
                if not camera_stale_warned:
                    camera_stale_warned = True
                    log.error(
                        "no camera frame for >%.2fs (%s); withholding liveness so the "
                        "watchdog can act",
                        stall_limit,
                        health.describe(),
                    )
                time.sleep(loop_dt)
                continue
            if camera_stale_warned:
                camera_stale_warned = False
                log.info("camera recovered (%s)", health.describe())

            frame = pump.latest()
            if frame is None or frame.timestamp == last_ts:
                # No NEW frame this tick. At a 60 Hz loop and a 30 fps camera this
                # is the normal case on half the ticks, so it is only a fault once
                # it persists past detection_gap_s.
                no_usable_pose(now, "no new frame")
                time.sleep(0.001)
                continue
            last_ts = frame.timestamp

            # ---- estimate -> joint targets -------------------------------
            frame = kp_smoother.smooth(frame)
            try:
                gu = resolve_gravity_up(tracker, gravity_up)  # IMU if available
                aligned = align_to_torso(frame, gravity_up=gu)
            except AlignmentError as exc:
                no_usable_pose(now, f"aligner: {exc}")
                continue

            if use_numeric:
                joint_targets = _solve_numeric_arms(
                    robot_model, aligned, min_arm_confidence, held_arms
                )
                if not joint_targets:
                    # Both arms held (low confidence, degenerate geometry, or an
                    # IK solve that did not converge).
                    no_usable_pose(now, "both arms held")
                    continue
            else:
                assert collector is not None and robot_geometry is not None
                if calibration is None:
                    collector.push(aligned)
                    if not collector.ready():
                        # Calibration is a normal startup phase, not a lost
                        # operator: keep the loop alive without arming the ramp.
                        safety.note_alive()
                        continue
                    try:
                        calibration = collector.finalise(
                            robot=robot_geometry,
                            decouple_pitch_elbow=decouple_pitch_elbow,
                            min_arm_confidence=min_arm_confidence,
                        )
                        # Config rest_offsets (if any) override auto-captured values
                        # per joint — auto for unspecified joints, manual for the rest.
                        rest_cfg = cfg["retarget"].get("rest_offsets") or {}
                        if rest_cfg:
                            calibration.rest_offsets = {
                                **calibration.rest_offsets,
                                **{k: float(v) for k, v in rest_cfg.items()},
                            }
                        log.info(
                            "calibration done: operator_arm_length=%.3f m, rest_offsets=%s",
                            calibration.operator_arm_length,
                            calibration.rest_offsets,
                        )
                    except SingularConfigurationError:
                        calibration = Calibration(operator_arm_length=fallback_arm_length)
                    post_calib_start = time.perf_counter()

                try:
                    joint_targets = retarget_full_upper_body(
                        aligned, robot_geometry, calibration,
                        decouple_pitch_elbow=decouple_pitch_elbow,
                        min_arm_confidence=min_arm_confidence,
                    )
                except SingularConfigurationError as exc:
                    no_usable_pose(now, f"retarget: {exc}")
                    continue

                if not any(j.startswith(("r_", "l_")) for j in joint_targets):
                    # retarget_full_upper_body always returns torso_yaw/head_pitch,
                    # so a frame where BOTH arms failed still looks like a result.
                    # Treat it the way the numeric path does, or the loss ramp
                    # never arms while the operator is missing.
                    no_usable_pose(now, "no arm solved")
                    continue

                if output_gain:
                    joint_targets = {
                        j: v * output_gain.get(j, 1.0) for j, v in joint_targets.items()
                    }

            if detection_gap_active:
                detection_gap_active = False
                log.info("pose recovered")
                held = safety.last_command()
                if held is not None:
                    # The loss ramp moved the robot while we were blind; re-baseline
                    # so the first tracking command is clamped from where it is now.
                    filt.reset_to(dict(held.positions), timestamp=now)

            cmd = filt(joint_targets, timestamp=now, source_frame_ts=frame.timestamp)
            safety.update(cmd, mean_confidence=frame.mean_confidence())
            last_cmd = cmd
            last_pose_wall = now
            # NOTE: no watchdog_tick() here. It used to be called right after
            # update(), which had just refreshed the liveness stamp — so the check
            # could never fail. SafetyLayer.start() now runs it on its own thread,
            # which is the only way to catch a loop blocked on a stalled tracker.

            if use_numeric:
                # Write the FILTERED angles back to the model so the next frame's
                # DLS warm-starts from the pose we actually commanded. Skipping this
                # lets the solver drift away from the smoothed output.
                for joint, value in cmd.positions.items():
                    addr = numeric_qadr.get(joint)
                    if addr is not None:
                        robot_model.data.qpos[addr] = float(value)

            # Latency = time from frame capture to *after* the command is forwarded.
            latencies.append(time.perf_counter() - frame.timestamp)
            frame_count += 1
            if frame_count % log_every == 0:
                p95 = _percentile(latencies[-200:], 95)
                base = post_calib_start if post_calib_start is not None else start
                rate = frame_count / max(time.perf_counter() - base, 1e-6)
                # Joint diagnostics — side-by-side L vs R 비교 (양팔 symmetry 진단).
                r_sp = cmd.positions.get("r_shoulder_pitch", 0.0)
                l_sp = cmd.positions.get("l_shoulder_pitch", 0.0)
                r_sr = cmd.positions.get("r_shoulder_roll", 0.0)
                l_sr = cmd.positions.get("l_shoulder_roll", 0.0)
                r_elb = cmd.positions.get("r_elbow", 0.0)
                l_elb = cmd.positions.get("l_elbow", 0.0)
                log.info(
                    "loop %.1f Hz p95 %.0fms | "
                    "SP r=%+.2f l=%+.2f Δ=%+.2f | "
                    "SR r=%+.2f l=%+.2f Δ=%+.2f | "
                    "ELB r=%+.2f l=%+.2f Δ=%+.2f | held R=%d L=%d",
                    rate, p95 * 1000,
                    r_sp, l_sp, r_sp - l_sp,
                    r_sr, l_sr, r_sr - l_sr,
                    r_elb, l_elb, r_elb - l_elb,
                    held_arms["right"], held_arms["left"],
                )
    finally:
        # Order matters. Disarm the watchdog FIRST: shutting down is not a stall,
        # but every step below stops feeding commands, and joining a pump that is
        # parked inside a tracker generator can take up to its join timeout — long
        # enough for the watchdog to fire and log a bogus E-stop on the way out.
        safety.stop()
        # Then the tracker, which is what unblocks a stream generator waiting on
        # frames, so the pump's own join returns immediately.
        tracker.stop()
        pump.stop()

    end = time.perf_counter()
    base = post_calib_start if post_calib_start is not None else start
    elapsed = max(end - base, 1e-6)
    summary = {
        "frames": float(frame_count),
        "rate_hz": frame_count / elapsed,
        "latency_p95_s": _percentile(latencies, 95),
        "latency_p50_s": _percentile(latencies, 50),
        "held_right": float(held_arms["right"]),
        "held_left": float(held_arms["left"]),
    }
    print(
        f"ik={ik_mode} "
        f"frames={summary['frames']:.0f} "
        f"rate={summary['rate_hz']:.1f} Hz "
        f"latency p50={summary['latency_p50_s']*1000:.1f} ms "
        f"p95={summary['latency_p95_s']*1000:.1f} ms"
    )
    if summary["latency_p95_s"] > p95_budget:
        log.warning("p95 latency %.1f ms exceeds budget %.1f ms",
                    summary['latency_p95_s']*1000, p95_budget*1000)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Imitation upper-body controller (Phase 1)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tracker", choices=("mock", "realsense"), default="mock")
    parser.add_argument(
        "--publisher",
        choices=("mock", "udp", "pybullet", "mujoco", "dynamixel"),
        default="mock",
    )
    parser.add_argument(
        "--ik",
        choices=("numeric", "analytic"),
        default=None,
        help="IK 방식 (기본: config main.ik → numeric)",
    )
    parser.add_argument(
        "--model", type=Path, default=None, help="robot model 경로 override (--ik numeric)"
    )
    parser.add_argument("--replay", type=Path, default=None, help="JSONL fixture for mock tracker")
    parser.add_argument("--duration", type=float, default=None, help="seconds before exit")
    parser.add_argument(
        "--deadman",
        action="store_true",
        help="require dead-man hotkey to be held for commands to flow "
             "(forced on for --publisher dynamixel)",
    )
    parser.add_argument(
        "--no-hotkeys",
        action="store_true",
        help="do not install the global keyboard listener (E-stop/reset keys off)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(
        config_path=args.config,
        tracker_kind=args.tracker,
        publisher_kind=args.publisher,
        replay=args.replay,
        duration_s=args.duration,
        require_deadman=args.deadman,
        ik=args.ik,
        model_override=args.model,
        enable_hotkeys=False if args.no_hotkeys else None,
    )


if __name__ == "__main__":
    main()
