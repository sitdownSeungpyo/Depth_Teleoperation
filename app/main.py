"""Phase 1 entrypoint — wires tracker, aligner, retargeter, filter, safety, publisher.

Usage (PowerShell):
    .\\scripts\\run.ps1 --config .\\config\\default.yaml --tracker mock --publisher mock \\
        --replay .\\tests\\fixtures\\arm_circle.jsonl
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.aligner import AlignmentError, align_to_torso
from core.filter import (
    FilterAndLimiter,
    JointLimits,
    JointLimiterConfig,
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
from publisher.mock_publisher import MockPublisher
from publisher.udp_publisher import UdpPublisherSkeleton
from tracker.body_backend import BodyBackend, MediaPipeBodyBackend
from tracker.hand_backend import HandBackend, MediaPipeHandBackend
from tracker.mock_tracker import MockTracker

log = logging.getLogger(__name__)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        return RealSenseTracker(
            color_resolution=tuple(rs_cfg["color_resolution"]),
            depth_resolution=tuple(rs_cfg["depth_resolution"]),
            fps=int(rs_cfg["fps"]),
            depth_max_m=float(rs_cfg["depth_max_m"]),
            body_backend=body_backend,
            hand_backend=hand_backend,
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
    cfg: dict[str, Any], publisher: Any, loop_dt: float, with_hotkey: bool
) -> SafetyLayer:
    s_cfg = cfg["safety"]
    safety_cfg = SafetyConfig(
        deadman_key=str(s_cfg["deadman"]["key"]),
        estop_key=str(s_cfg["estop_key"]),
        confidence_threshold=float(s_cfg["confidence_threshold"]),
        loss_grace_period_s=float(s_cfg["loss_grace_period_s"]),
        ramp_to_safe_s=float(s_cfg["ramp_to_safe_s"]),
        watchdog_factor=int(s_cfg["watchdog_factor"]),
        cycle_dt_s=loop_dt,
        safe_pose=dict(s_cfg["safe_pose"]),
    )
    hotkey = (
        PynputHotkey([safety_cfg.deadman_key, safety_cfg.estop_key]) if with_hotkey else None
    )
    return SafetyLayer(publisher=publisher, config=safety_cfg, hotkey=hotkey)


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
) -> dict[str, float]:
    cfg = _load_config(config_path)
    main_cfg = cfg.get("main", {})
    loop_rate = float(main_cfg.get("loop_rate_hz", 60))
    loop_dt = 1.0 / loop_rate
    log_every = int(main_cfg.get("latency_log_every", 30))
    p95_budget = float(main_cfg.get("latency_p95_budget_s", 0.1))

    tracker = _build_tracker(cfg, tracker_kind, replay)
    publisher = _build_publisher(cfg, publisher_kind)
    filt = _build_filter(cfg)
    safety = _build_safety(cfg, publisher, loop_dt, with_hotkey=require_deadman)

    # Keypoint smoothing — IK 직선팔 singularity 노이즈 억제. retarget 전에 적용.
    kp_cfg = cfg["filter"].get("keypoint_smoother", {})
    kp_smoother = KeypointSmoother(
        OneEuroParams(
            min_cutoff=float(kp_cfg.get("min_cutoff", 0.5)),
            beta=float(kp_cfg.get("beta", 0.005)),
            d_cutoff=float(kp_cfg.get("d_cutoff", 1.0)),
        )
    )

    # Per-joint motion scaling — robot은 운영자 motion의 N%만큼만 따라가도록 dampen.
    # 운영자 작은 motion → IK가 큰 robot angle 산출하는 amplification 억제.
    # config.retarget.output_gain[joint_name] (기본 1.0). 0.5 = 50% 만 따라감.
    output_gain: dict[str, float] = dict(cfg["retarget"].get("output_gain", {}))

    # Gravity-aligned aligner — 운영자 토르소 tilt 의 영향을 제거.
    # config.tracker.realsense.gravity_up (list[3]) 가 있으면 body-relative 대신 사용.
    gravity_up_cfg = cfg.get("tracker", {}).get("realsense", {}).get("gravity_up")
    gravity_up: "np.ndarray | None" = (
        np.asarray(gravity_up_cfg, dtype=np.float64) if gravity_up_cfg else None
    )

    # Decoupled shoulder_pitch ↔ elbow (Yi 2012 식 3에서 `-theta_5` 제거).
    # MediaPipe elbow 노이즈가 sh_p 로 propagate 되는 것 차단.
    decouple_pitch_elbow = bool(cfg.get("retarget", {}).get("decouple_pitch_elbow", False))

    robot_cfg = cfg["retarget"]["robot"]
    robot = RobotGeometry(
        upper_arm_length=float(robot_cfg["upper_arm_length"]),
        lower_arm_length=float(robot_cfg["lower_arm_length"]),
        shoulder_offset=tuple(robot_cfg["shoulder_offset"]),
    )
    fallback_arm_length = float(cfg["retarget"]["fixed_arm_length"])
    target_calib_frames = int(cfg["retarget"].get("calibration_frames", 30))
    collector = CalibrationCollector(target_frames=target_calib_frames)
    calibration: Calibration | None = None
    log.info(
        "[CALIBRATION] 양팔 자연스럽게 내리고 어깨 편안히 (~%.1fs hold). "
        "자동으로 rest pose 캡쳐 후 robot zero 로 매핑됩니다.",
        target_calib_frames / loop_rate,
    )

    latencies: list[float] = []
    start = time.perf_counter()
    post_calib_start: float | None = None
    frame_count = 0

    tracker.start()
    safety.start()

    # Wall-clock watchdog: if frame yields stop arriving, the for-loop never iterates
    # and the in-loop duration check would never fire. Give the tracker a chance to
    # be stopped cleanly so the loop unblocks.
    duration_thread: threading.Thread | None = None
    if duration_s is not None:
        def _stop_after_duration() -> None:
            time.sleep(duration_s + 2.0)
            try:
                tracker.stop()
            except Exception:  # noqa: BLE001
                pass
        duration_thread = threading.Thread(target=_stop_after_duration, daemon=True)
        duration_thread.start()

    try:
        for frame in tracker.stream():
            now = time.perf_counter()
            if duration_s is not None and now - start > duration_s:
                break
            # Smooth keypoints BEFORE alignment + IK (singularity 노이즈 억제)
            frame = kp_smoother.smooth(frame)
            try:
                aligned = align_to_torso(frame, gravity_up=gravity_up)
            except AlignmentError as exc:
                log.warning("aligner skipped frame: %s", exc)
                safety.note_alive()
                continue

            if calibration is None:
                collector.push(aligned)
                if collector.ready():
                    try:
                        calibration = collector.finalise(
                            robot=robot,
                            decouple_pitch_elbow=decouple_pitch_elbow,
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
                else:
                    safety.note_alive()
                    continue

            try:
                joint_targets = retarget_full_upper_body(
                    aligned, robot, calibration,
                    decouple_pitch_elbow=decouple_pitch_elbow,
                )
            except SingularConfigurationError as exc:
                log.warning("retarget skipped frame: %s", exc)
                safety.note_alive()
                continue

            # Apply per-joint motion scaling (1.0 = pass through)
            if output_gain:
                joint_targets = {
                    j: v * output_gain.get(j, 1.0)
                    for j, v in joint_targets.items()
                }

            cmd = filt(joint_targets, timestamp=now, source_frame_ts=frame.timestamp)
            safety.update(cmd, mean_confidence=frame.mean_confidence())
            safety.watchdog_tick()

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
                    "ELB r=%+.2f l=%+.2f Δ=%+.2f",
                    rate, p95 * 1000,
                    r_sp, l_sp, r_sp - l_sp,
                    r_sr, l_sr, r_sr - l_sr,
                    r_elb, l_elb, r_elb - l_elb,
                )
    finally:
        safety.stop()
        tracker.stop()

    end = time.perf_counter()
    base = post_calib_start if post_calib_start is not None else start
    elapsed = max(end - base, 1e-6)
    summary = {
        "frames": float(frame_count),
        "rate_hz": frame_count / elapsed,
        "latency_p95_s": _percentile(latencies, 95),
        "latency_p50_s": _percentile(latencies, 50),
    }
    print(
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
    parser.add_argument("--replay", type=Path, default=None, help="JSONL fixture for mock tracker")
    parser.add_argument("--duration", type=float, default=None, help="seconds before exit")
    parser.add_argument(
        "--deadman",
        action="store_true",
        help="require dead-man hotkey to be held for commands to flow (default: off)",
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
    )


if __name__ == "__main__":
    main()
