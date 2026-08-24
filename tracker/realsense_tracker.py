"""M1 — RealSense D435i + MediaPipe Pose tracker (spec §4.1).

Implemented to spec but verified by smoke test only when hardware is connected.
The Phase 1 acceptance path uses :class:`tracker.mock_tracker.MockTracker` instead.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.types import SkeletonFrame
from tracker.base import TrackerHealth
from tracker.body_backend import BodyBackend, median_depth_3x3
from tracker.hand_backend import HAND_LANDMARK_TO_SUFFIX, HandBackend

log = logging.getLogger(__name__)

# Re-exported for backward compatibility (used by app.viz_camera). Pose+depth
# logic now lives in tracker.body_backend.MediaPipeBodyBackend; the depth helper
# is defined there too and only re-exported here.
__all__ = ["MEDIAPIPE_LANDMARK_TO_NAME", "RealSenseTracker", "RealSenseUnavailableError",
           "median_depth_3x3"]
MEDIAPIPE_LANDMARK_TO_NAME: dict[int, str] = {
    0: "head",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
}


class RealSenseUnavailableError(RuntimeError):
    """Raised when pyrealsense2 / mediapipe wheels are missing or no D435i is found."""


def _import_realsense() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RealSenseUnavailableError(
            "pyrealsense2 not installed; install with `pip install pyrealsense2`"
        ) from exc
    return rs


class RealSenseTracker:
    """Capture color+depth from a D435i and emit 3D keypoints in the camera frame."""

    # Consecutive capture failures tolerated before the thread gives up and
    # reports itself dead via :meth:`health`. Reopening the pipeline recovers
    # from transient USB glitches; a persistent fault must not retry forever.
    MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        color_resolution: tuple[int, int] = (640, 480),
        depth_resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        depth_max_m: float = 4.0,
        body_backend: BodyBackend | None = None,
        hand_backend: HandBackend | None = None,
        enable_imu: bool = False,
        accel_fps: int = 200,
        gravity_lpf_alpha: float = 0.02,
        gravity_warmup_frames: int = 10,
        gravity_norm_tol: float = 0.30,
        gravity_axis_sign: float = 1.0,
    ) -> None:
        self._color_resolution = color_resolution
        self._depth_resolution = depth_resolution
        self._fps = fps
        self._depth_max_m = depth_max_m
        # IMU gravity: when enabled, the D435i accelerometer is streamed and a
        # measured torso-up vector is produced for the aligner (replaces the
        # hard-coded config gravity_up). Pure estimation logic in tracker.gravity.
        self._enable_imu = bool(enable_imu)
        self._accel_fps = int(accel_fps)
        self._gravity_est: Any = None
        if self._enable_imu:
            from tracker.gravity import GravityEstimator

            self._gravity_est = GravityEstimator(
                lpf_alpha=gravity_lpf_alpha,
                warmup_frames=gravity_warmup_frames,
                norm_tol=gravity_norm_tol,
                axis_sign=gravity_axis_sign,
            )
        self._R_accel_optical: NDArray[np.float64] | None = None  # accel->color rot
        self._latest_gravity_up: NDArray[np.float64] | None = None
        # Body backend produces the 11 canonical keypoints + bbox + wrist image
        # coords. MediaPipe by default; can be swapped to HMR2 for occlusion
        # robustness via config (see tracker.body_backend).
        self._body_backend = body_backend
        # Optional hand backend for wrist orientation + upper-arm twist (sh_yaw).
        # When None, hands are skipped; retargeter falls back to 0 for w_yaw/w_pitch.
        self._hand_backend = hand_backend
        self._lock = threading.Lock()
        self._latest: SkeletonFrame | None = None
        # Wall-clock stamp of the last frame accepted, so callers can tell a
        # *stale* frame from a fresh one. frame.timestamp alone can't: it never
        # changes once the capture thread stops producing.
        self._latest_wall: float | None = None
        # Wall-clock stamp of the last COLOR+DEPTH pair pulled off the camera,
        # whether or not a body was found in it. Tracked separately from
        # ``_latest_wall`` because a detector miss (operator out of shot) must not
        # look like a dead camera — conflating them made a person stepping aside
        # trip the caller's stall watchdog.
        self._latest_capture_wall: float | None = None
        # Smoothed capture interval, so callers can size a stall threshold against
        # the cadence this pipeline actually achieves rather than a fixed guess.
        # The first few intervals are skipped: they include model load and the
        # first CUDA inference, which are seconds long and would poison the EMA.
        self._capture_period: float | None = None
        self._capture_count = 0
        self._CAPTURE_EMA_ALPHA = 0.2
        self._CAPTURE_WARMUP_FRAMES = 5
        self._fatal_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rs_pipeline: Any = None
        self._rs_align: Any = None
        self._depth_scale: float = 0.001
        self._intrinsics: Any = None
        # Exposed for live monitors (app.viz_teleop): most recent BGR color frame
        # and pinhole intrinsics (fx, fy, ppx, ppy). Written by the capture thread,
        # read by the UI thread; a numpy reference swap is atomic in CPython so no
        # lock is needed for a best-effort viz snapshot.
        self._latest_color: NDArray[np.uint8] | None = None
        self._intr_params: tuple[float, float, float, float] | None = None
        # Most recent raw body detection (bbox + wrist image coords) for monitors.
        self._latest_detection: Any = None
        self._hw_offset: float | None = None  # rs_hw_ts - perf_counter offset
        self._last_mp_timestamp_ms: int = 0  # body backends in VIDEO mode need monotonic ts

    def _open_pipeline(self) -> None:
        rs = _import_realsense()
        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.color, self._color_resolution[0], self._color_resolution[1],
            rs.format.bgr8, self._fps,
        )
        cfg.enable_stream(
            rs.stream.depth, self._depth_resolution[0], self._depth_resolution[1],
            rs.format.z16, self._fps,
        )
        if self._enable_imu:
            # Accelerometer only (gyro not needed for gravity-up on a static mount).
            # D435i accel offers 100/200/400 fps — the rate MUST be given or
            # pipeline.start raises "Couldn't resolve requests".
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, self._accel_fps)
        self._rs_pipeline = rs.pipeline()
        profile = self._rs_pipeline.start(cfg)
        self._rs_align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        color_profile = profile.get_stream(rs.stream.color)
        self._intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        self._intr_params = (
            float(self._intrinsics.fx),
            float(self._intrinsics.fy),
            float(self._intrinsics.ppx),
            float(self._intrinsics.ppy),
        )
        if self._enable_imu:
            # Rotation that maps an accel-frame vector into the color optical frame,
            # so gravity comes out in the same frame as the keypoints. If anything
            # fails (no IMU / extrinsics), degrade gracefully to the fixed vector.
            try:
                accel_profile = profile.get_stream(rs.stream.accel)
                extr = accel_profile.get_extrinsics_to(color_profile)
                self._R_accel_optical = np.array(extr.rotation, dtype=np.float64).reshape(
                    3, 3, order="F"
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("IMU extrinsics unavailable (%s); disabling gravity IMU", exc)
                self._enable_imu = False
                self._R_accel_optical = None

    def _open_backends(self) -> None:
        if self._body_backend is None:
            raise RealSenseUnavailableError(
                "RealSenseTracker requires a body_backend (use MediaPipeBodyBackend or Hmr2BodyBackend)"
            )
        # If the body backend (or its inner mediapipe helper for HMR2) needs depth
        # deprojection, wire it up now that intrinsics + depth_scale are known.
        self._inject_deprojector(self._body_backend)
        self._body_backend.start()
        if self._hand_backend is not None:
            self._hand_backend.start()

    def _inject_deprojector(self, backend: Any) -> None:
        rs = _import_realsense()
        intr = self._intrinsics
        depth_scale = self._depth_scale

        def deproject(px: float, py: float, depth_m: float) -> tuple[float, float, float]:
            xyz = rs.rs2_deproject_pixel_to_point(intr, [px, py], depth_m)
            return float(xyz[0]), float(xyz[1]), float(xyz[2])

        if hasattr(backend, "set_deprojector"):
            backend.set_deprojector(deproject, depth_scale)
        # For composite backends (e.g. Hmr2BodyBackend wraps a MediaPipeBodyBackend),
        # also walk into the inner helper.
        inner = getattr(backend, "_mediapipe", None)
        if inner is not None and hasattr(inner, "set_deprojector"):
            inner.set_deprojector(deproject, depth_scale)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._open_pipeline()
        self._open_backends()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="realsense-tracker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._rs_pipeline is not None:
            try:
                self._rs_pipeline.stop()
            except Exception:  # noqa: BLE001
                pass
            self._rs_pipeline = None
        if self._body_backend is not None:
            try:
                self._body_backend.stop()
            except Exception:  # noqa: BLE001
                pass
        if self._hand_backend is not None:
            try:
                self._hand_backend.stop()
            except Exception:  # noqa: BLE001
                pass

    def latest(self) -> SkeletonFrame | None:
        with self._lock:
            return self._latest

    def health(self) -> TrackerHealth:
        """Liveness of the capture thread + age of the most recent frame.

        ``latest()`` keeps handing out the last frame after the thread dies, so
        this is the only way a caller can tell "operator is holding still" from
        "the camera stopped feeding us".
        """
        thread = self._thread
        with self._lock:
            wall = self._latest_wall
            capture_wall = self._latest_capture_wall
            period = self._capture_period
        now = time.perf_counter()
        detection_age = math.inf if wall is None else max(now - wall, 0.0)
        capture_age = (
            math.inf if capture_wall is None else max(now - capture_wall, 0.0)
        )
        return TrackerHealth(
            running=thread is not None and not self._stop.is_set(),
            alive=thread is not None and thread.is_alive(),
            frame_age_s=capture_age,
            error=self._fatal_error,
            detection_age_s=detection_age,
            capture_period_s=math.inf if period is None else period,
        )

    def latest_color(self) -> NDArray[np.uint8] | None:
        """Most recent BGR color frame (for live monitors). None until first capture."""
        return self._latest_color

    @property
    def intrinsics_params(self) -> tuple[float, float, float, float] | None:
        """Color-stream pinhole intrinsics (fx, fy, ppx, ppy). None until started."""
        return self._intr_params

    def latest_detection(self) -> Any:
        """Most recent body detection (bbox + wrist image coords), or None on miss.

        Reliable image-space info for monitors — independent of the backend's 3D
        coordinate scale (HMR2's virtual-camera metric won't reproject onto the
        RealSense image, but the bbox/wrist pixels always will)."""
        return self._latest_detection

    def stream(self) -> Iterator[SkeletonFrame]:
        # The threaded design exposes frames via latest(); a blocking iterator just
        # polls. Most callers should use the Protocol's latest() instead.
        last_ts = 0.0
        while not self._stop.is_set():
            if self._fatal_error is not None:
                # Ending the iterator surfaces the failure to the caller's loop
                # instead of spinning forever on a frame that will never change.
                log.error("capture thread is dead (%s); ending frame stream",
                          self._fatal_error)
                return
            frame = self.latest()
            if frame is not None and frame.timestamp != last_ts:
                last_ts = frame.timestamp
                yield frame
            else:
                time.sleep(0.001)

    def _update_gravity(self, frames: Any) -> None:
        """Extract one accel sample, rotate into the optical frame, feed estimator."""
        rs = _import_realsense()
        accel = frames.first_or_default(rs.stream.accel)
        if not accel or self._R_accel_optical is None:
            return
        m = accel.as_motion_frame().get_motion_data()
        a_sensor = np.array([m.x, m.y, m.z], dtype=np.float64)
        a_optical = self._R_accel_optical @ a_sensor
        up = self._gravity_est.update(a_optical)
        if up is not None:
            self._latest_gravity_up = up

    def latest_gravity_up(self) -> NDArray[np.float64] | None:
        """Measured torso-up vector (color optical frame), or None until the IMU
        estimate is warmed up / when IMU is disabled. Callers fall back to the
        fixed config gravity_up when this is None."""
        return self._latest_gravity_up

    def _convert_hw_ts(self, hw_ts_ms: float) -> float:
        # RealSense gives milliseconds in its own clock; lock to perf_counter on
        # the first frame so downstream code reads a monotonic perf_counter scale.
        now = time.perf_counter()
        if self._hw_offset is None:
            self._hw_offset = now - hw_ts_ms / 1000.0
        return self._hw_offset + hw_ts_ms / 1000.0

    def _process_one(self) -> SkeletonFrame | None:
        try:
            frames = self._rs_pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError as exc:
            log.warning("RealSense wait_for_frames failed (%s); will retry", exc)
            return None

        # IMU gravity: read the accelerometer from the RAW frameset (motion frames
        # are not part of the color-aligned set) and update the gravity estimate.
        if self._enable_imu and self._gravity_est is not None:
            self._update_gravity(frames)

        aligned = self._rs_align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image: NDArray[np.uint16] = np.asanyarray(depth_frame.get_data())
        # Expose the live BGR frame for monitors even when detection later fails.
        self._latest_color = color_image
        # The camera delivered a frame. Stamp that BEFORE running the detector, so
        # camera liveness stays true regardless of whether a body is found in it.
        now = time.perf_counter()
        with self._lock:
            previous = self._latest_capture_wall
            self._latest_capture_wall = now
            self._capture_count += 1
            if previous is not None and self._capture_count > self._CAPTURE_WARMUP_FRAMES:
                interval = now - previous
                self._capture_period = (
                    interval
                    if self._capture_period is None
                    else self._CAPTURE_EMA_ALPHA * interval
                    + (1.0 - self._CAPTURE_EMA_ALPHA) * self._capture_period
                )

        # 라이브 latency 측정에는 hw 클록 변환 대신 호스트 perf_counter 사용.
        # RealSense hw 클록과 perf_counter 사이 drift (1ms/sec 정도)로 인해
        # 변환된 timestamp가 미래 시간처럼 보여 latency가 음수가 되는 문제를 회피.
        _hw_ts_unused = self._convert_hw_ts(float(color_frame.get_timestamp()))
        ts = time.perf_counter()

        # Backends in VIDEO mode require strictly monotonic ms timestamps.
        timestamp_ms = max(int(ts * 1000), self._last_mp_timestamp_ms + 1)
        self._last_mp_timestamp_ms = timestamp_ms

        # MediaPipe / HMR2 expect RGB.
        rgb = color_image[..., ::-1].copy()

        assert self._body_backend is not None
        body_det = self._body_backend.detect(rgb, timestamp_ms, depth_image)
        self._latest_detection = body_det  # expose for monitors (None on miss)
        if body_det is None:
            return None

        keypoints = dict(body_det.keypoints)
        confidence = dict(body_det.confidence)

        # Hand landmarks — optional, for wrist orientation + upper-arm twist.
        if self._hand_backend is not None and body_det.wrist_image_xy:
            self._inject_hand_keypoints(
                rgb, timestamp_ms, body_det.wrist_image_xy, keypoints, confidence
            )

        return SkeletonFrame(timestamp=ts, keypoints=keypoints, confidence=confidence)

    def _inject_hand_keypoints(
        self,
        rgb_image: NDArray[np.uint8],
        timestamp_ms: int,
        wrist_image_xy: dict[str, tuple[float, float]],
        keypoints: dict[str, NDArray[np.float64]],
        confidence: dict[str, float],
    ) -> None:
        """Delegate to the configured HandBackend and merge results into the frame.

        Backends return wrist-relative metric vectors in image-aligned axes
        (same convention as pose_world_landmarks); we translate them onto the
        body wrist position so the aligner rotates everything together into the
        torso frame in one shot.
        """
        assert self._hand_backend is not None
        try:
            detections = self._hand_backend.detect(
                rgb_image, timestamp_ms, wrist_image_xy
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("hand backend detect failed: %s", exc)
            return

        for det in detections:
            body_wrist_key = f"{det.side}_wrist"
            if body_wrist_key not in keypoints:
                continue
            body_wrist_pos = keypoints[body_wrist_key]
            if float(np.linalg.norm(body_wrist_pos)) < 1e-6:
                # Body wrist itself failed visibility — anchoring on (0,0,0) would
                # corrupt downstream geometry.
                continue
            for idx, rel in det.landmarks.items():
                suffix = HAND_LANDMARK_TO_SUFFIX.get(idx)
                if suffix is None:
                    continue
                keypoints[f"{det.side}_{suffix}"] = body_wrist_pos + rel
                confidence[f"{det.side}_{suffix}"] = det.confidence

    def _run(self) -> None:
        backoff = 0.1
        consecutive_failures = 0
        while not self._stop.is_set():
            try:
                frame = self._process_one()
            except Exception as exc:  # noqa: BLE001
                # Deliberately broad. Previously only RuntimeError was caught, so
                # any other error out of a pose backend killed this thread
                # silently — latest() then returned the same stale frame forever
                # and nothing downstream could tell. A dead capture thread must
                # be an observable state, never a silent one.
                consecutive_failures += 1
                log.warning(
                    "RealSense capture error #%d (%s); reopening pipeline",
                    consecutive_failures, exc,
                )
                if consecutive_failures > self.MAX_CONSECUTIVE_FAILURES:
                    self._fatal_error = (
                        f"{type(exc).__name__}: {exc} "
                        f"(after {consecutive_failures} consecutive failures)"
                    )
                    log.error(
                        "RealSense capture thread giving up: %s", self._fatal_error
                    )
                    return
                try:
                    if self._rs_pipeline is not None:
                        self._rs_pipeline.stop()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 2.0)
                try:
                    self._open_pipeline()
                    backoff = 0.1
                except Exception as restart_exc:  # noqa: BLE001
                    log.warning("RealSense restart failed: %s", restart_exc)
                continue
            consecutive_failures = 0
            if frame is None:
                continue
            with self._lock:
                self._latest = frame
                self._latest_wall = time.perf_counter()
