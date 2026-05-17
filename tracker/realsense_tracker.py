"""M1 — RealSense D435i + MediaPipe Pose tracker (spec §4.1).

Implemented to spec but verified by smoke test only when hardware is connected.
The Phase 1 acceptance path uses :class:`tracker.mock_tracker.MockTracker` instead.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.types import SkeletonFrame
from tracker.body_backend import BodyBackend
from tracker.hand_backend import HAND_LANDMARK_TO_SUFFIX, HandBackend

log = logging.getLogger(__name__)

# Re-exported for backward compatibility (used by app.viz_camera). Pose+depth
# logic now lives in tracker.body_backend.MediaPipeBodyBackend.
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
        import pyrealsense2 as rs  # type: ignore
    except ImportError as exc:
        raise RealSenseUnavailableError(
            "pyrealsense2 not installed; install with `pip install pyrealsense2`"
        ) from exc
    return rs


def median_depth_3x3(depth_image: NDArray[np.uint16], px: int, py: int) -> float:
    """Return the 3x3 median depth (in meters, after applying ``depth_scale`` upstream).

    Caller passes depth in metres already scaled. We use a 3x3 window around (px, py),
    skipping zeros (invalid depth), to suppress speckle (FR-1.5).
    """
    h, w = depth_image.shape
    x0, x1 = max(0, px - 1), min(w, px + 2)
    y0, y1 = max(0, py - 1), min(h, py + 2)
    window = depth_image[y0:y1, x0:x1].ravel()
    valid = window[window > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


class RealSenseTracker:
    """Capture color+depth from a D435i and emit 3D keypoints in the camera frame."""

    def __init__(
        self,
        color_resolution: tuple[int, int] = (640, 480),
        depth_resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        depth_max_m: float = 4.0,
        body_backend: BodyBackend | None = None,
        hand_backend: HandBackend | None = None,
    ) -> None:
        self._color_resolution = color_resolution
        self._depth_resolution = depth_resolution
        self._fps = fps
        self._depth_max_m = depth_max_m
        # Body backend produces the 11 canonical keypoints + bbox + wrist image
        # coords. MediaPipe by default; can be swapped to HMR2 for occlusion
        # robustness via config (see tracker.body_backend).
        self._body_backend = body_backend
        # Optional hand backend for wrist orientation + upper-arm twist (sh_yaw).
        # When None, hands are skipped; retargeter falls back to 0 for w_yaw/w_pitch.
        self._hand_backend = hand_backend
        self._lock = threading.Lock()
        self._latest: SkeletonFrame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rs_pipeline: Any = None
        self._rs_align: Any = None
        self._depth_scale: float = 0.001
        self._intrinsics: Any = None
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
        self._rs_pipeline = rs.pipeline()
        profile = self._rs_pipeline.start(cfg)
        self._rs_align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        color_profile = profile.get_stream(rs.stream.color)
        self._intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

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

    def stream(self) -> Iterator[SkeletonFrame]:
        # The threaded design exposes frames via latest(); a blocking iterator just
        # polls. Most callers should use the Protocol's latest() instead.
        last_ts = 0.0
        while not self._stop.is_set():
            frame = self.latest()
            if frame is not None and frame.timestamp != last_ts:
                last_ts = frame.timestamp
                yield frame
            else:
                time.sleep(0.001)

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

        aligned = self._rs_align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image: NDArray[np.uint16] = np.asanyarray(depth_frame.get_data())

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
        rs = _import_realsense()
        backoff = 0.1
        while not self._stop.is_set():
            try:
                frame = self._process_one()
            except RuntimeError as exc:
                log.warning("RealSense pipeline error (%s); reopening", exc)
                try:
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
            _ = rs  # keep symbol referenced for potential future use
            if frame is None:
                continue
            with self._lock:
                self._latest = frame
