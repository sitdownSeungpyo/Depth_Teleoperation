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

log = logging.getLogger(__name__)

# MediaPipe pose-landmark indices that map to our canonical keypoint names.
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
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

# MediaPipe hand-landmark indices used for wrist orientation + sh_yaw recovery.
# Reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
HAND_WRIST_IDX = 0
HAND_INDEX_MCP_IDX = 5
HAND_MIDDLE_MCP_IDX = 9
HAND_PINKY_MCP_IDX = 17
HAND_LANDMARK_TO_SUFFIX: dict[int, str] = {
    HAND_WRIST_IDX: "hand_wrist",
    HAND_INDEX_MCP_IDX: "hand_index_mcp",
    HAND_MIDDLE_MCP_IDX: "hand_middle_mcp",
    HAND_PINKY_MCP_IDX: "hand_pinky_mcp",
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


def _import_mediapipe() -> Any:
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise RealSenseUnavailableError(
            "mediapipe not installed; install with `pip install mediapipe`"
        ) from exc
    return mp


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
        min_visibility: float = 0.5,
        model_asset_path: str | None = None,
        use_world_landmarks: bool = False,
        hand_model_asset_path: str | None = None,
        hand_min_confidence: float = 0.5,
    ) -> None:
        self._color_resolution = color_resolution
        self._depth_resolution = depth_resolution
        self._fps = fps
        self._depth_max_m = depth_max_m
        self._min_visibility = min_visibility
        self._model_asset_path = model_asset_path
        # If True, use MediaPipe's pose_world_landmarks (3D meters, hip-centered)
        # instead of RealSense depth deprojection. Avoids depth noise at thin body
        # parts (wrist, elbow) — major source of sh_p over-rotation.
        self._use_world_landmarks = use_world_landmarks
        # Optional hand landmarker for wrist orientation + upper-arm twist (sh_yaw).
        # When None, hands are skipped; retargeter falls back to 0 for w_yaw/w_pitch/sh_yaw.
        self._hand_model_asset_path = hand_model_asset_path
        self._hand_min_confidence = hand_min_confidence
        self._lock = threading.Lock()
        self._latest: SkeletonFrame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rs_pipeline: Any = None
        self._rs_align: Any = None
        self._depth_scale: float = 0.001
        self._intrinsics: Any = None
        self._landmarker: Any = None
        self._hand_landmarker: Any = None
        self._hw_offset: float | None = None  # rs_hw_ts - perf_counter offset
        self._last_mp_timestamp_ms: int = 0  # MediaPipe VIDEO mode requires strict monotonicity

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

    def _open_landmarker(self) -> None:
        mp = _import_mediapipe()
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        if self._model_asset_path is None:
            raise RealSenseUnavailableError(
                "MediaPipe pose model asset path is required (download pose_landmarker_lite.task)"
            )
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_asset_path),
            running_mode=VisionRunningMode.VIDEO,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

        if self._hand_model_asset_path:
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            hand_options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self._hand_model_asset_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=self._hand_min_confidence,
                min_tracking_confidence=self._hand_min_confidence,
            )
            self._hand_landmarker = HandLandmarker.create_from_options(hand_options)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._open_pipeline()
        self._open_landmarker()
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
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self._landmarker = None
        if self._hand_landmarker is not None:
            try:
                self._hand_landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self._hand_landmarker = None

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
        rs = _import_realsense()
        mp = _import_mediapipe()
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
        depth_image_raw = np.asanyarray(depth_frame.get_data())  # uint16, units=depth_scale
        # Convert to a u16 representation in millimetres-scaled units suitable for median.
        depth_image = depth_image_raw

        # 라이브 latency 측정에는 hw 클록 변환 대신 호스트 perf_counter 사용.
        # RealSense hw 클록과 perf_counter 사이 drift (1ms/sec 정도)로 인해
        # 변환된 timestamp가 미래 시간처럼 보여 latency가 음수가 되는 문제를 회피.
        # hw timestamp는 디버깅용으로만 보존 가능.
        _hw_ts_unused = self._convert_hw_ts(float(color_frame.get_timestamp()))
        ts = time.perf_counter()

        # MediaPipe expects RGB.
        rgb = color_image[..., ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # MediaPipe VIDEO mode requires strictly monotonically increasing timestamps.
        timestamp_ms = max(int(ts * 1000), self._last_mp_timestamp_ms + 1)
        self._last_mp_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        world_landmarks = (
            result.pose_world_landmarks[0]
            if self._use_world_landmarks and result.pose_world_landmarks
            else None
        )

        keypoints: dict[str, NDArray[np.float64]] = {}
        confidence: dict[str, float] = {}
        h, w = depth_image.shape
        for idx, name in MEDIAPIPE_LANDMARK_TO_NAME.items():
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            visibility = float(getattr(lm, "visibility", 1.0))
            if visibility < self._min_visibility:
                keypoints[name] = np.zeros(3, dtype=np.float64)
                confidence[name] = 0.0
                continue

            if world_landmarks is not None and idx < len(world_landmarks):
                # MediaPipe pose_world_landmarks: meters, origin at hip center.
                # BlazePose convention: +x=person's right, +y=up, +z=forward of chest.
                # 이 값을 그대로 사용. aligner가 u, v, w 정규화로 torso frame 생성.
                # (이전에 3축 flip 했던 게 cross product 방향을 뒤집어서 sh_p 편향 발생)
                wlm = world_landmarks[idx]
                keypoints[name] = np.array(
                    [float(wlm.x), float(wlm.y), float(wlm.z)],
                    dtype=np.float64,
                )
                confidence[name] = visibility
                continue

            # Default path: RealSense depth deprojection.
            px = int(round(lm.x * w))
            py = int(round(lm.y * h))
            if px < 0 or px >= w or py < 0 or py >= h:
                keypoints[name] = np.zeros(3, dtype=np.float64)
                confidence[name] = 0.0
                continue
            depth_units = median_depth_3x3(depth_image, px, py)
            depth_m = depth_units * self._depth_scale
            if depth_m <= 0.0 or depth_m > self._depth_max_m:
                keypoints[name] = np.zeros(3, dtype=np.float64)
                confidence[name] = 0.0
                continue
            xyz = rs.rs2_deproject_pixel_to_point(self._intrinsics, [float(px), float(py)], depth_m)
            keypoints[name] = np.asarray(xyz, dtype=np.float64)
            confidence[name] = visibility

        # Synthetic landmarks our pipeline expects.
        if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
            keypoints["neck"] = 0.5 * (keypoints["left_shoulder"] + keypoints["right_shoulder"])
            confidence["neck"] = min(confidence["left_shoulder"], confidence["right_shoulder"])
        if "left_hip" in keypoints and "right_hip" in keypoints and "neck" in keypoints:
            mid_hip = 0.5 * (keypoints["left_hip"] + keypoints["right_hip"])
            keypoints["torso"] = 0.5 * (keypoints["neck"] + mid_hip)
            confidence["torso"] = min(
                confidence["neck"], confidence["left_hip"], confidence["right_hip"]
            )

        # Hand landmarks — optional, for wrist orientation + upper-arm twist (sh_yaw).
        # Matched to body side by image-coord proximity to pose left/right wrist.
        if self._hand_landmarker is not None:
            self._inject_hand_keypoints(
                mp_image, timestamp_ms, landmarks, keypoints, confidence
            )

        return SkeletonFrame(timestamp=ts, keypoints=keypoints, confidence=confidence)

    def _inject_hand_keypoints(
        self,
        mp_image: Any,
        timestamp_ms: int,
        pose_landmarks: Any,
        keypoints: dict[str, NDArray[np.float64]],
        confidence: dict[str, float],
    ) -> None:
        """Run HandLandmarker and inject per-side hand keypoints into the frame.

        Hand world_landmarks are wrist-relative metric vectors (image-aligned axes,
        same convention as pose_world_landmarks). We translate them onto the body's
        wrist position so they share a coordinate system with the rest of the
        skeleton; the aligner then rotates everything into the torso frame in
        one shot.
        """
        try:
            hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:  # noqa: BLE001
            log.debug("hand detect failed: %s", exc)
            return
        if not hand_result.hand_landmarks:
            return

        # pose_landmarks 15=left_wrist, 16=right_wrist (image-normalized coords).
        if len(pose_landmarks) <= 16:
            return
        body_lw = pose_landmarks[15]
        body_rw = pose_landmarks[16]

        for i, lm_set in enumerate(hand_result.hand_landmarks):
            if not lm_set:
                continue
            hand_wrist_img = lm_set[HAND_WRIST_IDX]
            # Match to body side by 2D image-coord proximity (handedness from MediaPipe
            # is camera-perspective and ambiguous after image flips, so we ignore it).
            d_l = (hand_wrist_img.x - body_lw.x) ** 2 + (hand_wrist_img.y - body_lw.y) ** 2
            d_r = (hand_wrist_img.x - body_rw.x) ** 2 + (hand_wrist_img.y - body_rw.y) ** 2
            side = "left" if d_l < d_r else "right"
            body_wrist_key = f"{side}_wrist"
            if body_wrist_key not in keypoints:
                continue
            body_wrist_pos = keypoints[body_wrist_key]
            if float(np.linalg.norm(body_wrist_pos)) < 1e-6:
                # Body wrist itself failed visibility — skip so we don't inject
                # nonsense anchored on (0,0,0).
                continue

            world = (
                hand_result.hand_world_landmarks[i]
                if hand_result.hand_world_landmarks and i < len(hand_result.hand_world_landmarks)
                else None
            )
            if world is None:
                continue
            origin = np.array(
                [float(world[HAND_WRIST_IDX].x),
                 float(world[HAND_WRIST_IDX].y),
                 float(world[HAND_WRIST_IDX].z)],
                dtype=np.float64,
            )
            hand_conf = (
                float(hand_result.handedness[i][0].score)
                if hand_result.handedness and i < len(hand_result.handedness)
                else 1.0
            )
            for idx, suffix in HAND_LANDMARK_TO_SUFFIX.items():
                if idx >= len(world):
                    continue
                p = world[idx]
                rel = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64) - origin
                keypoints[f"{side}_{suffix}"] = body_wrist_pos + rel
                confidence[f"{side}_{suffix}"] = hand_conf

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
