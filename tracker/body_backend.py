"""Pluggable body-pose backends.

A ``BodyBackend`` consumes one RGB frame (plus optional depth) and returns
the 11 canonical body keypoints from :data:`core.types.KEYPOINT_NAMES`
in image-aligned metric axes (+x right, +y down, +z into the scene), the
same convention used by MediaPipe ``pose_world_landmarks``.

Backends:
    - :class:`MediaPipeBodyBackend` — MediaPipe Tasks PoseLandmarker. Two modes:
      world-landmarks (CPU, hip-centered meters) or RealSense depth deprojection
      (camera-frame meters via the supplied depth deprojector).
    - :class:`Hmr2BodyBackend` — 4D-Humans HMR2.0 (PyTorch, SMPL-prior, GPU).
      See :mod:`tracker.hmr2_body_backend`. Robust to upper-body occlusion via
      the SMPL kinematic prior.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


@dataclass
class BodyDetection:
    """One detected body. Keypoints in image-aligned metric axes."""

    keypoints: dict[str, NDArray[np.float64]]
    confidence: dict[str, float]
    # Image-coord pixel bbox (x_min, y_min, x_max, y_max). Used by hand backends
    # to seed their crops; for backends that don't naturally expose a bbox, this
    # can be derived from min/max of detected pose landmarks.
    body_bbox_xyxy: tuple[int, int, int, int] | None = None
    # Image-normalized [0, 1] wrist positions — handed to the hand backend
    # so it can locate each hand without re-detecting.
    wrist_image_xy: dict[str, tuple[float, float]] = field(default_factory=dict)


class BodyBackend(ABC):
    """Abstract base for body-pose backends."""

    @abstractmethod
    def start(self) -> None:
        """Allocate models/resources."""

    @abstractmethod
    def stop(self) -> None:
        """Release models/resources."""

    @abstractmethod
    def detect(
        self,
        rgb_image: NDArray[np.uint8],
        timestamp_ms: int,
        depth_image: NDArray[np.uint16] | None = None,
    ) -> BodyDetection | None:
        """Run body-pose detection on one RGB frame.

        Parameters
        ----------
        rgb_image:
            HxWx3 uint8, RGB.
        timestamp_ms:
            Monotonic ms timestamp (MediaPipe VIDEO mode requires monotonic).
        depth_image:
            Optional aligned depth (uint16, depth_scale applied upstream).
            Required only when a backend wants to deproject pixels — MediaPipe
            with ``use_world_landmarks=False`` uses this; HMR2 doesn't.

        Returns
        -------
        :class:`BodyDetection` or None if no body detected.
        """


# Function signature for backends that need to deproject (pixel, depth) → 3D camera point.
# Provided by RealSenseTracker — keeps the backend ignorant of pyrealsense2 details.
DepthDeprojector = Callable[[float, float, float], tuple[float, float, float]]


class MediaPipeBodyBackend(BodyBackend):
    """MediaPipe Tasks PoseLandmarker — default backend.

    When ``use_world_landmarks=True``, body keypoints come from MediaPipe's
    ``pose_world_landmarks`` (hip-centered 3D meters from BlazePose). Otherwise
    each 2D landmark is deprojected through the supplied ``depth_deprojector``.

    Parameters mirror what RealSenseTracker.__init__ used to take.
    """

    # Subset of MediaPipe's 33 landmarks that we forward.
    _LANDMARK_TO_NAME: dict[int, str] = {
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

    def __init__(
        self,
        model_asset_path: str,
        min_visibility: float = 0.5,
        use_world_landmarks: bool = False,
        depth_max_m: float = 4.0,
    ) -> None:
        self._model_asset_path = model_asset_path
        self._min_visibility = min_visibility
        self._use_world_landmarks = use_world_landmarks
        self._depth_max_m = depth_max_m
        # depth_scale + deprojector are injected by RealSenseTracker AFTER it
        # opens the pipeline (we don't know them at backend construction time).
        # When use_world_landmarks=True they're unused.
        self._depth_scale: float = 0.001
        self._deproject: DepthDeprojector | None = None
        self._landmarker: Any = None

    def set_deprojector(self, deprojector: DepthDeprojector, depth_scale: float) -> None:
        """Wire in the camera's depth → 3D unprojection. Required when
        ``use_world_landmarks=False`` so we can convert pixel keypoints + depth
        into camera-frame metric coordinates."""
        self._deproject = deprojector
        self._depth_scale = depth_scale

    def start(self) -> None:
        import mediapipe as mp  # type: ignore[import-not-found]

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_asset_path),
            running_mode=VisionRunningMode.VIDEO,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def stop(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self._landmarker = None

    def detect(
        self,
        rgb_image: NDArray[np.uint8],
        timestamp_ms: int,
        depth_image: NDArray[np.uint16] | None = None,
    ) -> BodyDetection | None:
        if self._landmarker is None:
            return None
        import mediapipe as mp  # type: ignore[import-not-found]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        world = (
            result.pose_world_landmarks[0]
            if self._use_world_landmarks and result.pose_world_landmarks
            else None
        )

        keypoints: dict[str, NDArray[np.float64]] = {}
        confidence: dict[str, float] = {}
        h, w = rgb_image.shape[:2]

        for idx, name in self._LANDMARK_TO_NAME.items():
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            visibility = float(getattr(lm, "visibility", 1.0))
            if visibility < self._min_visibility:
                keypoints[name] = np.zeros(3, dtype=np.float64)
                confidence[name] = 0.0
                continue

            if world is not None and idx < len(world):
                wlm = world[idx]
                keypoints[name] = np.array(
                    [float(wlm.x), float(wlm.y), float(wlm.z)], dtype=np.float64
                )
                confidence[name] = visibility
                continue

            # Depth deprojection path.
            if depth_image is None or self._deproject is None:
                keypoints[name] = np.zeros(3, dtype=np.float64)
                confidence[name] = 0.0
                continue
            px, py = int(round(lm.x * w)), int(round(lm.y * h))
            if px < 0 or px >= w or py < 0 or py >= h:
                keypoints[name] = np.zeros(3, dtype=np.float64)
                confidence[name] = 0.0
                continue
            depth_m = _median_depth_3x3(depth_image, px, py) * self._depth_scale
            if depth_m <= 0.0 or depth_m > self._depth_max_m:
                keypoints[name] = np.zeros(3, dtype=np.float64)
                confidence[name] = 0.0
                continue
            xyz = self._deproject(float(px), float(py), depth_m)
            keypoints[name] = np.asarray(xyz, dtype=np.float64)
            confidence[name] = visibility

        # Synthetic landmarks expected downstream.
        if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
            keypoints["neck"] = 0.5 * (keypoints["left_shoulder"] + keypoints["right_shoulder"])
            confidence["neck"] = min(
                confidence["left_shoulder"], confidence["right_shoulder"]
            )
        if "left_hip" in keypoints and "right_hip" in keypoints and "neck" in keypoints:
            mid_hip = 0.5 * (keypoints["left_hip"] + keypoints["right_hip"])
            keypoints["torso"] = 0.5 * (keypoints["neck"] + mid_hip)
            confidence["torso"] = min(
                confidence["neck"], confidence["left_hip"], confidence["right_hip"]
            )

        wrist_image_xy: dict[str, tuple[float, float]] = {}
        if len(landmarks) > 16:
            wrist_image_xy["left"] = (float(landmarks[15].x), float(landmarks[15].y))
            wrist_image_xy["right"] = (float(landmarks[16].x), float(landmarks[16].y))

        # Loose body bbox from observed landmarks.
        xs = [float(lm.x) for lm in landmarks if lm is not None]
        ys = [float(lm.y) for lm in landmarks if lm is not None]
        bbox: tuple[int, int, int, int] | None = None
        if xs and ys:
            x_min = max(0, int(min(xs) * w) - 10)
            y_min = max(0, int(min(ys) * h) - 10)
            x_max = min(w, int(max(xs) * w) + 10)
            y_max = min(h, int(max(ys) * h) + 10)
            bbox = (x_min, y_min, x_max, y_max)

        return BodyDetection(
            keypoints=keypoints,
            confidence=confidence,
            body_bbox_xyxy=bbox,
            wrist_image_xy=wrist_image_xy,
        )


def _median_depth_3x3(depth_image: NDArray[np.uint16], px: int, py: int) -> float:
    """3x3 median depth (in raw depth units), skipping zeros — speckle resistant."""
    h, w = depth_image.shape
    x0, x1 = max(0, px - 1), min(w, px + 2)
    y0, y1 = max(0, py - 1), min(h, py + 2)
    window = depth_image[y0:y1, x0:x1].ravel()
    valid = window[window > 0]
    return 0.0 if valid.size == 0 else float(np.median(valid))
