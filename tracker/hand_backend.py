"""Pluggable hand-tracking backends.

A ``HandBackend`` consumes one RGB frame plus the body's 2D wrist coordinates
(from MediaPipe Pose) and returns per-side hand landmarks in image-aligned
metric axes, wrist-relative. RealSenseTracker translates them onto the body
wrist position so the aligner can rotate everything together into the torso
frame.

Backends:
    - :class:`MediaPipeHandBackend` — MediaPipe Tasks HandLandmarker (CPU, 7.5 MB).
    - :class:`HamerHandBackend` — Berkeley HaMeR (PyTorch + MANO, GPU, ~1.5 GB).
      See :mod:`tracker.hamer_hand_backend`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# Canonical landmark indices we forward to the retargeter (subset of MediaPipe's
# 21 hand landmarks). Keep the names stable across backends.
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


@dataclass
class HandDetection:
    """One detected hand. ``landmarks`` keyed by MediaPipe-style index, values
    are wrist-relative metric vectors in image-aligned axes (+x right, +y down,
    +z away from camera)."""

    side: str  # "left" | "right"
    landmarks: dict[int, NDArray[np.float64]]
    confidence: float


class HandBackend(ABC):
    """Abstract base for hand-tracking backends."""

    @abstractmethod
    def start(self) -> None:
        """Allocate models/resources. Called once before the first detect()."""

    @abstractmethod
    def stop(self) -> None:
        """Release models/resources."""

    @abstractmethod
    def detect(
        self,
        rgb_image: NDArray[np.uint8],
        timestamp_ms: int,
        body_wrist_image_xy: dict[str, tuple[float, float]],
    ) -> list[HandDetection]:
        """Run hand detection on one RGB frame.

        Parameters
        ----------
        rgb_image:
            HxWx3 uint8 image (RGB).
        timestamp_ms:
            Monotonic millisecond timestamp (required by MediaPipe VIDEO mode).
        body_wrist_image_xy:
            Body left/right wrist positions in *normalized* image coords
            ([0, 1] range) — used to assign detected hands to a side.

        Returns
        -------
        Up to two :class:`HandDetection` (one per side).
        """


class MediaPipeHandBackend(HandBackend):
    """MediaPipe Tasks HandLandmarker (default CPU backend)."""

    def __init__(self, model_asset_path: str, min_confidence: float = 0.5) -> None:
        self._model_asset_path = model_asset_path
        self._min_confidence = min_confidence
        self._landmarker: Any = None

    def start(self) -> None:
        import mediapipe as mp  # type: ignore[import-not-found]

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_asset_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self._min_confidence,
            min_tracking_confidence=self._min_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

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
        body_wrist_image_xy: dict[str, tuple[float, float]],
    ) -> list[HandDetection]:
        if self._landmarker is None:
            return []
        import mediapipe as mp  # type: ignore[import-not-found]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        try:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:  # noqa: BLE001
            log.debug("MediaPipe hand detect failed: %s", exc)
            return []
        if not result.hand_landmarks:
            return []

        out: list[HandDetection] = []
        for i, lm_set in enumerate(result.hand_landmarks):
            if not lm_set:
                continue
            hand_wrist_img = lm_set[HAND_WRIST_IDX]
            # Side assignment from image-coord proximity (MediaPipe handedness is
            # camera-perspective and ambiguous after image flips).
            side = self._closest_side(
                hand_wrist_img.x, hand_wrist_img.y, body_wrist_image_xy
            )
            if side is None:
                continue
            world = (
                result.hand_world_landmarks[i]
                if result.hand_world_landmarks and i < len(result.hand_world_landmarks)
                else None
            )
            if world is None:
                continue
            origin = np.array(
                [
                    float(world[HAND_WRIST_IDX].x),
                    float(world[HAND_WRIST_IDX].y),
                    float(world[HAND_WRIST_IDX].z),
                ],
                dtype=np.float64,
            )
            landmarks: dict[int, NDArray[np.float64]] = {}
            for idx in HAND_LANDMARK_TO_SUFFIX:
                if idx >= len(world):
                    continue
                p = world[idx]
                rel = (
                    np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
                    - origin
                )
                landmarks[idx] = rel
            confidence = (
                float(result.handedness[i][0].score)
                if result.handedness and i < len(result.handedness)
                else 1.0
            )
            out.append(HandDetection(side=side, landmarks=landmarks, confidence=confidence))
        return out

    @staticmethod
    def _closest_side(
        x: float,
        y: float,
        body_wrist_image_xy: dict[str, tuple[float, float]],
    ) -> str | None:
        best: tuple[float, str] | None = None
        for side, (bx, by) in body_wrist_image_xy.items():
            d = (x - bx) ** 2 + (y - by) ** 2
            if best is None or d < best[0]:
                best = (d, side)
        return None if best is None else best[1]
