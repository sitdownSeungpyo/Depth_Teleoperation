"""HaMeR (Berkeley CVPR 2024) hand-tracking backend.

Replaces :class:`tracker.hand_backend.MediaPipeHandBackend` with a ViT-Huge
+ MANO regressor for stable per-joint hand orientation. Inputs are the same
(RGB frame + body wrist image coordinates) and outputs use the same
wrist-relative metric vectors in image-aligned axes — so the rest of the
pipeline doesn't change.

Setup
-----
1. ``scripts\\install_hamer.ps1`` installs PyTorch+CUDA + HaMeR + weights.
2. MANO model files (``MANO_RIGHT.pkl``) require manual download from
   https://mano.is.tue.mpg.de/ (non-commercial license).
   Place under ``<hamer_repo>/_DATA/data/mano/``.

This module imports ``hamer`` and ``torch`` lazily so the rest of the
codebase keeps working when those packages aren't installed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tracker.hand_backend import (
    HAND_INDEX_MCP_IDX,
    HAND_LANDMARK_TO_SUFFIX,
    HAND_MIDDLE_MCP_IDX,
    HAND_PINKY_MCP_IDX,
    HAND_WRIST_IDX,
    HandBackend,
    HandDetection,
)

log = logging.getLogger(__name__)


class HamerUnavailableError(RuntimeError):
    """Raised when HaMeR / PyTorch / MANO can't be loaded."""


class HamerHandBackend(HandBackend):
    """HaMeR-based hand backend.

    Pipeline per frame:
        1. Take body wrist pixel coords from MediaPipe Pose.
        2. Build a square crop around each visible wrist (scale by an estimated
           palm size). No ViTDet detection step — we already know where the
           hands are from body pose.
        3. Forward both crops as a batch through HaMeR.
        4. Decode predicted 21 keypoints into the camera (image-aligned) frame
           and return them wrist-relative.

    Parameters
    ----------
    device:
        ``"cuda"`` or ``"cpu"``. CPU works but ~500 ms/hand (unusable for live).
    checkpoint_path:
        Path to HaMeR checkpoint (``*.ckpt``). Defaults to HaMeR's bundled
        ``DEFAULT_CHECKPOINT`` if None.
    mano_dir:
        Directory containing ``MANO_RIGHT.pkl``. Defaults to HaMeR's
        ``_DATA/data/mano`` if None.
    crop_box_scale:
        Multiplier on the estimated palm size when building the square crop.
        Larger captures more context but loses resolution on the hand itself.
    """

    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: str | None = None,
        mano_dir: str | None = None,
        crop_box_scale: float = 2.0,
    ) -> None:
        self._device = device
        self._checkpoint_path = checkpoint_path
        self._mano_dir = mano_dir
        self._crop_box_scale = crop_box_scale
        self._model: Any = None
        self._model_cfg: Any = None
        self._torch: Any = None  # lazy import handle

    def start(self) -> None:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise HamerUnavailableError(
                "PyTorch not installed. Run scripts\\install_hamer.ps1 first."
            ) from exc

        try:
            from hamer.models import (  # type: ignore[import-not-found]
                DEFAULT_CHECKPOINT,
                load_hamer,
            )
        except ImportError as exc:
            raise HamerUnavailableError(
                "hamer package not installed. Run scripts\\install_hamer.ps1 first."
            ) from exc

        # MANO presence check (HaMeR loads MANO inside load_hamer).
        if self._mano_dir is not None:
            mano_pkl = Path(self._mano_dir) / "MANO_RIGHT.pkl"
            if not mano_pkl.exists():
                raise HamerUnavailableError(
                    f"MANO_RIGHT.pkl not found at {mano_pkl}. "
                    "Register at https://mano.is.tue.mpg.de/ and download — see README."
                )
            os.environ.setdefault("MANO_DIR", str(self._mano_dir))

        ckpt = self._checkpoint_path or DEFAULT_CHECKPOINT
        if not Path(ckpt).exists():
            raise HamerUnavailableError(
                f"HaMeR checkpoint not found at {ckpt}. "
                "Run scripts\\install_hamer.ps1 to fetch weights."
            )

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but not available; falling back to CPU (very slow)")
            self._device = "cpu"

        log.info("Loading HaMeR (%s) on %s ...", ckpt, self._device)
        model, model_cfg = load_hamer(ckpt)
        model.to(self._device).eval()
        self._model = model
        self._model_cfg = model_cfg
        self._torch = torch
        log.info("HaMeR ready.")

    def stop(self) -> None:
        if self._model is not None:
            try:
                del self._model
            except Exception:  # noqa: BLE001
                pass
            self._model = None
        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def detect(
        self,
        rgb_image: NDArray[np.uint8],
        timestamp_ms: int,
        body_wrist_image_xy: dict[str, tuple[float, float]],
    ) -> list[HandDetection]:
        if self._model is None:
            return []
        torch = self._torch
        h, w = rgb_image.shape[:2]

        # Build crops for each visible body wrist. Side identity follows the
        # body side directly (no MediaPipe-style handedness ambiguity).
        crops: list[tuple[str, NDArray[np.uint8], tuple[float, float, float]]] = []
        for side, (nx, ny) in body_wrist_image_xy.items():
            cx, cy = int(nx * w), int(ny * h)
            if cx <= 0 or cx >= w or cy <= 0 or cy >= h:
                continue
            # Palm size proxy: 25 % of image height (~ adult forearm length in
            # a typical desktop teleop setup). Refine with shoulder-elbow span
            # downstream if needed.
            half = int(0.5 * self._crop_box_scale * 0.25 * h)
            x0, y0 = max(0, cx - half), max(0, cy - half)
            x1, y1 = min(w, cx + half), min(h, cy + half)
            crop = rgb_image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crops.append((side, crop, (x0, y0, half * 2)))

        if not crops:
            return []

        # HaMeR expects right-hand input; mirror the left crop horizontally and
        # un-mirror the output keypoints below.
        sides: list[str] = []
        right_flag: list[bool] = []
        batch_imgs: list[NDArray[np.uint8]] = []
        bbox_meta: list[tuple[float, float, float]] = []
        for side, crop, meta in crops:
            sides.append(side)
            if side == "left":
                batch_imgs.append(np.ascontiguousarray(crop[:, ::-1, :]))
                right_flag.append(False)
            else:
                batch_imgs.append(crop)
                right_flag.append(True)
            bbox_meta.append(meta)

        with torch.no_grad():
            out = self._forward(batch_imgs, right_flag)
        pred_kp3d = out["pred_keypoints_3d"].cpu().numpy()  # (B, 21, 3)

        detections: list[HandDetection] = []
        for i, side in enumerate(sides):
            kp = pred_kp3d[i]  # (21, 3) in HaMeR's MANO-canonical/camera frame
            if side == "left":
                # Un-mirror the x axis we flipped in the input crop.
                kp = kp.copy()
                kp[:, 0] = -kp[:, 0]
            wrist = kp[HAND_WRIST_IDX].copy()
            landmarks: dict[int, NDArray[np.float64]] = {}
            for idx in (HAND_WRIST_IDX, HAND_INDEX_MCP_IDX, HAND_MIDDLE_MCP_IDX, HAND_PINKY_MCP_IDX):
                if idx in HAND_LANDMARK_TO_SUFFIX:
                    landmarks[idx] = (kp[idx] - wrist).astype(np.float64)
            detections.append(HandDetection(side=side, landmarks=landmarks, confidence=1.0))
        return detections

    def _forward(self, crops: list[NDArray[np.uint8]], right_flag: list[bool]) -> dict[str, Any]:
        """Run HaMeR forward on a batch of pre-cropped square hand images.

        Replicates the minimal preprocessing from HaMeR's ``ViTDetDataset`` —
        resize to model's input size and normalize with ImageNet mean/std.
        """
        torch = self._torch
        from hamer.datasets.vitdet_dataset import (  # type: ignore[import-not-found]
            DEFAULT_MEAN,
            DEFAULT_STD,
        )

        target = self._model_cfg.MODEL.IMAGE_SIZE
        mean = np.asarray(DEFAULT_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(DEFAULT_STD, dtype=np.float32).reshape(3, 1, 1)
        tensors: list[Any] = []
        for img in crops:
            # Resize via OpenCV if available; fall back to numpy stride trick.
            try:
                import cv2  # type: ignore[import-not-found]

                resized = cv2.resize(img, (target, target), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                resized = self._naive_resize(img, target)
            arr = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            arr = (arr - mean) / std
            tensors.append(torch.from_numpy(arr))
        batch = {
            "img": torch.stack(tensors).to(self._device),
            "right": torch.tensor(right_flag, dtype=torch.bool, device=self._device),
        }
        return self._model(batch)

    @staticmethod
    def _naive_resize(img: NDArray[np.uint8], target: int) -> NDArray[np.uint8]:
        h, w = img.shape[:2]
        ys = (np.arange(target) * h / target).astype(int)
        xs = (np.arange(target) * w / target).astype(int)
        return img[ys[:, None], xs[None, :]]
