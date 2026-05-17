"""HMR2.0 (4D-Humans, CVPR 2023) body-pose backend.

Replaces :class:`tracker.body_backend.MediaPipeBodyBackend` with a ViT-Huge
+ SMPL regressor. The SMPL kinematic prior makes joint positions stay plausible
even when arms/hands cross the torso — the main weakness of MediaPipe Pose.

Setup
-----
1. ``scripts\\install_hmr2.ps1`` clones 4D-Humans + applies Windows patches.
2. HMR2 weights auto-download to ``~/.cache/4DHumans`` on first ``load_hmr2``.
3. SMPL model file ``SMPL_NEUTRAL.pkl`` requires registration at
   https://smpl.is.tue.mpg.de/ (non-commercial license). Place at
   ``~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl``.

This module imports ``hmr2`` and ``torch`` lazily so the rest of the codebase
keeps working when those packages aren't installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tracker.body_backend import BodyBackend, BodyDetection, MediaPipeBodyBackend

log = logging.getLogger(__name__)


class Hmr2UnavailableError(RuntimeError):
    """Raised when HMR2 / PyTorch / SMPL can't be loaded."""


# SMPL 24 joint index → our canonical body keypoint name. The other 13 SMPL
# joints (spine, knees, feet, hands) we don't use upstream.
SMPL_JOINT_TO_NAME: dict[int, str] = {
    1: "left_hip",
    2: "right_hip",
    12: "neck",
    15: "head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
}


class Hmr2BodyBackend(BodyBackend):
    """HMR2-based body backend.

    Pipeline per frame:
        1. Use a coarse body bbox (caller supplies via a thin two-pass: first we
           run a one-shot MediaPipe Pose to get the bbox, then HMR2 on the crop).
           To avoid that two-pass overhead, RealSenseTracker can run MediaPipe
           Pose anyway (for the hand backend's wrist coords) and reuse the bbox.
        2. ViTDetDataset normalizes & crops to 256×192 with ImageNet stats.
        3. HMR2 forward returns ``pred_keypoints_3d`` (root-relative SMPL joints)
           plus ``pred_cam_t``; we sum to get camera-frame meters.
        4. Map SMPL joint indices → our keypoint names.

    Parameters
    ----------
    device:
        ``"cuda"`` or ``"cpu"``. CPU is unusable for live (>1 s/frame).
    checkpoint_path:
        Optional override for HMR2 checkpoint. Defaults to ``DEFAULT_CHECKPOINT``
        which auto-downloads to ``~/.cache/4DHumans`` on first load.
    bbox_padding:
        Fractional expansion of the supplied body bbox before sending to HMR2.
        HMR2 needs some context around the body — 0.15 (= 15 %) is a safe default.
    """

    def __init__(
        self,
        mediapipe_helper: MediaPipeBodyBackend,
        device: str = "cuda",
        checkpoint_path: str | None = None,
        bbox_padding: float = 0.15,
    ) -> None:
        # MediaPipe runs alongside HMR2 to (a) provide a body bbox cheaply and
        # (b) supply image-coord wrist positions for the hand backend's crop.
        # HMR2 then OVERRIDES MediaPipe's metric body keypoints with its
        # SMPL-prior-anchored joints.
        self._mediapipe = mediapipe_helper
        self._device = device
        self._checkpoint_path = checkpoint_path
        self._bbox_padding = bbox_padding
        self._model: Any = None
        self._model_cfg: Any = None
        self._torch: Any = None

    def start(self) -> None:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise Hmr2UnavailableError(
                "PyTorch not installed. Run scripts\\install_hmr2.ps1 first."
            ) from exc

        try:
            from hmr2.models import (  # type: ignore[import-not-found]
                DEFAULT_CHECKPOINT,
                load_hmr2,
            )
        except ImportError as exc:
            raise Hmr2UnavailableError(
                "hmr2 package not installed. Run scripts\\install_hmr2.ps1 first."
            ) from exc

        self._mediapipe.start()

        ckpt = self._checkpoint_path or DEFAULT_CHECKPOINT
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            log.warning(
                "HMR2 checkpoint %s not on disk — load_hmr2 will attempt to auto-download.",
                ckpt,
            )

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but not available; falling back to CPU (unusable for live)")
            self._device = "cpu"

        log.info("Loading HMR2 on %s ...", self._device)
        try:
            model, model_cfg = load_hmr2(ckpt)
        except (AssertionError, FileNotFoundError) as exc:
            # SMPL model file missing is the most common failure mode here.
            raise Hmr2UnavailableError(
                f"HMR2 load failed: {exc}. Check SMPL_NEUTRAL.pkl is at "
                "~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl (see README HMR2 setup)."
            ) from exc
        model.to(self._device).eval()
        self._model = model
        self._model_cfg = model_cfg
        self._torch = torch
        log.info("HMR2 ready.")

    def stop(self) -> None:
        try:
            self._mediapipe.stop()
        except Exception:  # noqa: BLE001
            pass
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
        depth_image: NDArray[np.uint16] | None = None,
    ) -> BodyDetection | None:
        if self._model is None:
            return None

        # First pass: MediaPipe for bbox + wrist image coords (cheap, ~5 ms CPU).
        mp_det = self._mediapipe.detect(rgb_image, timestamp_ms, depth_image)
        if mp_det is None or mp_det.body_bbox_xyxy is None:
            return None
        bbox = self._pad_bbox(mp_det.body_bbox_xyxy, rgb_image.shape[:2], self._bbox_padding)
        if bbox is None:
            return None

        torch = self._torch
        try:
            from hmr2.datasets.vitdet_dataset import (  # type: ignore[import-not-found]
                ViTDetDataset,
            )
        except ImportError:
            log.warning("hmr2.datasets.vitdet_dataset not importable; skipping HMR2 frame")
            return None

        boxes = np.asarray([bbox], dtype=np.float32)
        # HMR2 expects BGR (it uses cv2 conventions internally) — but our caller
        # gives RGB, so swap.
        bgr = rgb_image[..., ::-1].copy()
        dataset = ViTDetDataset(self._model_cfg, bgr, boxes)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in batch.items()}
                out = self._model(batch)
                # pred_keypoints_3d: (1, 44, 3) — root-relative SMPL+extra. First 24 = SMPL.
                # pred_cam_t: (1, 3) — translation in camera frame, in meters.
                kp = (out["pred_keypoints_3d"] + out["pred_cam_t"].unsqueeze(1)).cpu().numpy()[0]
                break
            else:
                return None

        keypoints: dict[str, NDArray[np.float64]] = {}
        confidence: dict[str, float] = {}
        for joint_idx, name in SMPL_JOINT_TO_NAME.items():
            if joint_idx >= kp.shape[0]:
                continue
            keypoints[name] = kp[joint_idx].astype(np.float64)
            confidence[name] = 1.0  # HMR2 doesn't expose per-joint confidence

        # Synthesize torso (midpoint of neck and mid-hip), matching MediaPipe path.
        if "neck" in keypoints and "left_hip" in keypoints and "right_hip" in keypoints:
            mid_hip = 0.5 * (keypoints["left_hip"] + keypoints["right_hip"])
            keypoints["torso"] = 0.5 * (keypoints["neck"] + mid_hip)
            confidence["torso"] = 1.0

        return BodyDetection(
            keypoints=keypoints,
            confidence=confidence,
            body_bbox_xyxy=bbox,
            # Wrist image coords come from MediaPipe — needed by hand backend's crop.
            wrist_image_xy=dict(mp_det.wrist_image_xy),
        )

    @staticmethod
    def _pad_bbox(
        bbox: tuple[int, int, int, int], hw: tuple[int, int], pad: float
    ) -> tuple[int, int, int, int] | None:
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return None
        bw, bh = x1 - x0, y1 - y0
        dx, dy = int(bw * pad), int(bh * pad)
        h, w = hw
        return (
            max(0, x0 - dx),
            max(0, y0 - dy),
            min(w, x1 + dx),
            min(h, y1 + dy),
        )
