"""RTMPose body-pose backend (RTMPose body-17, rtmlib + onnxruntime).

Drop-in alternative to :class:`tracker.body_backend.MediaPipeBodyBackend`.
RTMPose outputs 2D pixel keypoints (COCO-17) only, so 3D comes from the same
RealSense depth deprojection path MediaPipe uses with
``use_world_landmarks=False`` — the supplied ``depth_deprojector`` plus the
3x3-median + depth-clamp + score-gating heuristic. RTMPose's high-level
``Body`` carries its own person detector, so this backend produces the body
bbox and wrist image coords directly (no MediaPipe helper needed).

Output axes match the rest of the codebase: image-aligned metric
(+x right, +y down, +z into the scene) — whatever the deprojector returns.

``rtmlib`` is imported lazily in :meth:`start` so the rest of the codebase
keeps working when it isn't installed (mirrors the HMR2/HaMeR backends).
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tracker.body_backend import (
    BodyBackend,
    BodyDetection,
    DepthDeprojector,
    _median_depth_3x3,
)
from tracker.depth_lift import (
    ARM_SEGMENTS,
    BoneLengthStabilizer,
    RobustDepthLifter,
    SegmentConsistencyGate,
)

log = logging.getLogger(__name__)


class RTMPoseUnavailableError(RuntimeError):
    """Raised when rtmlib / onnxruntime can't be loaded."""


def _ensure_onnx_cuda_dll_path() -> None:
    """Best-effort: let onnxruntime-gpu's CUDA provider find the CUDA 12 / cuDNN 9
    runtime DLLs (cublasLt64_12.dll, cudnn64_9.dll, ...).

    onnxruntime-gpu ships the provider DLL but NOT the CUDA runtime itself. On a
    Windows box without a system CUDA Toolkit, the matching DLLs are bundled
    inside an installed PyTorch (``torch/lib``). Without them on PATH the CUDA
    provider fails to initialize and onnxruntime silently falls back to CPU
    (~775 ms/frame vs ~34 ms on GPU). We locate torch's lib dir without importing
    torch and prepend it to the DLL search path. No-op when torch isn't installed
    — then a system CUDA install (if any) is used.
    """
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return
    torch_lib = Path(next(iter(spec.submodule_search_locations))) / "lib"
    if not torch_lib.is_dir():
        return
    lib = str(torch_lib)
    if lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(lib)
    except OSError:  # pragma: no cover - non-Windows / already-added
        pass


class RTMPoseBodyBackend(BodyBackend):
    """RTMPose body-17 backend via rtmlib's high-level ``Body``.

    Parameters
    ----------
    mode:
        rtmlib ``Body`` mode — ``"performance"`` | ``"lightweight"`` | ``"balanced"``.
    device:
        ``"cuda"`` | ``"cpu"`` | ``"mps"``. CUDA via onnxruntime-gpu is the target.
    onnx_backend:
        rtmlib inference backend — ``"onnxruntime"`` | ``"opencv"`` | ``"openvino"``.
    min_visibility:
        COCO keypoint score below which a joint is rejected (zero-vector +
        0.0 confidence). Reuses the same ``tracker.pose.min_visibility`` knob.
    depth_max_m:
        Deprojected depths beyond this (meters) are rejected as invalid.
    """

    # rtmlib ``Body`` emits COCO-17; map the joints we forward to canonical names.
    # 1/2 (eyes), 3/4 (ears), 13-16 (knees/ankles) are unused upstream.
    _COCO_TO_NAME: dict[int, str] = {
        0: "head",  # nose — matches MediaPipe's 0->head
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
    }
    _LEFT_WRIST_IDX = 9
    _RIGHT_WRIST_IDX = 10

    def __init__(
        self,
        mode: str = "balanced",
        device: str = "cuda",
        onnx_backend: str = "onnxruntime",
        min_visibility: float = 0.5,
        depth_max_m: float = 4.0,
        depth_lift: dict[str, Any] | None = None,
    ) -> None:
        self._mode = mode
        self._device = device
        self._onnx_backend = onnx_backend
        self._min_visibility = min_visibility
        self._depth_max_m = depth_max_m
        # Robust depth-lift config (see tracker.depth_lift). None / {enabled:false}
        # falls back to the legacy 3x3-median path.
        self._depth_lift_cfg = depth_lift or {}
        # Injected by RealSenseTracker once intrinsics + depth_scale are known.
        self._depth_scale: float = 0.001
        self._deproject: DepthDeprojector | None = None
        self._body: Any = None
        self._lifter: RobustDepthLifter | None = None
        self._stabilizer: BoneLengthStabilizer | None = None
        self._seg_gate: SegmentConsistencyGate | None = None
        # Frontal-reach fix: kinematic segment-length jump rejection. Holds a
        # limb's last good direction when a bad depth read balloons its length
        # (background bleed when pointing at the camera), preventing the "arm
        # flips backward then returns" jump that length-pinning alone can't stop.
        sg_cfg = self._depth_lift_cfg.get("segment_gate", {}) or {}
        if sg_cfg.get("enabled", True):
            self._seg_gate = SegmentConsistencyGate(
                ARM_SEGMENTS,
                history=int(sg_cfg.get("history", 60)),
                ratio_tol=float(sg_cfg.get("ratio_tol", 0.35)),
                confirm_frames=int(sg_cfg.get("confirm_frames", 3)),
                min_history=int(sg_cfg.get("min_history", 8)),
            )
        if self._depth_lift_cfg.get("bone_stabilize", True):
            self._stabilizer = BoneLengthStabilizer(
                ARM_SEGMENTS,
                history=int(self._depth_lift_cfg.get("bone_history", 60)),
                ratio_min=float(self._depth_lift_cfg.get("bone_ratio_min", 0.5)),
                ratio_max=float(self._depth_lift_cfg.get("bone_ratio_max", 2.0)),
            )

    def set_deprojector(self, deprojector: DepthDeprojector, depth_scale: float) -> None:
        """Wire in the camera's depth -> 3D unprojection (required: RTMPose is
        2D-only, so every keypoint goes through depth deprojection)."""
        self._deproject = deprojector
        self._depth_scale = depth_scale
        if self._depth_lift_cfg.get("enabled", True):
            self._lifter = RobustDepthLifter(
                depth_scale=depth_scale,
                depth_max_m=self._depth_max_m,
                window=int(self._depth_lift_cfg.get("window", 7)),
                foreground_percentile=float(
                    self._depth_lift_cfg.get("foreground_percentile", 40.0)
                ),
                max_jump_m=float(self._depth_lift_cfg.get("max_jump_m", 0.25)),
                max_stale_frames=int(self._depth_lift_cfg.get("max_stale_frames", 5)),
            )

    def start(self) -> None:
        if self._device.startswith("cuda"):
            _ensure_onnx_cuda_dll_path()
        try:
            from rtmlib import Body  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RTMPoseUnavailableError(
                "rtmlib not installed. Run scripts\\install_rtmpose.ps1 "
                "(pip install rtmlib onnxruntime-gpu)."
            ) from exc

        log.info(
            "Loading RTMPose Body (mode=%s, backend=%s, device=%s) ...",
            self._mode,
            self._onnx_backend,
            self._device,
        )
        # Body downloads + caches its onnx models on first call.
        self._body = Body(
            mode=self._mode,
            backend=self._onnx_backend,
            device=self._device,
            to_openpose=False,
        )
        log.info("RTMPose ready.")

    def stop(self) -> None:
        self._body = None

    def detect(
        self,
        rgb_image: NDArray[np.uint8],
        timestamp_ms: int,
        depth_image: NDArray[np.uint16] | None = None,
    ) -> BodyDetection | None:
        if self._body is None:
            return None

        # rtmlib expects BGR; the tracker hands us RGB.
        bgr = rgb_image[..., ::-1].copy()
        keypoints_xy, scores = self._body(bgr)
        if keypoints_xy is None or len(keypoints_xy) == 0:
            return None

        # Single operator — use person 0 only.
        kpts = np.asarray(keypoints_xy[0], dtype=np.float64)  # (17, 2) pixel coords
        scrs = np.asarray(scores[0], dtype=np.float64)  # (17,)

        keypoints: dict[str, NDArray[np.float64]] = {}
        confidence: dict[str, float] = {}
        h, w = rgb_image.shape[:2]
        if self._lifter is not None:
            self._lifter.begin_frame()

        for idx, name in self._COCO_TO_NAME.items():
            if idx >= kpts.shape[0]:
                continue
            score = float(scrs[idx])
            if score < self._min_visibility:
                keypoints[name] = np.zeros(3, dtype=np.float64)
                confidence[name] = 0.0
                continue
            if depth_image is None or self._deproject is None:
                keypoints[name] = np.zeros(3, dtype=np.float64)
                confidence[name] = 0.0
                continue
            px, py = int(round(kpts[idx, 0])), int(round(kpts[idx, 1]))
            if px < 0 or px >= w or py < 0 or py >= h:
                keypoints[name] = np.zeros(3, dtype=np.float64)
                confidence[name] = 0.0
                continue
            # Robust depth (foreground select + temporal gate/hole-fill); falls
            # back to the legacy 3x3 median when no lifter is configured.
            if self._lifter is not None:
                depth_m = self._lifter.lift(name, depth_image, px, py)
            else:
                d = _median_depth_3x3(depth_image, px, py) * self._depth_scale
                depth_m = d if (0.0 < d <= self._depth_max_m) else None
            if depth_m is None:
                keypoints[name] = np.zeros(3, dtype=np.float64)
                confidence[name] = 0.0
                continue
            xyz = self._deproject(float(px), float(py), depth_m)
            keypoints[name] = np.asarray(xyz, dtype=np.float64)
            confidence[name] = score

        # Frontal-reach: reject bad-depth direction flips by segment-length gating
        # FIRST (holds last good direction), then pin segment length.
        if self._seg_gate is not None:
            keypoints = self._seg_gate(keypoints)
        # Bone-length stabilization on the arm chain (before synthetic neck/torso,
        # which derive from the more stable shoulders/hips).
        if self._stabilizer is not None:
            keypoints = self._stabilizer(keypoints)

        # Synthetic landmarks expected downstream (same logic as MediaPipe path).
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

        # Image-normalized wrist coords for the hand backend's crop (no MediaPipe).
        wrist_image_xy: dict[str, tuple[float, float]] = {}
        if kpts.shape[0] > self._RIGHT_WRIST_IDX:
            wrist_image_xy["left"] = (
                float(kpts[self._LEFT_WRIST_IDX, 0] / w),
                float(kpts[self._LEFT_WRIST_IDX, 1] / h),
            )
            wrist_image_xy["right"] = (
                float(kpts[self._RIGHT_WRIST_IDX, 0] / w),
                float(kpts[self._RIGHT_WRIST_IDX, 1] / h),
            )

        # Loose body bbox from all keypoint pixels (seed for hand crops).
        bbox: tuple[int, int, int, int] | None = None
        if kpts.shape[0] > 0:
            x_min = max(0, int(kpts[:, 0].min()) - 10)
            y_min = max(0, int(kpts[:, 1].min()) - 10)
            x_max = min(w, int(kpts[:, 0].max()) + 10)
            y_max = min(h, int(kpts[:, 1].max()) + 10)
            bbox = (x_min, y_min, x_max, y_max)

        return BodyDetection(
            keypoints=keypoints,
            confidence=confidence,
            body_bbox_xyxy=bbox,
            wrist_image_xy=wrist_image_xy,
        )
