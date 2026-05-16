"""M2 — Frame Aligner (spec §4.2).

Builds an operator torso basis from shoulder/head/hip keypoints, projects keypoints into
that frame so retargeting is camera-pose-invariant, and extracts torso roll/pitch/yaw.

References
----------
- Yi et al. 2012, Eq. 4-8.
- Horn et al. 1988 (closest-rotation via polar decomposition).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.types import SkeletonFrame


class AlignmentError(RuntimeError):
    """Raised when the operator basis cannot be constructed (rank-deficient)."""


@dataclass
class AlignedFrame:
    keypoints: dict[str, NDArray[np.float64]]
    rotation: NDArray[np.float64]  # 3x3, camera -> torso
    rpy: tuple[float, float, float]  # roll, pitch, yaw of operator torso (radians)


def _closest_rotation(m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Polar decomposition: nearest rotation to ``m`` in Frobenius norm.

    Uses SVD (R = U Vᵀ) to avoid pulling in scipy.linalg.sqrtm; equivalent to
    Eq. 4-5 of the paper. Det-flip guards against reflections when the basis
    happens to be left-handed.
    """
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def _rpy_from_rotation(r: NDArray[np.float64]) -> tuple[float, float, float]:
    """YXZ intrinsic Euler angles, returned as (roll, pitch, yaw).

    Convention matches the operator torso frame produced by the aligner:
    +x across shoulders, +y up (head→hip axis), +z forward (out of chest).
    So yaw (turning left/right) is rotation about +y, pitch (leaning forward/back)
    is about +x, roll (sideways tilt) is about +z. Decomposition:
    ``R = R_y(yaw) · R_x(pitch) · R_z(roll)``. Paper Eq. 6-8 with this axis assignment.
    """
    pitch = float(np.arcsin(-np.clip(r[1, 2], -1.0, 1.0)))
    cp = float(np.cos(pitch))
    if abs(cp) < 1e-6:
        # Gimbal lock at pitch = ±π/2 — yaw underdetermined; fold into roll.
        roll = float(np.arctan2(-r[0, 1], r[0, 0]))
        yaw = 0.0
    else:
        roll = float(np.arctan2(r[1, 0], r[1, 1]))
        yaw = float(np.arctan2(r[0, 2], r[2, 2]))
    return roll, pitch, yaw


def align_to_torso(
    frame: SkeletonFrame,
    gravity_up: NDArray[np.float64] | None = None,
) -> AlignedFrame:
    """Express a SkeletonFrame in the operator's torso frame (spec §4.2).

    Parameters
    ----------
    frame:
        Skeleton observation in the camera frame. Must contain ``left_shoulder``,
        ``right_shoulder``, ``head``. Hips are preferred but optional — when both
        hips are missing or have zero confidence (e.g. a seated operator with
        lower body out of frame), the vertical axis falls back to head-vs-shoulder.
    gravity_up:
        If provided, use this fixed vector as the torso-up direction (`v`) instead
        of computing it from `head - mid_hip`. Removes operator torso tilt effects
        (sh_r over-elevation + sh_p backward bias). Assumes camera is mounted
        level; typical desktop teleop setup. Pass the world-up direction in the
        same coordinate frame as the keypoints (e.g. ``[0,-1,0]`` for MediaPipe
        pose_world_landmarks where +y is image-down).

        Reference: standard motion capture (Vicon/OptiTrack), Apple ARKit Body
        Tracking, Open-TeleVision (MIT 2024), HumanPlus (Stanford 2024) all use
        a world-fixed gravity-aligned reference instead of operator-relative.

    Returns
    -------
    AlignedFrame with all input keypoints rotated into torso space, plus the
    rotation matrix and RPY of the operator torso.
    """
    required_upper = ("left_shoulder", "right_shoulder", "head")
    for name in required_upper:
        if name not in frame.keypoints:
            raise AlignmentError(f"missing required keypoint: {name}")

    ls = frame.keypoints["left_shoulder"].astype(np.float64)
    rs = frame.keypoints["right_shoulder"].astype(np.float64)
    head = frame.keypoints["head"].astype(np.float64)

    if gravity_up is not None:
        # Gravity-aligned mode: ignore operator body tilt entirely.
        v = np.asarray(gravity_up, dtype=np.float64)
    else:
        hips_present = (
            "left_hip" in frame.keypoints
            and "right_hip" in frame.keypoints
            and frame.confidence.get("left_hip", 0.0) > 0.0
            and frame.confidence.get("right_hip", 0.0) > 0.0
        )
        if hips_present:
            mid_hip = 0.5 * (
                frame.keypoints["left_hip"].astype(np.float64)
                + frame.keypoints["right_hip"].astype(np.float64)
            )
            v = head - mid_hip
        else:
            # Seated-operator fallback: use head minus shoulder midpoint, which is
            # short but points in the right direction for upper-body retargeting.
            mid_shoulder = 0.5 * (ls + rs)
            v = head - mid_shoulder

    u = rs - ls  # +x: across shoulders, right
    w = np.cross(u, v)  # +z: forward (out of the chest)

    if np.linalg.norm(u) < 1e-6 or np.linalg.norm(v) < 1e-6 or np.linalg.norm(w) < 1e-6:
        raise AlignmentError("operator basis is rank-deficient (degenerate pose)")

    m = np.column_stack((u / np.linalg.norm(u), v / np.linalg.norm(v), w / np.linalg.norm(w)))
    r = _closest_rotation(m)

    rt = r.T
    out: dict[str, NDArray[np.float64]] = {}
    for name, p in frame.keypoints.items():
        out[name] = rt @ p.astype(np.float64)

    return AlignedFrame(keypoints=out, rotation=r, rpy=_rpy_from_rotation(r))
