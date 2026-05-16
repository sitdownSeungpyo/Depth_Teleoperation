from __future__ import annotations

import math

import numpy as np
import pytest

from core.aligner import AlignmentError, align_to_torso
from core.types import SkeletonFrame


def _tpose_frame(rotation: np.ndarray | None = None) -> SkeletonFrame:
    base = {
        "head": np.array([0.0, 1.65, 0.0]),
        "left_shoulder": np.array([-0.18, 1.40, 0.0]),
        "right_shoulder": np.array([0.18, 1.40, 0.0]),
        "left_elbow": np.array([-0.46, 1.40, 0.0]),
        "right_elbow": np.array([0.46, 1.40, 0.0]),
        "left_wrist": np.array([-0.73, 1.40, 0.0]),
        "right_wrist": np.array([0.73, 1.40, 0.0]),
        "left_hip": np.array([-0.10, 1.00, 0.0]),
        "right_hip": np.array([0.10, 1.00, 0.0]),
    }
    if rotation is not None:
        base = {k: rotation @ v for k, v in base.items()}
    return SkeletonFrame(
        timestamp=0.0,
        keypoints=base,
        confidence={k: 1.0 for k in base},
    )


def test_identity_tpose_yields_zero_rpy() -> None:
    frame = _tpose_frame()
    aligned = align_to_torso(frame)
    assert np.allclose(aligned.rotation, np.eye(3), atol=1e-6)
    for v in aligned.rpy:
        assert abs(v) < 1e-6


def test_known_rotation_recovers_rpy() -> None:
    # Yaw of 30 degrees about the y axis.
    angle = math.radians(30.0)
    c, s = math.cos(angle), math.sin(angle)
    r_yaw = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    frame = _tpose_frame(r_yaw)
    aligned = align_to_torso(frame)
    roll, pitch, yaw = aligned.rpy
    assert abs(roll) < math.radians(1.0)
    assert abs(pitch) < math.radians(1.0)
    assert abs(yaw - angle) < math.radians(1.0)


def test_missing_required_keypoint_raises() -> None:
    frame = _tpose_frame()
    del frame.keypoints["head"]
    with pytest.raises(AlignmentError):
        align_to_torso(frame)


def test_missing_hips_falls_back_to_shoulder_midpoint() -> None:
    # Seated operator: hips out of frame. Aligner should still produce a sane basis.
    frame = _tpose_frame()
    frame.confidence["left_hip"] = 0.0
    frame.confidence["right_hip"] = 0.0
    aligned = align_to_torso(frame)
    assert np.allclose(aligned.rotation, np.eye(3), atol=1e-6)


def test_degenerate_basis_raises() -> None:
    frame = _tpose_frame()
    frame.keypoints["left_shoulder"] = frame.keypoints["right_shoulder"].copy()
    with pytest.raises(AlignmentError):
        align_to_torso(frame)
