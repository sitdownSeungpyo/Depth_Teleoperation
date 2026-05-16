from __future__ import annotations

import math

import numpy as np
import pytest

from core.aligner import align_to_torso
from core.retarget import (
    Calibration,
    RobotGeometry,
    SingularConfigurationError,
    estimate_arm_length,
    retarget_arm,
    retarget_full_upper_body,
)
from core.types import SkeletonFrame


SHOULDER = 0.18
UPPER = 0.28
LOWER = 0.27
ARM = UPPER + LOWER


def _tpose_keypoints() -> dict[str, np.ndarray]:
    return {
        "head": np.array([0.0, 1.65, 0.0]),
        "left_shoulder": np.array([-SHOULDER, 1.40, 0.0]),
        "right_shoulder": np.array([SHOULDER, 1.40, 0.0]),
        "left_elbow": np.array([-SHOULDER - UPPER, 1.40, 0.0]),
        "right_elbow": np.array([SHOULDER + UPPER, 1.40, 0.0]),
        "left_wrist": np.array([-SHOULDER - ARM, 1.40, 0.0]),
        "right_wrist": np.array([SHOULDER + ARM, 1.40, 0.0]),
        "left_hip": np.array([-0.10, 1.00, 0.0]),
        "right_hip": np.array([0.10, 1.00, 0.0]),
    }


def _arm_down_keypoints() -> dict[str, np.ndarray]:
    kp = _tpose_keypoints()
    kp["right_elbow"] = np.array([SHOULDER, 1.40 - UPPER, 0.0])
    kp["right_wrist"] = np.array([SHOULDER, 1.40 - ARM, 0.0])
    kp["left_elbow"] = np.array([-SHOULDER, 1.40 - UPPER, 0.0])
    kp["left_wrist"] = np.array([-SHOULDER, 1.40 - ARM, 0.0])
    return kp


def _frame(kp: dict[str, np.ndarray]) -> SkeletonFrame:
    return SkeletonFrame(timestamp=0.0, keypoints=kp, confidence={k: 1.0 for k in kp})


@pytest.fixture
def robot() -> RobotGeometry:
    return RobotGeometry(upper_arm_length=0.28, lower_arm_length=0.27, shoulder_offset=(0.0, 0.0, 0.0))


@pytest.fixture
def calib() -> Calibration:
    return Calibration(operator_arm_length=ARM)


def test_tpose_right_arm(robot: RobotGeometry, calib: Calibration) -> None:
    aligned = align_to_torso(_frame(_tpose_keypoints()))
    angles = retarget_arm(aligned, "right", robot, calib)
    assert abs(angles["r_elbow"]) < math.radians(1.0)
    assert abs(angles["r_shoulder_roll"] - math.pi / 2) < math.radians(1.0)


def test_tpose_left_arm(robot: RobotGeometry, calib: Calibration) -> None:
    aligned = align_to_torso(_frame(_tpose_keypoints()))
    angles = retarget_arm(aligned, "left", robot, calib)
    assert abs(angles["l_elbow"]) < math.radians(1.0)
    assert abs(angles["l_shoulder_roll"] - math.pi / 2) < math.radians(1.0)


def test_arm_down_all_near_zero(robot: RobotGeometry, calib: Calibration) -> None:
    aligned = align_to_torso(_frame(_arm_down_keypoints()))
    angles = retarget_full_upper_body(aligned, robot, calib)
    for joint in (
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_elbow",
        "l_shoulder_pitch",
        "l_shoulder_roll",
        "l_elbow",
    ):
        assert abs(angles[joint]) < math.radians(2.0), (joint, angles[joint])


def test_zero_length_raises(robot: RobotGeometry, calib: Calibration) -> None:
    kp = _tpose_keypoints()
    kp["right_elbow"] = kp["right_shoulder"].copy()
    kp["right_wrist"] = kp["right_shoulder"].copy()
    aligned = align_to_torso(_frame(kp))
    with pytest.raises(SingularConfigurationError):
        retarget_arm(aligned, "right", robot, calib)


def test_estimate_arm_length() -> None:
    kp = _tpose_keypoints()
    assert abs(estimate_arm_length(kp) - ARM) < 1e-6


def test_calibration_from_tpose_frames() -> None:
    aligned = [align_to_torso(_frame(_tpose_keypoints())) for _ in range(5)]
    cal = Calibration.from_tpose_frames(aligned)
    assert abs(cal.operator_arm_length - ARM) < 1e-6
