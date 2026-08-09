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


def _frame_with_conf(kp: dict[str, np.ndarray], conf: dict[str, float]) -> SkeletonFrame:
    base = {k: 1.0 for k in kp}
    base.update(conf)
    return SkeletonFrame(timestamp=0.0, keypoints=kp, confidence=base)


def test_low_arm_confidence_raises_when_gated(
    robot: RobotGeometry, calib: Calibration
) -> None:
    """A marginal keypoint places the arm plausibly-but-wrongly; the zero-vector
    check can't see it because the keypoint was never hard-rejected."""
    aligned = align_to_torso(
        _frame_with_conf(_tpose_keypoints(), {"right_wrist": 0.2})
    )
    with pytest.raises(SingularConfigurationError, match="confidence"):
        retarget_arm(aligned, "right", robot, calib, min_arm_confidence=0.4)


def test_low_arm_confidence_passes_when_gate_disabled(
    robot: RobotGeometry, calib: Calibration
) -> None:
    aligned = align_to_torso(
        _frame_with_conf(_tpose_keypoints(), {"right_wrist": 0.2})
    )
    angles = retarget_arm(aligned, "right", robot, calib, min_arm_confidence=0.0)
    assert "r_elbow" in angles


def test_gated_arm_omits_all_its_joints_including_passthroughs(
    robot: RobotGeometry, calib: Calibration
) -> None:
    """A failed arm must contribute NOTHING, so the filter holds its last
    command. Defaulting its unobserved joints to 0.0 would snap them to the
    robot's zero pose — the opposite of a hold."""
    aligned = align_to_torso(
        _frame_with_conf(_tpose_keypoints(), {"right_elbow": 0.1})
    )
    angles = retarget_full_upper_body(aligned, robot, calib, min_arm_confidence=0.4)
    for joint in (
        "r_shoulder_pitch", "r_shoulder_roll", "r_elbow",
        "r_shoulder_yaw", "r_wrist_yaw", "r_wrist_pitch",
    ):
        assert joint not in angles, f"{joint} leaked from a gated arm"
    # The healthy arm is unaffected, passthroughs included.
    for joint in ("l_shoulder_pitch", "l_elbow", "l_shoulder_yaw", "l_wrist_pitch"):
        assert joint in angles


def test_degenerate_arm_also_omits_its_passthroughs(
    robot: RobotGeometry, calib: Calibration
) -> None:
    kp = _tpose_keypoints()
    kp["right_elbow"] = np.zeros(3)   # hard reject, zero-vector convention
    kp["right_wrist"] = np.zeros(3)
    angles = retarget_full_upper_body(align_to_torso(_frame(kp)), robot, calib)
    assert "r_elbow" not in angles
    assert "r_shoulder_yaw" not in angles
    assert "l_shoulder_yaw" in angles
