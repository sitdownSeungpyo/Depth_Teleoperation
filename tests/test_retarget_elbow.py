"""Retargeter elbow 굽힘 검증 — Yi 2012 Eq. 1이 다양한 각도에서 정확한지 확인."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.aligner import align_to_torso
from core.retarget import Calibration, RobotGeometry, retarget_arm
from core.types import SkeletonFrame


SHOULDER = 0.18
UPPER = 0.28
LOWER = 0.27
ARM = UPPER + LOWER


def _make_frame(elbow_flex_rad: float, side: str = "right") -> SkeletonFrame:
    """양팔을 만든다. side 쪽 팔은 elbow_flex_rad 만큼 굽힘.

    Operator coordinate (camera-style: +x right, +y up, +z forward).
    어깨를 원점으로 두고 upper_arm은 -x 방향 (오른팔)이거나 +x (왼팔).
    팔꿈치를 +x축 둘레로 elbow_flex_rad 만큼 회전하여 굽힘을 만든다.
    """
    sign = -1 if side == "right" else +1
    # 기본 T-pose 자세 (어깨, 머리, 힙)
    kp = {
        "head": np.array([0.0, 1.65, 0.0]),
        "left_shoulder": np.array([+SHOULDER, 1.40, 0.0]),
        "right_shoulder": np.array([-SHOULDER, 1.40, 0.0]),
        "left_elbow": np.array([+SHOULDER + UPPER, 1.40, 0.0]),
        "left_wrist": np.array([+SHOULDER + UPPER + LOWER, 1.40, 0.0]),
        "right_elbow": np.array([-SHOULDER - UPPER, 1.40, 0.0]),
        "right_wrist": np.array([-SHOULDER - UPPER - LOWER, 1.40, 0.0]),
        "left_hip": np.array([+0.10, 1.00, 0.0]),
        "right_hip": np.array([-0.10, 1.00, 0.0]),
    }
    # 굽힘 변환: 우측 팔의 forearm을 elbow 기준으로 회전.
    # forearm vector at T-pose: (sign*LOWER, 0, 0) (팔 같은 방향)
    # 굽혀서 forearm이 +z (앞으로) 향하도록 = +x 축에 대해 회전
    # 각도 theta_5: 0=straight, π/2=90도 굽힘
    elbow_key = f"{side}_elbow"
    wrist_key = f"{side}_wrist"
    elbow_pos = kp[elbow_key]
    # forearm 벡터 회전: 원래 (sign*LOWER, 0, 0). theta=0 그대로, theta=π/2면 (0, 0, sign*LOWER) (앞쪽)
    # 굽힘 각도가 theta_5일 때 forearm은 elbow에서 theta_5 만큼 위/앞으로 굽혀짐.
    # forearm = (sign*LOWER*cos(theta), 0, sign*LOWER*sin(theta))? 부호 주의
    # 사실 cos(theta_5) = upper · lower / (|upper||lower|). upper = elbow-shoulder = (sign*UPPER, 0, 0).
    # 우리가 forearm을 (sign*LOWER*cos(α), 0, sign*LOWER*sin(α)) 형태로 만들면:
    # upper · lower = sign*UPPER * sign*LOWER * cos(α) = UPPER*LOWER*cos(α)
    # cos_elbow = cos(α). theta_5 = α. ✓
    alpha = elbow_flex_rad
    fx = sign * LOWER * math.cos(alpha)
    fz = sign * LOWER * math.sin(alpha)  # 양쪽 팔 모두 +z 쪽으로 굽히도록 부호 사용
    # 굽힘이 'forward' 방향(+z) 이도록: right 팔도 +z, left 팔도 +z 향하도록.
    # 위 식대로 하면 right 팔(sign=-1)일 때 fz=-LOWER*sin(α)로 -z 방향이 됨. 보정:
    fz = LOWER * math.sin(alpha)  # 양쪽 모두 +z (forward)
    new_wrist = elbow_pos + np.array([fx, 0.0, fz])
    kp[wrist_key] = new_wrist
    return SkeletonFrame(
        timestamp=0.0,
        keypoints=kp,
        confidence={k: 1.0 for k in kp},
    )


@pytest.fixture
def robot() -> RobotGeometry:
    return RobotGeometry(
        upper_arm_length=UPPER, lower_arm_length=LOWER, shoulder_offset=(0.0, 0.0, 0.0)
    )


@pytest.fixture
def calib() -> Calibration:
    return Calibration(operator_arm_length=ARM)


@pytest.mark.parametrize(
    "flex_rad",
    [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, 2 * math.pi / 3, math.pi - 0.01],
)
def test_elbow_flexion_right(flex_rad: float, robot: RobotGeometry, calib: Calibration) -> None:
    aligned = align_to_torso(_make_frame(flex_rad, "right"))
    angles = retarget_arm(aligned, "right", robot, calib)
    assert angles["r_elbow"] == pytest.approx(flex_rad, abs=math.radians(2.0)), (
        f"flex={math.degrees(flex_rad):.1f}° → r_elbow={math.degrees(angles['r_elbow']):.1f}°"
    )


@pytest.mark.parametrize(
    "flex_rad",
    [0.0, math.pi / 4, math.pi / 2, math.pi - 0.01],
)
def test_elbow_flexion_left(flex_rad: float, robot: RobotGeometry, calib: Calibration) -> None:
    aligned = align_to_torso(_make_frame(flex_rad, "left"))
    angles = retarget_arm(aligned, "left", robot, calib)
    assert angles["l_elbow"] == pytest.approx(flex_rad, abs=math.radians(2.0))
