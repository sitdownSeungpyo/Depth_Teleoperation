"""PyBullet publisher 통합 테스트 — DIRECT 모드 (GUI 없이).

URDF 로드 + JointCommand 적용 → joint state 변화 검증.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pybullet = pytest.importorskip("pybullet")

from core.types import JointCommand
from publisher.pybullet_publisher import PyBulletPublisher

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = REPO_ROOT / "models" / "ubp.urdf"


@pytest.fixture
def pybullet_pub():
    pub = PyBulletPublisher(
        urdf_path=URDF_PATH,
        rate_hz=100,
        gui=False,
        fixed_base=True,
    )
    pub.start()
    yield pub
    pub.stop()


def test_urdf_loads_and_lists_joints(pybullet_pub: PyBulletPublisher) -> None:
    names = pybullet_pub.joint_names
    # UBP: 14 DOF, no torso_yaw (fixed torso). Includes shoulder_yaw, wrist_yaw, neck_yaw.
    expected = {
        "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
        "r_elbow", "r_wrist_yaw", "r_wrist_pitch",
        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",
        "l_elbow", "l_wrist_yaw", "l_wrist_pitch",
        "neck_yaw", "head_pitch",
    }
    assert expected.issubset(set(names)), f"missing joints: {expected - set(names)}"


def test_command_drives_joint(pybullet_pub: PyBulletPublisher) -> None:
    # 목표: r_elbow를 1.0 rad로 보내고, 시뮬레이션 진행 후 상태가 그 방향으로 움직였는지 확인.
    pybullet_pub.set_target(
        JointCommand(
            timestamp=time.perf_counter(),
            positions={"r_elbow": 1.0},
            source_frame_ts=time.perf_counter(),
        )
    )
    time.sleep(0.5)  # 100Hz 보간 thread가 여러 step 진행
    state = pybullet_pub.current_joint_state()
    # 위치 제어 + force 10 + 짧은 시간이라 도달 못함, 다만 움직이긴 해야 함.
    assert state.get("r_elbow", 0.0) > 0.05, state


def test_unknown_joint_ignored(pybullet_pub: PyBulletPublisher) -> None:
    pybullet_pub.set_target(
        JointCommand(
            timestamp=time.perf_counter(),
            positions={"nonexistent_joint": 1.0, "r_elbow": 0.5},
            source_frame_ts=time.perf_counter(),
        )
    )
    time.sleep(0.3)
    # 모르는 관절은 silently 무시, 알려진 r_elbow는 처리되어야 함
    state = pybullet_pub.current_joint_state()
    assert "r_elbow" in state
