"""MuJoCo publisher 통합 테스트 — GUI 없이 sim 동작 검증."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from core.types import JointCommand
from publisher.mujoco_publisher import MuJoCoPublisher

REPO_ROOT = Path(__file__).resolve().parent.parent
MJCF_PATH = REPO_ROOT / "models" / "ubp.xml"


@pytest.fixture
def mujoco_pub():
    pub = MuJoCoPublisher(model_path=MJCF_PATH, rate_hz=100, gui=False)
    pub.start()
    yield pub
    pub.stop()


def test_mjcf_loads_and_lists_actuators(mujoco_pub: MuJoCoPublisher) -> None:
    names = mujoco_pub.actuator_names
    expected = {
        "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
        "r_elbow", "r_wrist_yaw", "r_wrist_pitch",
        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",
        "l_elbow", "l_wrist_yaw", "l_wrist_pitch",
        "neck_yaw", "head_pitch",
    }
    assert expected.issubset(set(names)), f"missing: {expected - set(names)}"


def test_command_drives_joint(mujoco_pub: MuJoCoPublisher) -> None:
    mujoco_pub.set_target(
        JointCommand(
            timestamp=time.perf_counter(),
            positions={"r_elbow": 1.0},
            source_frame_ts=time.perf_counter(),
        )
    )
    time.sleep(1.5)
    state = mujoco_pub.current_joint_state()
    # Position control with kp=50, frictionloss=0.05 → joint moves slowly but should
    # exceed 0.02 rad in 1.5s. Real-time runs are visually faster (camera updates push).
    assert state.get("r_elbow", 0.0) > 0.02, state


def test_unknown_joint_ignored(mujoco_pub: MuJoCoPublisher) -> None:
    mujoco_pub.set_target(
        JointCommand(
            timestamp=time.perf_counter(),
            positions={"nonexistent": 1.0, "r_elbow": 0.5},
            source_frame_ts=time.perf_counter(),
        )
    )
    time.sleep(0.3)
    state = mujoco_pub.current_joint_state()
    assert "r_elbow" in state
