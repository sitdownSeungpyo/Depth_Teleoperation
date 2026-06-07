"""Closed-loop test of numerical IK (core.numik + core.robot_model).

The win over the Yi-2012 analytic IK: numerical IK matches BOTH the upper-arm
(shoulder→elbow) and forearm (elbow→wrist) directions, so BENT arms track
accurately too (analytic had ~25-48 deg bent-arm error). Operator link lengths
differ from the robot's here, exercising the auto length-scaling.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from core.robot_model import RobotModel

R_OP2ROB = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
UPPER_OP, LOWER_OP = 0.30, 0.28  # operator arm (differs from robot model lengths)


def _cfg() -> dict:
    arm = lambda p: {  # noqa: E731
        "joints": [f"{p}_shoulder_pitch_joint", f"{p}_shoulder_roll_joint",
                   f"{p}_shoulder_yaw_joint", f"{p}_elbow_joint"],
        "shoulder_body": f"{p}_shoulder_pitch_link",
        "elbow_body": f"{p}_elbow_link",
        "wrist_body": f"{p}_hand_link",
    }
    return {
        "model_path": "models/upperbody_sim.xml",
        "operator_to_robot_R": R_OP2ROB,
        "ik": {"damping": 0.04, "max_iters": 60, "pos_tol": 1e-4, "step_clip": 0.5},
        "arms": {"right": arm("r"), "left": arm("l")},
    }


def _rot(v, axis, ang):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def _ang(a, b):
    return math.degrees(math.acos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("bend", [0.0, 0.6, 1.2, 1.8])
def test_ik_matches_upper_and_forearm_directions(side: str, bend: float) -> None:
    rm = RobotModel(_cfg())
    R = np.array(R_OP2ROB, float)
    arm = rm.arms[side]
    worst_u, worst_f = 0.0, 0.0
    for th in [0.3, 0.8, 1.3, 1.7]:
        for ph in [-1.0, 0.0, 1.0]:
            u = np.array([math.sin(th) * math.cos(ph), -math.cos(th), math.sin(th) * math.sin(ph)])
            perp = np.cross(u, [0, -1.0, 0])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.array([1.0, 0, 0])
            f = _rot(u, perp, bend)
            sh_op = np.zeros(3)
            el_op = sh_op + UPPER_OP * u
            wr_op = el_op + LOWER_OP * f
            rm.data.qpos[:] = 0  # solve from neutral (worst case, no warm start)
            sol = rm.solve_arm(side, sh_op, el_op, wr_op)
            assert sol is not None
            # robot directions in world vs operator directions mapped to world
            sh_r = arm.ik.body_pos(rm.data, "shoulder")
            el_r = arm.ik.body_pos(rm.data, "elbow")
            wr_r = arm.ik.body_pos(rm.data, "wrist")
            worst_u = max(worst_u, _ang(el_r - sh_r, R @ u))
            worst_f = max(worst_f, _ang(wr_r - el_r, R @ f))
    assert worst_u < 3.0, f"{side} bend={bend}: upper-arm dir err {worst_u:.2f} deg"
    assert worst_f < 3.0, f"{side} bend={bend}: forearm dir err {worst_f:.2f} deg"


def test_link_lengths_measured_from_model() -> None:
    rm = RobotModel(_cfg())
    for side, (u, l) in rm.link_lengths().items():
        assert 0.1 < u < 0.5 and 0.1 < l < 0.5, f"{side} lengths ({u},{l}) implausible"
