"""Closed-loop kinematic verification of models/upperbody_sim.xml.

operator pose → core.retarget (paper Eq.1-3) → model FK → compare to operator.
Asserts the model's joint axes realize the paper IK: STRAIGHT-arm wrist direction
matches exactly (<0.5 deg). Bent arms carry the paper's inherent 3-DOF
approximation and are only checked to run without NaN.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from core.aligner import AlignedFrame
from core.retarget import Calibration, RobotGeometry, retarget_arm

MODEL = Path("models/upperbody_sim.xml")
UPPER, LOWER = 0.26, 0.24
# root body euler="pi/2 0 0": maps torso-local (x,y,z) -> world (x,-z,y).
_R_ROOT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def _rot(v, axis, ang):
    axis = axis / np.linalg.norm(axis)
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def _ang(a, b):
    return math.degrees(math.acos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))


def _load():
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    return m, d


def _qadr(m, joint):
    return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)]


def _bid(m, body):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)


@pytest.mark.parametrize("side", ["right", "left"])
def test_straight_arm_direction_exact(side: str) -> None:
    m, d = _load()
    robot = RobotGeometry(UPPER, LOWER, (0, 0, 0))
    calib = Calibration(operator_arm_length=UPPER + LOWER)
    pre = side[0]
    sb = _bid(m, f"{pre}_shoulder_pitch_link")
    wb = _bid(m, f"{pre}_hand_link")
    qp = {n: _qadr(m, f"{pre}_{n}_joint") for n in ["shoulder_pitch", "shoulder_roll", "elbow"]}

    errs = []
    for th in [0.2, 0.6, 1.0, 1.4, 1.8]:
        for ph in [-1.2, -0.5, 0.2, 0.9, 1.5]:
            u = np.array([math.sin(th) * math.cos(ph), -math.cos(th), math.sin(th) * math.sin(ph)])
            sh = np.zeros(3)
            el = sh + UPPER * u
            wr = el + LOWER * u  # straight arm
            af = AlignedFrame(
                keypoints={f"{side}_shoulder": sh, f"{side}_elbow": el, f"{side}_wrist": wr},
                rotation=np.eye(3), rpy=(0, 0, 0),
            )
            a = retarget_arm(af, side, robot, calib)
            d.qpos[:] = 0
            d.qpos[qp["shoulder_pitch"]] = a[f"{pre}_shoulder_pitch"]
            d.qpos[qp["shoulder_roll"]] = a[f"{pre}_shoulder_roll"]
            d.qpos[qp["elbow"]] = a[f"{pre}_elbow"]
            mujoco.mj_forward(m, d)
            robot_dir = d.xpos[wb] - d.xpos[sb]
            op_dir = _R_ROOT @ (wr - sh)  # operator dir into world frame
            errs.append(_ang(robot_dir, op_dir))
    assert max(errs) < 0.5, f"{side}: max straight-arm wrist-dir err {max(errs):.3f} deg"


def test_rest_pose_arms_down() -> None:
    m, d = _load()
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    for pre in ("r", "l"):
        sh = d.xpos[_bid(m, f"{pre}_shoulder_pitch_link")]
        wr = d.xpos[_bid(m, f"{pre}_hand_link")]
        v = wr - sh
        # arm hangs straight down in world (-z) at the all-zero pose.
        assert v[2] < -0.4 and abs(v[0]) < 0.05 and abs(v[1]) < 0.05, f"{pre} rest not down: {v}"


def test_bent_arm_runs_without_nan() -> None:
    m, d = _load()
    robot = RobotGeometry(UPPER, LOWER, (0, 0, 0))
    calib = Calibration(operator_arm_length=UPPER + LOWER)
    sh = np.zeros(3)
    u = np.array([0.0, -1.0, 0.0])
    f = _rot(u, np.array([1.0, 0, 0]), 1.2)  # bent 1.2 rad
    el = sh + UPPER * u
    wr = el + LOWER * f
    af = AlignedFrame(
        keypoints={"right_shoulder": sh, "right_elbow": el, "right_wrist": wr},
        rotation=np.eye(3), rpy=(0, 0, 0),
    )
    a = retarget_arm(af, "right", robot, calib)
    assert all(math.isfinite(v) for v in a.values())
    assert a["r_elbow"] == pytest.approx(1.2, abs=math.radians(2))
