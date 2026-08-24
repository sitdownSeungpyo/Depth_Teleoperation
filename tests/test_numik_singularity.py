"""Singularity escape and the non-converged-solve gate (core.numik + core.robot_model).

``models/upperbody_sim.xml`` puts shoulder pitch and yaw on the same axis, and at
the rest pose the upper arm lies along it. Every column of the task Jacobian is
then zero for some target directions, so damped least squares computes a zero
step — and because the next frame warm-starts from the same configuration, the
arm never moves again. Measured before the fix: asked to point forward from rest,
the shoulder joints stayed at exactly 0.000 for 60 consecutive frames, and raising
``max_iters`` from 16 to 300 changed nothing.

The escape has to be narrow. A first attempt fired whenever an iteration stopped
making progress, which also happens during ordinary convergence — it then nudged
healthy joints every frame until shoulder yaw ratcheted into its limit and a plain
lateral arm raise was rejected on 89 of 90 frames. So these tests pin both
directions: the stuck pose must recover, and the healthy motions must be left
alone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from core.robot_model import RobotModel

R_OP2ROB = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
UPPER_OP, LOWER_OP = 0.28, 0.27
SHOULDER_OP = np.array([0.18, 1.4, 0.0])
UP = np.array([0.0, 1.0, 0.0])
FWD = np.array([0.0, 0.0, 1.0])


def _cfg(**ik_overrides: float) -> dict:
    def arm(p: str) -> dict:
        return {
            "joints": [f"{p}_shoulder_pitch_joint", f"{p}_shoulder_roll_joint",
                       f"{p}_shoulder_yaw_joint", f"{p}_elbow_joint"],
            "shoulder_body": f"{p}_shoulder_pitch_link",
            "elbow_body": f"{p}_elbow_link",
            "wrist_body": f"{p}_hand_link",
        }

    ik: dict = {"damping": 0.08, "max_iters": 16, "pos_tol": 2e-3, "step_clip": 0.35,
                "max_target_step_m": 0.0, "max_residual_m": 0.05}
    ik.update(ik_overrides)
    return {
        "model_path": "models/upperbody_sim.xml",
        "operator_to_robot_R": R_OP2ROB,
        "ik": ik,
        "arms": {"right": arm("r"), "left": arm("l")},
        "joint_limits": {
            "r_shoulder_yaw": [-1.57, 1.57], "r_elbow": [0.0, 2.79],
            "l_shoulder_yaw": [-1.57, 1.57], "l_elbow": [0.0, 2.79],
        },
    }


def _unit(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) / float(np.linalg.norm(v))


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    k = _unit(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * (kx @ kx)


def _arm_dirs(az: float, el: float, flex: float) -> tuple[np.ndarray, np.ndarray]:
    """Upper-arm and forearm directions in the operator torso frame (+x right,
    +y up, +z forward). ``el`` is measured from straight down."""
    upper = _unit(np.array([math.sin(el) * math.sin(az), -math.cos(el),
                            math.sin(el) * math.cos(az)]))
    ref = np.cross(upper, UP)
    if float(np.linalg.norm(ref)) < 1e-6:
        ref = np.cross(upper, FWD)
    return upper, _unit(_rot(_unit(ref), flex) @ upper)


def _points(upper: np.ndarray, fore: np.ndarray) -> tuple[np.ndarray, ...]:
    elbow = SHOULDER_OP + UPPER_OP * upper
    return SHOULDER_OP, elbow, elbow + LOWER_OP * fore


def _achieved(rm: RobotModel, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Robot's own upper-arm and forearm directions, back in the operator frame."""
    ik = rm.arms[side].ik
    inv = np.linalg.inv(rm.frame_R)
    ps = ik.body_pos(rm.data, "shoulder")
    pe = ik.body_pos(rm.data, "elbow")
    pw = ik.body_pos(rm.data, "wrist")
    return _unit(inv @ (pe - ps)), _unit(inv @ (pw - pe))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def test_forward_reach_escapes_the_rest_singularity() -> None:
    """From the rest pose, "point the arm forward" is the locked direction.

    At rest the upper arm lies along both the pitch and yaw axes, and the roll
    axis is parallel to the requested motion, so no joint can reduce the error.
    Without an escape this is a permanent fixed point, not a slow convergence.
    """
    rm = RobotModel(_cfg())
    upper = _unit(np.array([0.0, 0.0, 1.0]))
    shoulder, elbow, wrist = _points(upper, upper)

    solved_within = None
    for frame in range(1, 11):
        rm.solve_arm("right", shoulder, elbow, wrist)
        got_upper, _ = _achieved(rm, "right")
        if _angle_deg(got_upper, upper) < 2.0:
            solved_within = frame
            break
    assert solved_within is not None, (
        "shoulder never left the singularity; qpos="
        f"{np.round(rm.data.qpos[rm.arms['right'].ik.qadr], 4)}"
    )
    assert solved_within <= 5, f"took {solved_within} frames to escape"


def test_lateral_raise_is_left_alone_by_the_escape() -> None:
    """A plain sideways raise is fully reachable by shoulder roll.

    The regression this guards: an over-eager escape nudged shoulder yaw every
    frame in the same direction until it pinned at its limit, after which the
    residual gate rejected essentially the whole motion.
    """
    rm = RobotModel(_cfg())
    held = 0
    errors: list[float] = []
    for i in range(60):
        elevation = math.radians(15.0 + 75.0 * i / 59.0)
        upper, fore = _arm_dirs(math.radians(90.0), elevation, math.radians(10.0))
        solution = rm.solve_arm("right", *_points(upper, fore))
        if solution is None:
            held += 1
            errors.append(math.inf)
            continue
        got_upper, got_fore = _achieved(rm, "right")
        errors.append(max(_angle_deg(got_upper, upper), _angle_deg(got_fore, fore)))

    assert held == 0, f"{held}/60 frames rejected on a fully reachable motion"
    # The first frames start from the rest pose and need a few solves to catch up,
    # so the steady state is what matters. Note we deliberately do NOT assert on
    # shoulder yaw staying small: it legitimately runs out to its configured limit
    # here (measured 1.56 rad), and it does so identically with the escape disabled
    # — the arm has no null space to redistribute, so that is kinematics, not drift.
    assert max(errors[5:]) < 2.0, f"steady-state error {max(errors[5:]):.1f} deg"
    assert errors[0] < 10.0, f"cold-start error {errors[0]:.1f} deg"


@pytest.mark.parametrize(
    ("name", "az_deg", "el_deg", "flex_deg"),
    [
        ("overhead", 45.0, 150.0, 30.0),
        ("cross-body", 10.0, 70.0, 60.0),
        ("curl", 0.0, 15.0, 120.0),
        ("hand to head", 90.0, 120.0, 140.0),
        ("punch forward", 0.0, 85.0, 5.0),
    ],
)
def test_reachable_poses_track_within_a_degree(
    name: str, az_deg: float, el_deg: float, flex_deg: float
) -> None:
    rm = RobotModel(_cfg())
    upper, fore = _arm_dirs(math.radians(az_deg), math.radians(el_deg),
                            math.radians(flex_deg))
    points = _points(upper, fore)
    solution = None
    for _ in range(6):  # a couple of 30 Hz frames, as the live loop would
        solution = rm.solve_arm("right", *points)
    assert solution is not None, f"{name}: solve rejected a reachable pose"
    got_upper, got_fore = _achieved(rm, "right")
    assert _angle_deg(got_upper, upper) < 2.0, name
    assert _angle_deg(got_fore, fore) < 2.0, name


def test_unreachable_target_is_rejected_rather_than_approximated() -> None:
    """Fold the elbow past its stop: 175 deg of flexion against a [0, 2.79] rad
    (160 deg) limit. A least-squares solver still returns *a* pose; committing it
    would move the arm somewhere the operator never asked for."""
    rm = RobotModel(_cfg())
    upper, fore = _arm_dirs(math.radians(0.0), math.radians(85.0), math.radians(175.0))
    points = _points(upper, fore)
    for _ in range(10):
        solution = rm.solve_arm("right", *points)
    assert solution is None
    assert rm.last_residual("right") > 0.05


def test_a_reachable_deep_bend_is_still_accepted() -> None:
    """The gate must reject the impossible without also rejecting the merely
    extreme — 150 deg of flexion is inside the elbow's travel."""
    rm = RobotModel(_cfg())
    upper, fore = _arm_dirs(math.radians(0.0), math.radians(85.0), math.radians(150.0))
    points = _points(upper, fore)
    for _ in range(10):
        solution = rm.solve_arm("right", *points)
    assert solution is not None
    assert rm.last_residual("right") < 0.05


def test_residual_is_reported_for_a_converged_solve() -> None:
    rm = RobotModel(_cfg())
    upper, fore = _arm_dirs(math.radians(45.0), math.radians(60.0), math.radians(45.0))
    points = _points(upper, fore)
    for _ in range(6):
        rm.solve_arm("right", *points)
    assert rm.last_residual("right") < 0.01


def test_gate_can_be_disabled() -> None:
    rm = RobotModel(_cfg(max_residual_m=0.0))
    upper, fore = _arm_dirs(math.radians(0.0), math.radians(85.0), math.radians(175.0))
    solution = rm.solve_arm("right", *_points(upper, fore))
    assert solution is not None, "max_residual_m=0 must mean 'never reject'"
