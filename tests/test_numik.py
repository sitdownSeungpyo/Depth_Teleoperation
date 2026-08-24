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
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
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


def _straight_and_bent_targets():
    """Operator shoulder/elbow/wrist for a straight arm and a 1.2 rad bent one."""
    u = np.array([math.sin(1.0), -math.cos(1.0), 0.0])
    perp = np.cross(u, [0, -1.0, 0])
    sh = np.zeros(3)
    el = sh + UPPER_OP * u
    return (sh, el, el + LOWER_OP * u), (sh, el, el + LOWER_OP * _rot(u, perp, 1.2))


def test_elbow_seeded_out_of_the_straight_arm_singularity() -> None:
    """From a straight-arm start, shoulder yaw can't move the wrist, so DLS
    stalls unless the elbow is nudged off the singular set first."""
    rm = RobotModel(_cfg())
    _, bent = _straight_and_bent_targets()
    rm.data.qpos[:] = 0  # exactly on the singularity
    sol = rm.solve_arm("right", *bent)
    assert sol is not None
    assert sol["r_elbow_joint"] > 0.5, "solver never left the straight-arm pose"


def test_elbow_seeding_does_not_overwrite_a_bent_warm_start() -> None:
    """The seed used to be applied unconditionally, every frame — which silently
    replaced the caller's smoothed elbow angle (written back to qpos between
    frames) with the raw measurement, so the elbow alone tracked unfiltered while
    its three sibling joints were filtered.

    max_iters=0 isolates the seeding decision: solve_arm then returns exactly
    what qpos holds after seeding, with no solver motion mixed in.
    """
    cfg = _cfg()
    # max_iters=0 means no solver motion at all, so the residual is whatever the
    # seed left — the non-convergence gate would reject every call and hide the
    # seeding decision this test exists to check.
    cfg["ik"] = {**cfg["ik"], "max_iters": 0, "max_residual_m": 0.0}
    rm = RobotModel(cfg)
    _, bent = _straight_and_bent_targets()
    raw_flex = 1.2  # how _straight_and_bent_targets() bends the forearm
    eadr = int(rm.arms["right"].ik.qadr[-1])

    # Already bent (outside the singular neighbourhood): a caller-supplied,
    # smoothed value must survive untouched.
    warm = 0.55
    rm.data.qpos[:] = 0
    rm.data.qpos[eadr] = warm
    sol = rm.solve_arm("right", *bent)
    assert sol is not None
    assert sol["r_elbow_joint"] == pytest.approx(warm)
    assert abs(sol["r_elbow_joint"] - raw_flex) > 0.5   # NOT the raw measurement

    # Straight (inside the neighbourhood): seeding still applies.
    rm.data.qpos[:] = 0
    sol = rm.solve_arm("right", *bent)
    assert sol is not None
    assert sol["r_elbow_joint"] == pytest.approx(raw_flex, abs=1e-6)


def test_elbow_seed_threshold_is_configurable_and_disablable() -> None:
    cfg = _cfg()
    cfg["ik"] = {**cfg["ik"], "elbow_seed_below_rad": 0.0}   # seeding off
    rm = RobotModel(cfg)
    assert rm._elbow_seed_below == 0.0  # noqa: SLF001
    _, bent = _straight_and_bent_targets()
    rm.data.qpos[:] = 0
    # With seeding disabled the solver still runs; it just isn't helped off the
    # singularity. Assert only that it stays well-defined.
    sol = rm.solve_arm("right", *bent)
    assert sol is not None
    assert all(math.isfinite(v) for v in sol.values())


def test_link_lengths_measured_from_model() -> None:
    rm = RobotModel(_cfg())
    for side, (upper, lower) in rm.link_lengths().items():
        assert 0.1 < upper < 0.5 and 0.1 < lower < 0.5, (
            f"{side} lengths ({upper},{lower}) implausible"
        )


def test_config_joint_limits_override_model_and_are_enforced() -> None:
    """robot.joint_limits must override the model's jnt_range and clamp every
    solved angle, even for poses that 'want' to exceed them."""
    cfg = _cfg()
    cfg["joint_limits"] = {
        "r_shoulder_yaw": [-0.2, 0.2],   # much tighter than model ±3.14
        "r_elbow": [0.0, 1.0],           # tighter than model [0, 3.05]
    }
    # The sweep below deliberately asks for poses these limits forbid (bend=2.0
    # against an elbow capped at 1.0), so the non-convergence gate would reject
    # them — correctly, but that is a different behaviour, tested elsewhere. Turn
    # it off so every pose comes back and the clamp itself stays under test.
    cfg["ik"] = {**cfg["ik"], "max_residual_m": 0.0}
    rm = RobotModel(cfg)
    ik = rm.arms["right"].ik
    # Limits actually loaded into the solver.
    lo = dict(zip(ik.joint_names, ik.lo, strict=True))
    hi = dict(zip(ik.joint_names, ik.hi, strict=True))
    assert (lo["r_shoulder_yaw_joint"], hi["r_shoulder_yaw_joint"]) == (-0.2, 0.2)
    assert (lo["r_elbow_joint"], hi["r_elbow_joint"]) == (0.0, 1.0)
    # Sweep poses; no solved angle may leave its configured band.
    for th in [0.4, 0.9, 1.4]:
        for ph in [-1.2, 0.0, 1.2]:
            for bend in [0.0, 1.0, 2.0]:
                u = np.array([math.sin(th) * math.cos(ph), -math.cos(th),
                              math.sin(th) * math.sin(ph)])
                perp = np.cross(u, [0, -1.0, 0])
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.array([1.0, 0, 0])
                f = _rot(u, perp, bend)
                rm.data.qpos[:] = 0
                sol = rm.solve_arm("right", np.zeros(3), UPPER_OP * u,
                                   UPPER_OP * u + LOWER_OP * f)
                assert sol is not None
                assert -0.2 - 1e-6 <= sol["r_shoulder_yaw_joint"] <= 0.2 + 1e-6
                assert 0.0 - 1e-6 <= sol["r_elbow_joint"] <= 1.0 + 1e-6


def test_target_jump_limiter_clamps_endpoint_step() -> None:
    """An abnormal endpoint jump is clamped to max_target_step_m of the last
    target; a genuine move slews; the limiter is off by default."""
    cfg = _cfg()
    cfg["ik"] = {**cfg["ik"], "max_target_step_m": 0.05}
    rm = RobotModel(cfg)
    a = np.array([0.1, 0.0, 1.0])
    seeded = rm._limit_step("k", a.copy())  # first call seeds, no clamp
    np.testing.assert_allclose(seeded, a, atol=1e-12)
    jumped = a + np.array([0.20, 0.0, 0.0])  # 0.20 m teleport
    out = rm._limit_step("k", jumped.copy())
    assert float(np.linalg.norm(out - a)) <= 0.05 + 1e-9          # clamped
    np.testing.assert_allclose(out, a + np.array([0.05, 0.0, 0.0]), atol=1e-9)
    # next frame continues to slew toward the (still far) target, 0.05 m/step
    out2 = rm._limit_step("k", jumped.copy())
    assert float(np.linalg.norm(out2 - out)) <= 0.05 + 1e-9

    rm_off = RobotModel(_cfg())  # default max_target_step_m = 0 -> disabled
    np.testing.assert_allclose(rm_off._limit_step("k", jumped.copy()), jumped, atol=1e-12)


def test_config_in_repo_loads_joint_limits() -> None:
    """The shipped config (split files, merged via include) feeds robot.joint_limits
    from joint_limit.yaml into the robot section, and RobotModel picks it up."""
    from core.config import load_config

    cfg = load_config("config/ubp.yaml")
    rm = RobotModel(dict(cfg["robot"]))
    ik = rm.arms["right"].ik
    hi = dict(zip(ik.joint_names, ik.hi, strict=True))
    lo = dict(zip(ik.joint_names, ik.lo, strict=True))
    # shoulder_yaw tightened to ±1.57 in the shipped config (vs model ±3.14).
    assert hi["r_shoulder_yaw_joint"] == pytest.approx(1.57)
    assert lo["r_shoulder_yaw_joint"] == pytest.approx(-1.57)
