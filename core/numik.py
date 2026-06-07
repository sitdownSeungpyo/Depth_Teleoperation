"""Numerical task-space inverse kinematics (MuJoCo damped least squares).

Replaces the closed-form Yi-2012 analytic IK with the approach the recent
literature converges on (Ryan 2025 / OmniDP 2026 / MIRROR 2026; Jiang 2025's
joint+Cartesian blend): solve joint angles on the *actual robot model* so the
arm reaches task targets. We drive BOTH the elbow and the wrist body to target
positions — matching the operator's upper-arm AND forearm directions at once.
This removes the 3-DOF analytic IK's bent-arm ambiguity (it could not satisfy
upper-arm direction and wrist position simultaneously).

Model-agnostic: works on any MuJoCo model (MJCF or a loaded URDF) given the
arm's joint names + shoulder/elbow/wrist body names. Jacobians come from
``mujoco.mj_jacBody`` so no kinematics are hard-coded — drop in a real URDF and
it adapts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class ArmPositionIK:
    """Damped-least-squares IK driving one arm's elbow + wrist bodies to targets.

    Parameters
    ----------
    model:
        A ``mujoco.MjModel``.
    joint_names:
        Arm joints to actuate for the position task, base→tip. Typically the 4
        that move the wrist: shoulder pitch/roll/yaw + elbow. (Wrist twist/pitch
        are distal to the wrist body origin and don't change its position.)
    shoulder_body / elbow_body / wrist_body:
        Body names whose origins are the shoulder, elbow, and wrist (hand) points.
    damping:
        DLS Levenberg damping λ (m). Larger = more stable near singularities,
        slower convergence.
    max_iters, pos_tol, step_clip:
        Iteration budget, convergence tolerance (m), and per-iter joint step cap.
    """

    def __init__(
        self,
        model: Any,
        joint_names: list[str],
        shoulder_body: str,
        elbow_body: str,
        wrist_body: str,
        damping: float = 0.08,
        max_iters: int = 16,
        pos_tol: float = 2e-3,
        step_clip: float = 0.35,
    ) -> None:
        import mujoco

        self._mj = mujoco
        self.model = model
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        if any(j < 0 for j in jids):
            missing = [n for n, j in zip(joint_names, jids) if j < 0]
            raise ValueError(f"joints not found in model: {missing}")
        self.joint_names = list(joint_names)
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids], dtype=int)
        self.dof = np.array([model.jnt_dofadr[j] for j in jids], dtype=int)
        lo, hi = [], []
        for j in jids:
            if bool(model.jnt_limited[j]):
                lo.append(float(model.jnt_range[j, 0])); hi.append(float(model.jnt_range[j, 1]))
            else:
                lo.append(-np.inf); hi.append(np.inf)
        self.lo = np.array(lo); self.hi = np.array(hi)
        self.sb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, shoulder_body)
        self.eb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, elbow_body)
        self.wb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_body)
        if min(self.sb, self.eb, self.wb) < 0:
            raise ValueError("shoulder/elbow/wrist body not found in model")
        self.damping = damping
        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.step_clip = step_clip

    def body_pos(self, data: Any, which: str) -> NDArray[np.float64]:
        bid = {"shoulder": self.sb, "elbow": self.eb, "wrist": self.wb}[which]
        return np.array(data.xpos[bid], dtype=np.float64)

    def solve(
        self,
        data: Any,
        elbow_target: NDArray[np.float64],
        wrist_target: NDArray[np.float64],
    ) -> dict[str, float]:
        """Iterate DLS from the current ``data.qpos`` (warm start) toward targets.

        Mutates ``data.qpos`` for the arm joints and returns {joint_name: angle}.
        Warm-starting from the previous frame's solution gives temporal smoothness
        and resolves redundancy implicitly (stays near the last pose).
        """
        mj = self._mj
        nv = self.model.nv
        jp = np.zeros((3, nv))
        I6 = np.eye(6)
        for _ in range(self.max_iters):
            mj.mj_forward(self.model, data)
            pe = data.xpos[self.eb]
            pw = data.xpos[self.wb]
            err = np.concatenate([elbow_target - pe, wrist_target - pw])
            if float(np.linalg.norm(err)) < self.pos_tol:
                break
            mj.mj_jacBody(self.model, data, jp, None, self.eb)
            je = jp[:, self.dof].copy()
            mj.mj_jacBody(self.model, data, jp, None, self.wb)
            jw = jp[:, self.dof].copy()
            jac = np.vstack([je, jw])  # 6 x n
            # dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
            dq = jac.T @ np.linalg.solve(jac @ jac.T + (self.damping ** 2) * I6, err)
            dq = np.clip(dq, -self.step_clip, self.step_clip)
            q = np.clip(data.qpos[self.qadr] + dq, self.lo, self.hi)
            data.qpos[self.qadr] = q
        mj.mj_forward(self.model, data)
        return {n: float(data.qpos[a]) for n, a in zip(self.joint_names, self.qadr)}
