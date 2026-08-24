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
        joint_limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        import mujoco

        self._mj = mujoco
        self.model = model
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        if any(j < 0 for j in jids):
            missing = [n for n, j in zip(joint_names, jids, strict=True) if j < 0]
            raise ValueError(f"joints not found in model: {missing}")
        self.joint_names = list(joint_names)
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids], dtype=int)
        self.dof = np.array([model.jnt_dofadr[j] for j in jids], dtype=int)
        lo: list[float] = []
        hi: list[float] = []
        for j in jids:
            if bool(model.jnt_limited[j]):
                lo.append(float(model.jnt_range[j, 0]))
                hi.append(float(model.jnt_range[j, 1]))
            else:
                lo.append(-np.inf)
                hi.append(np.inf)
        self.lo = np.array(lo)
        self.hi = np.array(hi)
        # Dedicated joint-limit config (robot.joint_limits) overrides the model's
        # jnt_range so limits track the real robot independent of placeholder
        # model dims. Matched by full ("r_elbow_joint") or canonical ("r_elbow").
        if joint_limits:
            for i, n in enumerate(self.joint_names):
                canon = n[:-6] if n.endswith("_joint") else n
                lim = joint_limits.get(n, joint_limits.get(canon))
                if lim is not None:
                    self.lo[i] = float(lim[0])
                    self.hi[i] = float(lim[1])
        self.sb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, shoulder_body)
        self.eb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, elbow_body)
        self.wb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_body)
        if min(self.sb, self.eb, self.wb) < 0:
            raise ValueError("shoulder/elbow/wrist body not found in model")
        self.damping = damping
        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.step_clip = step_clip
        # Distance (m) still separating the bodies from their targets after the
        # last :meth:`solve`. Callers gate on this: a solve that did not converge
        # has produced *a* pose, just not the requested one, and committing it is
        # worse than holding the arm.
        self.last_residual_m = 0.0
        # Only escape once the error is this large (m). Set by RobotModel from the
        # arm's own link lengths. Without it the escape also fires when DLS has
        # simply reached the best pose the arm can hold — 4 joints against 6
        # position constraints rarely leaves zero error — and then it nudges
        # joints that were doing fine. Measured on upperbody_sim.xml: a genuinely
        # stuck solve sits at 0.08-0.80 m while healthy convergence plateaus at
        # or below 0.03 m, so anything in between separates them.
        self.escape_error_m = 0.05
        # Deterministic singularity escape, alternating signs so the nudge is not
        # biased toward one side. See :meth:`solve`.
        self._escape_pattern = np.array(
            [1.0 if i % 2 == 0 else -1.0 for i in range(len(self.joint_names))]
        )

    # A joint whose task-Jacobian column is this short moves the targets by
    # essentially nothing, so DLS has no reason to turn it -- that is what a
    # rank-deficient configuration looks like from inside the loop.
    _WEAK_COLUMN = 1e-3      # metres of body motion per radian
    # Progress counts as stalled below the larger of these: an absolute floor and
    # a fraction of the remaining error.
    #
    # Deliberately NOT a test on ||dq||: at the shoulder singularity the elbow
    # keeps taking large useless steps (it folds to reach the wrist) while the
    # shoulder is frozen, so the iteration looks busy while going nowhere.
    #
    # The relative term matters just as much. DLS approaches a solution
    # asymptotically, so per-iteration progress falls below any fixed floor while
    # the error is still above ``pos_tol``. An absolute-only test fired during
    # ordinary convergence and nudged joints that were doing fine -- observed as
    # shoulder yaw ratcheting to its limit over successive frames of a lateral
    # arm raise, which then never came back.
    _MIN_IMPROVE_M = 1e-4
    _MIN_IMPROVE_RATIO = 0.01
    # ...and the stall has to PERSIST. A descent that passes close to a
    # singularity legitimately crawls for an iteration or two before the useful
    # joint builds up leverage; a true fixed point never recovers. Escaping on the
    # first slow iteration wrecked poses the plain solver reached to 0.02 deg.
    _STALL_ITERS = 4
    _ESCAPE_STEP = 0.08      # rad; nudge applied to leave the singular manifold
    _MAX_ESCAPES = 3

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

        Sets :attr:`last_residual_m` so the caller can tell a converged solve from
        a pose the solver merely stopped at.

        **Singularity escape.** Warm-starting has a failure mode: some
        configurations are fixed points. On a shoulder whose pitch and yaw axes
        coincide while the upper arm lies along them, every column of the task
        Jacobian is zero for certain target directions, so ``dq`` comes out zero
        and the arm never moves again — not this iteration, not this frame, not
        any later frame, because the next warm start begins from the same
        configuration. Measured on ``models/upperbody_sim.xml``: from the rest
        pose, "point the arm forward" stayed 90 deg off for 60 consecutive frames
        with the shoulder joints at exactly 0.000, and raising ``max_iters`` from
        16 to 300 changed nothing. When the iteration stalls we therefore nudge
        the joints that are doing nothing, which leaves the singular manifold and
        lets DLS take over again.
        """
        mj = self._mj
        nv = self.model.nv
        jp = np.zeros((3, nv))
        I6 = np.eye(6)
        escapes = 0
        stalled_iters = 0
        prev_err = np.inf
        for _ in range(self.max_iters):
            mj.mj_forward(self.model, data)
            pe = data.xpos[self.eb]
            pw = data.xpos[self.wb]
            err = np.concatenate([elbow_target - pe, wrist_target - pw])
            err_norm = float(np.linalg.norm(err))
            if err_norm < self.pos_tol:
                break
            if err_norm > self.escape_error_m and (prev_err - err_norm) <= max(
                self._MIN_IMPROVE_M, self._MIN_IMPROVE_RATIO * err_norm
            ):
                stalled_iters += 1
            else:
                stalled_iters = 0
            prev_err = err_norm
            mj.mj_jacBody(self.model, data, jp, None, self.eb)
            je = jp[:, self.dof].copy()
            mj.mj_jacBody(self.model, data, jp, None, self.wb)
            jw = jp[:, self.dof].copy()
            jac = np.vstack([je, jw])  # 6 x n
            # dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
            dq = jac.T @ np.linalg.solve(jac @ jac.T + (self.damping ** 2) * I6, err)
            dq = np.clip(dq, -self.step_clip, self.step_clip)
            if stalled_iters >= self._STALL_ITERS:
                if escapes >= self._MAX_ESCAPES:
                    break  # genuinely unreachable; residual below tells the caller
                escapes += 1
                stalled_iters = 0
                # Nudge only the joints whose column is essentially zero — the
                # ones the rank deficiency is hiding. Nudging a joint that is
                # already contributing just fights the solver.
                weak = np.linalg.norm(jac, axis=0) < self._WEAK_COLUMN
                if not weak.any():
                    weak = np.ones(len(self.qadr), dtype=bool)
                dq = np.where(weak, self._escape_pattern * self._ESCAPE_STEP * escapes, 0.0)
            q = np.clip(data.qpos[self.qadr] + dq, self.lo, self.hi)
            data.qpos[self.qadr] = q
        # Final command clamp — guarantee the returned/applied angles are within
        # [lo, hi] even if the loop converged before an update this call.
        data.qpos[self.qadr] = np.clip(data.qpos[self.qadr], self.lo, self.hi)
        mj.mj_forward(self.model, data)
        self.last_residual_m = float(
            np.linalg.norm(
                np.concatenate(
                    [
                        elbow_target - data.xpos[self.eb],
                        wrist_target - data.xpos[self.wb],
                    ]
                )
            )
        )
        return {
            n: float(data.qpos[a])
            for n, a in zip(self.joint_names, self.qadr, strict=True)
        }
