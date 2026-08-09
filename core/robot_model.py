"""Robot-model adapter — the single place that knows the robot's geometry.

Loads a MuJoCo model (MJCF, or a URDF that MuJoCo can parse), auto-measures each
arm's link lengths and shoulder location at the rest pose, holds the
operator→robot frame transform, and turns operator keypoints into world-frame
elbow/wrist IK targets. Solving is delegated to :class:`core.numik.ArmPositionIK`.

Designed for DROP-IN robot models: when a real URDF/MJCF arrives, only
``config.robot`` changes — model_path, per-arm body/joint names, and the
operator→robot rotation. Link lengths are measured from the model (not hard-
coded), so arbitrary placeholder dimensions are replaced automatically.

URDF/xacro/SDF notes:
- URDF: MuJoCo's compiler loads many URDFs directly via from_xml_path. Add a
  ``<mujoco><compiler .../></mujoco>`` block inside the URDF if meshes/inertia
  need options. SDF is NOT supported by MuJoCo — convert to MJCF/URDF first.
- xacro is a preprocessor: run ``xacro robot.xacro > robot.urdf`` first, then
  point model_path at the generated URDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.numik import ArmPositionIK


def _unit(v: NDArray[np.float64]) -> NDArray[np.float64] | None:
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


class _Arm:
    def __init__(self, ik: ArmPositionIK, upper_len: float, lower_len: float) -> None:
        self.ik = ik
        self.upper_len = upper_len
        self.lower_len = lower_len


class RobotModel:
    """Loaded robot + per-arm IK, parameterized entirely by ``config.robot``."""

    def __init__(self, robot_cfg: dict[str, Any]) -> None:
        import mujoco

        self._mj = mujoco
        model_path = Path(robot_cfg["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"robot model not found: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)  # rest pose for length measurement

        # operator torso frame (+x right, +y up, +z fwd) → robot world frame.
        R = robot_cfg.get("operator_to_robot_R")
        self.frame_R = np.asarray(R, dtype=np.float64) if R is not None else np.eye(3)

        ik_cfg = robot_cfg.get("ik", {})
        # Task-space target jump limiter: cap how far each arm's elbow/wrist IK
        # TARGET may move per frame, so an abnormal pose-estimate jump can't
        # teleport the endpoint — it gets clamped toward the last good target and
        # a genuine fast reach slews over a few frames. 0 disables. (Redundancy /
        # "nearest of multiple IK solutions" is already handled by warm-starting
        # the DLS from the previous qpos, so this guards the INPUT, not the solver.)
        self._max_target_step = float(ik_cfg.get("max_target_step_m", 0.0))
        self._last_targets: dict[str, NDArray[np.float64]] = {}
        # Elbow seeding threshold (rad). See :meth:`solve_arm`. 0 disables seeding.
        self._elbow_seed_below = float(ik_cfg.get("elbow_seed_below_rad", 0.15))
        # Dedicated joint-limit block (rad), overrides the model's jnt_range in
        # the IK. Keyed by joint name (full or canonical); each arm's IK picks
        # the joints it actuates. Empty -> fall back to model limits.
        jl_cfg = robot_cfg.get("joint_limits") or {}
        self.joint_limits: dict[str, tuple[float, float]] = {
            str(k): (float(v[0]), float(v[1])) for k, v in jl_cfg.items()
        }
        self.arms: dict[str, _Arm] = {}
        for side, c in robot_cfg["arms"].items():
            ik = ArmPositionIK(
                self.model,
                joint_names=list(c["joints"]),
                shoulder_body=c["shoulder_body"],
                elbow_body=c["elbow_body"],
                wrist_body=c["wrist_body"],
                damping=float(ik_cfg.get("damping", 0.08)),
                max_iters=int(ik_cfg.get("max_iters", 16)),
                pos_tol=float(ik_cfg.get("pos_tol", 2e-3)),
                step_clip=float(ik_cfg.get("step_clip", 0.35)),
                joint_limits=self.joint_limits,
            )
            sh = ik.body_pos(self.data, "shoulder")
            el = ik.body_pos(self.data, "elbow")
            wr = ik.body_pos(self.data, "wrist")
            self.arms[side] = _Arm(
                ik,
                upper_len=float(np.linalg.norm(el - sh)),
                lower_len=float(np.linalg.norm(wr - el)),
            )

    @property
    def joint_qpos(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for j in range(self.model.njnt):
            jn = self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_JOINT, j)
            if jn is None:
                continue
            canonical = jn[:-6] if jn.endswith("_joint") else jn
            out[canonical] = float(self.data.qpos[self.model.jnt_qposadr[j]])
        return out

    def solve_arm(
        self,
        side: str,
        op_shoulder: NDArray[np.float64],
        op_elbow: NDArray[np.float64],
        op_wrist: NDArray[np.float64],
    ) -> dict[str, float] | None:
        """Solve one arm. Operator points are in the torso frame (meters).

        Targets are placed at the robot's OWN link lengths along the operator's
        upper-arm and forearm directions, so they're always reachable and the
        robot reproduces both directions (full arm configuration), not just the
        hand. Returns {joint_name: angle} or None if a direction is degenerate.
        """
        arm = self.arms[side]
        du = _unit(self.frame_R @ (op_elbow - op_shoulder))
        df = _unit(self.frame_R @ (op_wrist - op_elbow))
        if du is None or df is None:
            return None
        # Seed the elbow joint with the directly-observable flex angle
        # (arccos(u·f), 0=straight) ONLY while the robot elbow is still inside the
        # straight-arm neighbourhood, where shoulder-yaw cannot move the wrist and
        # DLS would stall in a local minimum. (Rotation preserves the angle, so
        # computing it from robot-frame dirs is fine.)
        #
        # It used to be seeded unconditionally, every frame. That silently undid
        # the warm start for this one joint and — worse — overwrote the smoothed
        # elbow angle that the caller writes back to qpos, so the elbow alone
        # tracked the raw, unfiltered measurement while its three siblings were
        # filtered. Seeding only near the singularity keeps the escape behaviour
        # without leaking raw measurements into the solved pose.
        eadr = int(arm.ik.qadr[-1])  # elbow joint listed last (base→tip)
        elo, ehi = float(arm.ik.lo[-1]), float(arm.ik.hi[-1])
        if self._elbow_seed_below > 0.0 and abs(
            float(self.data.qpos[eadr])
        ) < self._elbow_seed_below:
            flex = float(np.arccos(np.clip(np.dot(du, df), -1.0, 1.0)))
            self.data.qpos[eadr] = float(np.clip(flex, elo, ehi))
        shoulder = arm.ik.body_pos(self.data, "shoulder")
        elbow_target = self._limit_step(
            f"{side}_elbow", shoulder + arm.upper_len * du
        )
        wrist_target = self._limit_step(
            f"{side}_wrist", elbow_target + arm.lower_len * df
        )
        return arm.ik.solve(self.data, elbow_target, wrist_target)

    def _limit_step(
        self, key: str, target: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Clamp a target's per-frame displacement to ``max_target_step`` (m).

        Holds the endpoint near its last value when the pose estimate jumps, so an
        abnormal frame can't teleport the arm; a sustained move slews over a few
        frames and then tracks normally. No-op when the limiter is disabled."""
        if self._max_target_step <= 0.0:
            return target
        last = self._last_targets.get(key)
        if last is not None:
            delta = target - last
            dist = float(np.linalg.norm(delta))
            if dist > self._max_target_step:
                target = last + delta * (self._max_target_step / dist)
        self._last_targets[key] = target
        return target

    def link_lengths(self) -> dict[str, tuple[float, float]]:
        return {s: (a.upper_len, a.lower_len) for s, a in self.arms.items()}
