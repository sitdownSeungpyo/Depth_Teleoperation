"""최소화된 MuJoCo 진단 — 모든 actuator 우회하고 qpos 직접 설정해서 모델이 그래픽적으로
동작 가능한지만 확인. 만약 이것도 안 움직이면 모델 자체 문제."""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    try:
        import mujoco
    except ImportError:
        print("mujoco not installed", file=sys.stderr)
        return 2

    mjcf = Path("./models/ubp.xml")
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)

    r_elbow_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_elbow_joint")
    r_elbow_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r_elbow_act")
    qpos_addr = model.jnt_qposadr[r_elbow_jid]
    qvel_addr = model.jnt_dofadr[r_elbow_jid]

    # Print all actuator parameters
    print(f"actuator idx={r_elbow_aid}")
    print(f"  gainprm   = {model.actuator_gainprm[r_elbow_aid]}")
    print(f"  biasprm   = {model.actuator_biasprm[r_elbow_aid]}")
    print(f"  gaintype  = {model.actuator_gaintype[r_elbow_aid]}")
    print(f"  biastype  = {model.actuator_biastype[r_elbow_aid]}")
    print(f"  dyntype   = {model.actuator_dyntype[r_elbow_aid]}")
    print(f"  forcerange = {model.actuator_forcerange[r_elbow_aid]}")
    print(f"  ctrlrange  = {model.actuator_ctrlrange[r_elbow_aid]}")
    print(f"  ctrllimited= {model.actuator_ctrllimited[r_elbow_aid]}")
    print(f"  forcelimited={model.actuator_forcelimited[r_elbow_aid]}")
    print(f"  trntype    = {model.actuator_trntype[r_elbow_aid]}")
    print(f"  trnid      = {model.actuator_trnid[r_elbow_aid]}")
    print()
    print(f"joint r_elbow: idx={r_elbow_jid}, qpos_addr={qpos_addr}, qvel_addr={qvel_addr}")
    print(f"  range         = {model.jnt_range[r_elbow_jid]}")
    print(f"  limited       = {model.jnt_limited[r_elbow_jid]}")
    print(f"  damping (DOF) = {model.dof_damping[qvel_addr]}")
    print(f"  friction(DOF) = {model.dof_frictionloss[qvel_addr]}")
    print(f"  armature(DOF) = {model.dof_armature[qvel_addr]}")
    print(f"  mass    (DOF) = {model.dof_M0[qvel_addr]}")
    print()

    # Test 1: directly set qpos to 1.5, run mj_forward, see if it stays
    print("TEST 1: direct qpos write")
    data.qpos[qpos_addr] = 1.5
    mujoco.mj_forward(model, data)
    print(f"  after mj_forward: qpos={data.qpos[qpos_addr]:.3f}")
    data.qpos[qpos_addr] = 0.0
    data.qvel[qvel_addr] = 0.0
    mujoco.mj_forward(model, data)

    # Test 2: set actuator ctrl and run mj_step 1000 times
    print("\nTEST 2: actuator ctrl=2.0, 1000 mj_step calls")
    data.ctrl[r_elbow_aid] = 2.0
    for i in range(1000):
        mujoco.mj_step(model, data)
        if i in (0, 1, 10, 100, 500, 999):
            print(f"  step {i:4d}: qpos={data.qpos[qpos_addr]:+.4f}  qvel={data.qvel[qvel_addr]:+.4f}  "
                  f"qfrc_actuator={data.qfrc_actuator[qvel_addr]:+.3f}  "
                  f"qfrc_bias={data.qfrc_bias[qvel_addr]:+.3f}  "
                  f"qfrc_constraint={data.qfrc_constraint[qvel_addr]:+.3f}")

    # Test 3: zero everything except elbow, see force
    print("\nTEST 3: reset, set ctrl=1.0 step-by-step force tracing")
    mujoco.mj_resetData(model, data)
    data.ctrl[r_elbow_aid] = 1.0
    for i in range(5):
        mujoco.mj_step(model, data)
        print(f"  step {i}: ctrl={data.ctrl[r_elbow_aid]:.2f}  qpos={data.qpos[qpos_addr]:+.5f}  "
              f"qvel={data.qvel[qvel_addr]:+.4f}  qacc={data.qacc[qvel_addr]:+.2f}  "
              f"qfrc_actuator={data.qfrc_actuator[qvel_addr]:+.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
