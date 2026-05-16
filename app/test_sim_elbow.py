"""MuJoCo sim 자체가 r_elbow를 움직일 수 있는지 격리 검증.

retargeter / filter / publisher 일체 우회. 직접 MJCF 로드해서 r_elbow_act에
sin wave를 commanded ctrl로 넣고 실제 qpos가 따라가는지 확인.

사용법:
    python -m app.test_sim_elbow              # GUI 창 + 0.25Hz sin wave (4초 주기)
    python -m app.test_sim_elbow --headless   # 창 없이 qpos만 로그

이 테스트가 통과하면 sim은 정상. 통과 안하면 actuator gain / damping 등 sim 설정 문제.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, default=Path("./models/ubp.xml"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--joint", type=str, default="r_elbow",
                        help="test할 관절 이름 (예: r_elbow, l_elbow, r_shoulder_pitch)")
    parser.add_argument("--target-min", type=float, default=0.0)
    parser.add_argument("--target-max", type=float, default=2.0)
    parser.add_argument("--frequency", type=float, default=0.25, help="sin wave 주파수 (Hz)")
    args = parser.parse_args()

    try:
        import mujoco
    except ImportError:
        print("mujoco not installed", file=sys.stderr)
        return 2

    if not args.mjcf.exists():
        print(f"MJCF not found: {args.mjcf}", file=sys.stderr)
        return 2

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)

    act_name = f"{args.joint}_act"
    jname = f"{args.joint}_joint"
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if act_id < 0 or jid < 0:
        print(f"actuator/joint not found: {act_name} / {jname}", file=sys.stderr)
        return 2
    qpos_addr = model.jnt_qposadr[jid]
    print(f"testing {args.joint} → actuator idx {act_id}, joint qpos addr {qpos_addr}")
    print(f"target range [{args.target_min}, {args.target_max}], freq {args.frequency} Hz")
    print(f"actuator kp/forcerange will appear below if non-default")
    # Print actuator's gain config
    ctrlrange = model.actuator_ctrlrange[act_id]
    forcerange = model.actuator_forcerange[act_id]
    gainprm = model.actuator_gainprm[act_id]
    print(f"  ctrlrange={ctrlrange}, forcerange={forcerange}, gainprm={gainprm}")

    amp = (args.target_max - args.target_min) / 2
    mid = (args.target_max + args.target_min) / 2

    def step_and_log() -> None:
        now = time.perf_counter() - start
        if now > args.duration:
            return
        target = mid + amp * math.sin(2 * math.pi * args.frequency * now)
        data.ctrl[act_id] = target
        mujoco.mj_step(model, data)
        actual = data.qpos[qpos_addr]
        # log every 0.25s
        if int(now * 4) > step_and_log.last_log:  # type: ignore[attr-defined]
            step_and_log.last_log = int(now * 4)  # type: ignore[attr-defined]
            err = target - actual
            print(f"[{now:5.2f}s] target={target:+.3f}  actual={actual:+.3f}  err={err:+.3f}")

    step_and_log.last_log = -1  # type: ignore[attr-defined]
    start = time.perf_counter()

    if args.headless:
        while time.perf_counter() - start < args.duration:
            step_and_log()
            time.sleep(model.opt.timestep)
        return 0

    from mujoco import viewer

    with viewer.launch_passive(model, data) as v:
        while v.is_running() and time.perf_counter() - start < args.duration:
            step_and_log()
            v.sync()
            time.sleep(model.opt.timestep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
