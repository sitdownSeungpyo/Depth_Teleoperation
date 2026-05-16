"""MuJoCo 시뮬레이션 + 뷰어로 UBP 모델 확인 / 조작.

사용법:
    python -m app.viz_mujoco                                # MJCF 로드 (기본)
    python -m app.viz_mujoco --mjcf .\\models\\ubp.xml
    python -m app.viz_mujoco --urdf .\\models\\ubp.urdf     # URDF도 MuJoCo 로드 가능
    python -m app.viz_mujoco --headless                     # 창 없이 조인트 정보만

뷰어 조작 (mujoco.viewer 기본 단축키):
- 마우스 좌클릭 드래그: 카메라 회전
- 마우스 우클릭 드래그: 카메라 이동
- 휠: 줌
- 더블 클릭: 객체 선택
- Ctrl+드래그(선택된 body): force/torque 인가
- Space: 시뮬레이션 일시정지/재개
- BackSpace: 시뮬레이션 리셋
- 'h': 도움말 토글
- 'q' 또는 창 닫기: 종료

PyBullet viz_urdf 와의 차이:
- MuJoCo는 자체 GUI에 슬라이더 위젯이 없으므로 actuator control은
  Python script로 ctrl 값을 변경하거나, 'L'키로 actuator control panel 열기.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("./models/ubp.xml"),
        help="MJCF 파일 경로",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="대신 URDF 로드 (MJCF 우선)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="창 없이 모델 정보만 dump",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="자동 종료 시간 (초). 미지정 시 사용자가 닫을 때까지 실행",
    )
    args = parser.parse_args()

    try:
        import mujoco
    except ImportError:
        print("mujoco not installed; pip install mujoco", file=sys.stderr)
        return 2

    if args.urdf is not None:
        if not args.urdf.exists():
            print(f"URDF not found: {args.urdf}", file=sys.stderr)
            return 2
        path = args.urdf
    else:
        if not args.mjcf.exists():
            print(f"MJCF not found: {args.mjcf}", file=sys.stderr)
            return 2
        path = args.mjcf

    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        print(f"failed to load {path}: {exc}", file=sys.stderr)
        return 1

    data = mujoco.MjData(model)

    print(f"\nLoaded {path.name}")
    print(f"  nq (관절 위치 개수): {model.nq}")
    print(f"  nv (관절 속도 개수): {model.nv}")
    print(f"  nu (액추에이터 개수): {model.nu}")
    print(f"  nbody (link 개수): {model.nbody}")
    print(f"\n{'idx':>3}  {'joint name':<28} {'type':<10} {'range':<20}")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"<{j}>"
        jtype = model.jnt_type[j]
        jtype_name = {
            mujoco.mjtJoint.mjJNT_FREE: "FREE",
            mujoco.mjtJoint.mjJNT_BALL: "BALL",
            mujoco.mjtJoint.mjJNT_SLIDE: "SLIDE",
            mujoco.mjtJoint.mjJNT_HINGE: "HINGE",
        }.get(jtype, str(jtype))
        lo, hi = model.jnt_range[j]
        limited = "limited" if model.jnt_limited[j] else "free"
        range_str = f"[{lo:.3f}, {hi:.3f}]" if model.jnt_limited[j] else "(unlimited)"
        print(f"{j:>3}  {name:<28} {jtype_name:<10} {range_str:<20} ({limited})")

    print(f"\n{'idx':>3}  {'actuator name':<28} {'joint':<28}")
    for a in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"<{a}>"
        jid = model.actuator_trnid[a, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"<{jid}>"
        print(f"{a:>3}  {aname:<28} {jname:<28}")

    if args.headless:
        return 0

    try:
        from mujoco import viewer
    except ImportError:
        print("mujoco.viewer not available", file=sys.stderr)
        return 2

    print("\n뷰어 열림. 종료: 창 닫기 또는 'q'.")
    with viewer.launch_passive(model, data) as v:
        start = time.perf_counter()
        last_step = start
        while v.is_running():
            now = time.perf_counter()
            dt = now - last_step
            if dt < model.opt.timestep:
                time.sleep(model.opt.timestep - dt)
            mujoco.mj_step(model, data)
            last_step = time.perf_counter()
            v.sync()
            if args.duration is not None and now - start > args.duration:
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
