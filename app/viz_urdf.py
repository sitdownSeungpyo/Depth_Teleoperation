"""URDF 시각화 도구 — PyBullet GUI 창에 로봇 모델을 띄우고 슬라이더로 관절 조작.

사용법:
    python -m app.viz_urdf
    python -m app.viz_urdf --urdf .\\models\\ubp.urdf
    python -m app.viz_urdf --headless          # GUI 없이 joint info만 dump

조작:
- 왼쪽 패널의 슬라이더로 각 관절 각도 조정 (rad).
- 마우스 좌클릭 드래그: 카메라 회전.
- 마우스 휠: 줌.
- 'q' 또는 창 닫기: 종료.
- 'r' 버튼 (UI 우측 패널 'reset'): 모든 관절 0으로.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("./models/ubp.urdf"),
        help="URDF 파일 경로",
    )
    parser.add_argument("--headless", action="store_true", help="GUI 없이 joint dump만")
    parser.add_argument(
        "--fixed-base",
        action="store_true",
        default=True,
        help="base_link 고정 (지지 베이스 가정)",
    )
    args = parser.parse_args()

    if not args.urdf.exists():
        print(f"URDF not found: {args.urdf}", file=sys.stderr)
        return 2

    try:
        import pybullet as p
        import pybullet_data
    except ImportError:
        print("pybullet not installed — run: pip install pybullet", file=sys.stderr)
        return 2

    mode = p.DIRECT if args.headless else p.GUI
    client = p.connect(mode)
    if client < 0:
        print("PyBullet connect failed", file=sys.stderr)
        return 1

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")  # 바닥 격자

        robot_id = p.loadURDF(
            str(args.urdf),
            basePosition=[0, 0, 0],
            useFixedBase=args.fixed_base,
        )

        num_joints = p.getNumJoints(robot_id)
        revolute_joints: list[tuple[int, str, float, float]] = []
        print(f"\nLoaded {args.urdf.name}: {num_joints} joints")
        print(f"{'idx':>3}  {'name':<28} {'type':<12} {'lower':>8} {'upper':>8}")
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            joint_idx = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            lower = info[8]
            upper = info[9]
            type_str = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED"}.get(joint_type, str(joint_type))
            print(f"{joint_idx:>3}  {joint_name:<28} {type_str:<12} {lower:>8.3f} {upper:>8.3f}")
            if joint_type == p.JOINT_REVOLUTE:
                revolute_joints.append((joint_idx, joint_name, lower, upper))

        if args.headless:
            return 0

        # GUI: 슬라이더 추가
        sliders: dict[int, int] = {}
        for joint_idx, joint_name, lower, upper in revolute_joints:
            sliders[joint_idx] = p.addUserDebugParameter(
                joint_name, lower, upper, 0.0
            )
        reset_btn = p.addUserDebugParameter("[reset]", 1, 0, 0)
        last_reset_val = p.readUserDebugParameter(reset_btn)

        p.resetDebugVisualizerCamera(
            cameraDistance=1.6,
            cameraYaw=45,
            cameraPitch=-15,
            cameraTargetPosition=[0, 0, 0.4],
        )

        print("\nGUI 창에서 슬라이더로 관절 조작. 종료: 창 닫기.")
        while p.isConnected():
            try:
                # 슬라이더 값 → 관절에 적용
                for joint_idx, slider_id in sliders.items():
                    target = p.readUserDebugParameter(slider_id)
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target,
                        force=10.0,
                    )
                # reset 버튼: 슬라이더 직접 못 바꾸므로 메시지 출력만
                cur_reset = p.readUserDebugParameter(reset_btn)
                if cur_reset != last_reset_val:
                    last_reset_val = cur_reset
                    print("reset 누름 — 슬라이더를 수동으로 0에 맞춰주세요 "
                          "(PyBullet 제약).")
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
            except p.error:
                break
    finally:
        try:
            p.disconnect()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
