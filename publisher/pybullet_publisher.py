"""PyBullet 시뮬레이션 Publisher.

URDF를 로드하고, `JointCommand`를 받을 때마다 PyBullet 조인트에 position-control로
명령을 적용한다. GUI 모드에서는 사용자가 로봇이 동작하는 모습을 시각적으로 확인 가능.

이 publisher는 실 로봇 직전 단계의 sim-in-the-loop 검증용. DIRECT 모드는 자동 테스트용.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from core.types import JointCommand
from publisher.base import InterpolatingPublisherBase

log = logging.getLogger(__name__)


class PyBulletPublisher(InterpolatingPublisherBase):
    """URDF 모델을 PyBullet에 로드 + JointCommand → setJointMotorControl2."""

    def __init__(
        self,
        urdf_path: Path,
        rate_hz: int = 100,
        gui: bool = True,
        fixed_base: bool = True,
        init_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        max_force: float = 10.0,
    ) -> None:
        super().__init__(rate_hz=rate_hz)
        self._urdf_path = urdf_path
        self._gui = gui
        self._fixed_base = fixed_base
        self._init_pose = init_pose
        self._max_force = max_force
        self._client: int | None = None
        self._robot_id: int | None = None
        self._joint_name_to_idx: dict[str, int] = {}
        self._sim_lock = threading.Lock()

    def _load(self) -> None:
        import pybullet as p
        import pybullet_data

        mode = p.GUI if self._gui else p.DIRECT
        self._client = p.connect(mode)
        if self._client < 0:
            raise RuntimeError("PyBullet connect failed")
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        if self._gui:
            try:
                p.loadURDF("plane.urdf", physicsClientId=self._client)
            except p.error:
                pass  # plane.urdf가 없으면 무시

        if not self._urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self._urdf_path}")
        self._robot_id = p.loadURDF(
            str(self._urdf_path),
            basePosition=list(self._init_pose),
            useFixedBase=self._fixed_base,
            physicsClientId=self._client,
        )

        # joint name → index 캐싱 (revolute만)
        num_joints = p.getNumJoints(self._robot_id, physicsClientId=self._client)
        for i in range(num_joints):
            info = p.getJointInfo(self._robot_id, i, physicsClientId=self._client)
            joint_type = info[2]
            if joint_type != p.JOINT_REVOLUTE:
                continue
            # URDF의 joint 이름은 보통 "<name>_joint" 형태. command key와 매칭하려면 suffix 제거.
            raw_name = info[1].decode("utf-8")
            canonical = raw_name[:-6] if raw_name.endswith("_joint") else raw_name
            self._joint_name_to_idx[canonical] = info[0]

        log.info(
            "PyBullet loaded %s — %d revolute joints: %s",
            self._urdf_path.name,
            len(self._joint_name_to_idx),
            sorted(self._joint_name_to_idx.keys()),
        )

    def start(self) -> None:
        # PyBullet 호출은 모두 같은 스레드에서 해야 안정적 → 메인 스레드에서 _load.
        # 이후 InterpolatingPublisherBase의 백그라운드 스레드는 setJointMotorControl만 호출.
        # PyBullet는 setJointMotorControl을 다른 스레드에서 호출해도 일반적으로 동작하지만,
        # 안전을 위해 lock 사용.
        self._load()
        super().start()

    def stop(self) -> None:
        super().stop()
        if self._client is not None:
            try:
                import pybullet as p
                p.disconnect(self._client)
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def _emit(self, command: JointCommand) -> None:
        if self._robot_id is None or self._client is None:
            return
        try:
            import pybullet as p
        except ImportError:
            return
        with self._sim_lock:
            for joint_name, target in command.positions.items():
                idx = self._joint_name_to_idx.get(joint_name)
                if idx is None:
                    continue
                try:
                    p.setJointMotorControl2(
                        bodyUniqueId=self._robot_id,
                        jointIndex=idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=float(target),
                        force=self._max_force,
                        physicsClientId=self._client,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug("setJointMotorControl2 failed on %s: %s", joint_name, exc)
            try:
                # 시뮬레이션 step — GUI 모드면 화면 업데이트, DIRECT면 물리 진행.
                p.stepSimulation(physicsClientId=self._client)
            except Exception as exc:  # noqa: BLE001
                log.debug("stepSimulation failed: %s", exc)

    @property
    def joint_names(self) -> list[str]:
        return sorted(self._joint_name_to_idx.keys())

    def current_joint_state(self) -> dict[str, float]:
        """현재 PyBullet 조인트 위치를 dict로 반환 (테스트 / 디버깅용)."""
        if self._robot_id is None or self._client is None:
            return {}
        import pybullet as p

        out: dict[str, float] = {}
        with self._sim_lock:
            for name, idx in self._joint_name_to_idx.items():
                try:
                    state = p.getJointState(self._robot_id, idx, physicsClientId=self._client)
                    out[name] = float(state[0])
                except Exception:  # noqa: BLE001
                    pass
        return out
