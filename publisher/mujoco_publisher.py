"""MuJoCo 시뮬레이션 Publisher.

MJCF (`.xml`) 또는 URDF 모델을 로드하고, `JointCommand`를 받을 때마다
`data.ctrl[i]` 값을 갱신 + `mj_step` 으로 sim 진행. `launch_passive` 뷰어를 띄우면
사용자가 GUI에서 동작 확인 가능.

PyBulletPublisher 와 유사하지만 MuJoCo native (더 정확한 dynamics, 더 좋은 그래픽).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from core.types import JointCommand
from publisher.base import InterpolatingPublisherBase

log = logging.getLogger(__name__)


class MuJoCoPublisher(InterpolatingPublisherBase):
    """MJCF/URDF 모델을 MuJoCo에 로드 + JointCommand → data.ctrl + mj_step."""

    def __init__(
        self,
        model_path: Path,
        rate_hz: int = 100,
        gui: bool = True,
    ) -> None:
        super().__init__(rate_hz=rate_hz)
        self._model_path = model_path
        self._gui = gui
        self._mj: Any = None  # mujoco module
        self._model: Any = None
        self._data: Any = None
        self._viewer: Any = None
        self._actuator_idx: dict[str, int] = {}
        self._sim_lock = threading.Lock()
        self._steps_per_emit: int = 1  # set after model load: real-time / timestep ratio

    def _load(self) -> None:
        import mujoco

        self._mj = mujoco
        if not self._model_path.exists():
            raise FileNotFoundError(f"model not found: {self._model_path}")
        self._model = mujoco.MjModel.from_xml_path(str(self._model_path))
        self._data = mujoco.MjData(self._model)

        # joint name → actuator index 매핑
        # actuator name 형식: "<joint_name>_act" 또는 별도. 여기선 actuator_trnid로 joint 추적.
        for a in range(self._model.nu):
            jid = self._model.actuator_trnid[a, 0]
            jname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname is None:
                continue
            canonical = jname[:-6] if jname.endswith("_joint") else jname
            self._actuator_idx[canonical] = a

        # 매 _emit 마다 sim을 real-time만큼 진행하려면 (1/rate_hz) / timestep 번 step 필요.
        real_dt = 1.0 / self._rate_hz
        ts = float(self._model.opt.timestep)
        self._steps_per_emit = max(1, int(round(real_dt / ts)))

        log.info(
            "MuJoCo loaded %s — %d actuators, %d steps/emit (timestep=%.4fs, rate=%dHz): %s",
            self._model_path.name,
            len(self._actuator_idx),
            self._steps_per_emit,
            ts,
            self._rate_hz,
            sorted(self._actuator_idx.keys()),
        )

        if self._gui:
            from mujoco import viewer
            self._viewer = viewer.launch_passive(self._model, self._data)

    def start(self) -> None:
        self._load()
        super().start()

    def stop(self) -> None:
        super().stop()
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # noqa: BLE001
                pass
            self._viewer = None

    def _emit(self, command: JointCommand) -> None:
        if self._model is None or self._data is None:
            return
        with self._sim_lock:
            for joint_name, target in command.positions.items():
                idx = self._actuator_idx.get(joint_name)
                if idx is None:
                    continue
                self._data.ctrl[idx] = float(target)
            try:
                # real-time advancement: step (1/rate_hz) / timestep 번
                for _ in range(self._steps_per_emit):
                    self._mj.mj_step(self._model, self._data)
            except Exception as exc:  # noqa: BLE001
                log.debug("mj_step failed: %s", exc)
            if self._viewer is not None:
                try:
                    if self._viewer.is_running():
                        self._viewer.sync()
                except Exception as exc:  # noqa: BLE001
                    log.debug("viewer.sync failed: %s", exc)

    @property
    def actuator_names(self) -> list[str]:
        return sorted(self._actuator_idx.keys())

    def current_joint_state(self) -> dict[str, float]:
        """MuJoCo의 현재 joint position을 dict로 반환 (테스트/디버깅용)."""
        if self._model is None or self._data is None:
            return {}
        mujoco = self._mj
        out: dict[str, float] = {}
        with self._sim_lock:
            for j in range(self._model.njnt):
                jname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, j)
                if jname is None:
                    continue
                canonical = jname[:-6] if jname.endswith("_joint") else jname
                qpos_addr = self._model.jnt_qposadr[j]
                out[canonical] = float(self._data.qpos[qpos_addr])
        return out
