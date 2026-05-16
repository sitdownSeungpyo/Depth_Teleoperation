"""Dynamixel SDK 직접 제어 Publisher — UBP 실 로봇용.

[PC] ─USB─ [U2D2 or USB→RS485] ─RS485 daisy chain─ [DXL servos]

지원 모델 (모두 Protocol 2.0):
- MX-64 / MX-28: GOAL_POSITION addr 116, 4 byte, range [0, 4095] = [0°, 360°]
- XL430:         GOAL_POSITION addr 116, 4 byte, range [0, 4095] = [0°, 360°]

Radian → DXL unit 변환:
- 모든 모델 4096 unit / 2π rad. 중앙(angle 0 rad)이 unit 2048.
- 변환: unit = int(2048 + angle * 4096 / (2π))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from core.types import JointCommand
from publisher.base import InterpolatingPublisherBase

log = logging.getLogger(__name__)

# Protocol 2.0 control table — MX-64, MX-28, XL430 모두 동일
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4

DXL_RESOLUTION = 4096  # units per revolution
DXL_CENTER = 2048  # unit at angle 0


def angle_rad_to_dxl_unit(angle_rad: float) -> int:
    """Convert radian to DXL unit, clamped to [0, 4095]."""
    unit = int(round(DXL_CENTER + angle_rad * DXL_RESOLUTION / (2.0 * math.pi)))
    return max(0, min(unit, DXL_RESOLUTION - 1))


@dataclass
class ServoSpec:
    """One servo's identity."""
    id: int
    model: str  # "MX-64", "MX-28", "XL430"


class DynamixelPublisher(InterpolatingPublisherBase):
    """Position-control publisher over Dynamixel SDK.

    `servos` 는 joint name (예: 'r_shoulder_pitch') → ServoSpec 매핑.
    """

    def __init__(
        self,
        port: str,
        baud: int,
        servos: dict[str, ServoSpec],
        rate_hz: int = 100,
        protocol: float = 2.0,
        torque_on_start: bool = True,
    ) -> None:
        super().__init__(rate_hz=rate_hz)
        self._port = port
        self._baud = baud
        self._servos = servos
        self._protocol = protocol
        self._torque_on_start = torque_on_start
        self._port_handler: Any = None
        self._packet_handler: Any = None
        self._group_sync_write: Any = None

    def _open(self) -> None:
        try:
            from dynamixel_sdk import (
                COMM_SUCCESS,
                GroupSyncWrite,
                PacketHandler,
                PortHandler,
            )
        except ImportError as exc:
            raise RuntimeError(
                "dynamixel-sdk not installed; pip install dynamixel-sdk"
            ) from exc

        self._port_handler = PortHandler(self._port)
        self._packet_handler = PacketHandler(self._protocol)

        if not self._port_handler.openPort():
            raise RuntimeError(f"failed to open port {self._port}")
        if not self._port_handler.setBaudRate(self._baud):
            raise RuntimeError(f"failed to set baud {self._baud}")

        self._group_sync_write = GroupSyncWrite(
            self._port_handler, self._packet_handler,
            ADDR_GOAL_POSITION, LEN_GOAL_POSITION,
        )

        if self._torque_on_start:
            for joint, spec in self._servos.items():
                rc, err = self._packet_handler.write1ByteTxRx(
                    self._port_handler, spec.id, ADDR_TORQUE_ENABLE, 1
                )
                if rc != COMM_SUCCESS:
                    log.warning(
                        "torque enable failed for %s (id=%d): rc=%d err=%d",
                        joint, spec.id, rc, err,
                    )
                else:
                    log.info("torque enabled: %s id=%d model=%s", joint, spec.id, spec.model)

    def start(self) -> None:
        self._open()
        super().start()

    def stop(self) -> None:
        super().stop()
        if self._port_handler is not None:
            try:
                # 토크는 안전을 위해 다음 세션에서 명시적으로 켜도록 disable
                if self._torque_on_start and self._packet_handler is not None:
                    for spec in self._servos.values():
                        try:
                            self._packet_handler.write1ByteTxRx(
                                self._port_handler, spec.id, ADDR_TORQUE_ENABLE, 0
                            )
                        except Exception:  # noqa: BLE001
                            pass
                self._port_handler.closePort()
            except Exception as exc:  # noqa: BLE001
                log.warning("port close failed: %s", exc)
            self._port_handler = None

    def _emit(self, command: JointCommand) -> None:
        if self._port_handler is None or self._group_sync_write is None:
            return
        from dynamixel_sdk import COMM_SUCCESS

        self._group_sync_write.clearParam()
        for joint, target_rad in command.positions.items():
            spec = self._servos.get(joint)
            if spec is None:
                continue
            unit = angle_rad_to_dxl_unit(target_rad)
            param = [
                unit & 0xFF,
                (unit >> 8) & 0xFF,
                (unit >> 16) & 0xFF,
                (unit >> 24) & 0xFF,
            ]
            ok = self._group_sync_write.addParam(spec.id, bytes(param))
            if not ok:
                log.warning("addParam failed for id=%d", spec.id)
        rc = self._group_sync_write.txPacket()
        if rc != COMM_SUCCESS:
            log.warning("sync write rc=%d", rc)


def build_servos_from_config(servos_cfg: dict[str, Any]) -> dict[str, ServoSpec]:
    """Helper: convert YAML 'servos' dict (e.g. config/ubp.yaml) → ServoSpec map."""
    return {
        joint: ServoSpec(id=int(spec["id"]), model=str(spec["model"]))
        for joint, spec in servos_cfg.items()
    }
