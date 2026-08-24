"""Dynamixel SDK 직접 제어 Publisher — UBP 실 로봇용.

[PC] ─USB─ [U2D2 or USB→RS485] ─RS485 daisy chain─ [DXL servos]

지원 모델은 :data:`SERVO_MODELS` 에 명시한다. **표에 없는 모델은 예외**로 막는다:
모델별 분해능이 다르면 각도가 조용히 틀어지는데, 예전 코드는 `model` 문자열을
저장만 하고 쓰지 않아 오타든 미지원 모델이든 전부 4096 으로 처리했다.

Radian → DXL unit 변환 (:func:`angle_rad_to_dxl_unit`):
- unit = center + direction × angle × resolution / 2π + offset_unit
- ``direction`` (±1) 과 ``offset_unit`` 은 **서보마다** 다르다. 조립 방향과 혼(horn)
  장착 각도는 관절마다 제각각이라, "모든 서보가 2048 = 로봇 0 rad, 회전 방향도 동일"
  이라는 가정은 실기에서 거의 항상 틀린다. config 의 servos 블록에서 지정한다.
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

# ★ 미검증(2026-08 기준 데이터시트 대조 안 함). 아래 두 주소는 profile 제한을 쓰는
#   경우에만 사용되고, 기본값은 None(사용 안 함)이다. 실제로 쓰기 전에 해당 모델
#   e-Manual 의 control table 을 확인할 것. 쓰기 실패는 조용히 넘기지 않고 예외로
#   올라오므로, 주소가 틀리면 시작 시점에 바로 드러난다.
ADDR_PROFILE_VELOCITY = 112
ADDR_PROFILE_ACCELERATION = 108
LEN_PROFILE = 4

DXL_RESOLUTION = 4096  # units per revolution (표의 기본값)
DXL_CENTER = 2048  # unit at angle 0


@dataclass(frozen=True)
class ServoModel:
    """One servo model's electrical facts that affect the command we send."""

    resolution: int  # units per full revolution
    center: int  # unit that corresponds to 0 rad


# 표에 없는 모델은 build_servos_from_config 에서 거부된다.
SERVO_MODELS: dict[str, ServoModel] = {
    "MX-64": ServoModel(resolution=4096, center=2048),
    "MX-28": ServoModel(resolution=4096, center=2048),
    "XL430": ServoModel(resolution=4096, center=2048),
    # 2XL430: 축 2개가 각자 ID 를 갖고, 축마다 XL430 과 같은 control table 을 쓴다는
    # 전제. ★ 데이터시트 미확인 — 실기 연결 전 확인할 것.
    "2XL430": ServoModel(resolution=4096, center=2048),
}


def angle_rad_to_dxl_unit(
    angle_rad: float,
    resolution: int = DXL_RESOLUTION,
    center: int = DXL_CENTER,
    offset_unit: int = 0,
    direction: int = 1,
) -> int:
    """Convert radian to DXL unit, clamped to [0, resolution - 1].

    ``direction`` (+1/-1) and ``offset_unit`` map the robot's joint convention
    onto how this particular servo is physically mounted.
    """
    unit = int(
        round(center + direction * angle_rad * resolution / (2.0 * math.pi) + offset_unit)
    )
    return max(0, min(unit, resolution - 1))


@dataclass(frozen=True)
class ServoSpec:
    """One servo's identity and mounting convention."""

    id: int
    model: str  # must be a key of SERVO_MODELS
    # unit added after the radian conversion — the servo's mechanical zero
    # relative to the robot's joint zero.
    offset_unit: int = 0
    # +1 when the servo turns the same way as the joint convention, -1 when mirrored.
    direction: int = 1

    def spec(self) -> ServoModel:
        return SERVO_MODELS[self.model]

    def to_unit(self, angle_rad: float) -> tuple[int, bool]:
        """Return (unit, clamped) for a joint angle in radians."""
        model = self.spec()
        raw = int(
            round(
                model.center
                + self.direction * angle_rad * model.resolution / (2.0 * math.pi)
                + self.offset_unit
            )
        )
        unit = max(0, min(raw, model.resolution - 1))
        return unit, unit != raw


class DynamixelPublisher(InterpolatingPublisherBase):
    """Position-control publisher over Dynamixel SDK.

    `servos` 는 joint name (예: 'r_shoulder_pitch') → ServoSpec 매핑.

    An emergency stop is inherited from :class:`InterpolatingPublisherBase`: it
    freezes the goal at the last commanded pose and keeps torque on, so the arm
    holds instead of falling. Torque is deliberately NOT dropped — for an upper
    body, cutting torque is a collapse, not a stop.
    """

    def __init__(
        self,
        port: str,
        baud: int,
        servos: dict[str, ServoSpec],
        rate_hz: int = 100,
        protocol: float = 2.0,
        torque_on_start: bool = True,
        profile_velocity: int | None = None,
        profile_acceleration: int | None = None,
        torque_off_on_stop: bool = False,
    ) -> None:
        super().__init__(rate_hz=rate_hz)
        self._port = port
        self._baud = baud
        self._servos = servos
        self._protocol = protocol
        self._torque_on_start = torque_on_start
        self._profile_velocity = profile_velocity
        self._profile_acceleration = profile_acceleration
        self._torque_off_on_stop = torque_off_on_stop
        self._port_handler: Any = None
        self._packet_handler: Any = None
        self._group_sync_write: Any = None
        self._comm_success: int = 0
        self._unmapped_warned: set[str] = set()
        self._clamp_warned: set[str] = set()

    # ---- wiring ---------------------------------------------------------

    def _write_checked(self, what: str, spec: ServoSpec, addr: int, value: int, size: int) -> None:
        """Write one register and RAISE on any failure.

        Every one of these configures how the joint will move. A warning here
        would mean starting with, say, one joint at full speed and no torque —
        the operator finds out by watching it happen.
        """
        if size == 1:
            rc, err = self._packet_handler.write1ByteTxRx(
                self._port_handler, spec.id, addr, value
            )
        elif size == 4:
            rc, err = self._packet_handler.write4ByteTxRx(
                self._port_handler, spec.id, addr, value
            )
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unsupported register width: {size}")
        if rc != self._comm_success or err != 0:
            raise RuntimeError(
                f"{what} failed on id={spec.id} ({spec.model}, addr={addr}): "
                f"rc={rc} err={err} — check wiring, ID, and baud before running the robot"
            )

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

        self._comm_success = COMM_SUCCESS
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

        # Profile limits BEFORE torque: with torque already on, the first goal
        # would be executed at whatever speed the servo happens to be configured
        # for, which on a fresh servo is its maximum.
        for joint, spec in self._servos.items():
            if self._profile_velocity is not None:
                self._write_checked(
                    "profile velocity", spec, ADDR_PROFILE_VELOCITY,
                    int(self._profile_velocity), LEN_PROFILE,
                )
            if self._profile_acceleration is not None:
                self._write_checked(
                    "profile acceleration", spec, ADDR_PROFILE_ACCELERATION,
                    int(self._profile_acceleration), LEN_PROFILE,
                )
            if self._profile_velocity is None and self._profile_acceleration is None:
                log.warning(
                    "%s (id=%d): no profile velocity/acceleration set — the servo will "
                    "drive to each goal at its configured maximum speed",
                    joint, spec.id,
                )

        if self._torque_on_start:
            for joint, spec in self._servos.items():
                self._write_checked("torque enable", spec, ADDR_TORQUE_ENABLE, 1, 1)
                log.info(
                    "torque enabled: %s id=%d model=%s dir=%+d offset=%d",
                    joint, spec.id, spec.model, spec.direction, spec.offset_unit,
                )

    def start(self) -> None:
        self._open()
        super().start()

    def stop(self) -> None:
        super().stop()
        if self._port_handler is None:
            return
        try:
            if self._torque_off_on_stop and self._packet_handler is not None:
                # Cutting torque on an upper body means the arms fall. Off by
                # default; only meaningful once the caller has already ramped to a
                # pose that is safe to release from.
                log.warning("disabling torque on stop — arms will drop if unsupported")
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
        finally:
            self._port_handler = None

    # ---- output ---------------------------------------------------------

    def _emit(self, command: JointCommand) -> None:
        if self._port_handler is None or self._group_sync_write is None:
            return

        self._group_sync_write.clearParam()
        for joint, target_rad in command.positions.items():
            spec = self._servos.get(joint)
            if spec is None:
                if joint not in self._unmapped_warned:
                    self._unmapped_warned.add(joint)
                    log.warning("no servo mapped for joint %s; it will never move", joint)
                continue
            unit, clamped = spec.to_unit(float(target_rad))
            if clamped and joint not in self._clamp_warned:
                self._clamp_warned.add(joint)
                log.warning(
                    "%s target %.3f rad is outside servo id=%d travel; clamping. "
                    "Check config/joint_limit.yaml and this servo's offset/direction.",
                    joint, target_rad, spec.id,
                )
            param = [
                unit & 0xFF,
                (unit >> 8) & 0xFF,
                (unit >> 16) & 0xFF,
                (unit >> 24) & 0xFF,
            ]
            if not self._group_sync_write.addParam(spec.id, bytes(param)):
                log.warning("addParam failed for id=%d", spec.id)
        rc = self._group_sync_write.txPacket()
        if rc != self._comm_success:
            log.warning("sync write rc=%d", rc)


def build_servos_from_config(servos_cfg: dict[str, Any]) -> dict[str, ServoSpec]:
    """Convert the YAML ``publisher.dynamixel.servos`` block → ServoSpec map.

    Rejects an unknown model outright rather than falling back to a default
    resolution: a typo there would silently scale every command for that joint.
    """
    out: dict[str, ServoSpec] = {}
    for joint, spec in servos_cfg.items():
        model = str(spec["model"])
        if model not in SERVO_MODELS:
            raise ValueError(
                f"{joint}: unknown servo model {model!r}. "
                f"Known: {sorted(SERVO_MODELS)}. Add it to SERVO_MODELS with the "
                f"resolution from its e-Manual instead of assuming 4096."
            )
        direction = int(spec.get("direction", 1))
        if direction not in (1, -1):
            raise ValueError(f"{joint}: direction must be +1 or -1, got {direction}")
        out[joint] = ServoSpec(
            id=int(spec["id"]),
            model=model,
            offset_unit=int(spec.get("offset_unit", 0)),
            direction=direction,
        )
    return out
