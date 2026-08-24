"""Dynamixel publisher 단위 테스트 (실 서보 없이).

`angle_rad_to_dxl_unit` 변환 정확성 + ServoSpec 빌드 헬퍼 검증.
실제 USB-RS485 통신은 테스트 안 함 (하드웨어 필요).
"""

from __future__ import annotations

import math

import pytest

from publisher.dynamixel_publisher import (
    DXL_CENTER,
    DXL_RESOLUTION,
    ServoSpec,
    angle_rad_to_dxl_unit,
    build_servos_from_config,
)


def test_center_position() -> None:
    assert angle_rad_to_dxl_unit(0.0) == DXL_CENTER


def test_quarter_turn_positive() -> None:
    # +π/2 rad = +90° = 1024 units 추가 = 2048 + 1024 = 3072
    assert angle_rad_to_dxl_unit(math.pi / 2) == DXL_CENTER + 1024


def test_quarter_turn_negative() -> None:
    assert angle_rad_to_dxl_unit(-math.pi / 2) == DXL_CENTER - 1024


def test_clamp_high() -> None:
    # 4π → 단순 변환 시 unit 약 6144 (> 4095). 클램프되어야 함.
    assert angle_rad_to_dxl_unit(4.0 * math.pi) == DXL_RESOLUTION - 1


def test_clamp_low() -> None:
    assert angle_rad_to_dxl_unit(-4.0 * math.pi) == 0


def test_build_servos_from_config() -> None:
    cfg = {
        "r_shoulder_pitch": {"id": 1, "model": "MX-64"},
        "r_wrist_pitch": {"id": 7, "model": "XL430"},
    }
    out = build_servos_from_config(cfg)
    assert out["r_shoulder_pitch"] == ServoSpec(id=1, model="MX-64")
    assert out["r_wrist_pitch"] == ServoSpec(id=7, model="XL430")


def test_build_servos_rejects_an_unknown_model() -> None:
    """A typo'd or unsupported model used to be accepted and silently treated as
    4096 units/rev, which scales every command for that joint."""
    with pytest.raises(ValueError, match="unknown servo model"):
        build_servos_from_config({"r_elbow": {"id": 1, "model": "MX-46"}})


def test_build_servos_rejects_a_bad_direction() -> None:
    with pytest.raises(ValueError, match="direction must be"):
        build_servos_from_config({"r_elbow": {"id": 1, "model": "MX-28", "direction": 0}})


def test_servo_direction_mirrors_the_command() -> None:
    """Real arms are assembled mirrored; without a per-servo direction the left
    side drives the wrong way."""
    forward = ServoSpec(id=1, model="MX-28")
    mirrored = ServoSpec(id=2, model="MX-28", direction=-1)
    unit_fwd, _ = forward.to_unit(math.pi / 2)
    unit_mir, _ = mirrored.to_unit(math.pi / 2)
    assert unit_fwd == DXL_CENTER + 1024
    assert unit_mir == DXL_CENTER - 1024


def test_servo_offset_shifts_the_mechanical_zero() -> None:
    spec = ServoSpec(id=3, model="XL430", offset_unit=150)
    unit, clamped = spec.to_unit(0.0)
    assert unit == DXL_CENTER + 150
    assert not clamped


def test_servo_to_unit_reports_clamping() -> None:
    spec = ServoSpec(id=4, model="MX-64")
    unit, clamped = spec.to_unit(4.0 * math.pi)
    assert unit == DXL_RESOLUTION - 1
    assert clamped, "an out-of-travel target must be reported, not silently clamped"


def test_servo_spec_immutable_like() -> None:
    # ServoSpec is a dataclass — verifies field types.
    s = ServoSpec(id=5, model="MX-28")
    assert s.id == 5
    assert s.model == "MX-28"


def test_roundtrip_known_angles() -> None:
    # 알려진 각도 → unit → degrees 환산 후 일치 확인
    for angle_deg in (-180.0, -90.0, -45.0, 0.0, 45.0, 90.0, 180.0):
        angle_rad = math.radians(angle_deg)
        unit = angle_rad_to_dxl_unit(angle_rad)
        # 0° = 2048, 360° = 4096 → 1° = 11.378 units
        expected_unit = int(round(DXL_CENTER + angle_deg * DXL_RESOLUTION / 360.0))
        expected_unit = max(0, min(expected_unit, DXL_RESOLUTION - 1))
        assert unit == pytest.approx(expected_unit, abs=1), f"angle={angle_deg}"
