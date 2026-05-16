from core.lower_body import LowerBodyController, PassThroughLowerBody
from core.types import JointCommand, SensorBundle


def test_passthrough_returns_input_unchanged() -> None:
    cmd = JointCommand(timestamp=1.0, positions={"r_elbow": 0.5, "r_shoulder_pitch": 0.1})
    sensors = SensorBundle(extra={"imu_pitch": 0.02})
    result = PassThroughLowerBody().update(cmd, sensors)
    assert result is cmd


def test_passthrough_satisfies_protocol() -> None:
    assert isinstance(PassThroughLowerBody(), LowerBodyController)
