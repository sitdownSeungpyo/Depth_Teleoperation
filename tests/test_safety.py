from __future__ import annotations

import threading
from typing import Any

from core.safety import SafetyConfig, SafetyLayer
from core.types import JointCommand
from publisher.mock_publisher import MockPublisher


class FakeHotkey:
    def __init__(self) -> None:
        self._held: set[str] = set()
        self._lock = threading.Lock()

    def press(self, key: str) -> None:
        with self._lock:
            self._held.add(key.lower())

    def release(self, key: str) -> None:
        with self._lock:
            self._held.discard(key.lower())

    def is_pressed(self, key: str) -> bool:
        with self._lock:
            return key.lower() in self._held

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _cmd(positions: dict[str, float], ts: float = 0.0) -> JointCommand:
    return JointCommand(timestamp=ts, positions=dict(positions), source_frame_ts=ts)


def _safe_pose() -> dict[str, float]:
    return {
        "r_shoulder_pitch": 1.4,
        "r_shoulder_roll": 0.1,
        "r_elbow": 0.2,
    }


def test_deadman_release_stops_commands() -> None:
    pub = MockPublisher(rate_hz=200)
    hot = FakeHotkey()
    cfg = SafetyConfig(safe_pose=_safe_pose(), cycle_dt_s=1 / 30.0)
    safety = SafetyLayer(publisher=pub, config=cfg, hotkey=hot)
    safety.start()
    try:
        hot.press("space")
        safety.update(_cmd({"r_elbow": 1.0}), mean_confidence=0.99)
        last_target_with_press: Any = pub._next  # noqa: SLF001
        hot.release("space")
        safety.update(_cmd({"r_elbow": 1.5}), mean_confidence=0.99)
        # After release, the new value must not have been forwarded.
        assert pub._next is last_target_with_press  # noqa: SLF001
    finally:
        safety.stop()


def test_estop_hotkey_zeroes_commands() -> None:
    pub = MockPublisher(rate_hz=200)
    hot = FakeHotkey()
    cfg = SafetyConfig(safe_pose=_safe_pose(), cycle_dt_s=1 / 30.0)
    safety = SafetyLayer(publisher=pub, config=cfg, hotkey=hot)
    safety.start()
    try:
        hot.press("space")
        hot.press("esc")
        safety.update(_cmd({"r_elbow": 1.0}), mean_confidence=0.99)
        forwarded: Any = pub._next  # noqa: SLF001
        assert forwarded is not None
        assert forwarded.positions["r_elbow"] == 0.0
        assert safety.estopped
    finally:
        safety.stop()


def test_confidence_drop_ramps_to_safe_pose() -> None:
    pub = MockPublisher(rate_hz=200)
    hot = FakeHotkey()
    hot.press("space")

    t = [0.0]

    def fake_clock() -> float:
        return t[0]

    cfg = SafetyConfig(
        safe_pose=_safe_pose(),
        cycle_dt_s=1 / 30.0,
        loss_grace_period_s=0.5,
        ramp_to_safe_s=1.0,
    )
    safety = SafetyLayer(publisher=pub, config=cfg, hotkey=hot, clock=fake_clock)
    safety.start()
    try:
        # High confidence first → command passes through.
        safety.update(_cmd({"r_elbow": 1.0, "r_shoulder_pitch": 0.0, "r_shoulder_roll": 0.0}),
                      mean_confidence=0.99)

        # Drop confidence; advance time past grace + full ramp.
        for step in range(50):
            t[0] += 0.05
            safety.update(
                _cmd({"r_elbow": 1.0, "r_shoulder_pitch": 0.0, "r_shoulder_roll": 0.0}),
                mean_confidence=0.0,
            )
        latest: Any = pub._next  # noqa: SLF001
        assert abs(latest.positions["r_elbow"] - 0.2) < 1e-3
        assert abs(latest.positions["r_shoulder_pitch"] - 1.4) < 1e-3
    finally:
        safety.stop()


def test_watchdog_triggers_estop_on_stall() -> None:
    pub = MockPublisher(rate_hz=200)
    hot = FakeHotkey()
    hot.press("space")
    t = [0.0]

    def clock() -> float:
        return t[0]

    cfg = SafetyConfig(safe_pose=_safe_pose(), cycle_dt_s=0.01, watchdog_factor=3)
    safety = SafetyLayer(publisher=pub, config=cfg, hotkey=hot, clock=clock)
    safety.start()
    try:
        safety.update(_cmd({"r_elbow": 0.0}), mean_confidence=0.99)
        t[0] += 0.5  # well past 3 * 0.01 = 0.03 s
        safety.watchdog_tick()
        assert safety.estopped
    finally:
        safety.stop()
