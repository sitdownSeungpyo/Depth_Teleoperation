"""E-stop semantics: freeze, latch, reset, and its independence from the dead-man.

Three defects these pin down, all of which the old wiring had:

1. E-stop sent ``{joint: 0.0}``. On a position-controlled robot that is a
   full-speed move to the zero pose, and it bypassed the velocity limiter
   because the safety layer sits downstream of it.
2. ``reset_estop()`` existed but nothing called it, so a latched E-stop could
   only be cleared by killing the process.
3. The E-stop key was only wired up when the dead-man was enabled, so running
   without ``--deadman`` silently removed the keyboard stop as well.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from core.filter import (
    FilterAndLimiter,
    JointLimiterConfig,
    JointLimits,
    OneEuroParams,
)
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


def _cmd(positions: dict[str, float], ts: float | None = None) -> JointCommand:
    stamp = time.perf_counter() if ts is None else ts
    return JointCommand(timestamp=stamp, positions=dict(positions), source_frame_ts=stamp)


def test_publisher_freeze_holds_the_last_pose_and_drops_new_targets() -> None:
    pub = MockPublisher(rate_hz=200)
    pub.start()
    try:
        pub.set_target(_cmd({"r_elbow": 0.7}))
        time.sleep(0.05)
        pub.emergency_stop()
        assert pub.frozen

        pub.set_target(_cmd({"r_elbow": -1.5}))
        time.sleep(0.05)
        for emitted in pub.history[-3:]:
            assert emitted.positions["r_elbow"] == pytest.approx(0.7)
    finally:
        pub.stop()


def test_publisher_release_resumes_tracking() -> None:
    pub = MockPublisher(rate_hz=200)
    pub.start()
    try:
        pub.set_target(_cmd({"r_elbow": 0.7}))
        time.sleep(0.05)
        pub.emergency_stop()
        pub.release_emergency_stop()
        assert not pub.frozen
        assert pub.held_command() is None

        pub.set_target(_cmd({"r_elbow": 0.2}))
        time.sleep(0.05)
        assert pub.history[-1].positions["r_elbow"] == pytest.approx(0.2)
    finally:
        pub.stop()


def test_watchdog_estop_freezes_the_publisher_with_no_help_from_the_loop() -> None:
    """The stall the watchdog catches is a loop that stopped calling update().

    So the freeze has to happen inside trigger_estop() on the watchdog thread. If
    it only happened inside update(), the E-stop would do nothing in exactly the
    situation it exists for — which is what the old code did.
    """
    pub = MockPublisher(rate_hz=200)
    cfg = SafetyConfig(watchdog_timeout_s=0.05)
    safety = SafetyLayer(publisher=pub, config=cfg, hotkey=None)
    safety.start()
    try:
        safety.update(_cmd({"r_elbow": 0.9}), mean_confidence=0.99)
        deadline = time.perf_counter() + 2.0
        while not pub.frozen and time.perf_counter() < deadline:
            time.sleep(0.01)  # feed nothing: the loop is "blocked"
        assert safety.estopped
        assert pub.frozen, "watchdog latched the E-stop but never stopped the publisher"
        held: Any = pub.held_command()
        assert held is not None
        assert held.positions["r_elbow"] == pytest.approx(0.9)
    finally:
        safety.stop()


def test_estop_key_still_works_when_the_deadman_is_off() -> None:
    pub = MockPublisher(rate_hz=200)
    hot = FakeHotkey()
    cfg = SafetyConfig(require_deadman=False, watchdog_timeout_s=10.0)
    safety = SafetyLayer(publisher=pub, config=cfg, hotkey=hot, watchdog=False)
    safety.start()
    try:
        # No dead-man held, yet commands flow because it is not required.
        safety.update(_cmd({"r_elbow": 0.4}), mean_confidence=0.99)
        assert not safety.estopped
        hot.press("esc")
        safety.update(_cmd({"r_elbow": 0.5}), mean_confidence=0.99)
        assert safety.estopped
        assert pub.frozen
    finally:
        safety.stop()


def test_reset_key_clears_the_latch_and_demands_a_deadman_recycle() -> None:
    pub = MockPublisher(rate_hz=200)
    hot = FakeHotkey()
    cfg = SafetyConfig(require_deadman=True, watchdog_timeout_s=10.0)
    safety = SafetyLayer(publisher=pub, config=cfg, hotkey=hot, watchdog=False)
    safety.start()
    try:
        hot.press("space")
        safety.update(_cmd({"r_elbow": 0.4}), mean_confidence=0.99)
        time.sleep(0.03)

        hot.press("esc")
        safety.update(_cmd({"r_elbow": 0.4}), mean_confidence=0.99)
        assert safety.estopped

        # Reset while the dead-man is STILL held: the latch clears, but motion
        # must not resume until the key has been seen released and pressed again.
        hot.release("esc")
        hot.press("r")
        safety.update(_cmd({"r_elbow": 0.4}), mean_confidence=0.99)
        assert not safety.estopped
        assert not pub.frozen

        hot.release("r")
        before: Any = pub._next  # noqa: SLF001
        safety.update(_cmd({"r_elbow": 1.2}), mean_confidence=0.99)
        assert pub._next is before, "commands resumed while the dead-man was still held"  # noqa: SLF001

        hot.release("space")
        safety.update(_cmd({"r_elbow": 1.2}), mean_confidence=0.99)
        hot.press("space")
        safety.update(_cmd({"r_elbow": 1.2}), mean_confidence=0.99)
        assert pub._next is not None  # noqa: SLF001
        assert pub._next.positions["r_elbow"] == pytest.approx(1.2)  # noqa: SLF001
    finally:
        safety.stop()


def test_reset_rebaselines_the_limiter_to_the_pose_actually_held() -> None:
    """After a reset the robot sits where it froze, not where the limiter last
    commanded — so without a re-baseline the first new command is unwrapped and
    velocity-clamped against a stale value."""
    limiter = FilterAndLimiter(
        one_euro=OneEuroParams(min_cutoff=5.0, beta=0.0),
        limiter=JointLimiterConfig(
            limits={"r_elbow": JointLimits(soft_min=-2.0, soft_max=2.0, max_velocity=2.0)}
        ),
    )
    pub = MockPublisher(rate_hz=200)
    cfg = SafetyConfig(require_deadman=False, watchdog_timeout_s=10.0)
    safety = SafetyLayer(publisher=pub, config=cfg, hotkey=None, watchdog=False)
    safety.on_estop_reset = limiter.reset_to
    safety.start()
    try:
        now = time.perf_counter()
        cmd = limiter({"r_elbow": 0.8}, timestamp=now, source_frame_ts=now)
        safety.update(cmd, mean_confidence=0.99)
        time.sleep(0.03)

        safety.trigger_estop("test")
        assert pub.frozen
        held: Any = pub.held_command()
        assert held is not None

        safety.reset_estop()
        assert limiter._last is not None  # noqa: SLF001
        assert limiter._last.positions == pytest.approx(held.positions)  # noqa: SLF001
    finally:
        safety.stop()
