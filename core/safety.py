"""M6 — Safety Layer (spec §4.6).

Wraps a Publisher with:
- Dead-man switch (pynput global hotkey on Windows; commands flow only when held).
- E-stop hotkey — freezes the publisher at the pose the robot already holds, and
  latches until the reset key is pressed.
- Watchdog (E-stop if the main loop stops feeding commands) — runs on its OWN
  thread, because the stall it must catch is precisely the case where the main
  loop is blocked and therefore cannot tick anything itself.
- Tracking-loss policy (ramp to a configured safe pose when confidence drops).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.types import JointCommand
from publisher.base import Publisher

log = logging.getLogger(__name__)


class HotkeyBackend(Protocol):
    def is_pressed(self, key: str) -> bool: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class PynputHotkey:
    """Global hotkey state via pynput. Listener runs on a background thread.

    ``strict`` makes a missing pynput a hard failure instead of a warning. Use it
    whenever the dead-man is required: a dead-man we cannot read is not a
    degraded dead-man, it is no dead-man at all, and starting anyway would let a
    real robot move with no way to stop it from the keyboard.
    """

    def __init__(self, keys: list[str], strict: bool = False) -> None:
        self._wanted = {k.lower() for k in keys}
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._listener: Any = None
        self._strict = strict
        self.available = False

    def _normalise(self, key: Any) -> str | None:
        try:
            from pynput.keyboard import Key, KeyCode
        except ImportError:
            return None
        if isinstance(key, KeyCode) and key.char is not None:
            return str(key.char).lower()
        if isinstance(key, Key):
            return str(key.name).lower()
        return None

    def _on_press(self, key: Any) -> None:
        name = self._normalise(key)
        if name is not None and name in self._wanted:
            with self._lock:
                self._held.add(name)

    def _on_release(self, key: Any) -> None:
        name = self._normalise(key)
        if name is not None:
            with self._lock:
                self._held.discard(name)

    def is_pressed(self, key: str) -> bool:
        with self._lock:
            return key.lower() in self._held

    def start(self) -> None:
        try:
            from pynput.keyboard import Listener
        except ImportError as exc:
            if self._strict:
                raise RuntimeError(
                    "pynput is unavailable, so the dead-man and E-stop keys cannot be "
                    "read; refusing to start. Install it with `pip install pynput`."
                ) from exc
            log.warning("pynput unavailable; hotkey disabled (commands will not flow)")
            return
        self._listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()
        self.available = True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self.available = False


@dataclass
class SafetyConfig:
    deadman_key: str = "space"
    estop_key: str = "esc"
    # Clears a latched E-stop. The latch is deliberate — whatever tripped it is
    # usually still true one frame later — but without a reset key the only way
    # out is killing the process, which leaves the robot powered and commanded by
    # nobody. Reset additionally demands the dead-man be re-pressed.
    reset_key: str = "r"
    # When False the dead-man is not required for commands to flow. Kept separate
    # from "are hotkeys wired up at all" so the E-stop key still works without it;
    # the two used to be the same flag, which silently disabled E-stop whenever
    # the dead-man was off.
    require_deadman: bool = True
    confidence_threshold: float = 0.5
    loss_grace_period_s: float = 0.5
    ramp_to_safe_s: float = 1.5
    watchdog_factor: int = 3  # cycles
    cycle_dt_s: float = 1.0 / 30.0
    # Explicit stall timeout in seconds. None -> cycle_dt_s * watchdog_factor.
    # Prefer setting this: the derived value is a *loop period* multiple, which at
    # 60 Hz is only 50 ms — far tighter than a 30 fps camera's real frame jitter,
    # so it would trip on normal hiccups rather than on an actual stall.
    watchdog_timeout_s: float | None = None
    safe_pose: dict[str, float] = field(default_factory=dict)

    def stall_timeout_s(self) -> float:
        if self.watchdog_timeout_s is not None:
            return float(self.watchdog_timeout_s)
        return self.cycle_dt_s * self.watchdog_factor


@dataclass
class _LossState:
    started_at: float | None = None
    ramping: bool = False
    ramp_start: float = 0.0
    ramp_from: dict[str, float] = field(default_factory=dict)


class SafetyLayer:
    """Wraps a Publisher; can freeze, override, or ramp commands at any time."""

    def __init__(
        self,
        publisher: Publisher,
        config: SafetyConfig,
        hotkey: HotkeyBackend | None = None,
        clock: Callable[[], float] = time.perf_counter,
        watchdog: bool = True,
    ) -> None:
        self._pub = publisher
        self._cfg = config
        self._hotkey = hotkey
        self._clock = clock
        self._estopped = False
        self._last_update = clock()
        self._loss = _LossState()
        self._last_cmd: JointCommand | None = None
        # After a reset the dead-man must be observed RELEASED once before it
        # counts as held again, so a key held down through the fault cannot
        # resume motion the instant the reset lands.
        self._deadman_recycle = False
        # Optional hook invoked with the held pose when a latched E-stop is
        # cleared, so the caller can re-baseline its filter/limiter to where the
        # robot actually is. Without it the first post-reset command is unwrapped
        # and velocity-clamped against a stale pre-fault value.
        self.on_estop_reset: Callable[[dict[str, float]], None] | None = None
        # The watchdog MUST NOT be driven from the caller's loop: the failure it
        # exists to catch (the loop blocked on a dead camera) is exactly the case
        # where the caller can't call anything. Own thread, real-time sleeps.
        self._watchdog_enabled = watchdog
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._hotkey is not None:
            self._hotkey.start()
        self._pub.start()
        self._last_update = self._clock()
        if self._watchdog_enabled and self._watchdog_thread is None:
            self._watchdog_stop.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, name="safety-watchdog", daemon=True
            )
            self._watchdog_thread.start()

    def stop(self) -> None:
        self._watchdog_stop.set()
        thread, self._watchdog_thread = self._watchdog_thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        try:
            self._pub.stop()
        finally:
            if self._hotkey is not None:
                self._hotkey.stop()

    def _watchdog_loop(self) -> None:
        # Poll at a fraction of the stall timeout so detection latency is small
        # relative to it. Waits on the Event (wall clock) so an injected test
        # clock can't stall the poller itself.
        period = max(self._cfg.stall_timeout_s() / 4.0, 0.01)
        while not self._watchdog_stop.wait(period):
            self.watchdog_tick()

    @property
    def estopped(self) -> bool:
        return self._estopped

    def last_command(self) -> JointCommand | None:
        """The last command actually forwarded to the publisher."""
        return self._last_cmd

    def trigger_estop(self, reason: str = "manual") -> None:
        """Latch the E-stop and freeze the publisher immediately.

        The publisher action happens HERE, not in :meth:`update`. The watchdog's
        whole reason to exist is a main loop that stopped calling ``update()``, so
        an E-stop that only took effect inside ``update()`` would do nothing in
        exactly the case it was built for.
        """
        if self._estopped:
            return
        self._estopped = True
        log.warning("E-STOP triggered (%s)", reason)
        self._pub.emergency_stop()

    def reset_estop(self) -> None:
        """Clear a latched E-stop and resume normal setpoint flow."""
        if not self._estopped:
            return
        held = self._pub.current()
        self._estopped = False
        self._pub.release_emergency_stop()
        self._last_update = self._clock()
        self._loss = _LossState()
        self._deadman_recycle = self._cfg.require_deadman and self._hotkey is not None
        if held is not None:
            self._last_cmd = held
            if self.on_estop_reset is not None:
                self.on_estop_reset(dict(held.positions))
        log.warning(
            "E-stop reset; dead-man must be released and re-pressed before commands flow"
        )

    def _deadman_held(self) -> bool:
        if not self._cfg.require_deadman:
            return True
        if self._hotkey is None:
            return True  # tests / headless mode default to allowing flow
        held = self._hotkey.is_pressed(self._cfg.deadman_key)
        if self._deadman_recycle:
            if not held:
                self._deadman_recycle = False
            return False
        return held

    def _check_hotkeys(self) -> None:
        if self._hotkey is None:
            return
        if self._hotkey.is_pressed(self._cfg.estop_key):
            self.trigger_estop("estop hotkey")
            return
        if self._estopped and self._hotkey.is_pressed(self._cfg.reset_key):
            self.reset_estop()

    def _maybe_ramp(
        self, command: JointCommand, mean_confidence: float, now: float
    ) -> JointCommand:
        if mean_confidence < self._cfg.confidence_threshold:
            if self._loss.started_at is None:
                self._loss.started_at = now
            elif now - self._loss.started_at > self._cfg.loss_grace_period_s and not self._loss.ramping:
                self._loss.ramping = True
                self._loss.ramp_start = now
                self._loss.ramp_from = dict(command.positions)
                log.warning("tracking loss > %.2fs; ramping to safe pose", self._cfg.loss_grace_period_s)
        else:
            self._loss = _LossState()

        if not self._loss.ramping:
            return command

        elapsed = now - self._loss.ramp_start
        u = min(1.0, elapsed / max(self._cfg.ramp_to_safe_s, 1e-3))
        # cosine ease-in/out for a smoother transition.
        e = 0.5 - 0.5 * math.cos(math.pi * u)
        positions = {
            joint: self._loss.ramp_from.get(joint, command.positions.get(joint, 0.0)) * (1 - e)
            + self._cfg.safe_pose.get(joint, command.positions.get(joint, 0.0)) * e
            for joint in command.positions
        }
        return JointCommand(
            timestamp=command.timestamp,
            positions=positions,
            source_frame_ts=command.source_frame_ts,
        )

    def update(self, command: JointCommand, mean_confidence: float = 1.0) -> None:
        """Forward a (possibly safety-adjusted) command to the wrapped publisher."""
        now = self._clock()
        self._check_hotkeys()

        if self._estopped:
            # The publisher is already frozen at the pose the robot holds. Sending
            # anything here would be a *motion* command, which is exactly what an
            # E-stop must not produce. Just keep the loop's liveness stamp fresh
            # so the watchdog doesn't re-trip on an already-stopped system.
            self._last_update = now
            return

        if not self._deadman_held():
            # Hold the last command rather than send anything new.
            if self._last_cmd is not None:
                self._pub.set_target(self._last_cmd)
            self._last_update = now
            return

        cmd = self._maybe_ramp(command, mean_confidence, now)
        self._pub.set_target(cmd)
        self._last_update = now
        self._last_cmd = cmd

    def watchdog_tick(self) -> None:
        """Triggers E-stop when no command has been forwarded within the stall
        timeout. Driven by :meth:`start`'s background thread; still public so
        tests can step it with an injected clock."""
        now = self._clock()
        if now - self._last_update > self._cfg.stall_timeout_s():
            self.trigger_estop("watchdog")

    def note_alive(self) -> None:
        """Mark the loop as alive without forwarding a command.

        Use this during phases where we are processing frames but cannot yet emit
        valid joint targets — operator-arm calibration, or a tick with no new pose
        while the camera is still healthy. Without this, the watchdog would read
        those as a stall even though nothing is wrong.
        """
        self._last_update = self._clock()
