"""M6 — Safety Layer (spec §4.6).

Wraps a Publisher with:
- Dead-man switch (pynput global hotkey on Windows; commands flow only when held).
- E-stop hotkey (zero output, disable command flow).
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
    """Global hotkey state via pynput. Listener runs on a background thread."""

    def __init__(self, keys: list[str]) -> None:
        self._wanted = {k.lower() for k in keys}
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._listener: Any = None

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
        except ImportError:
            log.warning("pynput unavailable; hotkey disabled (commands will not flow)")
            return
        self._listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


@dataclass
class SafetyConfig:
    deadman_key: str = "space"
    estop_key: str = "esc"
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
    """Wraps a Publisher; can override or zero commands at any time."""

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

    def trigger_estop(self, reason: str = "manual") -> None:
        if not self._estopped:
            log.warning("E-STOP triggered (%s)", reason)
        self._estopped = True

    def reset_estop(self) -> None:
        self._estopped = False

    def _deadman_held(self) -> bool:
        if self._hotkey is None:
            return True  # tests / headless mode default to allowing flow
        return self._hotkey.is_pressed(self._cfg.deadman_key)

    def _check_estop_key(self) -> None:
        if self._hotkey is not None and self._hotkey.is_pressed(self._cfg.estop_key):
            self.trigger_estop("estop hotkey")

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
        self._check_estop_key()

        if self._estopped:
            zero = JointCommand(
                timestamp=command.timestamp,
                positions={j: 0.0 for j in command.positions},
                source_frame_ts=command.source_frame_ts,
            )
            self._pub.set_target(zero)
            self._last_update = now
            self._last_cmd = zero
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

        Use this during initialisation phases (e.g. operator-arm calibration) where
        we are processing frames but cannot yet emit valid joint targets — without
        this, the watchdog would fire because no ``update()`` has happened yet.
        """
        self._last_update = self._clock()
