"""Publisher Protocol and an interpolating base class shared by all backends (spec §4.5)."""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Protocol, runtime_checkable

from core.types import JointCommand


def _raise_windows_timer_resolution() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:  # noqa: BLE001
        pass

log = logging.getLogger(__name__)

STALE_INPUT_THRESHOLD_S = 0.2


@runtime_checkable
class Publisher(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_target(self, command: JointCommand) -> None: ...
    def current(self) -> JointCommand | None: ...
    def emergency_stop(self) -> None: ...
    def release_emergency_stop(self) -> None: ...


class InterpolatingPublisherBase:
    """Linear-interpolation publisher running at a fixed rate.

    Holds the two most recent setpoints and interpolates between them on a worker
    thread. Subclasses override :meth:`_emit` to send the interpolated command
    elsewhere (UDP, ROS, mock log).
    """

    def __init__(self, rate_hz: int = 100) -> None:
        self._dt = 1.0 / rate_hz
        self._rate_hz = rate_hz
        self._lock = threading.Lock()
        self._prev: JointCommand | None = None
        self._next: JointCommand | None = None
        self._latest_emit: JointCommand | None = None
        self._stop = threading.Event()
        self._stale_warned = False
        self._thread: threading.Thread | None = None
        # Emergency freeze state. See :meth:`emergency_stop`.
        self._frozen = False
        self._frozen_cmd: JointCommand | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        _raise_windows_timer_resolution()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="publisher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def set_target(self, command: JointCommand) -> None:
        with self._lock:
            if self._frozen:
                return  # emergency stop in force — every new setpoint is dropped
            self._prev = self._next if self._next is not None else command
            self._next = command
            self._stale_warned = False

    def current(self) -> JointCommand | None:
        with self._lock:
            return self._latest_emit

    @property
    def rate_hz(self) -> int:
        return self._rate_hz

    # ---- emergency stop -------------------------------------------------
    #
    # An E-stop on a POSITION-controlled robot cannot mean "send zeros": zero is
    # a *pose*, so commanding it is a full-speed move to the robot's zero pose —
    # the opposite of stopping. It also bypasses the upstream velocity limiter,
    # because the safety layer sits downstream of it. The only meaningful stop
    # here is to keep repeating the pose the robot already holds and to drop
    # every setpoint that arrives afterwards.

    def emergency_stop(self) -> None:
        """Freeze output at the last emitted command; ignore new targets.

        Idempotent, and safe to call from the watchdog thread — which is the
        point: the stall the watchdog catches is a main loop that stopped calling
        anything, so the E-stop must not depend on that loop to take effect.
        """
        with self._lock:
            if self._frozen:
                return
            held = self._latest_emit or self._next or self._prev
            self._frozen = True
            self._frozen_cmd = (
                JointCommand(
                    timestamp=held.timestamp,
                    positions=dict(held.positions),
                    source_frame_ts=held.source_frame_ts,
                )
                if held is not None
                else None
            )
        log.warning("publisher frozen by emergency stop (holding last commanded pose)")

    def release_emergency_stop(self) -> None:
        """Resume normal setpoint flow after a latched E-stop is cleared."""
        with self._lock:
            if not self._frozen:
                return
            self._frozen = False
            self._frozen_cmd = None
            # Drop the pre-freeze interpolation state: its timestamps are now far
            # in the past, so keeping it would make the first post-release
            # setpoint interpolate over a bogus span.
            self._prev = None
            self._next = None
            self._stale_warned = False
        log.warning("publisher released from emergency stop")

    @property
    def frozen(self) -> bool:
        with self._lock:
            return self._frozen

    def held_command(self) -> JointCommand | None:
        """The pose being repeated while frozen, or None when not frozen."""
        with self._lock:
            return self._frozen_cmd

    # ---------------------------------------------------------------------

    def _emit(self, command: JointCommand) -> None:
        raise NotImplementedError

    def _interpolate(self, now: float) -> JointCommand | None:
        with self._lock:
            if self._frozen:
                frozen = self._frozen_cmd
                if frozen is None:
                    return None
                # Re-stamp so downstream age arithmetic keeps working; the
                # positions themselves never move while frozen.
                return JointCommand(
                    timestamp=now,
                    positions=dict(frozen.positions),
                    source_frame_ts=frozen.source_frame_ts,
                )
            prev = self._prev
            nxt = self._next
            stale_warned = self._stale_warned
        if nxt is None:
            return None
        if prev is None or prev is nxt:
            return JointCommand(
                timestamp=now,
                positions=dict(nxt.positions),
                source_frame_ts=nxt.source_frame_ts,
            )
        span = max(nxt.timestamp - prev.timestamp, 1e-6)
        age = now - nxt.timestamp
        if age > STALE_INPUT_THRESHOLD_S:
            if not stale_warned:
                log.warning(
                    "publisher input is stale (>%dms), holding last value",
                    int(STALE_INPUT_THRESHOLD_S * 1000),
                )
                with self._lock:
                    self._stale_warned = True
            return JointCommand(
                timestamp=now,
                positions=dict(nxt.positions),
                source_frame_ts=nxt.source_frame_ts,
            )
        # Interpolate between the two most recent setpoints. u is clamped to
        # [0, 1], so a late tick holds at nxt instead of extrapolating past it.
        u = (now - prev.timestamp) / span
        u = max(0.0, min(u, 1.0))
        positions = {
            joint: prev.positions[joint] * (1 - u) + nxt.positions[joint] * u
            for joint in nxt.positions
            if joint in prev.positions
        }
        # Carry over any joints unique to nxt unchanged.
        for joint, value in nxt.positions.items():
            positions.setdefault(joint, value)
        return JointCommand(
            timestamp=now,
            positions=positions,
            source_frame_ts=nxt.source_frame_ts,
        )

    def _loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            cmd = self._interpolate(now)
            if cmd is not None:
                self._emit(cmd)
                with self._lock:
                    self._latest_emit = cmd
            next_tick += self._dt
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Behind schedule; reset baseline so we don't busy-loop forever.
                next_tick = time.perf_counter()
