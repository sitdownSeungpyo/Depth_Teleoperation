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
            self._prev = self._next if self._next is not None else command
            self._next = command
            self._stale_warned = False

    def current(self) -> JointCommand | None:
        with self._lock:
            return self._latest_emit

    @property
    def rate_hz(self) -> int:
        return self._rate_hz

    def _emit(self, command: JointCommand) -> None:
        raise NotImplementedError

    def _interpolate(self, now: float) -> JointCommand | None:
        with self._lock:
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
        # Extrapolate slightly past nxt by the same dt cadence; clamp to [0, 1+small].
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
