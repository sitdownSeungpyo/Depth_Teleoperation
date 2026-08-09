"""SkeletonTracker Protocol shared by all tracker backends (spec §4.1)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from core.types import SkeletonFrame


@dataclass(frozen=True)
class TrackerHealth:
    """Liveness snapshot of a tracker's capture producer.

    Exists because ``latest()`` alone cannot express failure: when a capture
    thread dies it keeps returning the last frame it ever produced, so a frozen
    pose is indistinguishable from a perfectly still operator. Callers need the
    *age* of that frame and whether the producer is still running.
    """

    running: bool          # start() called and stop() not yet called
    alive: bool            # producer still able to deliver frames
    frame_age_s: float     # seconds since the last accepted frame (inf if none)
    error: str | None = None   # fatal error text when the producer gave up

    def is_stale(self, max_age_s: float) -> bool:
        return self.frame_age_s > max_age_s

    def describe(self) -> str:
        if self.error is not None:
            return f"DEAD ({self.error})"
        if not self.running:
            return "STOPPED"
        if not math.isfinite(self.frame_age_s):
            return "NO FRAMES YET"
        return f"age {self.frame_age_s * 1000:.0f} ms"


def health_of(tracker: Any) -> TrackerHealth:
    """Best-effort health for any tracker, including ones predating the API.

    Trackers without ``health()`` report as alive with an unknown (zero) age, so
    callers can use this unconditionally without special-casing.
    """
    getter = getattr(tracker, "health", None)
    if getter is None:
        return TrackerHealth(running=True, alive=True, frame_age_s=0.0)
    result = getter()
    return result if isinstance(result, TrackerHealth) else TrackerHealth(
        running=True, alive=True, frame_age_s=0.0
    )


@runtime_checkable
class SkeletonTracker(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self) -> SkeletonFrame | None: ...
    def stream(self) -> Iterator[SkeletonFrame]: ...
    def health(self) -> TrackerHealth: ...
