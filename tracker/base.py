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
    frame_age_s: float     # seconds since the last CAMERA frame (inf if none)
    error: str | None = None   # fatal error text when the producer gave up
    # Seconds since the last frame that actually carried a pose (inf if none).
    #
    # Split from ``frame_age_s`` because "the operator stepped out of shot" and
    # "the camera stopped feeding us" are different faults needing opposite
    # responses, and one age conflated them: a detector miss read as a dead
    # camera, so a person briefly leaving the frame tripped the stall watchdog.
    detection_age_s: float = math.inf
    # Smoothed interval between captured frames, or inf before enough have
    # arrived to measure one. Exists so callers can judge staleness against what
    # the producer actually achieves — see :meth:`stall_limit_s`.
    capture_period_s: float = math.inf

    def is_stale(self, max_age_s: float) -> bool:
        """True when the CAMERA has stopped delivering frames — a real fault."""
        return self.frame_age_s > max_age_s

    def stall_limit_s(self, floor_s: float, factor: float) -> float:
        """Age above which this pipeline's camera has genuinely stopped.

        A fixed threshold cannot serve both a 30 fps GPU backend and a 5 fps CPU
        one. Set it for the fast case and the slow case trips on ordinary jitter;
        set it for the slow case and a real stall goes unnoticed for a second.

        This matters more than it looks: capture and detection share one thread,
        so the capture stamp only advances as fast as the detector runs. A cold
        CUDA warm-up on the first inference can hold it for seconds on a camera
        that is working perfectly. Scaling the limit by the measured cadence is
        what keeps that from reading as a dead camera.

        Falls back to ``floor_s`` until a cadence has been measured.
        """
        if not math.isfinite(self.capture_period_s):
            return floor_s
        return max(floor_s, factor * self.capture_period_s)

    def is_detection_stale(self, max_age_s: float) -> bool:
        """True when no pose has been produced recently. Says nothing about the
        camera: a healthy camera pointed at an empty room reports exactly this."""
        return self.detection_age_s > max_age_s

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
        return TrackerHealth(
            running=True, alive=True, frame_age_s=0.0, detection_age_s=0.0
        )
    result = getter()
    return result if isinstance(result, TrackerHealth) else TrackerHealth(
        running=True, alive=True, frame_age_s=0.0, detection_age_s=0.0
    )


@runtime_checkable
class SkeletonTracker(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self) -> SkeletonFrame | None: ...
    def stream(self) -> Iterator[SkeletonFrame]: ...
    def health(self) -> TrackerHealth: ...
