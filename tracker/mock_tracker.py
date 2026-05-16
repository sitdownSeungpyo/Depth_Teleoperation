"""MockTracker — replay a JSONL skeleton recording at a configured rate (spec §4.1)."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from core.types import SkeletonFrame


def _raise_windows_timer_resolution() -> None:
    """Best-effort: raise the multimedia timer to 1 ms so time.sleep is precise.

    Windows defaults to ~15 ms granularity; without this, 30 Hz pacing aliases to
    ~22 Hz. ``timeBeginPeriod`` is process-wide and persists for the run.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:  # noqa: BLE001
        pass


def _frame_from_dict(d: dict[str, object]) -> SkeletonFrame:
    keypoints = {
        name: np.asarray(coords, dtype=np.float64)
        for name, coords in (d["keypoints"] or {}).items()  # type: ignore[union-attr]
    }
    confidence = {
        name: float(c) for name, c in (d["confidence"] or {}).items()  # type: ignore[union-attr]
    }
    return SkeletonFrame(
        timestamp=float(d["timestamp"]),  # type: ignore[arg-type]
        keypoints=keypoints,
        confidence=confidence,
    )


class MockTracker:
    """Replay JSONL frames, pacing on perf_counter at the original timestamps.

    Synchronous-only: ``start()`` loads the file, ``stream()`` paces the yields. There
    is no background thread, so consuming ``stream()`` from the main loop is the only
    way to get frames. ``latest()`` returns the most recent yielded frame.
    """

    def __init__(self, jsonl_path: Path, loop: bool = True) -> None:
        self._path = jsonl_path
        self._loop = loop
        self._frames: list[SkeletonFrame] = []
        self._latest: SkeletonFrame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _load(self) -> None:
        if self._frames:
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._frames.append(_frame_from_dict(json.loads(line)))
        if not self._frames:
            raise RuntimeError(f"no frames in {self._path}")

    def start(self) -> None:
        self._load()
        self._stop.clear()
        _raise_windows_timer_resolution()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> SkeletonFrame | None:
        with self._lock:
            return self._latest

    def stream(self) -> Iterator[SkeletonFrame]:
        self._load()
        base = time.perf_counter()
        first_ts = self._frames[0].timestamp
        idx = 0
        while not self._stop.is_set():
            frame = self._frames[idx]
            target = base + (frame.timestamp - first_ts)
            sleep_for = target - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            stamped = SkeletonFrame(
                timestamp=time.perf_counter(),
                keypoints=frame.keypoints,
                confidence=frame.confidence,
            )
            with self._lock:
                self._latest = stamped
            yield stamped
            idx += 1
            if idx >= len(self._frames):
                if not self._loop:
                    break
                idx = 0
                base = time.perf_counter()
                first_ts = self._frames[0].timestamp
