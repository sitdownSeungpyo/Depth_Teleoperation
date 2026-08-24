"""Control-loop behaviour in app.main — camera faults vs tracking faults, and IK modes.

The loop used to *be* the tracker's stream consumer, so it stopped iterating
whenever the tracker stopped producing. A detector miss (operator steps out of
shot) therefore looked identical to a dead camera: the loop froze, nothing fed
the safety layer, and the stall watchdog latched an E-stop. These tests pin the
split — the camera stalling must E-stop, losing the pose must not.

They also cover the numeric IK running through the full safety + publisher path,
which previously existed only in app/sim_teleop.py (no safety, no publisher).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app import main
from core.types import SkeletonFrame
from tracker.base import TrackerHealth

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "config" / "default.yaml"


def _tpose_frame(timestamp: float) -> SkeletonFrame:
    from tests.generate_fixtures import _tpose_keypoints

    keypoints = {k: np.asarray(v, dtype=np.float64) for k, v in _tpose_keypoints().items()}
    return SkeletonFrame(
        timestamp=timestamp,
        keypoints=keypoints,
        confidence={k: 0.95 for k in keypoints},
    )


class _StubTracker:
    """A tracker whose camera and detector can be failed independently.

    ``detection_stops_after_s`` keeps delivering camera frames but stops finding a
    body — the operator-left-the-shot case. ``camera_stops_after_s`` stops
    delivering frames altogether — the real fault.
    """

    def __init__(
        self,
        detection_stops_after_s: float | None = None,
        camera_stops_after_s: float | None = None,
        fps: float = 30.0,
    ) -> None:
        self._detection_stop = detection_stops_after_s
        self._camera_stop = camera_stops_after_s
        self._period = 1.0 / fps
        self._stop = threading.Event()
        self._t0 = 0.0
        self._running = False
        self._lock = threading.Lock()
        self._last_capture: float | None = None
        self._last_detection: float | None = None
        self._capture_period: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._running = True

    def stop(self) -> None:
        self._stop.set()
        self._running = False

    def latest(self) -> SkeletonFrame | None:
        return None  # the loop reads frames through the pump, not this

    def stream(self) -> Any:
        while not self._stop.is_set():
            elapsed = time.perf_counter() - self._t0
            camera_up = self._camera_stop is None or elapsed < self._camera_stop
            detecting = camera_up and (
                self._detection_stop is None or elapsed < self._detection_stop
            )
            if camera_up:
                now = time.perf_counter()
                with self._lock:
                    if self._last_capture is not None:
                        self._capture_period = now - self._last_capture
                    self._last_capture = now
            if detecting:
                frame = _tpose_frame(time.perf_counter())
                with self._lock:
                    self._last_detection = time.perf_counter()
                yield frame
            time.sleep(self._period)

    def health(self) -> TrackerHealth:
        now = time.perf_counter()
        with self._lock:
            capture, detection = self._last_capture, self._last_detection
            period = self._capture_period
        alive = self._running and not self._stop.is_set()
        return TrackerHealth(
            running=alive,
            alive=alive,
            frame_age_s=math.inf if capture is None else max(now - capture, 0.0),
            detection_age_s=math.inf if detection is None else max(now - detection, 0.0),
            capture_period_s=math.inf if period is None else period,
        )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    tracker: _StubTracker,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Point app.main at the stub tracker and capture the publisher it builds.

    ``overrides`` patches loaded config sections, so a timing test can shrink the
    real thresholds instead of sleeping through them.
    """
    created: dict[str, Any] = {}
    monkeypatch.setattr(
        main, "_build_tracker", lambda cfg, kind, replay: tracker  # noqa: ARG005
    )
    real_build_publisher = main._build_publisher  # noqa: SLF001

    def capture(cfg: dict[str, Any], kind: str) -> Any:
        publisher = real_build_publisher(cfg, kind)
        created["publisher"] = publisher
        return publisher

    monkeypatch.setattr(main, "_build_publisher", capture)

    if overrides:
        real_load = main._load_config  # noqa: SLF001

        def load(path: Path) -> dict[str, Any]:
            cfg = real_load(path)
            for section, values in overrides.items():
                cfg.setdefault(section, {}).update(values)
            return cfg

        monkeypatch.setattr(main, "_load_config", load)
    return created


def _estop_logged(caplog: pytest.LogCaptureFixture) -> bool:
    return any("E-STOP triggered" in rec.message for rec in caplog.records)


def test_losing_the_pose_does_not_estop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tracker = _StubTracker(detection_stops_after_s=0.6)
    _install(monkeypatch, tracker)
    with caplog.at_level(logging.WARNING):
        summary = main.run(
            config_path=CONFIG,
            tracker_kind="mock",
            publisher_kind="mock",
            duration_s=2.2,
            enable_hotkeys=False,
        )
    assert summary["frames"] > 0
    assert not _estop_logged(caplog), (
        "a healthy camera with no visible operator must not trip the stall watchdog"
    )
    assert any("tracking-loss policy" in rec.message for rec in caplog.records), (
        "the loss ramp is the designed response and should have engaged"
    )


def test_a_slow_backend_is_not_mistaken_for_a_dead_camera(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """1.5 fps is slower than the 0.5 s stale floor, but the camera is fine.

    Capture and detection share a thread, so the capture stamp only advances as
    fast as the detector runs. Comparing it against a fixed threshold made any
    backend slower than that threshold look dead — observed on real hardware,
    where a cold CUDA warm-up on the first inference latched an E-stop the
    ``--no-hotkeys`` run could never clear.
    """
    # 3 fps -> a 333 ms capture period, comfortably past the 100 ms stale floor.
    # With the floor alone the loop would withhold liveness for 233 ms of every
    # cycle, which outlasts the 150 ms watchdog and latches an E-stop. Scaled by
    # the measured cadence the limit becomes 4 x 333 ms, and nothing trips.
    tracker = _StubTracker(fps=3.0)
    _install(
        monkeypatch,
        tracker,
        overrides={
            "main": {"camera_stale_s": 0.1, "camera_startup_grace_s": 1.0},
            "safety": {"watchdog_timeout_s": 0.15},
        },
    )
    with caplog.at_level(logging.WARNING):
        main.run(
            config_path=CONFIG,
            tracker_kind="mock",
            publisher_kind="mock",
            duration_s=2.0,
            enable_hotkeys=False,
        )
    assert not _estop_logged(caplog), (
        "a slow-but-healthy pipeline must not read as a stalled camera"
    )


def test_camera_stall_still_estops(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tracker = _StubTracker(camera_stops_after_s=0.6)
    _install(monkeypatch, tracker)
    with caplog.at_level(logging.WARNING):
        main.run(
            config_path=CONFIG,
            tracker_kind="mock",
            publisher_kind="mock",
            duration_s=2.5,
            enable_hotkeys=False,
        )
    assert _estop_logged(caplog), "a dead camera must still reach the watchdog"


def test_numeric_ik_runs_through_safety_and_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of the integration: accurate IK *and* the safety layer at once."""
    tracker = _StubTracker()
    created = _install(monkeypatch, tracker)
    summary = main.run(
        config_path=CONFIG,
        tracker_kind="mock",
        publisher_kind="mock",
        duration_s=1.5,
        ik="numeric",
        enable_hotkeys=False,
    )
    assert summary["frames"] > 0
    publisher = created["publisher"]
    assert publisher.history, "numeric IK produced no published commands"
    positions = publisher.history[-1].positions
    for joint in ("r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
                  "l_shoulder_pitch", "l_elbow"):
        assert joint in positions, f"{joint} missing from the numeric IK command"
    # Joints the numeric solver does not estimate are still present, so the joint
    # set stays constant frame to frame.
    assert "r_wrist_yaw" in positions
    assert "head_pitch" in positions


def test_analytic_ik_is_still_selectable(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _StubTracker()
    created = _install(monkeypatch, tracker)
    summary = main.run(
        config_path=CONFIG,
        tracker_kind="mock",
        publisher_kind="mock",
        duration_s=2.5,
        ik="analytic",
        enable_hotkeys=False,
    )
    assert summary["frames"] > 0
    assert created["publisher"].history


def test_unknown_ik_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ik mode"):
        main.run(
            config_path=CONFIG,
            tracker_kind="mock",
            publisher_kind="mock",
            ik="magic",
            enable_hotkeys=False,
        )
