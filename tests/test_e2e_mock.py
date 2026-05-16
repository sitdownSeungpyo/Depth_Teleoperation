from __future__ import annotations

from pathlib import Path

import numpy as np

from app.main import run

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "config" / "default.yaml"


def test_e2e_mock_arm_circle_meets_rate_and_latency_budget(arm_circle_jsonl: Path) -> None:
    summary = run(
        config_path=CONFIG,
        tracker_kind="mock",
        publisher_kind="mock",
        replay=arm_circle_jsonl,
        duration_s=2.5,
        require_deadman=False,
    )
    assert summary["frames"] > 0
    assert summary["rate_hz"] >= 28.0, f"loop rate too low: {summary['rate_hz']:.1f} Hz"
    assert summary["latency_p95_s"] < 0.1, (
        f"p95 latency budget exceeded: {summary['latency_p95_s']*1000:.1f} ms"
    )


def test_e2e_mock_tpose_calibrates(tpose_jsonl: Path) -> None:
    # Need duration > calibration_frames/fps + a margin; calibration is 30 frames
    # at 30 Hz = ~1 s, so 2.5 s leaves ~1.5 s for post-calibration frames.
    summary = run(
        config_path=CONFIG,
        tracker_kind="mock",
        publisher_kind="mock",
        replay=tpose_jsonl,
        duration_s=2.5,
        require_deadman=False,
    )
    assert summary["frames"] > 30, summary
    # Latencies are computed only after calibration completes; just sanity-check.
    assert np.isfinite(summary["latency_p95_s"])
