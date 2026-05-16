"""Drive a tracker and persist each SkeletonFrame to a JSONL file.

Default mode (free movement):
    python -m app.record --config .\\config\\default.yaml --out .\\recordings\\session.jsonl --duration 30

T-pose calibration capture (5-second hold, useful for operator_arm_length tuning):
    python -m app.record --config .\\config\\default.yaml --out .\\recordings\\tpose.jsonl --tpose --duration 5
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from core.types import SkeletonFrame

log = logging.getLogger(__name__)


def _frame_to_dict(frame: SkeletonFrame) -> dict[str, Any]:
    return {
        "timestamp": frame.timestamp,
        "keypoints": {name: arr.tolist() for name, arr in frame.keypoints.items()},
        "confidence": dict(frame.confidence),
    }


def _countdown(seconds: int, prompt: str) -> None:
    print(prompt)
    for s in range(seconds, 0, -1):
        print(f"  starting in {s}...", flush=True)
        time.sleep(1.0)
    print("  GO", flush=True)


def record(
    config_path: Path,
    tracker_kind: str,
    out_path: Path,
    duration_s: float,
    tpose: bool = False,
) -> int:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if tracker_kind == "realsense":
        from tracker.realsense_tracker import RealSenseTracker

        rs_cfg = cfg["tracker"]["realsense"]
        tracker: Any = RealSenseTracker(
            color_resolution=tuple(rs_cfg["color_resolution"]),
            depth_resolution=tuple(rs_cfg["depth_resolution"]),
            fps=int(rs_cfg["fps"]),
            depth_max_m=float(rs_cfg["depth_max_m"]),
            min_visibility=float(cfg["tracker"]["pose"]["min_visibility"]),
            model_asset_path=rs_cfg.get("model_asset_path"),
        )
    else:
        raise ValueError("record only supports --tracker realsense")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if tpose:
        _countdown(3, "Stand 1.5-2 m from the camera with arms straight out (T-pose). Hold still.")
    else:
        _countdown(3, "Stand 1.5-2 m from the camera. Recording will start.")

    interrupted = False

    def _on_sigint(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _on_sigint)

    n = 0
    log_every = 0.5  # seconds
    confidence_running = 0.0
    print("  loading camera + MediaPipe (one-time ~10s)...", flush=True)
    tracker.start()
    print("  recording...", flush=True)
    start = time.perf_counter()
    last_log = start
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for frame in tracker.stream():
                f.write(json.dumps(_frame_to_dict(frame)) + "\n")
                n += 1
                confidence_running = 0.9 * confidence_running + 0.1 * frame.mean_confidence()
                now = time.perf_counter()
                if now - last_log >= log_every:
                    fps = n / max(now - start, 1e-6)
                    print(
                        f"\rframes={n:5d}  fps={fps:5.1f}  mean_conf={confidence_running:.2f}",
                        end="",
                        flush=True,
                    )
                    last_log = now
                if now - start > duration_s or interrupted:
                    break
    finally:
        tracker.stop()
        print()  # newline after the live progress line

    log.info("wrote %d frames to %s", n, out_path)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Record skeleton frames to JSONL")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tracker", choices=("realsense",), default="realsense")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--tpose",
        action="store_true",
        help="prompt for T-pose hold; useful for operator arm-length calibration",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(0 if record(args.config, args.tracker, args.out, args.duration, args.tpose) > 0 else 1)


if __name__ == "__main__":
    main()
