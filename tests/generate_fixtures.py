"""Generate synthetic SkeletonFrame JSONL fixtures used by tests.

Run from the repo root:
    python -m tests.generate_fixtures

Produces:
- tests/fixtures/tpose.jsonl       — 200 frames @ 30 Hz, perfect T-pose
- tests/fixtures/arm_circle.jsonl  — 200 frames @ 30 Hz, right wrist tracing a vertical
                                     circle in front of the operator
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Geometry (metres)
SHOULDER_HALF_WIDTH = 0.18
SHOULDER_Y = 1.40
HIP_HALF_WIDTH = 0.10
HIP_Y = 1.00
HEAD_Y = 1.65
UPPER_ARM = 0.28
LOWER_ARM = 0.27
ARM_LENGTH = UPPER_ARM + LOWER_ARM


def _tpose_keypoints() -> dict[str, list[float]]:
    return {
        "head": [0.0, HEAD_Y, 0.0],
        "neck": [0.0, SHOULDER_Y + 0.05, 0.0],
        "torso": [0.0, 0.5 * (SHOULDER_Y + HIP_Y), 0.0],
        "left_shoulder": [-SHOULDER_HALF_WIDTH, SHOULDER_Y, 0.0],
        "right_shoulder": [SHOULDER_HALF_WIDTH, SHOULDER_Y, 0.0],
        "left_elbow": [-SHOULDER_HALF_WIDTH - UPPER_ARM, SHOULDER_Y, 0.0],
        "right_elbow": [SHOULDER_HALF_WIDTH + UPPER_ARM, SHOULDER_Y, 0.0],
        "left_wrist": [-SHOULDER_HALF_WIDTH - ARM_LENGTH, SHOULDER_Y, 0.0],
        "right_wrist": [SHOULDER_HALF_WIDTH + ARM_LENGTH, SHOULDER_Y, 0.0],
        "left_hip": [-HIP_HALF_WIDTH, HIP_Y, 0.0],
        "right_hip": [HIP_HALF_WIDTH, HIP_Y, 0.0],
    }


def _arm_circle_keypoints(t: float, radius: float = 0.25) -> dict[str, list[float]]:
    """Right wrist traces a vertical circle in front of the chest; left arm stays in T-pose."""
    base = _tpose_keypoints()
    rs = base["right_shoulder"]
    cx, cy, cz = rs[0], rs[1] - 0.10, 0.45
    angle = 2.0 * math.pi * t * 0.5  # 0.5 Hz
    wrist = [cx + radius * math.cos(angle), cy + radius * math.sin(angle), cz]
    elbow = [
        rs[0] + 0.5 * (wrist[0] - rs[0]),
        rs[1] + 0.5 * (wrist[1] - rs[1]),
        rs[2] + 0.5 * (wrist[2] - rs[2]),
    ]
    base["right_elbow"] = elbow
    base["right_wrist"] = wrist
    return base


def _confidence(value: float = 0.95) -> dict[str, float]:
    keys = list(_tpose_keypoints().keys())
    return {k: value for k in keys}


def _write_jsonl(path: Path, build, n_frames: int = 200, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / fps
    with path.open("w", encoding="utf-8") as f:
        for i in range(n_frames):
            t = i * dt
            kp = build(t)
            record = {
                "timestamp": t,
                "keypoints": kp,
                "confidence": _confidence(),
            }
            f.write(json.dumps(record) + "\n")


def main() -> None:
    here = Path(__file__).resolve().parent / "fixtures"
    _write_jsonl(here / "tpose.jsonl", lambda _t: _tpose_keypoints())
    _write_jsonl(here / "arm_circle.jsonl", _arm_circle_keypoints)
    print(f"wrote fixtures to {here}")


if __name__ == "__main__":
    main()
