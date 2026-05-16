"""Core dataclasses and naming constants shared across modules.

Mirrors spec §3.1–§3.3 — keypoint names, paper joint IDs, frame and command structs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

# Required skeleton keypoint names (spec §3.1).
KEYPOINT_NAMES: Final[tuple[str, ...]] = (
    "head",
    "neck",
    "torso",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)

# Canonical joint names keyed by paper ID (spec §3.3).
# IDs 7-12 / 21 are UBP extensions — Yi 2012 doesn't define them.
# Note: torso_yaw (ID 19) is computed by retargeter but UBP has no such joint
# (torso fixed). Other platforms that include torso_yaw will pick it up automatically.
JOINT_NAMES_BY_PAPER_ID: Final[dict[int, str]] = {
    1: "r_shoulder_pitch",
    2: "l_shoulder_pitch",
    3: "r_shoulder_roll",
    4: "l_shoulder_roll",
    9: "r_shoulder_yaw",   # UBP: between shoulder_roll and upper_arm (forearm twist of upper arm)
    10: "l_shoulder_yaw",
    5: "r_elbow",
    6: "l_elbow",
    11: "r_wrist_yaw",     # forearm twist (between elbow and wrist_pitch)
    12: "l_wrist_yaw",
    7: "r_wrist_pitch",
    8: "l_wrist_pitch",
    19: "torso_yaw",       # not on UBP, but kept for cross-platform retarget output
    21: "neck_yaw",
    20: "head_pitch",
}

JOINT_NAMES: Final[tuple[str, ...]] = tuple(JOINT_NAMES_BY_PAPER_ID.values())


@dataclass
class SkeletonFrame:
    """One pose observation in the camera frame (spec §3.1)."""

    timestamp: float
    keypoints: dict[str, np.ndarray]
    confidence: dict[str, float]

    def mean_confidence(self) -> float:
        if not self.confidence:
            return 0.0
        return float(np.mean(list(self.confidence.values())))


@dataclass
class JointCommand:
    """One joint setpoint frame sent to the robot (spec §3.2)."""

    timestamp: float
    positions: dict[str, float]
    velocities: dict[str, float] | None = None
    source_frame_ts: float = 0.0


@dataclass
class SensorBundle:
    """Placeholder for future lower-body controller sensor inputs (spec §4.7)."""

    extra: dict[str, float] = field(default_factory=dict)
