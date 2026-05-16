"""M3 — Kinematic Retargeter (spec §4.3, paper Eq. 1-3).

Analytic IK from operator wrist position (in torso frame, +y up) to three robot joint
angles per arm: shoulder pitch, shoulder roll, elbow. Operator arm length is captured
once at startup via :class:`Calibration` so the operator's reach is normalised before
scaling onto the robot's geometry.

Joint sign convention (Yi 2012 / DARwIn-OP):
- +shoulder_pitch = arm rotates forward (sagittal flexion)
- +shoulder_roll  = arm abducts outward (away from body center)
- +elbow          = forearm flexes (biceps curl)

URDF axes (models/ubp.urdf) are chosen so that positive joint values in this
convention drive the robot in the matching direction:
    r_shoulder_pitch axis="0 -1 0"   l_shoulder_pitch axis="0 -1 0"
    r_shoulder_roll  axis="-1 0 0"   l_shoulder_roll  axis="+1 0 0"
    r_elbow          axis="0 -1 0"   l_elbow          axis="0 -1 0"
If a different robot uses different axis conventions, adapt either the URDF
axes or wrap the retargeter output with sign flips per-joint.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray

from core.aligner import AlignedFrame

log = logging.getLogger(__name__)

Side = Literal["left", "right"]


class SingularConfigurationError(RuntimeError):
    """Raised when a joint angle cannot be computed (zero-length vector or near-zero norm)."""


@dataclass
class RobotGeometry:
    upper_arm_length: float
    lower_arm_length: float
    shoulder_offset: tuple[float, float, float]


@dataclass
class Calibration:
    """Cached operator arm length + per-joint rest-pose offsets (FR-3.5).

    ``rest_offsets`` map joint name → angle (rad). Subtracted from retargeter output
    so the operator's natural rest pose maps to the robot's zero pose. Compensates
    for individual anatomical asymmetry (e.g., right-handed operator whose right
    shoulder sits slightly forward by default).
    """

    operator_arm_length: float
    rest_offsets: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_tpose_frames(cls, frames: Iterable[AlignedFrame]) -> "Calibration":
        """Average shoulder→wrist distance across both arms over a held T-pose."""
        lengths: list[float] = []
        for frame in frames:
            for side in ("left", "right"):
                shoulder = frame.keypoints[f"{side}_shoulder"]
                wrist = frame.keypoints[f"{side}_wrist"]
                lengths.append(float(np.linalg.norm(wrist - shoulder)))
        if not lengths:
            raise SingularConfigurationError("calibration received no frames")
        return cls(operator_arm_length=float(np.mean(lengths)))


def _safe_arccos(x: float) -> float:
    # Clip first: floating-point drift just past ±1 makes arccos return NaN.
    return float(np.arccos(np.clip(x, -1.0, 1.0)))


def _wrap_pi(angle: float) -> float:
    """Wrap a scalar angle into (-π, π]. Needed after offset subtraction so that
    joints whose raw value lives near ±π (sh_yaw, w_yaw) don't blow up across
    the discontinuity."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _circular_mean(values: list[float]) -> float:
    """Mean of angles, handling the ±π wrap. Arithmetic mean of e.g. [+3.0, -3.0]
    is 0 (wrong); circular mean is π (correct)."""
    if not values:
        return 0.0
    s = float(np.mean(np.sin(values)))
    c = float(np.mean(np.cos(values)))
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return 0.0
    return float(np.arctan2(s, c))


def _project_perpendicular(
    v: NDArray[np.float64], axis: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Remove the axis-parallel component. ``axis`` assumed unit length."""
    return v - float(np.dot(v, axis)) * axis


def _safe_normalize(
    v: NDArray[np.float64], min_norm: float = 1e-6
) -> NDArray[np.float64] | None:
    n = float(np.linalg.norm(v))
    if n < min_norm:
        return None
    return v / n


def _estimate_shoulder_yaw(
    shoulder: NDArray[np.float64],
    elbow: NDArray[np.float64],
    wrist: NDArray[np.float64],
    side: Side,
) -> float:
    """Upper-arm twist: rotation of the elbow bend-axis around the upper arm.

    Uses ``cross(upper, world_down)`` as the sh_yaw=0 reference so the formula
    stays well-conditioned for any upper-arm orientation EXCEPT the one parallel
    to gravity (arm hanging straight or fully overhead) — that pose is genuinely
    underdetermined and we return 0.

    Also underdetermined when the elbow is straight (forearm ‖ upper); ``cross``
    magnitude is proportional to sin(elbow_flex), so a 0.1 threshold ≈ 6° flex.
    """
    upper = _safe_normalize(elbow - shoulder)
    if upper is None:
        return 0.0
    forearm = _safe_normalize(wrist - elbow)
    if forearm is None:
        return 0.0

    elbow_axis = np.cross(upper, forearm)
    elbow_axis = _safe_normalize(elbow_axis, min_norm=0.1)
    if elbow_axis is None:
        return 0.0  # nearly straight arm — yaw not observable

    world_down = np.array([0.0, -1.0, 0.0])
    ref_axis = _safe_normalize(np.cross(upper, world_down), min_norm=0.1)
    if ref_axis is None:
        return 0.0  # upper arm parallel to gravity (hanging or fully overhead)

    sin_yaw = float(np.dot(np.cross(ref_axis, elbow_axis), upper))
    cos_yaw = float(np.dot(ref_axis, elbow_axis))
    yaw = float(np.arctan2(sin_yaw, cos_yaw))
    return yaw if side == "right" else -yaw


def _estimate_wrist_yaw_pitch(
    elbow: NDArray[np.float64],
    wrist: NDArray[np.float64],
    hand_middle: NDArray[np.float64],
    hand_index: NDArray[np.float64],
    hand_pinky: NDArray[np.float64],
    side: Side,
) -> tuple[float, float]:
    """Forearm twist (yaw) and hand flexion (pitch) from hand keypoints.

    Absolute reference choices are arbitrary — :class:`Calibration` rest_offsets
    auto-captured during calibration absorb the per-operator baseline. We only
    need the formula to be **consistent** (same input pose → same angle) and
    **monotonic** through the joint's range of motion.

    Returns (wrist_yaw, wrist_pitch) in radians, both 0.0 when hand keypoints
    are missing or degenerate.
    """
    forearm = _safe_normalize(wrist - elbow)
    if forearm is None:
        return 0.0, 0.0
    hand_dir = _safe_normalize(hand_middle - wrist, min_norm=1e-4)
    if hand_dir is None:
        return 0.0, 0.0
    palm_across = _safe_normalize(hand_index - hand_pinky, min_norm=1e-4)
    if palm_across is None:
        return 0.0, 0.0

    # Pitch = angle between forearm and hand pointing direction. Signed by
    # cross-product alignment with the palm-across axis (flexion vs extension).
    pitch_mag = _safe_arccos(float(np.dot(forearm, hand_dir)))
    pitch_axis = np.cross(forearm, hand_dir)
    sign = float(np.sign(np.dot(pitch_axis, palm_across)))
    if sign == 0.0:
        sign = 1.0
    wrist_pitch = (sign if side == "right" else -sign) * pitch_mag

    # Yaw = rotation of palm-across around the forearm axis. Reference baseline
    # (w_yaw=0): palm_across aligned with cross(forearm, world_down) — palm faces
    # downward at neutral, regardless of forearm orientation. Underdetermined
    # only when forearm is parallel to gravity (arm extended straight up/down).
    pap = _safe_normalize(_project_perpendicular(palm_across, forearm), min_norm=1e-4)
    if pap is None:
        return 0.0, wrist_pitch
    world_down = np.array([0.0, -1.0, 0.0])
    ref = _safe_normalize(np.cross(forearm, world_down), min_norm=0.1)
    if ref is None:
        return 0.0, wrist_pitch
    sin_yaw = float(np.dot(np.cross(ref, pap), forearm))
    cos_yaw = float(np.dot(ref, pap))
    yaw = float(np.arctan2(sin_yaw, cos_yaw))
    wrist_yaw = yaw if side == "right" else -yaw

    return wrist_yaw, wrist_pitch


def retarget_arm(
    aligned: AlignedFrame,
    side: Side,
    robot: RobotGeometry,
    calibration: Calibration,
    decouple_pitch_elbow: bool = False,
) -> dict[str, float]:
    """Map one operator arm onto three robot joint angles (radians).

    Convention: the aligner produces +y pointing up (head minus mid-hip). The paper's
    Eq. 2 expects +y down, so we invert when computing shoulder elevation. For the left
    arm we mirror x so the same formulas give symmetric joint values to the right arm.

    If ``decouple_pitch_elbow`` is True, drop the ``- theta_5`` term from Eq. 3 so
    shoulder_pitch depends only on wrist horizontal direction (atan2(z, x)), making it
    immune to MediaPipe elbow noise propagating into sh_p. Trade-off: full elbow flex
    may shift hand position slightly relative to operator. Recommended for noisy data.
    """
    shoulder = aligned.keypoints[f"{side}_shoulder"]
    elbow = aligned.keypoints[f"{side}_elbow"]
    wrist = aligned.keypoints[f"{side}_wrist"]

    # confidence=0인 keypoint는 RealSenseTracker가 [0,0,0]으로 zero-vector 처리한다.
    # 이 상태로 IK를 풀면 carbage 값이 나오므로 명시적으로 거부.
    # NOTE: aligned는 confidence를 따로 들고 있지 않으므로 zero-vector check로 대체.
    if (
        float(np.linalg.norm(elbow)) < 1e-6
        or float(np.linalg.norm(wrist)) < 1e-6
    ):
        raise SingularConfigurationError(f"{side} elbow/wrist keypoint missing (zero vector)")

    upper = elbow - shoulder
    lower = wrist - elbow
    s_to_w = wrist - shoulder

    upper_n = float(np.linalg.norm(upper))
    lower_n = float(np.linalg.norm(lower))
    c = float(np.linalg.norm(s_to_w))
    if upper_n < 1e-6 or lower_n < 1e-6 or c < 1e-6:
        raise SingularConfigurationError(f"degenerate {side} arm geometry")

    scale = (robot.upper_arm_length + robot.lower_arm_length) / max(
        calibration.operator_arm_length, 1e-6
    )
    # Scale only affects link lengths, not angles, so we don't need to apply it here —
    # but the scale factor is exposed for downstream Cartesian targets if a future
    # controller wants them. The angles below are scale-invariant by construction.
    _ = scale

    # Eq. 1 — elbow flexion.
    # Yi 2012 원본은 upper · lower로 계산하지만, MediaPipe elbow 위치가 부정확하면
    # theta_5가 과대 평가되어 theta_1과 redundant하게 보상 (sh_p 과도 진동 원인).
    # 대안: law of cosines로 c (shoulder→wrist 거리)만 사용해서 elbow 각도 계산.
    # → 안정된 wrist position만 의존, MediaPipe elbow 노이즈로부터 격리.
    a = calibration.operator_arm_length / 2  # 단순 1:1 split (upper:lower)
    b = a
    # Clamp c to feasible triangle range [|a-b|, a+b]. Beyond → invalid geometry.
    c_clamped = min(max(c, abs(a - b)), a + b)
    cos_inner = float((a * a + b * b - c_clamped * c_clamped) / (2 * a * b))
    inner_angle = _safe_arccos(cos_inner)  # angle at elbow vertex (arccos clamps too)
    theta_5 = math.pi - inner_angle  # 0 = straight, π = fully folded

    # Eq. 2 — shoulder elevation. Aligner is +y up, paper is +y down → negate.
    y_paper = -float(s_to_w[1])
    theta_3 = _safe_arccos(y_paper / c)

    # Eq. 3 — shoulder pitch. Mirror x for the left arm so left/right give matched values.
    # Add +0.0 to canonicalise -0.0 → +0.0; otherwise atan2(0.0, -0.0) returns π.
    x = (-float(s_to_w[0]) if side == "left" else float(s_to_w[0])) + 0.0
    z = float(s_to_w[2]) + 0.0
    azimuth = float(np.arctan2(z, x))
    theta_1 = azimuth if decouple_pitch_elbow else (azimuth - theta_5)

    prefix = "r" if side == "right" else "l"
    # Subtract rest-pose offsets so operator's natural rest maps to robot's zero.
    # 비대칭 operator (예: 오른어깨 약간 forward) 의 자연 bias를 제거.
    # _wrap_pi keeps the result well-defined for joints whose raw values live
    # near ±π (sh_yaw, w_yaw); benign for joints that don't wrap (small motion
    # ranges) since wrap_pi is identity on (-π, π].
    offsets = calibration.rest_offsets

    def _out(name: str, raw: float) -> float:
        return _wrap_pi(raw - offsets.get(name, 0.0))

    out = {
        f"{prefix}_shoulder_pitch": _out(f"{prefix}_shoulder_pitch", theta_1),
        f"{prefix}_shoulder_roll": _out(f"{prefix}_shoulder_roll", theta_3),
        f"{prefix}_elbow": _out(f"{prefix}_elbow", theta_5),
    }

    # Shoulder yaw — always observable (just elbow bend direction), no hand needed.
    sh_yaw = _estimate_shoulder_yaw(shoulder, elbow, wrist, side)
    out[f"{prefix}_shoulder_yaw"] = _out(f"{prefix}_shoulder_yaw", sh_yaw)

    # Wrist yaw / pitch — require hand keypoints (injected by tracker when
    # HandLandmarker is enabled). Missing → skipped, so retarget_full_upper_body's
    # setdefault keeps the joint at 0.
    hm = aligned.keypoints.get(f"{side}_hand_middle_mcp")
    hi = aligned.keypoints.get(f"{side}_hand_index_mcp")
    hp = aligned.keypoints.get(f"{side}_hand_pinky_mcp")
    if (
        hm is not None
        and hi is not None
        and hp is not None
        and float(np.linalg.norm(hm - wrist)) > 1e-4
    ):
        w_yaw, w_pitch = _estimate_wrist_yaw_pitch(elbow, wrist, hm, hi, hp, side)
        out[f"{prefix}_wrist_yaw"] = _out(f"{prefix}_wrist_yaw", w_yaw)
        out[f"{prefix}_wrist_pitch"] = _out(f"{prefix}_wrist_pitch", w_pitch)

    return out


def retarget_full_upper_body(
    aligned: AlignedFrame,
    robot: RobotGeometry,
    calibration: Calibration,
    decouple_pitch_elbow: bool = False,
) -> dict[str, float]:
    """Run :func:`retarget_arm` for both arms and append torso yaw / head pitch.

    Per-arm errors are caught and the failing arm's joints are simply omitted from
    the result, so a downstream filter can hold them at their previous values
    rather than freezing the whole frame.

    ``decouple_pitch_elbow`` (default False) makes shoulder_pitch independent of
    elbow flex — recommended when MediaPipe elbow estimates are noisy.
    """
    out: dict[str, float] = {}
    for side in ("right", "left"):
        try:
            out.update(retarget_arm(
                aligned, side, robot, calibration,
                decouple_pitch_elbow=decouple_pitch_elbow,
            ))
        except SingularConfigurationError as exc:
            log.debug("skipping %s arm: %s", side, exc)
    out["torso_yaw"] = aligned.rpy[2]
    out["head_pitch"] = aligned.rpy[1]
    # Phase 1+ passthrough = 0 for joints Yi 2012 IK doesn't compute.
    # Phase 4 may extend via MediaPipe hand/face landmarks.
    out.setdefault("r_shoulder_yaw", 0.0)  # upper arm twist
    out.setdefault("l_shoulder_yaw", 0.0)
    out.setdefault("r_wrist_yaw", 0.0)     # forearm twist
    out.setdefault("l_wrist_yaw", 0.0)
    out.setdefault("r_wrist_pitch", 0.0)
    out.setdefault("l_wrist_pitch", 0.0)
    out.setdefault("neck_yaw", 0.0)        # head turning left/right
    return out


class CalibrationCollector:
    """Helper that buffers AlignedFrames and finalises a Calibration on demand.

    When ``finalise`` is called with a ``robot``, the collector also runs the
    retargeter on every buffered frame and uses the per-joint mean as
    :attr:`Calibration.rest_offsets` — so the operator's natural calibration
    pose maps to the robot's zero pose. Without ``robot`` (legacy callers like
    debug_retarget/tune_filter), only arm length is captured.
    """

    def __init__(self, target_frames: int = 30) -> None:
        self._target = target_frames
        self._buf: deque[AlignedFrame] = deque(maxlen=target_frames)

    def push(self, frame: AlignedFrame) -> None:
        self._buf.append(frame)

    def ready(self) -> bool:
        return len(self._buf) >= self._target

    def finalise(
        self,
        robot: RobotGeometry | None = None,
        decouple_pitch_elbow: bool = False,
    ) -> Calibration:
        if not self.ready():
            raise SingularConfigurationError(
                f"need {self._target} frames, have {len(self._buf)}"
            )
        cal = Calibration.from_tpose_frames(self._buf)
        if robot is None:
            return cal

        # Auto rest-pose offsets: run retargeter on every buffered frame with a
        # temporary zero-offset Calibration, then average per joint. Operator's
        # natural asymmetry (e.g. right shoulder slightly forward) becomes the
        # robot's zero so SP/SR/ELB ≈ 0 in rest pose.
        temp = Calibration(operator_arm_length=cal.operator_arm_length)
        accum: dict[str, list[float]] = {}
        for frame in self._buf:
            try:
                angles = retarget_full_upper_body(
                    frame, robot, temp, decouple_pitch_elbow=decouple_pitch_elbow
                )
            except SingularConfigurationError:
                continue
            for joint, value in angles.items():
                accum.setdefault(joint, []).append(value)
        # Use circular mean: arithmetic mean of angles near ±π gives garbage
        # (e.g. [+3.0, -3.0] → 0 when correct answer is π). Joints whose values
        # live close to the wrap boundary (sh_yaw, w_yaw) need this; safe for
        # the others since circular mean ≈ arithmetic mean inside (-π/2, π/2).
        cal.rest_offsets = {
            joint: _circular_mean(values) for joint, values in accum.items() if values
        }
        return cal


def estimate_arm_length(keypoints: dict[str, NDArray[np.float64]]) -> float:
    """Single-frame estimate of operator arm length (for tests / fallback paths)."""
    lengths = []
    for side in ("left", "right"):
        shoulder = keypoints[f"{side}_shoulder"]
        wrist = keypoints[f"{side}_wrist"]
        lengths.append(float(np.linalg.norm(wrist - shoulder)))
    return float(np.mean(lengths))
