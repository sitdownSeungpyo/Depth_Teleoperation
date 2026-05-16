"""M4 — Filter & Limiter (spec §4.4).

OneEuro filter (Casiez et al. 2012) per joint, plus position/velocity clamping and a
NaN/inf rejection rule that holds the previous valid command.

KeypointSmoother는 retarget 전에 3D 키포인트 좌표를 미리 부드럽게 만들어 직선팔
singularity 근처에서 IK 사잇각이 폭주하는 것을 막는다.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.types import JointCommand

if TYPE_CHECKING:
    import numpy as np

    from core.types import SkeletonFrame

log = logging.getLogger(__name__)


def _unwrap(target: float, prev: float) -> float:
    """Add ±2π to ``target`` so its principal-value distance to ``prev`` is ≤ π.

    atan2-based joint angles wrap at ±π; without unwrap, a small operator motion that
    crosses the branch cut shows up as a ~2π jump and trips the velocity guard.
    """
    diff = target - prev
    if diff > math.pi:
        return target - 2.0 * math.pi
    if diff < -math.pi:
        return target + 2.0 * math.pi
    return target


@dataclass
class OneEuroParams:
    min_cutoff: float = 1.0
    beta: float = 0.05
    d_cutoff: float = 1.0


class OneEuroFilter:
    """Single-channel One Euro filter.

    Reference: Casiez, Roussel, Vogel — "1€ Filter" (CHI 2012).
    """

    def __init__(self, params: OneEuroParams) -> None:
        self._p = params
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def update(self, x: float, t: float) -> float:
        if self._t_prev is None or self._x_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x
        dt = max(t - self._t_prev, 1e-6)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self._p.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self._p.min_cutoff + self._p.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class KeypointSmoother:
    """OneEuro filter per (keypoint, coordinate) — 3D keypoint를 매끄럽게 만든다.

    confidence == 0 (zero-vector로 거부된) keypoint는 smoothing을 우회하고 그대로 전달
    (다음에 회복했을 때 큰 점프가 생기지 않도록 필터 상태도 reset).

    retarget 전에 적용하면 singularity(직선팔) 근처에서 IK 각도 노이즈가 크게 줄어듦.
    """

    def __init__(self, params: OneEuroParams) -> None:
        self._params = params
        self._filters: dict[str, list[OneEuroFilter]] = {}

    def _filters_for(self, name: str) -> list[OneEuroFilter]:
        f = self._filters.get(name)
        if f is None:
            f = [OneEuroFilter(self._params) for _ in range(3)]
            self._filters[name] = f
        return f

    def smooth(self, frame: "SkeletonFrame") -> "SkeletonFrame":
        import numpy as np

        from core.types import SkeletonFrame

        out_kp: dict[str, "np.ndarray"] = {}
        for name, pos in frame.keypoints.items():
            conf = frame.confidence.get(name, 0.0)
            if conf < 1e-6:
                # 거부된 keypoint: 그대로 전달 + 필터 reset (다음 valid frame에서 점프 방지)
                self._filters.pop(name, None)
                out_kp[name] = pos
                continue
            filters = self._filters_for(name)
            out_kp[name] = np.array(
                [filters[i].update(float(pos[i]), frame.timestamp) for i in range(3)],
                dtype=np.float64,
            )
        return SkeletonFrame(
            timestamp=frame.timestamp,
            keypoints=out_kp,
            confidence=frame.confidence,
        )

    def reset(self) -> None:
        self._filters.clear()


@dataclass
class JointLimits:
    soft_min: float
    soft_max: float
    max_velocity: float


@dataclass
class JointLimiterConfig:
    limits: dict[str, JointLimits]
    velocity_violation_factor: float = 5.0


@dataclass
class FilterAndLimiter:
    """Per-joint OneEuro + soft limits + per-step velocity clamp + NaN guard.

    Capsule self-collision (FR-4.5) is left as :meth:`check_self_collision` — a no-op
    here so the interface is stable when a real check lands.
    """

    one_euro: OneEuroParams
    limiter: JointLimiterConfig
    _filters: dict[str, OneEuroFilter] = field(default_factory=dict)
    _last: JointCommand | None = None

    def _filter_for(self, joint: str) -> OneEuroFilter:
        f = self._filters.get(joint)
        if f is None:
            f = OneEuroFilter(self.one_euro)
            self._filters[joint] = f
        return f

    def check_self_collision(self, _positions: dict[str, float]) -> bool:
        # Capsule check — to be implemented when robot geometry is wired up.
        return False

    def __call__(
        self, raw_positions: dict[str, float], timestamp: float, source_frame_ts: float
    ) -> JointCommand:
        if any(not math.isfinite(v) for v in raw_positions.values()):
            log.warning("non-finite joint value, holding last command")
            if self._last is not None:
                return self._last
            # No prior command yet — substitute zeros (clamp will pull into limits below).
            raw_positions = {j: 0.0 for j in raw_positions}

        prev_positions = self._last.positions if self._last is not None else None
        prev_ts = self._last.timestamp if self._last is not None else timestamp
        dt = max(timestamp - prev_ts, 1e-6)

        # Unwrap atan2-style discontinuities against the previous command.
        if prev_positions is not None:
            raw_positions = {
                joint: _unwrap(target, prev_positions[joint]) if joint in prev_positions else target
                for joint, target in raw_positions.items()
            }

        out_positions: dict[str, float] = {}
        for joint, target in raw_positions.items():
            limits = self.limiter.limits.get(joint)

            # 5x velocity violation → drop *this joint's* update and hold its previous
            # value; let the other joints flow. (Per-joint instead of whole-command;
            # in practice, one noisy joint shouldn't freeze all of them.)
            if prev_positions is not None and joint in prev_positions and limits is not None:
                if abs(target - prev_positions[joint]) > limits.max_velocity * dt * (
                    self.limiter.velocity_violation_factor
                ):
                    log.warning("velocity violation on %s, holding that joint", joint)
                    out_positions[joint] = prev_positions[joint]
                    continue

            smoothed = self._filter_for(joint).update(target, timestamp)

            if limits is not None:
                # Per-step velocity clamp.
                if prev_positions is not None and joint in prev_positions:
                    delta_max = limits.max_velocity * dt
                    smoothed = max(
                        prev_positions[joint] - delta_max,
                        min(smoothed, prev_positions[joint] + delta_max),
                    )
                # Position clamp.
                smoothed = max(limits.soft_min, min(smoothed, limits.soft_max))

            out_positions[joint] = smoothed

        cmd = JointCommand(
            timestamp=timestamp,
            positions=out_positions,
            source_frame_ts=source_frame_ts,
        )
        self._last = cmd
        return cmd


def default_limits_from_mechanical(
    mechanical: dict[str, tuple[float, float]],
    factor: float,
    max_velocity: float,
) -> JointLimiterConfig:
    """Build a JointLimiterConfig from per-joint mechanical (min, max) bounds."""
    limits = {
        joint: JointLimits(
            soft_min=lo * factor,
            soft_max=hi * factor,
            max_velocity=max_velocity,
        )
        for joint, (lo, hi) in mechanical.items()
    }
    return JointLimiterConfig(limits=limits)
