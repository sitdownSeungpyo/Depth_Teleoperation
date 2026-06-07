"""IMU gravity estimation — turn D435i accelerometer samples into a torso-up
vector for the aligner (replaces the hard-coded ``gravity_up``).

A *stationary* accelerometer measures specific force = −g (it reads the reaction
that holds it up), so the measured vector points UP with magnitude ≈ 9.8 m/s².
Expressed in the color optical frame (+x right, +y down, +z forward), a level
camera therefore reads ≈ (0, −9.8, 0) → up = (0, −1, 0), which is exactly the
value the config used to hard-code. With the camera mounted statically the IMU
sees pure gravity (plus vibration), so a light low-pass gives a clean, true-up
direction even as the operator moves.

This module is pure/no-hardware so it is unit-testable. The RealSense-specific
parts (enabling the accel stream, the accel→color extrinsics rotation) live in
``tracker.realsense_tracker``; it pre-rotates each accel sample into the optical
frame and feeds it to :class:`GravityEstimator` here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_EPS = 1e-9
_G = 9.80665  # standard gravity (m/s²)


def _normalize(v: NDArray[np.float64]) -> NDArray[np.float64] | None:
    n = float(np.linalg.norm(v))
    return None if n < _EPS else v / n


class GravityEstimator:
    """Low-pass + outlier-gated gravity-up estimate from accelerometer samples.

    Feed accel samples already expressed in the optical frame (the tracker applies
    the accel→color extrinsics first). Each :meth:`update` returns the current
    smoothed unit up-vector, or ``None`` until enough good samples have arrived
    (caller should fall back to the fixed config vector during warm-up / when the
    camera is being moved).

    Parameters
    ----------
    lpf_alpha:
        EMA weight for a new sample (0<α≤1). Small = heavy smoothing / slow to
        follow a real re-mount; ~0.02 is a good static-camera default.
    warmup_frames:
        Number of accepted samples before :meth:`update` returns a value.
    norm_tol:
        Accept a sample as gravity only when ``|‖a‖−g|/g ≤ norm_tol``. A moving or
        shaken camera has |a| far from g, so those samples are rejected (the last
        good up is held) instead of corrupting the estimate.
    axis_sign:
        Scalar ±1 applied to the up vector — escape hatch if the reported accel
        convention points the opposite way than expected (verify on hardware).
    """

    def __init__(
        self,
        lpf_alpha: float = 0.02,
        warmup_frames: int = 10,
        norm_tol: float = 0.30,
        axis_sign: float = 1.0,
    ) -> None:
        self.lpf_alpha = float(lpf_alpha)
        self.warmup_frames = int(warmup_frames)
        self.norm_tol = float(norm_tol)
        self.axis_sign = float(axis_sign)
        self._up: NDArray[np.float64] | None = None  # smoothed unit up (optical frame)
        self._accepted = 0
        self._rejected = 0

    @property
    def ready(self) -> bool:
        return self._up is not None and self._accepted >= self.warmup_frames

    @property
    def up(self) -> NDArray[np.float64] | None:
        """Current smoothed up vector regardless of warm-up (None before 1st sample)."""
        return None if self._up is None else self._up.copy()

    @property
    def stats(self) -> tuple[int, int]:
        """(accepted, rejected) sample counts — for diagnostics."""
        return (self._accepted, self._rejected)

    def update(self, accel_optical: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """Ingest one accel sample (optical frame); return up if ready else None."""
        a = np.asarray(accel_optical, dtype=np.float64)
        n = float(np.linalg.norm(a))
        if n < _EPS:
            self._rejected += 1
            return self.up if self.ready else None
        # Outlier gate: only a roughly-1g magnitude is trustworthy as gravity.
        if abs(n - _G) / _G > self.norm_tol:
            self._rejected += 1
            return self.up if self.ready else None
        u = (a / n) * self.axis_sign
        if self._up is None:
            self._up = u
        else:
            blended = self.lpf_alpha * u + (1.0 - self.lpf_alpha) * self._up
            self._up = _normalize(blended) if _normalize(blended) is not None else self._up
        self._accepted += 1
        return self.up if self.ready else None


def tilt_degrees(up: NDArray[np.float64], level_up: NDArray[np.float64]) -> float:
    """Angle (deg) between a measured up vector and the assumed level-up — a
    convenient readout for the verification tool (0° = camera perfectly level)."""
    a, b = _normalize(np.asarray(up, float)), _normalize(np.asarray(level_up, float))
    if a is None or b is None:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))
