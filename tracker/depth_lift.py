"""Robust depth lifting for 2D→3D keypoint estimation (RTMPose / MediaPipe-depth).

The naive path (`_median_depth_3x3` + clamp) makes thin limbs (wrist/elbow) jitter
in z: a small window centered on a wrist straddling the body silhouette mixes
near (body) and far (background) depth, so the median lands *between* them and
flickers frame-to-frame as the window composition changes. That z-noise
propagates straight into shoulder-pitch / elbow angles (the observed `r_sp`
wobble).

Two stages fix this, both camera-free testable:

- :func:`foreground_depth` — sample the *nearest cluster* in the window (the body
  surface) instead of the plain median, rejecting background bleed at silhouette
  edges.
- :class:`RobustDepthLifter` — per-keypoint temporal gating: hole-fill from the
  last valid depth, and reject single-frame spikes (with 1-frame confirmation so
  genuine fast motion still commits).
- :class:`BoneLengthStabilizer` — online running-median segment-length
  normalization (shoulder→elbow→wrist). Preserves the measured *direction* but
  pins each segment to its recent median length, killing residual z-jitter in the
  arm. Ratio-clamped so a degenerate measurement can't fling a joint off-screen.

All operate in image-aligned metric axes (+x right, +y down, +z into scene).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

_EPS = 1e-6


def foreground_depth(
    depth_image: NDArray[np.uint16],
    px: int,
    py: int,
    window: int,
    foreground_percentile: float,
    depth_scale: float,
    depth_max_m: float,
) -> float:
    """Return the foreground (body-surface) depth in metres at (px, py), or 0.0.

    Takes valid depths in a ``window``×``window`` box, keeps the nearest
    ``foreground_percentile``% (the body sits in front of the background), and
    returns their median. This rejects the far-background contamination that a
    plain median suffers at limb silhouettes.
    """
    h, w = depth_image.shape
    r = window // 2
    x0, x1 = max(0, px - r), min(w, px + r + 1)
    y0, y1 = max(0, py - r), min(h, py + r + 1)
    win = depth_image[y0:y1, x0:x1].ravel().astype(np.float64)
    valid = win[win > 0] * depth_scale
    valid = valid[(valid > 0.0) & (valid <= depth_max_m)]
    if valid.size == 0:
        return 0.0
    thresh = float(np.percentile(valid, foreground_percentile))
    near = valid[valid <= thresh]
    return float(np.median(near)) if near.size else float(np.median(valid))


class RobustDepthLifter:
    """Per-keypoint robust depth sampling with temporal hole-fill + spike rejection.

    Call :meth:`begin_frame` once per frame, then :meth:`lift` per keypoint.
    Returns a depth in metres, or ``None`` when no usable depth is available
    (caller should treat that as a rejected keypoint).
    """

    def __init__(
        self,
        depth_scale: float,
        depth_max_m: float,
        window: int = 7,
        foreground_percentile: float = 40.0,
        max_jump_m: float = 0.25,
        max_stale_frames: int = 5,
    ) -> None:
        self.depth_scale = depth_scale
        self.depth_max_m = depth_max_m
        self.window = window
        self.foreground_percentile = foreground_percentile
        self.max_jump_m = max_jump_m
        self.max_stale_frames = max_stale_frames
        self._frame = 0
        self._last: dict[str, tuple[float, int]] = {}  # name -> (depth_m, frame_idx)
        self._pending: dict[str, float] = {}  # name -> candidate awaiting confirmation

    def begin_frame(self) -> None:
        self._frame += 1

    def lift(
        self, name: str, depth_image: NDArray[np.uint16], px: int, py: int
    ) -> float | None:
        d = foreground_depth(
            depth_image, px, py, self.window, self.foreground_percentile,
            self.depth_scale, self.depth_max_m,
        )
        prev = self._last.get(name)
        # Narrowed to a non-None `prev` rather than a separate `fresh` flag, so the
        # "still fresh" fact and the value it licenses stay tied together.
        fresh_prev = (
            prev
            if prev is not None and (self._frame - prev[1]) <= self.max_stale_frames
            else None
        )

        if d <= 0.0:
            # Depth hole — reuse the last valid value if still fresh, else drop.
            return fresh_prev[0] if fresh_prev is not None else None

        if fresh_prev is not None and abs(d - fresh_prev[0]) > self.max_jump_m:
            # Implausible single-frame jump. Likely background bleed. Require the
            # next frame to confirm before committing, so genuine fast motion
            # still gets through after ~1 frame while lone spikes are dropped.
            pend = self._pending.get(name)
            if pend is not None and abs(d - pend) <= self.max_jump_m:
                self._last[name] = (d, self._frame)
                self._pending.pop(name, None)
                return d
            self._pending[name] = d
            return fresh_prev[0]

        self._last[name] = (d, self._frame)
        self._pending.pop(name, None)
        return d


class BoneLengthStabilizer:
    """Online running-median segment-length normalization for an arm chain.

    For each (parent, child) segment, keeps a rolling history of measured lengths
    and repositions the child along the *measured direction* at the running-median
    length. Segments are processed parent→child so the chain stays connected
    (shoulder→elbow then elbow→wrist uses the already-stabilized elbow).

    Pure length stabilization — directional noise is left to the downstream
    OneEuro KeypointSmoother. Ratio-clamped to ``[ratio_min, ratio_max]`` so a
    near-zero measured length can't produce a huge reposition.
    """

    def __init__(
        self,
        segments: Iterable[tuple[str, str]],
        history: int = 60,
        ratio_min: float = 0.5,
        ratio_max: float = 2.0,
    ) -> None:
        self._segments = list(segments)
        self._ratio_min = ratio_min
        self._ratio_max = ratio_max
        self._hist: dict[tuple[str, str], deque[float]] = {
            seg: deque(maxlen=history) for seg in self._segments
        }

    def __call__(
        self, keypoints: dict[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        out = dict(keypoints)
        for seg in self._segments:
            parent, child = seg
            p_in = keypoints.get(parent)
            c_in = keypoints.get(child)
            if p_in is None or c_in is None:
                continue
            if float(np.linalg.norm(p_in)) < _EPS or float(np.linalg.norm(c_in)) < _EPS:
                continue  # rejected (zero-vector) keypoint — skip
            v = c_in - p_in
            measured = float(np.linalg.norm(v))
            if measured < _EPS:
                continue
            self._hist[seg].append(measured)
            ref = float(np.median(self._hist[seg]))
            ratio = min(max(ref / measured, self._ratio_min), self._ratio_max)
            out[child] = out[parent] + v * ratio  # chain via stabilized parent
        return out


class TemporalMedianFilter:
    """Per-keypoint causal temporal median — kills lone-frame 3D spikes.

    Keeps the last ``window`` samples per keypoint and returns their per-coordinate
    median. A single outlier frame is rejected (the median picks a good neighbor),
    which is exactly what a low-pass like OneEuro CANNOT do — it smears the spike
    instead of rejecting it. Genuine motion tracks with ~``window//2`` frames of
    lag, so a small odd window (3) removes momentary jumps with minimal delay.

    Targets the residual jumps the SegmentConsistencyGate misses: a one-frame
    direction outlier (2D-keypoint jitter, or a depth error that keeps the segment
    length plausible). Rejected (zero-vector) keypoints pass through and clear
    their history so a recovered keypoint doesn't median against stale positions.
    """

    def __init__(self, window: int = 3) -> None:
        self._w = max(1, window)
        self._hist: dict[str, deque[NDArray[np.float64]]] = {}

    def __call__(
        self, keypoints: dict[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        out = dict(keypoints)
        for name, pos in keypoints.items():
            if float(np.linalg.norm(pos)) < _EPS:
                self._hist.pop(name, None)  # rejected — reset so no stale median
                continue
            h = self._hist.get(name)
            if h is None:
                h = deque(maxlen=self._w)
                self._hist[name] = h
            h.append(np.asarray(pos, dtype=np.float64))
            out[name] = np.median(np.stack(h, axis=0), axis=0)
        return out


class SegmentConsistencyGate:
    """Reject implausible per-frame 3D depth by gating arm-segment LENGTH.

    An upper-arm / forearm length is ~constant regardless of pointing direction,
    so when the operator extends the arm toward the camera the segment length
    should stay fixed while only its direction (and the wrist's z) changes. A bad
    depth read — background bleed at a thin / pointed-at-camera limb — instead
    *balloons* (or collapses) the segment length. We use that as a discriminator:

    - length within a ratio band of its running median  → trust it (accept).
    - length outside the band                           → the child's depth is
      untrusted, so HOLD the child at ``parent + last-accepted offset`` (keeping
      the last good direction *and* length) instead of letting the arm flip
      backward.

    This is the right tool for frontal reach where the absolute-depth jump gate
    (RobustDepthLifter.max_jump_m) is wrong: z legitimately changes fast there, so
    a depth-delta gate fights real motion, while segment length stays invariant.
    A short confirmation window lets a genuine scale change (operator stepping
    closer) recover instead of being held forever. Processes parent→child so the
    chain stays connected (uses the already-gated parent).
    """

    def __init__(
        self,
        segments: Iterable[tuple[str, str]],
        history: int = 60,
        ratio_tol: float = 0.35,
        confirm_frames: int = 3,
        min_history: int = 8,
    ) -> None:
        self._segments = list(segments)
        self._ratio_tol = ratio_tol
        self._confirm = max(1, confirm_frames)
        self._min_history = max(1, min_history)
        self._len_hist: dict[tuple[str, str], deque[float]] = {
            seg: deque(maxlen=history) for seg in self._segments
        }
        self._offset: dict[tuple[str, str], NDArray[np.float64]] = {}
        # candidate offset awaiting confirmation + how many consecutive frames.
        self._pending: dict[tuple[str, str], tuple[NDArray[np.float64], int]] = {}
        self._rejected = 0

    @property
    def rejected(self) -> int:
        return self._rejected

    def __call__(
        self, keypoints: dict[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        out = dict(keypoints)
        for seg in self._segments:
            parent, child = seg
            p = out.get(parent)
            c = keypoints.get(child)
            if p is None or c is None:
                continue
            if float(np.linalg.norm(p)) < _EPS or float(np.linalg.norm(c)) < _EPS:
                continue  # rejected (zero-vector) keypoint — skip
            v = c - p
            length = float(np.linalg.norm(v))
            if length < _EPS:
                continue
            hist = self._len_hist[seg]
            if len(hist) < self._min_history:  # warm-up: seed unconditionally
                hist.append(length)
                self._offset[seg] = v.copy()
                self._pending.pop(seg, None)
                out[child] = p + v
                continue
            ref = float(np.median(hist))
            if ref < _EPS or abs(length - ref) / ref <= self._ratio_tol:
                hist.append(length)            # plausible -> accept
                self._offset[seg] = v.copy()
                self._pending.pop(seg, None)
                out[child] = p + v
                continue
            # Implausible length: a depth spike (hold) unless several consecutive
            # frames agree on a new value (a genuine scale change -> recover).
            self._rejected += 1
            pend = self._pending.get(seg)
            if pend is not None and float(np.linalg.norm(v - pend[0])) <= self._ratio_tol * ref:
                count = pend[1] + 1
                if count >= self._confirm:
                    hist.append(length)
                    self._offset[seg] = v.copy()
                    self._pending.pop(seg, None)
                    out[child] = p + v       # confirmed genuine change
                    continue
                self._pending[seg] = (v.copy(), count)
            else:
                self._pending[seg] = (v.copy(), 1)
            out[child] = p + self._offset.get(seg, v)  # hold last good offset
        return out


# Default upper-body arm chain for BoneLengthStabilizer / SegmentConsistencyGate.
ARM_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)
