"""Unit tests for RTMPoseBodyBackend with a fake rtmlib model injected.

These run without rtmlib/onnxruntime installed: we bypass ``start()`` and set
``_body`` (the rtmlib callable) + ``_deproject`` directly. They lock the
COCO-17 -> canonical mapping, neck/torso synthesis, score gating, depth
deprojection, and wrist_image_xy normalization — the parts of the contract the
downstream pipeline depends on.
"""

from __future__ import annotations

import numpy as np

from tracker.rtmpose_body_backend import RTMPoseBodyBackend

# COCO-17 order: 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_sh, 6 r_sh,
# 7 l_el, 8 r_el, 9 l_wr, 10 r_wr, 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee,
# 15 l_ankle, 16 r_ankle.
W, H = 640, 480


def _full_keypoints() -> np.ndarray:
    """17 distinct in-bounds pixel coords; x = idx*30+50, y = idx*20+40."""
    return np.array([[i * 30 + 50, i * 20 + 40] for i in range(17)], dtype=np.float64)


def _fake_body(keypoints: np.ndarray, scores: np.ndarray):
    """Return an rtmlib-Body-like callable yielding (N,17,2),(N,17)."""

    def _call(_bgr_img):
        return keypoints[None, ...], scores[None, ...]

    return _call


def _identity_deprojector():
    """deproject(px, py, depth_m) -> (px, py, depth_m) so we can assert exactly."""
    return lambda px, py, depth_m: (px, py, depth_m)


def _make_backend(keypoints: np.ndarray, scores: np.ndarray, *, min_visibility=0.3):
    be = RTMPoseBodyBackend(min_visibility=min_visibility, depth_max_m=4.0)
    be._body = _fake_body(keypoints, scores)
    be.set_deprojector(_identity_deprojector(), depth_scale=0.001)
    return be


def _depth_image(value_raw: int = 1500) -> np.ndarray:
    """Uniform raw depth; *0.001 scale -> 1.5 m, inside depth_max_m."""
    return np.full((H, W), value_raw, dtype=np.uint16)


def _rgb() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def test_coco17_mapping_and_deprojection() -> None:
    kp = _full_keypoints()
    scores = np.ones(17, dtype=np.float64)
    be = _make_backend(kp, scores)

    det = be.detect(_rgb(), timestamp_ms=0, depth_image=_depth_image())
    assert det is not None

    # Each forwarded joint deprojects to (px, py, 1.5) via the identity deprojector.
    expected_idx = {
        "head": 0,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
    }
    for name, idx in expected_idx.items():
        assert name in det.keypoints, name
        px, py = float(round(kp[idx, 0])), float(round(kp[idx, 1]))
        assert np.allclose(det.keypoints[name], [px, py, 1.5]), name
        assert det.confidence[name] == 1.0

    # Unused COCO joints (eyes/ears/knees/ankles) must not leak through.
    assert "left_eye" not in det.keypoints
    assert "left_knee" not in det.keypoints


def test_neck_and_torso_synthesis() -> None:
    kp = _full_keypoints()
    be = _make_backend(kp, np.ones(17, dtype=np.float64))
    det = be.detect(_rgb(), 0, _depth_image())
    assert det is not None

    neck = 0.5 * (det.keypoints["left_shoulder"] + det.keypoints["right_shoulder"])
    assert np.allclose(det.keypoints["neck"], neck)
    mid_hip = 0.5 * (det.keypoints["left_hip"] + det.keypoints["right_hip"])
    torso = 0.5 * (det.keypoints["neck"] + mid_hip)
    assert np.allclose(det.keypoints["torso"], torso)


def test_score_gating_yields_zero_vector() -> None:
    kp = _full_keypoints()
    scores = np.ones(17, dtype=np.float64)
    scores[7] = 0.1  # left_elbow below min_visibility=0.3
    be = _make_backend(kp, scores, min_visibility=0.3)
    det = be.detect(_rgb(), 0, _depth_image())
    assert det is not None

    assert np.array_equal(det.keypoints["left_elbow"], np.zeros(3))
    assert det.confidence["left_elbow"] == 0.0
    # A neighbor above threshold is unaffected.
    assert det.confidence["right_elbow"] == 1.0


def test_invalid_depth_yields_zero_vector() -> None:
    kp = _full_keypoints()
    be = _make_backend(kp, np.ones(17, dtype=np.float64))
    det = be.detect(_rgb(), 0, _depth_image(value_raw=0))  # zero depth -> rejected
    assert det is not None
    for name in ("head", "left_wrist", "right_hip"):
        assert np.array_equal(det.keypoints[name], np.zeros(3)), name
        assert det.confidence[name] == 0.0


def test_wrist_image_xy_normalized() -> None:
    kp = _full_keypoints()
    be = _make_backend(kp, np.ones(17, dtype=np.float64))
    det = be.detect(_rgb(), 0, _depth_image())
    assert det is not None

    assert det.wrist_image_xy["left"] == (kp[9, 0] / W, kp[9, 1] / H)
    assert det.wrist_image_xy["right"] == (kp[10, 0] / W, kp[10, 1] / H)
    for axis in (*det.wrist_image_xy["left"], *det.wrist_image_xy["right"]):
        assert 0.0 <= axis <= 1.0


def test_bbox_spans_keypoints() -> None:
    kp = _full_keypoints()
    be = _make_backend(kp, np.ones(17, dtype=np.float64))
    det = be.detect(_rgb(), 0, _depth_image())
    assert det is not None and det.body_bbox_xyxy is not None

    x0, y0, x1, y1 = det.body_bbox_xyxy
    assert x0 == max(0, int(kp[:, 0].min()) - 10)
    assert y0 == max(0, int(kp[:, 1].min()) - 10)
    assert x1 == min(W, int(kp[:, 0].max()) + 10)
    assert y1 == min(H, int(kp[:, 1].max()) + 10)


def test_no_person_returns_none() -> None:
    empty = np.empty((0, 17, 2), dtype=np.float64)
    empty_scores = np.empty((0, 17), dtype=np.float64)
    be = RTMPoseBodyBackend()
    be._body = lambda _img: (empty, empty_scores)
    be.set_deprojector(_identity_deprojector(), 0.001)
    assert be.detect(_rgb(), 0, _depth_image()) is None


def test_detect_before_start_returns_none() -> None:
    be = RTMPoseBodyBackend()  # _body is None until start()
    assert be.detect(_rgb(), 0, _depth_image()) is None
