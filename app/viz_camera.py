"""Live monitoring window: D435i color stream + body keypoints + per-joint
acceptance status. Useful for diagnosing why frames get rejected ("degenerate
arm geometry", low confidence, depth holes, etc.).

    python -m app.viz_camera --config .\\config\\ubp.yaml                  # MediaPipe (default)
    python -m app.viz_camera --config .\\config\\ubp.yaml --backend rtmpose  # RTMPose body-17 (GPU)

Press 'q' or ESC to quit. Press 's' to save a snapshot to .\\recordings\\snap_<ts>.png.

Color legend on overlay:
- GREEN  filled circle  : keypoint accepted (score/visibility >= threshold AND valid depth)
- ORANGE filled circle  : score-rejected (below min_visibility)
- RED    filled circle  : depth-rejected (depth=0 or beyond depth_max_m)
- WHITE  text           : depth in metres at that pixel

For ``--backend rtmpose`` the "confidence" shown is the RTMPose COCO keypoint
score (same min_visibility gate). The FPS panel reflects backend inference cost
— watch it stays comfortably above the camera FPS and that wrist/elbow depth
values are stable (not flickering between a metre value and "depth-bad").

Right-side panel shows per-keypoint confidence and the upper/lower arm vector norms
that drive the SingularConfigurationError when either is below 1e-6.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

KEYPOINT_DRAW_ORDER = (
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)

SKELETON_EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--backend",
        choices=("mediapipe", "rtmpose"),
        default="mediapipe",
        help="Pose backend for the overlay (default: mediapipe).",
    )
    parser.add_argument(
        "--no-mediapipe",
        action="store_true",
        help="Skip pose detection entirely (color+depth feed only).",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rs_cfg = cfg["tracker"]["realsense"]
    pose_cfg = cfg["tracker"]["pose"]
    min_visibility = float(pose_cfg["min_visibility"])
    depth_max_m = float(rs_cfg["depth_max_m"])
    model_path = Path(rs_cfg.get("model_asset_path") or "./models/pose_landmarker_lite.task")

    try:
        import pyrealsense2 as rs  # type: ignore
    except ImportError:
        print("pyrealsense2 not installed", file=sys.stderr)
        return 2

    landmarker: Any = None
    mp: Any = None
    rtm_body: Any = None
    rtm_coco_map: dict[int, str] = {}
    use_rtmpose = (not args.no_mediapipe) and args.backend == "rtmpose"
    use_mediapipe = (not args.no_mediapipe) and args.backend == "mediapipe"

    if use_rtmpose:
        from tracker.rtmpose_body_backend import (
            RTMPoseBodyBackend,
            _ensure_onnx_cuda_dll_path,
        )

        rtm_coco_map = RTMPoseBodyBackend._COCO_TO_NAME
        device = str(rs_cfg.get("rtmpose_device", "cuda"))
        if device.startswith("cuda"):
            _ensure_onnx_cuda_dll_path()
        try:
            from rtmlib import Body  # type: ignore
        except ImportError:
            print(
                "rtmlib not installed — run scripts/install_rtmpose.ps1",
                file=sys.stderr,
            )
            return 2
        print(f"loading RTMPose Body (mode={rs_cfg.get('rtmpose_mode', 'balanced')}, "
              f"device={device}) — first run downloads ONNX models ...")
        rtm_body = Body(
            mode=str(rs_cfg.get("rtmpose_mode", "balanced")),
            backend=str(rs_cfg.get("rtmpose_onnx_backend", "onnxruntime")),
            device=device,
            to_openpose=False,
        )

    if use_mediapipe:
        if not model_path.exists():
            print(f"MediaPipe model not found at {model_path}", file=sys.stderr)
            return 2
        import mediapipe as mp  # type: ignore

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    config_rs = rs.config()
    config_rs.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config_rs.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipeline = rs.pipeline()
    profile = pipeline.start(config_rs)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    from tracker.realsense_tracker import (
        MEDIAPIPE_LANDMARK_TO_NAME,
        median_depth_3x3,
    )

    last_mp_ts = 0
    fps_smoothed = 0.0
    last_t = time.perf_counter()

    print("press 'q' or ESC to quit, 's' to save snapshot")
    snap_dir = Path("./recordings")
    snap_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data())
            h, w = depth.shape
            t = time.perf_counter()
            dt = max(t - last_t, 1e-6)
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * (1.0 / dt)
            last_t = t

            keypoint_status: dict[str, tuple[tuple[int, int] | None, str, float, float]] = {}

            if rtm_body is not None:
                # rtmlib wants BGR — `color` already is (rs.format.bgr8).
                kpts_all, scores_all = rtm_body(color)
                if kpts_all is not None and len(kpts_all) > 0:
                    kpts = np.asarray(kpts_all[0], dtype=np.float64)
                    scrs = np.asarray(scores_all[0], dtype=np.float64)
                    for idx, name in rtm_coco_map.items():
                        if idx >= kpts.shape[0]:
                            continue
                        score = float(scrs[idx])
                        px = int(round(kpts[idx, 0]))
                        py = int(round(kpts[idx, 1]))
                        if not (0 <= px < w and 0 <= py < h):
                            keypoint_status[name] = (None, "off-frame", score, 0.0)
                            continue
                        if score < min_visibility:
                            keypoint_status[name] = ((px, py), "vis-low", score, 0.0)
                            continue
                        depth_m = median_depth_3x3(depth, px, py) * depth_scale
                        if depth_m <= 0.0 or depth_m > depth_max_m:
                            keypoint_status[name] = ((px, py), "depth-bad", score, depth_m)
                            continue
                        keypoint_status[name] = ((px, py), "ok", score, depth_m)

            if landmarker is not None and mp is not None:
                rgb = color[..., ::-1].copy()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = max(int(t * 1000), last_mp_ts + 1)
                last_mp_ts = ts_ms
                result = landmarker.detect_for_video(mp_image, ts_ms)
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    for idx, name in MEDIAPIPE_LANDMARK_TO_NAME.items():
                        if idx >= len(landmarks):
                            continue
                        lm = landmarks[idx]
                        visibility = float(getattr(lm, "visibility", 1.0))
                        px = int(round(lm.x * w))
                        py = int(round(lm.y * h))
                        if not (0 <= px < w and 0 <= py < h):
                            keypoint_status[name] = (None, "off-frame", visibility, 0.0)
                            continue
                        if visibility < min_visibility:
                            keypoint_status[name] = ((px, py), "vis-low", visibility, 0.0)
                            continue
                        depth_units = median_depth_3x3(depth, px, py)
                        depth_m = depth_units * depth_scale
                        if depth_m <= 0.0 or depth_m > depth_max_m:
                            keypoint_status[name] = ((px, py), "depth-bad", visibility, depth_m)
                            continue
                        keypoint_status[name] = ((px, py), "ok", visibility, depth_m)

            # Draw skeleton edges first (so circles cover endpoints).
            for a, b in SKELETON_EDGES:
                if a in keypoint_status and b in keypoint_status:
                    pa, sa, _, _ = keypoint_status[a]
                    pb, sb, _, _ = keypoint_status[b]
                    if pa is not None and pb is not None and sa == "ok" and sb == "ok":
                        cv2.line(color, pa, pb, (0, 200, 0), 2)

            for name in KEYPOINT_DRAW_ORDER:
                info = keypoint_status.get(name)
                if info is None:
                    continue
                pt, status, vis, depth_m = info
                if pt is None:
                    continue
                if status == "ok":
                    color_bgr = (0, 220, 0)
                elif status == "vis-low":
                    color_bgr = (0, 165, 255)
                else:  # depth-bad
                    color_bgr = (0, 0, 255)
                cv2.circle(color, pt, 6, color_bgr, -1)
                cv2.putText(
                    color,
                    f"{depth_m:.2f}m" if status == "ok" else status,
                    (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color_bgr,
                    1,
                    cv2.LINE_AA,
                )

            # Right-side panel: per-keypoint status + arm-vector norms.
            panel_w = 320
            panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
            cv2.putText(
                panel,
                f"FPS: {fps_smoothed:.1f}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"min_vis={min_visibility:.2f}  depth_max={depth_max_m:.1f}m",
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            y = 72
            for name in KEYPOINT_DRAW_ORDER:
                info = keypoint_status.get(name)
                if info is None:
                    continue
                _, status, vis, depth_m = info
                col = (0, 220, 0) if status == "ok" else (0, 165, 255) if status == "vis-low" else (0, 0, 255)
                cv2.putText(
                    panel,
                    f"{name:<15} v={vis:.2f} d={depth_m:.2f} {status}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    col,
                    1,
                    cv2.LINE_AA,
                )
                y += 18

            # Arm vector norms (for SingularConfigurationError diagnosis).
            def _norm(a: str, b: str) -> float | None:
                ia = keypoint_status.get(a)
                ib = keypoint_status.get(b)
                if ia is None or ib is None:
                    return None
                if ia[1] != "ok" or ib[1] != "ok":
                    return None
                # Pixel space norm is enough as a sanity proxy here.
                pa = ia[0]
                pb = ib[0]
                if pa is None or pb is None:
                    return None
                return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))

            y += 8
            cv2.putText(panel, "arm vector norms (px):", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 22
            for label, (a, b) in [
                ("R upper s->e", ("right_shoulder", "right_elbow")),
                ("R lower e->w", ("right_elbow", "right_wrist")),
                ("L upper s->e", ("left_shoulder", "left_elbow")),
                ("L lower e->w", ("left_elbow", "left_wrist")),
            ]:
                n = _norm(a, b)
                txt = f"{label}: {n:.0f}" if n is not None else f"{label}: --"
                col = (0, 220, 0) if n is not None and n > 5.0 else (0, 0, 255)
                cv2.putText(panel, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                y += 18

            view = np.hstack([color, panel])
            cv2.imshow("imitation_upper viz (press q/ESC to quit, s to snap)", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = snap_dir / f"snap_{int(time.time())}.png"
                cv2.imwrite(str(out), view)
                print(f"saved {out}")
    finally:
        pipeline.stop()
        if landmarker is not None:
            landmarker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
