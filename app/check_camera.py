"""Standalone D435i smoke test (spec §7.4).

Default mode:
    python -m app.check_camera                    # 3-second pipeline + depth stats

With pose-detection sanity check:
    python -m app.check_camera --with-pose --model .\\models\\pose_landmarker_lite.task

Prints depth statistics (min / median / max), color+depth resolutions, achieved FPS,
and (if --with-pose) the count of detected pose landmarks per frame.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=3.0, help="seconds to capture")
    parser.add_argument(
        "--with-pose",
        action="store_true",
        help="run MediaPipe Pose on each color frame and report detection rate",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("./models/pose_landmarker_lite.task"),
        help="MediaPipe pose_landmarker_lite.task path (only used with --with-pose)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    try:
        import pyrealsense2 as rs  # type: ignore
    except ImportError:
        print("pyrealsense2 not installed", file=sys.stderr)
        return 2

    landmarker = None
    mp = None
    if args.with_pose:
        if not args.model.exists():
            print(
                f"--with-pose requested but model not found at {args.model}; "
                "run scripts/download_mediapipe_model.ps1 first",
                file=sys.stderr,
            )
            return 2
        try:
            import mediapipe as mp  # type: ignore
        except ImportError:
            print("mediapipe not installed", file=sys.stderr)
            return 2
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipeline = rs.pipeline()
    try:
        profile = pipeline.start(cfg)
    except RuntimeError as exc:
        print(f"failed to start pipeline: {exc}", file=sys.stderr)
        if landmarker is not None:
            landmarker.close()
        return 1

    try:
        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        depth_samples_m: list[float] = []
        landmark_counts: list[int] = []
        deadline = time.perf_counter() + args.duration
        start = time.perf_counter()
        frame_count = 0

        while time.perf_counter() < deadline:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned = align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue
            depth_image = np.asanyarray(depth.get_data())
            valid = depth_image[depth_image > 0]
            if valid.size > 0:
                depth_samples_m.append(float(np.median(valid)) * depth_scale)
            frame_count += 1

            if landmarker is not None and mp is not None:
                color_image = np.asanyarray(color.get_data())
                rgb = color_image[..., ::-1].copy()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((time.perf_counter() - start) * 1000)
                result = landmarker.detect_for_video(mp_image, ts_ms)
                landmark_counts.append(len(result.pose_landmarks[0]) if result.pose_landmarks else 0)

        elapsed = time.perf_counter() - start
        fps = frame_count / max(elapsed, 1e-6)

        print(f"color: {args.width}x{args.height} @ {args.fps} fps requested")
        print(f"depth: {args.width}x{args.height} @ {args.fps} fps requested (scale={depth_scale:.5f})")
        print(f"frames captured: {frame_count} in {elapsed:.2f}s -> {fps:.1f} fps achieved")
        if depth_samples_m:
            arr = np.asarray(depth_samples_m)
            print(
                f"per-frame median depth (m): "
                f"min={arr.min():.3f} median={np.median(arr):.3f} max={arr.max():.3f}"
            )
        if landmark_counts:
            arr2 = np.asarray(landmark_counts)
            detected = (arr2 > 0).mean() * 100
            print(
                f"pose detection: rate={detected:.0f}% landmarks_avg={arr2.mean():.1f}"
            )
        return 0 if frame_count > 0 else 1
    finally:
        pipeline.stop()
        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    sys.exit(main())
