"""Standalone MediaPipe Pose smoke test (no camera required).

Runs the Pose Landmarker on a synthetic image plus a downloaded sample image of a
real person, reports timings, and confirms the model loads and emits landmarks.

    python -m app.check_mediapipe
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_IMAGE_URL = (
    "https://storage.googleapis.com/mediapipe-assets/pose_landmarker.jpg"
)


def _load_sample_image() -> np.ndarray | None:
    """Try to download MediaPipe's sample pose image. Returns RGB ndarray or None."""
    try:
        with urllib.request.urlopen(SAMPLE_IMAGE_URL, timeout=15) as resp:
            data = resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not download sample image: {exc})")
        return None
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow not installed; cannot decode JPEG)")
        return None
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("./models/pose_landmarker_lite.task"),
    )
    parser.add_argument("--no-network", action="store_true", help="skip the sample-image test")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"model not found at {args.model}", file=sys.stderr)
        print(
            "  → run scripts/download_mediapipe_model.ps1 first "
            "(or follow README bring-up step 1)",
            file=sys.stderr,
        )
        return 2

    print(f"== MediaPipe Pose smoke test ==")
    print(f"  model: {args.model} ({args.model.stat().st_size / 1e6:.1f} MB)")

    t0 = time.perf_counter()
    import mediapipe as mp

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model)),
        running_mode=RunningMode.IMAGE,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    print(f"  loaded MediaPipe + landmarker in {(time.perf_counter() - t0)*1000:.0f} ms")

    try:
        # 1. Synthetic black image — should not detect anybody, just prove the API runs.
        synth = np.zeros((480, 640, 3), dtype=np.uint8)
        synth_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=synth)
        t = time.perf_counter()
        result = landmarker.detect(synth_img)
        synth_ms = (time.perf_counter() - t) * 1000
        synth_landmarks = len(result.pose_landmarks[0]) if result.pose_landmarks else 0
        print(
            f"  synthetic 640x480 black: inference={synth_ms:.1f} ms, "
            f"detected_landmarks={synth_landmarks}"
        )

        # 2. Synthetic noise — also should not detect.
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        noise_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=noise)
        t = time.perf_counter()
        result = landmarker.detect(noise_img)
        noise_ms = (time.perf_counter() - t) * 1000
        noise_landmarks = len(result.pose_landmarks[0]) if result.pose_landmarks else 0
        print(
            f"  synthetic 640x480 noise: inference={noise_ms:.1f} ms, "
            f"detected_landmarks={noise_landmarks}"
        )

        # 3. Real sample image — should detect 33 landmarks.
        if not args.no_network:
            print("  downloading MediaPipe sample image (real person)...")
            arr = _load_sample_image()
            if arr is not None:
                sample_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
                # warm-up
                landmarker.detect(sample_img)
                runs = []
                for _ in range(5):
                    t = time.perf_counter()
                    result = landmarker.detect(sample_img)
                    runs.append((time.perf_counter() - t) * 1000)
                avg_ms = sum(runs) / len(runs)
                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    nose = lm[0]
                    print(
                        f"  real sample {arr.shape[1]}x{arr.shape[0]}: "
                        f"avg inference={avg_ms:.1f} ms over {len(runs)} runs, "
                        f"landmarks={len(lm)}, nose=(x={nose.x:.3f}, y={nose.y:.3f}, z={nose.z:.3f}, vis={nose.visibility:.2f})"
                    )
                    if len(lm) != 33:
                        print(f"  WARN: expected 33 landmarks, got {len(lm)}")
                else:
                    print(f"  real sample: inference={avg_ms:.1f} ms but NO landmarks detected (unexpected)")
                    return 1
    finally:
        landmarker.close()

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
