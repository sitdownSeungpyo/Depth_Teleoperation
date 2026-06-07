"""실시간 텔레옵 모니터 — 카메라(추적) + 로봇(추종) 한 창에 나란히.

왼쪽 패널 : RealSense 컬러 영상 + 재투영된 스켈레톤 오버레이 + 관절 수락 상태/HUD.
오른쪽 패널: 동일 프레임에서 retarget된 관절각으로 구동되는 MuJoCo 로봇(오프스크린 렌더).

main.py 와 같은 파이프라인(keypoint smoother → aligner → calibration → retarget
→ filter)을 그대로 재사용하므로, 화면의 로봇 자세 = 실제 publisher 로 나가는 명령과
동일하다. 앞뒤(z) 추종이 맞는지 확인하는 데에는 **오른쪽 로봇 패널이 ground truth** 다
(카메라 정면 재투영으로는 깊이가 스케일로만 보이기 때문).

오버레이 정확도 주의:
- body_backend = mediapipe (depth-deproject) → 스켈레톤이 영상에 정확히 정렬.
- body_backend = hmr2 → HMR2 가 자체 가상 카메라(focal≈5000) 기준 3D 를 내므로,
  RealSense intrinsics 재투영은 위치/스케일이 다소 어긋날 수 있다(근사). 그래도
  bbox 와 로봇 패널은 정확하다.

사용법:
    python -m app.viz_teleop --config .\\config\\ubp.yaml
    python -m app.viz_teleop --config .\\config\\ubp.yaml --duration 60

종료: 'q' 또는 ESC.  스냅샷 저장: 's' → .\\recordings\\teleop_<ts>.png
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from core.aligner import AlignmentError, align_to_torso
from core.filter import KeypointSmoother, OneEuroParams
from core.retarget import (
    Calibration,
    CalibrationCollector,
    RobotGeometry,
    SingularConfigurationError,
    retarget_full_upper_body,
)

log = logging.getLogger(__name__)

# Canonical-name skeleton drawn on the camera panel.
SKELETON_EDGES = (
    ("neck", "head"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
)
DRAW_POINTS = (
    "head", "neck", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip",
)


def _project(kp: np.ndarray, intr: tuple[float, float, float, float]) -> tuple[int, int] | None:
    """카메라-프레임 3D 점(+x right, +y down, +z forward)을 픽셀로 재투영."""
    fx, fy, ppx, ppy = intr
    x, y, z = float(kp[0]), float(kp[1]), float(kp[2])
    if z <= 1e-6:
        return None
    return int(round(fx * x / z + ppx)), int(round(fy * y / z + ppy))


def _draw_detection(color: np.ndarray, det: Any) -> bool:
    """신뢰 가능한 image-space 검출표시: bbox + 손목 마커. det가 backend 3D 스케일과
    무관하게 픽셀좌표를 주므로 HMR2에서도 항상 정확. 검출 성공 여부 반환."""
    if det is None:
        return False
    h, w = color.shape[:2]
    bbox = getattr(det, "body_bbox_xyxy", None)
    if bbox is not None:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        cv2.rectangle(color, (x0, y0), (x1, y1), (255, 200, 0), 2)
    for side, (nx, ny) in getattr(det, "wrist_image_xy", {}).items():
        px, py = int(nx * w), int(ny * h)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(color, (px, py), 7, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(color, f"{side[0]}w", (px + 8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    return True


def _draw_camera_panel(
    color: np.ndarray,
    frame: Any,
    intr: tuple[float, float, float, float] | None,
) -> None:
    """color(BGR) 위에 (best-effort) 3D 재투영 스켈레톤을 그린다 (in-place).

    NOTE: HMR2 백엔드는 자체 가상카메라 스케일이라 이 재투영이 어긋날 수 있다.
    신뢰 가능한 검출표시는 _draw_detection(bbox/wrist)가 담당."""
    if intr is None:
        return
    h, w = color.shape[:2]
    px: dict[str, tuple[int, int]] = {}
    for name in DRAW_POINTS + ("neck", "torso"):
        kp = frame.keypoints.get(name)
        if kp is None or float(np.linalg.norm(kp)) < 1e-6:
            continue
        p = _project(kp, intr)
        if p is None or not (0 <= p[0] < w and 0 <= p[1] < h):
            continue
        px[name] = p

    for a, b in SKELETON_EDGES:
        if a in px and b in px:
            cv2.line(color, px[a], px[b], (0, 200, 0), 2, cv2.LINE_AA)
    for name in DRAW_POINTS:
        if name not in px:
            continue
        conf = float(frame.confidence.get(name, 0.0))
        col = (0, 220, 0) if conf >= 0.5 else (0, 165, 255)
        cv2.circle(color, px[name], 5, col, -1, cv2.LINE_AA)


def _hud_lines(
    color: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
) -> None:
    y = 24
    for text, col in lines:
        cv2.putText(color, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(color, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        y += 24


class RobotRenderer:
    """MJCF 모델을 오프스크린으로 렌더 — qpos 를 직접 세팅(kinematic)하여
    retarget 명령 자세를 그대로 보여준다."""

    def __init__(self, model_path: Path, width: int, height: int, azimuth: float,
                 elevation: float, distance_scale: float) -> None:
        import mujoco

        self._mj = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        # 오프스크린 버퍼 한계(기본 640x480)를 요청 크기로 확장.
        self._model.vis.global_.offwidth = max(int(self._model.vis.global_.offwidth), width)
        self._model.vis.global_.offheight = max(int(self._model.vis.global_.offheight), height)
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=height, width=width)

        self._qadr: dict[str, int] = {}
        for j in range(self._model.njnt):
            jname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jname is None:
                continue
            canonical = jname[:-6] if jname.endswith("_joint") else jname
            self._qadr[canonical] = int(self._model.jnt_qposadr[j])

        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.lookat = np.asarray(self._model.stat.center, dtype=np.float64)
        self._cam.distance = float(distance_scale * self._model.stat.extent)
        self._cam.azimuth = azimuth
        self._cam.elevation = elevation

    @property
    def joints(self) -> list[str]:
        return sorted(self._qadr)

    def render(self, positions: dict[str, float]) -> np.ndarray:
        for name, val in positions.items():
            adr = self._qadr.get(name)
            if adr is not None:
                self._data.qpos[adr] = float(val)
        self._mj.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, self._cam)
        rgb = self._renderer.render()
        return np.ascontiguousarray(rgb[..., ::-1])  # RGB -> BGR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=None, help="자동 종료(초)")
    parser.add_argument("--azimuth", type=float, default=150.0, help="로봇 카메라 방위각")
    parser.add_argument("--elevation", type=float, default=-15.0, help="로봇 카메라 고도각")
    parser.add_argument("--distance-scale", type=float, default=1.4, help="로봇 카메라 거리 = scale*extent")
    parser.add_argument("--no-robot", action="store_true", help="로봇 패널 끄기(카메라만)")
    parser.add_argument("--model", type=Path, default=Path("./models/upperbody_sim.xml"),
                        help="MuJoCo 모델 경로 (기본: IK 규약 검증된 upperbody_sim.xml)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    # --- 파이프라인 컴포넌트 (main.run 과 동일 구성) ---
    from app.main import _build_filter, _build_tracker

    tracker: Any = _build_tracker(cfg, "realsense", replay=None)
    filt = _build_filter(cfg)

    kp_cfg = cfg["filter"].get("keypoint_smoother", {})
    smoother = KeypointSmoother(OneEuroParams(
        min_cutoff=float(kp_cfg.get("min_cutoff", 0.5)),
        beta=float(kp_cfg.get("beta", 0.005)),
        d_cutoff=float(kp_cfg.get("d_cutoff", 1.0)),
    ))

    gravity_up_cfg = cfg.get("tracker", {}).get("realsense", {}).get("gravity_up")
    gravity_up = np.asarray(gravity_up_cfg, dtype=np.float64) if gravity_up_cfg else None
    decouple = bool(cfg.get("retarget", {}).get("decouple_pitch_elbow", False))
    output_gain: dict[str, float] = dict(cfg["retarget"].get("output_gain", {}))

    robot_cfg = cfg["retarget"]["robot"]
    robot = RobotGeometry(
        upper_arm_length=float(robot_cfg["upper_arm_length"]),
        lower_arm_length=float(robot_cfg["lower_arm_length"]),
        shoulder_offset=tuple(robot_cfg["shoulder_offset"]),
    )
    fallback = float(cfg["retarget"]["fixed_arm_length"])
    collector = CalibrationCollector(target_frames=int(cfg["retarget"].get("calibration_frames", 30)))
    calibration: Calibration | None = None
    rest_cfg = cfg["retarget"].get("rest_offsets") or {}

    # --- 로봇 렌더러 ---
    renderer: RobotRenderer | None = None
    if not args.no_robot:
        model_path = args.model
        try:
            renderer = RobotRenderer(
                model_path, width=480, height=480,
                azimuth=args.azimuth, elevation=args.elevation,
                distance_scale=args.distance_scale,
            )
            print(f"robot model: {model_path.name}, joints={renderer.joints}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"robot panel disabled (MuJoCo load failed: {exc})", file=sys.stderr)
            renderer = None

    backend_name = (cfg["tracker"]["realsense"].get("body_backend") or "mediapipe").lower()
    print(f"body_backend={backend_name}  loading camera + model (~10s for hmr2)...", flush=True)
    tracker.start()

    snap_dir = Path("./recordings")
    snap_dir.mkdir(parents=True, exist_ok=True)
    win = "imitation_upper teleop monitor (q/ESC quit, s snapshot)"

    last_positions: dict[str, float] = {}
    lat_window: deque[float] = deque(maxlen=120)
    fps_sm = 0.0
    last_t = time.perf_counter()
    last_ts = 0.0
    frames_total = 0  # unique detected skeleton frames since start
    start = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            if args.duration is not None and now - start > args.duration:
                break

            frame = tracker.latest()
            color = tracker.latest_color()
            if color is None:
                if (cv2.waitKey(15) & 0xFF) in (ord("q"), 27):
                    break
                continue
            color = color.copy()
            intr = tracker.intrinsics_params

            status = "WAIT"
            conf = 0.0
            if frame is not None and frame.timestamp != last_ts:
                last_ts = frame.timestamp
                dt = max(now - last_t, 1e-6)
                fps_sm = 0.9 * fps_sm + 0.1 * (1.0 / dt)
                last_t = now
                conf = frame.mean_confidence()
                frames_total += 1

                sframe = smoother.smooth(frame)
                try:
                    aligned = align_to_torso(sframe, gravity_up=gravity_up)
                    if calibration is None:
                        collector.push(aligned)
                        if collector.ready():
                            try:
                                calibration = collector.finalise(
                                    robot=robot, decouple_pitch_elbow=decouple)
                                if rest_cfg:
                                    calibration.rest_offsets = {
                                        **calibration.rest_offsets,
                                        **{k: float(v) for k, v in rest_cfg.items()},
                                    }
                            except SingularConfigurationError:
                                calibration = Calibration(operator_arm_length=fallback)
                        status = "CALIB"
                    if calibration is not None:
                        targets = retarget_full_upper_body(
                            aligned, robot, calibration, decouple_pitch_elbow=decouple)
                        if output_gain:
                            targets = {j: v * output_gain.get(j, 1.0) for j, v in targets.items()}
                        cmd = filt(targets, timestamp=now, source_frame_ts=frame.timestamp)
                        last_positions = dict(cmd.positions)
                        lat_window.append(now - frame.timestamp)
                        status = "OK"
                except AlignmentError:
                    status = "NOFRAME"
                except SingularConfigurationError as exc:
                    status = "ZEROKP" if "zero vector" in str(exc) else "SING"

            # 카메라 패널 — 신뢰 가능한 검출표시(bbox/wrist) 먼저, best-effort 스켈레톤 위에.
            det = tracker.latest_detection()
            detected_now = _draw_detection(color, det)
            if frame is not None:
                _draw_camera_panel(color, frame, intr)
            h, w = color.shape[:2]

            # 큰 검출 배너 — "인식 안 됨" 인지 오버레이만 깨진 건지 한눈에.
            if detected_now:
                banner, bcol = "BODY DETECTED", (0, 200, 0)
            else:
                banner, bcol = "NO BODY - check 1.5-2.5m / lighting / full torso in frame", (0, 0, 255)
            cv2.rectangle(color, (0, h - 30), (w, h), (0, 0, 0), -1)
            cv2.putText(color, f"{banner}  (frames={frames_total})", (8, h - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bcol, 2, cv2.LINE_AA)

            # HUD
            p50 = float(np.percentile(lat_window, 50)) * 1000 if lat_window else 0.0
            scol = (0, 220, 0) if status == "OK" else (0, 165, 255) if status in ("CALIB", "WAIT") else (0, 0, 255)
            hud = [
                (f"{backend_name}  {status}", scol),
                (f"conf {conf:.2f}  fps {fps_sm:.1f}  lat {p50:.0f}ms", (255, 255, 255)),
            ]
            if last_positions:
                hud.append((
                    f"r_sp {last_positions.get('r_shoulder_pitch', 0):+.2f}  "
                    f"r_elb {last_positions.get('r_elbow', 0):+.2f}  "
                    f"r_sr {last_positions.get('r_shoulder_roll', 0):+.2f}",
                    (200, 255, 200),
                ))
            _hud_lines(color, hud)

            # 로봇 패널
            if renderer is not None:
                try:
                    robot_img = renderer.render(last_positions)
                    if robot_img.shape[0] != h:
                        scale = h / robot_img.shape[0]
                        robot_img = cv2.resize(robot_img, (int(robot_img.shape[1] * scale), h))
                    view = np.hstack([color, robot_img])
                except Exception as exc:  # noqa: BLE001
                    log.debug("robot render failed: %s", exc)
                    view = color
            else:
                view = color

            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = snap_dir / f"teleop_{int(now)}.png"
                cv2.imwrite(str(out), view)
                print(f"saved {out}", flush=True)
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
