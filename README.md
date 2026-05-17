# imitation_upper — RealSense D435i 기반 상반신 텔레오퍼레이션 컨트롤러

Intel RealSense D435i + MediaPipe Pose / Hand 로 운영자 상반신 자세를 캡쳐해서
14 DOF 휴머노이드 로봇 (UBP) 의 관절 각도로 실시간 매핑합니다. Yi et al.
(Humanoids 2012) 의 analytical IK 를 기반으로 시작했으며, 다음 항목들을 추가로
구현했습니다.

- **Gravity-aligned aligner** — 운영자 토르소 tilt 에 무관한 torso frame.
- **Decoupled shoulder pitch ↔ elbow** — MediaPipe elbow 노이즈가 sh_pitch 로
  전파되지 않도록 law-of-cosines 기반 elbow + atan2 기반 sh_pitch.
- **Auto rest-pose calibration** — 캘리브레이션 frame 들에 retargeter 를 돌려
  per-joint circular mean 을 `rest_offsets` 로 자동 캡쳐. 운영자의 해부학적
  비대칭이 robot zero 에 자동 매핑됨.
- **MediaPipe Hand keypoints → sh_yaw / w_yaw / w_pitch** — pose-only 로는 관측
  불가능했던 4 DOF 를 hand 21-landmark 의 손가락 MCP 위치로 추정 (gravity 기준
  reference 축, ±π wrap-aware).
- **Sim + 실 로봇 publisher 4 종** — PyBullet (URDF), MuJoCo (MJCF), Dynamixel
  (UART/RS485 직결), Mock JSONL 로깅.
- **Safety layer** — dead-man hotkey, E-stop, watchdog, ramp-to-safe-pose.

플랫폼: Windows 11 + Python 3.11, MediaPipe Tasks API, OpenCV, PyRealSense2.

## 디렉토리

```
imitation_upper/
├── config/
│   ├── ubp.yaml             # UBP 14 DOF 로봇 전용 (메인 운용 config)
│   ├── default.yaml         # 일반 mock/replay 용
│   └── loose_visibility.yaml
├── models/
│   ├── ubp.urdf / .xml / .xacro     # 로봇 모델 (URDF / MJCF / xacro)
│   └── (pose|hand)_landmarker_*.task  # download script 로 가져옴 (gitignored)
├── core/
│   ├── types.py             # SkeletonFrame, JointCommand, KEYPOINT_NAMES
│   ├── aligner.py           # 토르소 basis 생성, gravity-aligned 옵션
│   ├── retarget.py          # Yi 2012 IK + hand-based sh_yaw/w_yaw/w_pitch + auto calibration
│   ├── filter.py            # One Euro + joint limit / velocity clamp
│   └── safety.py            # Deadman, E-stop, watchdog, ramp-to-safe
├── tracker/
│   ├── mock_tracker.py      # JSONL replay
│   └── realsense_tracker.py # D435i + MediaPipe Pose + Hand
├── publisher/
│   ├── mock_publisher.py    # JSONL 로깅
│   ├── udp_publisher.py     # 외부 로봇으로 UDP
│   ├── pybullet_publisher.py
│   ├── mujoco_publisher.py
│   └── dynamixel_publisher.py
├── app/
│   ├── main.py              # 메인 entrypoint
│   ├── debug_retarget.py    # filter/safety/publisher 우회, raw 각도 콘솔 출력
│   ├── tune_filter.py       # One Euro 파라미터 sweep
│   ├── record.py / replay.py
│   └── check_camera.py / viz_*.py / test_sim_*.py
├── scripts/
│   ├── setup_env.ps1
│   ├── run.ps1
│   └── download_mediapipe_model.ps1
└── tests/                   # 52 unit + e2e mock tests
```

## 셋업 (Windows / PowerShell)

```powershell
# 1. RealSense SDK 2.0 + Python 3.11 (64-bit) 설치 (사전)
.\scripts\setup_env.ps1                          # .venv 생성 + pip install -e .[dev]
.\.venv\Scripts\Activate.ps1

# 2. MediaPipe 모델 다운로드 (~14 MB pose + 7.5 MB hand)
.\scripts\download_mediapipe_model.ps1 heavy     # pose_landmarker_heavy.task
# hand 는 README 의 URL 로 별도 다운로드:
#   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
#   → models/hand_landmarker.task

# 3. 테스트
pytest -q                                        # 52 passing
```

### (Optional) HMR2 (4D-Humans) body backend — 상체 occlusion robust

기본 MediaPipe Pose 는 팔이 몸통 앞을 가리면 keypoint 가 흔들립니다. **HMR2.0**
(Berkeley CVPR 2023, ViT-Huge + SMPL prior) 로 교체 시 SMPL 의 운동학 prior 가
가려진 joint 도 자연스럽게 채워줍니다. GPU 필수 (RTX 4070 8 GB 권장,
~50 ms/frame). MediaPipe 대비 약 5x 느림 대신 occlusion 시 안정성 ↑↑.

```powershell
.\scripts\install_hmr2.ps1
# step 1: 공통 deps 확인 (install_hamer.ps1 와 공유 — torch, chumpy, smplx ...)
# step 2: 4D-Humans clone + 패치 (renderer optional) + pip install --no-deps -e .
#         (detectron2 skip — MediaPipe Pose bbox 로 ViTDet 우회)
# step 3: HMR2 checkpoint ~670 MB 자동 다운로드 (~/.cache/4DHumans)
# step 4: import sanity check
# step 5: SMPL_NEUTRAL.pkl 안내 (MANUAL — 라이센스)
```

SMPL 다운로드 (수동, 라이센스):
1. https://smpl.is.tue.mpg.de/ register (free, non-commercial only)
2. "Download SMPL for Python users" 또는 SMPL v1.0/v1.1 다운로드
3. zip 안의 `basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl` 을
   `~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl` 로 복사 (rename 필수)

활성화:
```yaml
# config/ubp.yaml
tracker:
  realsense:
    body_backend: hmr2        # mediapipe → hmr2
    hmr2_device: cuda
```

> install_hamer.ps1 와 install_hmr2.ps1 은 동일한 PyTorch/chumpy/smplx 설치를
> 공유합니다. 한쪽만 쓰는 경우에도 둘 다 의존성 셋업은 동일하게 통과.

### (Optional) HaMeR backend 셋업 — 손 정확도 ↑↑

`config/ubp.yaml` 의 `tracker.realsense.hand_backend` 를 `hamer` 로 바꾸면
MediaPipe Hand 대신 HaMeR (Berkeley CVPR 2024, ViT-Huge + MANO) 사용.
검출률 ~100 %, w_yaw/w_pitch jitter 큰 폭 감소. GPU 필수 (RTX 4070 8 GB
권장, ~30 ms/hand). MediaPipe 대비 latency 는 ~2x.

총 디스크: ~13 GB (weights 5.7 GB tarball + 압축 해제 + torch). 시간: ~15분.

```powershell
.\scripts\install_hamer.ps1
# step 1: PyTorch + CUDA 12.1 (이미 있으면 skip)
# step 2: deps (gdown, pyrender, pytorch-lightning, scikit-image, yacs, timm, einops,
#               chumpy --no-build-isolation, smplx==0.1.28)
# step 3: HaMeR clone + 패치 (renderer optional) + pip install --no-deps -e .
#         (detectron2 skip — MediaPipe Pose 의 wrist 좌표로 bbox 만드므로 불필요)
# step 4: HaMeR pretrained weights (~5.7 GB via UT Austin 직링크)
# step 5: import sanity check
# step 6: MANO 다운로드 안내 (MANUAL)
```

스크립트는 idempotent — 이미 설치된 단계는 자동 skip.

MANO 다운로드 (수동, 라이센스):
1. https://mano.is.tue.mpg.de/ register (free, non-commercial only)
2. "Models & Code" → `MANO_v1_2.zip` 다운로드
3. `MANO_RIGHT.pkl` 만 `third_party/hamer/_DATA/data/mano/` 에 배치
   (HaMeR 는 left hand 를 right mirror 로 처리하므로 RIGHT 만 필요)

활성화:
```yaml
# config/ubp.yaml
tracker:
  realsense:
    hand_backend: hamer       # mediapipe → hamer
    hamer_device: cuda
```

> **Note**: Windows 에서 HaMeR 의 일부 import 가 깨집니다 — pyrender 의 OpenGL.EGL
> (Linux 전용) + detectron2 C++ build. install_hamer.ps1 이 자동 우회:
> (a) `hamer/utils/__init__.py` 의 renderer imports 를 try/except wrap (in-place patch),
> (b) HaMeR 를 `--no-deps` 로 설치 + 필요한 deps 만 별도. 패치는 `third_party/hamer/`
> (gitignored) 안에서만 적용됨.

## 실행

### 시뮬레이션 (MuJoCo, 권장)

```powershell
.\scripts\run.ps1 --config .\config\ubp.yaml --tracker realsense --publisher mujoco
```

캘리브레이션 안내가 콘솔에 출력됩니다 (양팔 자연스럽게 내리고 어깨 편안히, ~0.5s hold).
캘리브레이션 완료되면 운영자 동작이 시뮬 robot 으로 매핑됨.

### Raw 각도 디버그 (filter / publisher 우회)

```powershell
python -m app.debug_retarget --config .\config\ubp.yaml --duration 30
```

콘솔 컬럼: `r_elb` `l_elb` | `r_sp` `r_sr` | `r_shy` `r_wy` `r_wp` | confidence + 손 검출 여부.

### 실 로봇 (Dynamixel)

```powershell
.\scripts\run.ps1 --config .\config\ubp.yaml --tracker realsense --publisher dynamixel --deadman
```

`--deadman` 옵션: `space` 를 누르고 있을 때만 명령 흐름, `esc` 로 E-stop.

### Mock JSONL replay (카메라 없이)

```powershell
python -m tests.generate_fixtures                # 한 번만
python -m app.main --config .\config\default.yaml --tracker mock --publisher mock `
                  --replay .\tests\fixtures\arm_circle.jsonl --duration 5
```

## 주요 파이프라인

1. **Tracker** — RealSense color+depth → MediaPipe Pose (33 landmarks) + Hand
   (2 × 21 landmarks). pose_world_landmarks (hip-centered, 미터 단위) +
   hand_world_landmarks (wrist-relative) → `SkeletonFrame` 의 keypoints dict.

2. **Aligner** — shoulder/head/hip 으로 torso basis 생성. `gravity_up`
   설정 있으면 운영자 tilt 무시 (gravity-aligned), 없으면 body-relative.
   결과: 모든 keypoint 를 torso frame 으로 회전 + 토르소 RPY 산출.

3. **Retargeter** —
   - sh_pitch, sh_roll, elbow: Yi 2012 Eq. 1–3 (law of cosines 기반 elbow,
     decoupled atan2 기반 sh_pitch).
   - sh_yaw: `cross(upper, world_down)` 을 reference 로 elbow flex 평면의 회전
     각도. 팔이 gravity 와 평행하면 underdetermined → 0.
   - w_yaw, w_pitch: hand_middle_mcp / index_mcp / pinky_mcp 로 palm 방향
     계산, gravity 기준 reference. hand 미검출 시 0.
   - 모든 출력은 `Calibration.rest_offsets` 차감 후 (-π, π] 로 wrap.

4. **Filter / Limiter** — One Euro smoother + 기계적 joint limits +
   velocity clamp.

5. **Safety** — deadman hotkey 누름 여부, confidence threshold, frame loss
   grace period, watchdog, E-stop 시 safe_pose 로 ramp.

6. **Publisher** — mock / udp / pybullet / mujoco / dynamixel. 100 Hz
   interpolating loop (운영자 30 Hz < 로봇 명령 100 Hz).

## Config 핵심 키 (`config/ubp.yaml`)

| key | 의미 |
|-----|------|
| `tracker.realsense.use_world_landmarks` | true 면 MediaPipe pose_world_landmarks, false 면 RealSense depth deprojection |
| `tracker.realsense.gravity_up` | gravity-aligned aligner 의 up 방향. MediaPipe pose_world 는 `[0,-1,0]` |
| `tracker.realsense.hand_model_asset_path` | hand_landmarker.task 경로. null 이면 hand 비활성 |
| `retarget.decouple_pitch_elbow` | true 면 atan2(z,x) 만, MediaPipe elbow 노이즈 격리 |
| `retarget.rest_offsets` | null 이면 auto calibration. dict 지정 시 manual override |
| `retarget.output_gain` | per-joint 1.0 (그대로) / 1.2 (20% amplify) / 0.5 (절반) 등 |
| `filter.one_euro.min_cutoff, beta` | 작을수록 smoothing 강함 (응답성 ↓), 클수록 빠른 motion 반응 |
| `safety.deadman.key` | 기본 `space` |

## 의존성

- pyrealsense2, mediapipe, opencv-python, numpy, scipy, pyyaml
- pynput (deadman hotkey)
- pybullet, mujoco (시뮬레이션), dynamixel-sdk (실 로봇)
- (dev) pytest, ruff, mypy

## 테스트 / CI

`pytest -q` — 52 unit/e2e tests, 약 10 초. `ruff check .` + `mypy --strict core` 권장.

## 알려진 제약

- 카메라 단일, 운영자 정면 응시 가정 (옆에서 보면 self-occlusion 으로 정확도 ↓).
- MediaPipe Hand 는 약 70 % frame 에서만 검출 (occlusion / 빠른 motion 시 dropout).
- sh_yaw 는 elbow 가 거의 stretched 거나 upper arm 이 gravity 와 평행할 때
  underdetermined → 0 으로 반환.
- 운영자 캘리브레이션 자세 (양팔 자연 hang) 가 robot zero 와 큰 차이 있으면
  rest_offset 이 그것을 흡수하지만, 캘리브레이션 중 움직이면 baseline 불안정.

## 향후 작업

- Numerical IK (mink / pink) 도입 — 14 DOF 동시 최적화 + joint limit / smoothness 통합.
- Hand 정확도 부족 시 HaMeR / WiLoR 등 GPU 기반 hand mesh reconstruction 으로 swap.
- 카메라 다중화 (front + side) 로 depth 모호성 해소.
- Lower-body controller (외부 balance) — 현재 out of scope.
