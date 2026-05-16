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
