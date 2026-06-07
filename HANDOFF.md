# HANDOFF — 작업 재개 가이드 (2026-06-08)

PC 재부팅 후 이 파일만 보면 **현재 상태 / 무엇이 끝났는지 / 다음에 뭘 할지**를 알 수 있습니다.

## 0) 재개 방법
- 대화 이어가기: 터미널에서 `claude --resume` → 이 프로젝트 세션 선택.
- 대화 백업 위치: `C:\Users\ssp80\.claude\projects\C--pj-depthcam-teleop\session-backups\`
  - `2026-06-08_imu-and-depth-robustness.jsonl` (이 세션 명명 백업)
  - `live_<세션id>.jsonl` (매 턴 자동 갱신, Stop hook)
- 메모리 색인: `…\memory\MEMORY.md` (세션 시작 시 자동 로드).
- 앱 실행: `python -m app.sim_teleop --config .\config\ubp.yaml`

## 1) 지금 동작하는 것 (현재 상태)
카메라(D435i, RTMPose 2D + RealSense depth) → 상체 3D 추정 → 수치 IK → MuJoCo 로봇.
실행하면 **로봇 창 + 카메라 스켈레톤 창**이 함께 뜨고, 굽힘 팔까지 추종, 정면 팔 뻗기 추종됨.

추정 z 파이프라인 (정면 튐 방지가 여기 쌓여 있음):
```
RTMPose 2D → foreground depth + 스칼라 spike거부 → deproject(3D)
  → temporal median (1프레임 outlier 제거)
  → segment gate (길이폭주=뒤로튐 거부, 방향 유지)
  → bone-length 고정 → aligner(+IMU 중력) → OneEuro
  → 수치 IK(elbow+wrist, warm-start) → target 점프 clamp → joint OneEuro → MuJoCo
```

## 2) 이번 세션에 끝낸 것 (전부 커밋됨, 테스트 100개 통과)
| 커밋 | 내용 |
|---|---|
| `fc7fa35` | 수치 task-space IK(MuJoCo DLS) + URDF 드롭인 로봇 모델 |
| `6607653` | sim_teleop 카메라 스켈레톤 창 |
| `3b11544` | 수치 IK 출력에 joint OneEuro (진동↓) |
| `70e0ffa` | IK 전용 joint_limits (모델 한계 override) |
| `2a843c6` | config 용도별 분리 + include/merge 로더 |
| `228c649`+`296583f` | **IMU 중력 정렬** — 하드웨어 검증·활성화(`enable_imu: true`) |
| `f1bc656` | **정면 z 튐 수정** — segment-length jump rejection |
| `1063c82` | **순간 튐 수정** — temporal median |
| `f1ff4d1` | **endpoint 점프 clamp** — task-space target 이동량 제한 |

## 3) 다음에 할 일 (우선순위)
1. **[검증]** sim_teleop 실행해 정면 팔 뻗기 + 순간 튐이 충분한지 체감. 부족하면 아래 노브 튜닝.
2. **[다음 기능] confidence-aware hold** — RTMPose per-keypoint score가 떨어진 동안 target을 hold.
   현재 위치-clamp는 *지속적* 오인식(좌우 뒤바뀜 등)은 못 막음; confidence가 그걸 잡는 정공법.
   (구현 메모: `AlignedFrame`에 confidence가 안 실려 있어 `core/aligner.py`에서 한 줄 plumbing 필요.)
3. **[선택] 정면 z 검증 HUD** — viz_camera에 손목 z(m) 실시간 그래프(어느 관절이 언제 튀는지 정량화).
4. **[대기] 실제 로봇 URDF** 들어오면 `config/joint_limit.yaml` + `config/robot.yaml`만 수정하면 적용.

## 4) 튜닝 노브 (튐이 남았을 때)
- `config/tracker.yaml`
  - `depth_lift.temporal_median.window` (기본 3) → **5**: 2프레임 spike까지 제거(지연↑)
  - `depth_lift.segment_gate.ratio_tol` (기본 0.35) → **0.25**: 더 엄격(뒤로 튐)
  - `depth_lift.max_jump_m` (기본 0.25) → **0.4**: 정면 뻗기가 굼뜰 때 빠른 z 허용
  - `imu.axis_sign`/`lpf_alpha`: 중력 부호/평활
- `config/robot.yaml`
  - `ik.max_target_step_m` (기본 0.08) → **0.06**: 확 튐 더 억제 / **0.12**: 고무줄 느낌이면 완화

## 5) 검증 명령
- 전체 테스트: `python -m pytest -q`
- IMU 중력 확인: `python -m app.debug_imu --config .\config\ubp.yaml --duration 30`
- 카메라 추정만 보기: `python -m app.viz_camera --config .\config\ubp.yaml --backend rtmpose`
- 텔레옵(로봇+카메라): `python -m app.sim_teleop --config .\config\ubp.yaml`

## 6) 알아둘 것
- 커밋 메시지에 co-author(Claude) 안 넣음 (사용자 요청).
- 카메라/GPU는 사용자 PC에만 있음 → 라이브는 사용자가 실행, 코드 쪽은 단위테스트로 검증.
- 상세 배경은 메모리 참조: `imu_gravity.md`, `estimation_first_rtmpose.md`, `spec_and_kinematics.md`.
