# HANDOFF — 작업 재개 가이드 (2026-08-12)

PC 재부팅 후 이 파일만 보면 **현재 상태 / 무엇이 끝났는지 / 다음에 뭘 할지**를 알 수 있습니다.

## 0) 재개 방법
```powershell
cd C:\pj_depthcam_teleop\imitation_upper
claude --resume          # 이 프로젝트의 이전 세션 선택 → 대화 그대로 이어감
#   또는 그냥 `claude` 로 새 세션 (memory/MEMORY.md 가 자동 로드됨)
```
- **대화 백업**: `C:\Users\ssp80\.claude\projects\C--pj-depthcam-teleop\session-backups\`
  - `2026-08-12_code-review-hardening.jsonl` — ★ 이 세션(코드리뷰→전면 수정→푸시)
  - `2026-06-08_imu-and-depth-robustness.jsonl` — IMU/깊이 강건화 세션
  - `live_<세션id>.jsonl` — Stop hook(`~\.claude\backup-transcript.ps1`)이 갱신하는
    롤링 복사본. **주의: 2026-08-12 확인 시 이번 세션 것이 자동 생성되지 않았음**
    (스크립트는 수동 실행하면 정상 동작 → hook 이 안 걸리는 쪽으로 의심).
    수동 갱신:
    ```powershell
    $p="C:\Users\ssp80\.claude\projects\C--pj-depthcam-teleop"
    Copy-Item "$p\<세션id>.jsonl" "$p\session-backups\<원하는이름>.jsonl" -Force
    ```
  - 원본 트랜스크립트는 한 단계 위 `…\C--pj-depthcam-teleop\<세션id>.jsonl`
- **메모리 색인**: `…\memory\MEMORY.md` (세션 시작 시 자동 로드).
- 앱 실행: `python -m app.sim_teleop --config .\config\ubp.yaml`

## 0-1) ★ 지금 당장 필요한 git 상태 (2026-08-12)
```
origin/main                     c4af15b  [docs] translate README to English
origin/hardening/2026-08-review a10bfd3  ← 이번 작업 10 커밋, 푸시 완료
로컬 main                        faea97c  (원격보다 1 커밋 뒤 — pull 필요)
```
- 작업 브랜치는 `origin/main` 위로 rebase 되어 있어 **fast-forward 머지 가능**.
- **머지가 아직 안 됐습니다.** 이어서 할 첫 작업은 보통 이것:
  ```powershell
  git checkout main
  git pull --ff-only
  git merge --ff-only hardening/2026-08-review
  git push origin main
  ```
  또는 PR: https://github.com/sitdownSeungpyo/Depth_Teleoperation/pull/new/hardening/2026-08-review
- 주의: README 는 원격에서 **영문**입니다(c4af15b). 한국어로 되돌리지 말 것.
  HANDOFF.md(이 파일)와 config 주석은 한국어 유지.
- 원격에 `feature/c++` 브랜치가 따로 있음 — 이번 작업과 무관.

## 1) 지금 동작하는 것 (현재 상태)
카메라(D435i, RTMPose 2D + RealSense depth) → 상체 3D 추정 → 수치 IK → MuJoCo 로봇.
실행하면 **로봇 창 + 카메라 스켈레톤 창**이 함께 뜨고, 굽힘 팔까지 추종, 정면 팔 뻗기 추종됨.

추정 z 파이프라인 (정면 튐 방지가 여기 쌓여 있음):
```
RTMPose 2D → foreground depth + 스칼라 spike거부 → deproject(3D)
  → temporal median (1프레임 outlier 제거)
  → segment gate (길이폭주=뒤로튐 거부, 방향 유지)
  → bone-length 고정 → aligner(+IMU 중력) → OneEuro
  → [confidence hold] → 수치 IK(elbow+wrist, warm-start)
  → target 점프 clamp → joint OneEuro → MuJoCo
```

검증 상태: `pytest -q` **133 통과**, `ruff check .` 클린, `mypy` (strict,
core+tracker+publisher) 클린. `.github/workflows/ci.yml` 이 셋을 CI 에서 돌림.

## 2) 2026-08-09~12 세션에서 고친 것 (코드 리뷰 → 전체 반영 → 푸시)

브랜치 `hardening/2026-08-review` 의 10 커밋. 검증: 테스트 133 통과 / ruff 클린 /
mypy 클린 (rebase 후 재검증 완료).
```
a10bfd3 [docs]     README·HANDOFF 현행화 (README 는 영문 유지)
307a07f [chore]    mujoco 의존성, mypy 확대, lint 정리, CI 추가
265844b [refactor] median_depth_3x3 중복 제거 + tracker/publisher 타입 정리
87b28ec [change]   safety/health/confidence 신호를 앱에 연결
e551913 [fix]      config include 순환 검출 + default.yaml drift 제거
d7c4634 [fix]      IK 팔꿈치 시드를 직선팔 근방으로 한정
bc12052 [add]      confidence 기반 팔 hold + 실패한 팔 0 스냅 제거
170420c [fix]      FilterAndLimiter: 누락 관절 hold, 상태 일관성
3b8c41d [fix]      캡처 스레드 사망을 관측 가능하게 (TrackerHealth)
7f5fe3e [fix]      safety watchdog 을 전용 스레드로
```


**실제 결함**
| 무엇 | 왜 문제였나 |
|---|---|
| watchdog 무력화 | `main.py` 가 `update()` 직후 `watchdog_tick()` 호출 → 방금 갱신한 스탬프와 비교하니 **절대 발동 불가**. 게다가 tracker 가 멈추면 루프 자체가 안 돌아 tick 도 안 됨. → `SafetyLayer` **전용 스레드**로 이동 (`safety.watchdog_timeout_s`, 기본 0.5 s) |
| 캡쳐 스레드 사망 무감지 | `_run` 이 `RuntimeError` 만 잡아 다른 예외에 조용히 죽고, `latest()` 는 옛 프레임을 영원히 반환 → 로봇이 굳은 채 무경고. → 광범위 예외 처리 + 재시작 백오프 + `TrackerHealth`(생존/프레임 나이/치명 에러). sim_teleop 에 `CAMERA STALLED` 배너 |
| 관절 집합 변동 | 실패한 팔의 관절이 dict 에서 빠지면 다음 복구 프레임에서 `prev_positions` 에 없어 **unwrap·속도클램프가 통째로 건너뛰어짐** → 스냅. → 필터가 없는 관절을 직전 값으로 hold |
| 실패 팔이 0 으로 스냅 | `retarget_full_upper_body` 가 실패한 팔의 sh_yaw/w_yaw/w_pitch 까지 `setdefault(0.0)` → hold 가 아니라 robot zero 로 튐. → 성공한 팔에만 passthrough 적용 |
| 필터 상태 불일치 | 속도 위반 시 OneEuro update 를 건너뛰어 내부 x_prev 가 emit 안 한 값에 고정 / 비유한값일 때 **저장된 command 객체를 그대로 반환**해 timestamp 정지(publisher 보간 왜곡). → 각각 hold 값 주입 / 재스탬프 |
| 팔꿈치 warm-start 파괴 | `solve_arm` 이 매 프레임 팔꿈치 qpos 를 raw 측정각으로 덮어써, 호출자가 되쓴 **필터된 값이 무효화** → 팔꿈치만 필터 없이 동작. → 직선팔 근방에서만 시드 (`ik.elbow_seed_below_rad`) |

**기능 추가**
- **confidence plumbing + hold** — `AlignedFrame.confidence` / `arm_confidence(side)`
  (shoulder·elbow·wrist 중 **최솟값**). 해석 IK 는 `min_arm_confidence` 로 게이트,
  수치 IK 는 그 팔을 건너뛰어 hold. 지속적 오인식(좌우 뒤바뀜 등)에 대한 정공법 —
  position clamp 로는 못 막는 케이스.
- `tracker.pose.min_visibility` 0.1 → **0.3** (0.1 은 사실상 무필터였음),
  `min_arm_confidence` **0.4** 신설.

**구조/위생**
- `config/default.yaml`, `loose_visibility.yaml` 을 **include 오버레이**로 재작성 —
  ubp.yaml split 이후 두 벌이 따로 흘러가던 drift 제거.
- `core/config.py` include **순환 검출** + 누락 파일 명시 (`ConfigError`).
- `retarget.output_gain.r/l_elbow` **3.0 → 1.0** — 3.0 은 HMR2 깊이 압축 보정값인데
  기본 백엔드가 rtmpose 로 바뀐 뒤에도 남아 팔꿈치만 3배 과장하고 있었음.
- `pyproject`: **mujoco 를 의존성에 추가**(기본 IK 경로인데 빠져 있었음),
  mypy 대상을 core → core+tracker+publisher 로 확대, ruff/mypy 클린화.
- 중복/죽은 코드 제거(`median_depth_3x3` 이중 정의, 미사용 `scale`), 루트 로그 5개 삭제.
- README 를 현재 상태로 갱신(RTMPose 기본, 수치 IK, depth_lift, IMU, config 표).

## 3) 다음에 할 일 (우선순위)
0. **[머지]** `hardening/2026-08-review` → `main`. 명령은 §0-1 참고. 아직 안 됐음.
1. **[검증 — 사용자만 가능]** `python -m app.sim_teleop --config .\config\ubp.yaml`
   실행해 체감 확인. 카메라/GPU 가 없는 쪽에서는 못 하는 유일한 항목.
   특히 이번에 바뀐 두 가지: (a) 팔꿈치가 이제 필터를 타므로 **덜 떨리지만 살짝 느릴 수**
   있음 — 답답하면 `filter.one_euro.min_cutoff` ↑. (b) `min_arm_confidence: 0.4` 로
   팔이 자주 멈추면(콘솔 `held R=/L=` 카운터가 계속 증가) 0.25~0.3 으로 낮출 것.
   콘솔 1초 줄 예시:
   ```
   r_sp=+0.31 r_sr=+0.12 r_sy=-0.04 r_elb=+1.02  conf=0.87  arm_conf R=0.82 L=0.79  held R=0 L=0
   ```
2. **[다음 기능] 수치 IK 를 `app/main.py` 에 통합** — 지금 수치 IK 는 sim_teleop 에만
   있고 safety/publisher/calibration 은 main.py 에만 있음. 실기 투입 전 필수.
3. **[선택] 정면 z 검증 HUD** — viz_camera 에 손목 z(m) 실시간 그래프.
4. **[대기] 실제 로봇 URDF** 들어오면 `config/joint_limit.yaml` + `config/robot.yaml` 만 수정.

## 4) 튜닝 노브 (튐이 남았을 때)
- `config/tracker.yaml`
  - `depth_lift.temporal_median.window` (기본 3) → **5**: 2프레임 spike까지 제거(지연↑)
  - `depth_lift.segment_gate.ratio_tol` (기본 0.35) → **0.25**: 더 엄격(뒤로 튐)
  - `depth_lift.max_jump_m` (기본 0.25) → **0.4**: 정면 뻗기가 굼뜰 때 빠른 z 허용
  - `pose.min_arm_confidence` (기본 0.4) → **0.25**: 팔이 자주 hold 될 때 완화
  - `imu.axis_sign`/`lpf_alpha`: 중력 부호/평활
- `config/robot.yaml`
  - `ik.max_target_step_m` (기본 0.08) → **0.06**: 확 튐 더 억제 / **0.12**: 고무줄 느낌이면 완화
  - `ik.elbow_seed_below_rad` (기본 0.15) → **0**: 시드 완전히 끔(직선팔에서 수렴 느려질 수 있음)
- `config/safety.yaml`
  - `watchdog_timeout_s` (기본 0.5) → 오탐이면 ↑, 정지 감지를 빨리 하려면 ↓
- 진단 모드: `--config .\config\loose_visibility.yaml` (게이트 전부 해제)

## 5) 검증 명령
```powershell
cd C:\pj_depthcam_teleop\imitation_upper; .\.venv\Scripts\Activate.ps1

# 카메라 없이 전부
pytest -q; if ($?) { ruff check . }; if ($?) { mypy }   # 133 passed / clean / clean

# 이번 수정분만 (28개)
pytest -v -k "watchdog or stall_timeout or note_alive"
pytest -v -k "confidence or held or hold or seed or stale or health or circular or drift"

# 파이프라인 E2E (카메라 없이)
python -m tests.generate_fixtures     # 최초 1회
python -m app.main --config .\config\default.yaml --tracker mock --publisher mock `
                   --replay .\tests\fixtures\arm_circle.jsonl --duration 5

# 카메라 필요
python -m app.debug_imu   --config .\config\ubp.yaml --duration 30      # IMU 중력
python -m app.viz_camera  --config .\config\ubp.yaml --backend rtmpose  # 추정만
python -m app.sim_teleop  --config .\config\ubp.yaml                    # 텔레옵 전체
```

## 6) 알아둘 것 (작업 스타일 / 함정)
- **커밋 메시지에 co-author(Claude) 안 넣음** (사용자 요청). 커밋 본문은 "무엇이 왜
  깨져 있었는지"를 적는 스타일로 유지.
- **카메라/GPU는 사용자 PC에만 있음** → 라이브 실행은 사용자가, 코드 쪽은 카메라 없는
  단위테스트로 검증. "이거 돌려보고 알려주세요" 를 반복하지 말 것(사용자가 싫어함).
  가능하면 테스트를 먼저 쓰고 계측 핑퐁을 줄인다.
- **git 함정**: `origin/main` 이 로컬보다 앞서 있을 수 있음. 푸시/머지 전 반드시
  `git fetch` 후 확인. 지난번 origin 에 README 영문 번역 커밋이 먼저 들어와 있어
  rebase 로 해소했음.
- **PowerShell 함정**: `git commit -m @'...'@` here-string 이 깨진 적 있음 →
  긴 커밋 메시지는 파일에 쓰고 `git commit -F <file>` 로.
  `&&`/`||` 없음, `A; if ($?) { B }` 사용.
- 상세 배경은 메모리 참조: `imu_gravity.md`, `estimation_first_rtmpose.md`,
  `spec_and_kinematics.md`, `rtmpose_migration.md`, `project_overview.md`.
