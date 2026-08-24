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

## 0-1) ★ 지금 당장 필요한 git 상태 (2026-08-25)
```
origin/main                     6b002b6  ← 머지·푸시 완료
origin/hardening/2026-08-review 6b002b6  (main 과 동일)
로컬 main                        6b002b6  (upstream 설정됨)
```
- **2026-08-25 머지 완료.** hardening/2026-08-review 가 fast-forward 로 main 에
  들어갔고 둘 다 푸시됐다. 이제 main 에서 바로 작업해도 되고, 새 작업은
  새 브랜치를 파면 된다. 지난 세션의 "머지 안 됨" 항목은 해소됐다.
- 로컬 main 은 upstream 이 없어서 `git pull` 이 조용히 아무것도 안 했었다.
  `git branch --set-upstream-to=origin/main main` 로 설정해 뒀다.
- 되돌리려면(머지를 취소하고 싶을 때): `git reset --hard c4af15b` 후 강제 푸시.
  c4af15b 가 머지 직전의 origin/main 이다.
- 주의: README 는 **영문**이다. 한국어로 되돌리지 말 것.
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

검증 상태: `pytest -q` **162 통과**, `ruff check .` 클린, `mypy` (strict,
core+tracker+publisher) 클린. `.github/workflows/ci.yml` 이 셋을 CI 에서 돌림.

★ 2026-08-24 부터 **`app/main.py` 가 수치 IK 도 돌린다**(`--ik numeric` 기본).
즉 "정확한 IK + safety + publisher" 를 한 경로에서 쓸 수 있다. `app/sim_teleop.py`
는 추정 확인용 뷰어로 남았고 safety/publisher 가 없으므로 실기 경로가 아니다.

```powershell
# 실기/전체 파이프라인 (수치 IK)
.\scripts\run.ps1 --config .\config\ubp.yaml --tracker realsense --publisher mujoco
# 해석 IK 로 비교
.\scripts\run.ps1 --config .\config\ubp.yaml --tracker realsense --publisher mujoco --ik analytic
```

## 1-1) 2026-08-24 세션에서 고친 것 (안전 의미론 + 파이프라인 통합)

린터/타입체커가 못 잡는 **동작 레벨** 결함 위주. 전부 카메라 없이 테스트로 고정했다.

| # | 무엇 | 왜 문제였나 | 어떻게 고쳤나 |
|---|---|---|---|
| 1 | **E-stop 이 정지가 아니라 zero pose 로 전속 이동** | `SafetyLayer.update` 가 `{joint: 0.0}` 을 publisher 에 넣었다. 위치제어 서보에서 0 은 *자세*라 "0으로 가라"는 명령이다. 게다가 safety 는 `FilterAndLimiter` **하류**라 속도 클램프도 안 받는다 | publisher 에 `emergency_stop()` / `release_emergency_stop()` 추가 — 마지막 emit 한 자세를 **동결**해 계속 반복하고 이후 setpoint 는 전부 버린다. Dynamixel 은 토크 유지(상체에서 토크 off 는 낙하) |
| 2 | **E-stop 이 watchdog 에서는 아무 일도 안 했다** | 동결/영출력이 `update()` 안에만 있었는데, watchdog 이 잡는 상황이 바로 "루프가 `update()` 를 못 부르는" 경우다 | 동결을 `trigger_estop()` 안으로 이동 → watchdog 스레드에서 바로 적용 |
| 3 | **E-stop 래치 해제 경로 없음** | `reset_estop()` 호출처가 코드 전체에 0개. 한 번 걸리면 프로세스 재시작뿐 | `safety.reset_key`(기본 `r`) 신설. 해제 시 filter 를 **동결 자세로 재기준**(`FilterAndLimiter.reset_to`)하고, deadman 을 뗐다 다시 눌러야 명령이 흐르게 함 |
| 4 | **`--deadman` 없으면 ESC(E-stop)도 죽음** | hotkey 객체 자체를 deadman 플래그로 만들/안만들 었다 | `require_deadman` 과 `enable_hotkeys` 를 분리. `--publisher dynamixel` 이면 deadman 강제 + pynput 없으면 `strict` 로 시작 거부 |
| 5 | **사람이 프레임을 벗어나면 watchdog 오발동 → 영구 정지** | 메인 루프가 `for frame in tracker.stream()` 자체였다. 미검출이면 루프가 안 돌고 → `safety.update`/`note_alive` 도 안 되고 → 0.5 s 뒤 watchdog E-stop(#3 때문에 해제 불가). 정작 이 케이스용으로 만든 safe-pose 램프는 `update()` 가 안 불려 실행조차 안 됐다 | `TrackerHealth` 에 `detection_age_s` 를 분리(카메라 프레임 나이 vs 포즈 나이). stream 소비를 `_FramePump` 스레드로 빼고 메인 루프는 논블로킹 폴링. **카메라 정지**(`main.camera_stale_s`) → liveness 보류해 watchdog 작동, **포즈만 없음**(`main.detection_gap_s`) → confidence 램프 |
| 6 | **`velocity_violation_factor: 50.0` — 이상치 거부가 꺼져 있었음** | 8.0 rad/s × 1/30 s × 50 = 13.2 rad/frame(>2π). 한 프레임에 2π 넘게 튀어야 발동 → 오검출이 거부 대신 속도클램프에 걸려 "느리게 틀린 곳으로" 갔다 | 5.0 (≈76°/frame). ★ 실기에서 팔이 자주 hold 되면 8~10 으로 올릴 것 |
| 7 | **수치 IK 와 safety 가 서로 다른 파일에 있었음** | 정확한 IK 는 `sim_teleop.py`(safety/publisher 없음), safety 는 `main.py`(해석 IK만) → 둘을 동시에 쓰는 경로가 없었다 | `app/main.py --ik numeric|analytic`. 필터된 각을 `qpos` 에 되써서 DLS warm-start 유지 |
| 8 | **Dynamixel: 서보별 영점/방향 없음, 모델 문자열 미사용** | `2048 = 0 rad, 방향 동일` 을 전 서보에 가정. 조립된 실기에서 맞는 경우는 거의 없다. `ServoSpec.model` 은 로그에만 쓰여 오타/미지원 모델도 통과 | `offset_unit` / `direction` 을 ServoSpec + config 에 추가. `SERVO_MODELS` 표에 없는 모델은 **예외**. 클램프 발생 시 관절별 1회 경고 |
| 9 | **Dynamixel: 프로파일 제한 없음, 토크 실패 무시, 종료 시 토크 off** | 서보가 goal 까지 최대 속도로 감. torque enable 실패가 warning 이라 죽은 관절로 계속 돎. 종료 시 토크를 끊어 팔이 떨어짐 | 프로파일 속도/가속 config 화(기본 null=미사용, 쓰면 실패 시 예외). 토크/프로파일 쓰기 전부 rc+err 검사 후 실패 시 `RuntimeError`. `torque_off_on_stop` 기본 false |
| 10 | **종료 시 watchdog 오발동** | `finally` 에서 pump join(최대 1 s) 하는 동안 명령이 안 나가 E-stop 로그가 찍혔다 | `safety.stop()` → `tracker.stop()` → `pump.stop()` 순서로 변경(주석에 이유 명시) |

**추가된 테스트 (17개, 카메라 불필요)**
- `tests/test_safety_estop.py` — 동결/래치/리셋/deadman 재무장, watchdog 동결
- `tests/test_main_loop.py` — 포즈 상실은 E-stop 아님 / 카메라 정지는 E-stop / 수치·해석 IK 통합
- `tests/test_filter.py::test_reset_to_rebaselines_the_limiter`
- `tests/test_dynamixel_publisher.py` — 미지원 모델 거부, direction/offset, 클램프 보고

**★ 실기 전 반드시 사람이 확인해야 하는 것**
- `publisher.dynamixel.servos.*.offset_unit` / `direction` 은 전부 기본값(0 / +1)이다.
  조립된 로봇에서 관절별로 **실측해서 채워야** 한다 (절차는 config/publisher.yaml 주석).
- `ADDR_PROFILE_VELOCITY`(112) / `ADDR_PROFILE_ACCELERATION`(108) 은 **데이터시트 미대조**.
  기본값 null 이라 안 쓰이고, 쓰면 시작 시 rc/err 검사로 즉시 실패한다.
- `2XL430` 의 축별 control table 이 XL430 과 같은지 **미확인**.

## 1-2) 2026-08-25 검증 세션 — 다양한 동작으로 IK 검증 → 결함 3개 발견/수정

**검증 방법**: 합성 동작(정답 방향을 알고 있음)으로 IK가 실제로 그 자세를 재현하는지
각도 오차로 측정. 워크스페이스 스윕 1296자세 + 9종 연속 동작 × 90프레임.
(RGB 영상은 못 씀 — RTMPose 는 2D 만 내고 3D 는 RealSense depth 역투영에서 나온다.)

| # | 결함 | 증상 (실측) | 수정 |
|---|---|---|---|
| 1 | **수치 IK 특이점 영구 고착** | `upperbody_sim.xml` 은 shoulder pitch/yaw 축이 겹치고 휴식자세에서 상완이 그 축과 나란함 → 자코비안 전 열이 0. "팔을 앞으로" 요청 시 어깨 관절이 **60프레임 내내 정확히 0.000**, `max_iters` 16→300 도 무효. 즉 조작자가 팔을 든 채 시작하면 그 팔은 영원히 안 움직임 | `core/numik.py` 에 결정적 특이점 탈출 — 연속 4반복 정체 + 오차가 팔길이의 10% 초과일 때만, 자코비안 열이 0인 관절만 nudge. 프레임 2에서 0.2°, 5부터 0.0° 로 수렴 |
| 2 | **수렴 실패한 자세를 그대로 명령** | DLS 는 수렴 못 해도 "어떤" 자세를 반환. 실측: 수렴 프레임 잔차 중앙 0.001/최대 0.031 m vs 실패 프레임 0.37 m 이상 — 실패 자세는 방향오차 48~127° | `ArmPositionIK.last_residual_m` 노출 + `robot.ik.max_residual_m`(0.05) 게이트. 초과하면 `solve_arm` 이 None → 그 팔 hold |
| 3 | **손실 정책이 발동 안 되는 경로 + 로그 폭주** | 카메라 정상인데 aligner 가 basis 를 못 만들면(어깨 zero-vector) `note_alive()` 만 호출 → safe pose 램프가 **영원히 발동 안 함**. 게다가 프레임마다 WARNING — 25초에 **464줄** | `app/main.py` 의 `no_usable_pose()` 로 통합(프레임 없음 / aligner 실패 / 양팔 hold / 해석IK 양팔 실패). 진입·복구 각 1회만 로그 → 464줄 → **7줄** |

**작업 중 스스로 만든 회귀 2건도 잡음** (측정으로 확인 후 수정)
- 이스케이프 1차 구현이 정상 수렴을 정지로 오판 → `shoulder_yaw` 를 매 프레임 같은
  방향으로 밀어 한계에 래칫. lateral raise 가 90프레임 중 **89개 거부**됨.
  → 오차 크기 조건 + 연속 정체 횟수 조건 추가.
- `camera_stale_s` 0.5 s 고정값이 실제 캡처 주기(실측 62 ms, 콜드 CUDA 워밍업은 초 단위)와
  무관 → 기동 중 watchdog E-stop 래치. → `max(camera_stale_s, factor × 측정 캡처주기)`.

**동작별 최종 실측** (오차 = 조작자 방향 대비 로봇 방향, 90프레임 중앙값/p95)

| 동작 | held | upper 오차 | 동작 | held | upper 오차 |
|---|---|---|---|---|---|
| lateral raise | 0 | 0.22 / 0.45° | overhead arc | 0 | 0.12 / 0.26° |
| reach fwd/back | 0 | 0.24 / 0.41° | wave (fast) | 0 | 0.14 / 1.19° |
| biceps curl | 0 | 0.07 / 1.08° | punch fwd | 0 | 0.10 / 0.25° |
| cross-body | 0 | 0.18 / 0.18° | hands to head | 0 | 0.08 / 0.09° |
| twist sweep | **37** | 0.21 / 18.69° | | | |

twist sweep 만 거부가 많은데, `shoulder_yaw ±90°` 한계 밖이라 **물리적으로 불가능**한
구간이다. 틀린 자세를 명령하는 대신 hold 하는 것이 맞는 동작.

**★ 실기 전 확인 필요 — `shoulder_yaw` 한계**: `joint_limit.yaml` 은 ±90°, 모델은 ±180°.
6종 동작 540프레임 기준 한계 2% 이내에 붙어 있는 프레임이 ±90°→363, ±135°→15, ±180°→4
(정확도는 셋 다 동일). 정확도 손해는 없지만 **여유가 0** 이다. 실제 로봇 가동범위를
확인한 뒤 값을 정할 것. (여유자유도가 없어서 — 관절 4개 vs 구속 6개 — 자세 편향
같은 널스페이스 기법으로는 못 뺀다. 실측으로 확인함.)

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
0. ~~[머지] hardening/2026-08-review → main~~ → **2026-08-25 완료** (§0-1).
1. **[검증 — 사용자만 가능]** `python -m app.sim_teleop --config .\config\ubp.yaml`
   실행해 체감 확인. 카메라/GPU 가 없는 쪽에서는 못 하는 유일한 항목.
   특히 이번에 바뀐 두 가지: (a) 팔꿈치가 이제 필터를 타므로 **덜 떨리지만 살짝 느릴 수**
   있음 — 답답하면 `filter.one_euro.min_cutoff` ↑. (b) `min_arm_confidence: 0.4` 로
   팔이 자주 멈추면(콘솔 `held R=/L=` 카운터가 계속 증가) 0.25~0.3 으로 낮출 것.
   콘솔 1초 줄 예시:
   ```
   r_sp=+0.31 r_sr=+0.12 r_sy=-0.04 r_elb=+1.02  conf=0.87  arm_conf R=0.82 L=0.79  held R=0 L=0
   ```
2. ~~수치 IK 를 `app/main.py` 에 통합~~ → **2026-08-24 완료**(§1-1 #7).
   대신 확인할 것: `python -m app.main --config .\config\ubp.yaml --tracker realsense
   --publisher mujoco` 로 sim_teleop 과 같은 움직임이 나오는지.
3. **[검증 — 사용자만 가능] E-stop / 리셋 동작** — ESC 로 멈추면 로봇이 **그 자세에서
   멈추는지**(0 자세로 확 가면 안 됨), `r` 로 해제 후 space 를 다시 눌러야 움직이는지.
   그리고 카메라 앞에서 **비켜서 보기** — 예전엔 E-stop 이 걸려 재시작해야 했는데,
   이제는 safe pose 로 램프했다가 돌아오면 복귀해야 한다.
4. **[선택] 정면 z 검증 HUD** — viz_camera 에 손목 z(m) 실시간 그래프.
5. **[대기] 실제 로봇 URDF** 들어오면 `config/joint_limit.yaml` + `config/robot.yaml` 만 수정.
6. **[실기 전 필수] Dynamixel 브링업** — §1-1 의 "실기 전 반드시 사람이 확인해야 하는 것".

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
- `config/filter.yaml`
  - `velocity_violation_factor` (기본 5.0) → **8~10**: 팔이 자주 hold 되면 완화 /
    **3**: 튐이 그대로 통과하면 강화. (예전 50.0 은 사실상 꺼짐)
- `config/safety.yaml`
  - `watchdog_timeout_s` (기본 0.5) → 오탐이면 ↑, 정지 감지를 빨리 하려면 ↓
  - `reset_key` (기본 `r`) → E-stop 래치 해제 키
- `config/runtime.yaml`
  - `camera_stale_s` (기본 0.5) → 카메라 정지 판정. 이 시간이 지나면 watchdog 이 E-stop
  - `detection_gap_s` (기본 0.3) → 포즈 상실 판정. 짧으면 잠깐만 가려도 safe pose 로 램프,
    길면 오검출 상태로 더 오래 버팀. 카메라 프레임 간격(33 ms)보다 충분히 커야 함
  - `ik` (기본 numeric) → `--ik` 로 매 실행마다 덮어쓸 수 있음
- 진단 모드: `--config .\config\loose_visibility.yaml` (게이트 전부 해제)

## 5) 검증 명령
```powershell
cd C:\pj_depthcam_teleop\imitation_upper; .\.venv\Scripts\Activate.ps1

# 카메라 없이 전부
pytest -q; if ($?) { ruff check . }; if ($?) { mypy }   # 162 passed / clean / clean

# 2026-08-24~25 수정분만
pytest -v tests/test_safety_estop.py tests/test_main_loop.py tests/test_numik_singularity.py
pytest -v -k "reset_to or servo or direction or unknown_model or clamp"

# 이전 수정분
pytest -v -k "watchdog or stall_timeout or note_alive"
pytest -v -k "confidence or held or hold or seed or stale or health or circular or drift"

# 파이프라인 E2E (카메라 없이) — 두 IK 경로 모두
python -m tests.generate_fixtures     # 최초 1회
python -m app.main --config .\config\default.yaml --tracker mock --publisher mock `
                   --replay .\tests\fixtures\arm_circle.jsonl --duration 5
python -m app.main --config .\config\default.yaml --tracker mock --publisher mock `
                   --replay .\tests\fixtures\arm_circle.jsonl --duration 5 --ik analytic

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
