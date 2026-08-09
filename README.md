# imitation_upper — RealSense D435i upper-body teleoperation controller

Captures the operator's upper-body pose with an Intel RealSense D435i and maps it
in real time to the joint angles of a 14-DOF humanoid robot (UBP). It started
from the analytical IK in Yi et al. (Humanoids 2012); the default path is now
**RTMPose 2D + RealSense depth estimation → numerical task-space IK**.

> For current status and next steps see **`HANDOFF.md`**. This README covers
> structure and setup.

**Estimation**

- **RTMPose body-17** (rtmlib + onnxruntime-gpu, ~34 ms/frame) is the default
  backend. MediaPipe and HMR2 are kept as fallbacks.
- **Robust depth lifting** (`tracker/depth_lift.py`) — foreground sampling →
  spike rejection → temporal median → segment-length gate → bone-length pinning.
  Removes, stage by stage, the forward/back jitter caused by a thin limb at a
  silhouette edge mixing body and background depth.
- **IMU gravity alignment** — measures real gravity with the D435i
  accelerometer and uses it as the aligner's up vector, giving a torso frame
  invariant to camera and operator tilt (verified on hardware).
- **Confidence-aware hold** — when the *lowest* score among an arm's
  shoulder/elbow/wrist falls below a threshold, that arm's IK is skipped and its
  previous command is held.

**Control**

- **Numerical task-space IK** (`core/numik.py`, MuJoCo DLS) — drives elbow and
  wrist to targets simultaneously. Tracks bent arms to <3° (analytic: 25–48°),
  needs no arm-length calibration, and takes a URDF drop-in.
- **Analytic IK (Yi 2012)** remains as the `--ik analytic` fallback, with auto
  rest-pose calibration and hand-keypoint-based sh_yaw / w_yaw / w_pitch.
- **Filter / Limiter** — One Euro + joint limits + velocity clamp. Joints that
  were not observed are held at their previous value, so the command's joint set
  does not change from frame to frame.
- **Safety layer** — dead-man hotkey, E-stop, a **watchdog on its own thread**,
  and tracking-loss ramp-to-safe-pose.
- **5 publishers** — PyBullet (URDF), MuJoCo (MJCF), Dynamixel (UART/RS485
  direct), UDP (skeleton), and a Mock JSONL logger.

Platform: Windows 11 + Python 3.11, OpenCV, PyRealSense2, MuJoCo, onnxruntime-gpu.

## Directory layout

```
imitation_upper/
├── config/                  # ubp.yaml includes the purpose-split files → deep-merge
│   ├── ubp.yaml             # ★ main operating entrypoint (include orchestrator)
│   ├── tracker.yaml         # camera / pose estimation / depth_lift / IMU
│   ├── retarget.yaml        # analytic retargeter (Yi 2012)
│   ├── robot.yaml           # numerical IK robot model
│   ├── filter.yaml          # OneEuro / velocity limits
│   ├── joint_limit.yaml     # ★ all joint limits — the only file to edit for a real robot
│   ├── publisher.yaml / safety.yaml / runtime.yaml
│   ├── default.yaml         # mock/replay (includes the above + overrides)
│   └── loose_visibility.yaml  # diagnostics: all gates off (not for operation)
├── models/
│   ├── upperbody_sim.xml    # ★ default numerical-IK model (axes verified)
│   ├── ubp.urdf / .xml / .xacro     # robot model (URDF / MJCF / xacro)
│   └── (pose|hand)_landmarker_*.task  # fetched by the download script (gitignored)
├── core/
│   ├── types.py             # SkeletonFrame, JointCommand, KEYPOINT_NAMES
│   ├── config.py            # include resolution + deep-merge loader (cycle detection)
│   ├── aligner.py           # torso basis, gravity/IMU up, confidence pass-through
│   ├── retarget.py          # Yi 2012 analytic IK + hand-based yaw/pitch + calibration
│   ├── numik.py             # numerical task-space IK (MuJoCo DLS)
│   ├── robot_model.py       # robot adapter (auto link lengths, target clamp)
│   ├── filter.py            # One Euro + joint limit / velocity clamp / hold
│   └── safety.py            # deadman, E-stop, watchdog thread, ramp-to-safe
├── tracker/
│   ├── base.py              # SkeletonTracker Protocol + TrackerHealth
│   ├── mock_tracker.py      # JSONL replay
│   ├── realsense_tracker.py # D435i capture thread + IMU + backend delegation
│   ├── body_backend.py / rtmpose_body_backend.py / hmr2_body_backend.py
│   ├── hand_backend.py / hamer_hand_backend.py
│   ├── depth_lift.py        # the 5 robust-depth stages
│   └── gravity.py           # accelerometer → up vector estimation
├── publisher/               # mock / udp / pybullet / mujoco / dynamixel
├── app/
│   ├── sim_teleop.py        # ★ camera → numerical IK → MuJoCo (default run path)
│   ├── main.py              # analytic IK + safety + publisher pipeline
│   ├── debug_retarget.py / debug_elbow.py / debug_imu.py
│   ├── viz_camera.py / viz_teleop.py / viz_mujoco.py / viz_urdf.py
│   ├── tune_filter.py / record.py / replay.py / check_camera.py
├── scripts/                 # setup_env / run / install_{rtmpose,hamer,hmr2}
└── tests/                   # 133 unit + e2e mock tests (no camera needed)
```

## Setup (Windows / PowerShell)

```powershell
# 1. Install RealSense SDK 2.0 + Python 3.11 (64-bit) first
.\scripts\setup_env.ps1                          # create .venv + pip install -e .[dev]
.\.venv\Scripts\Activate.ps1

# 2. Download the MediaPipe models (~14 MB pose + 7.5 MB hand)
.\scripts\download_mediapipe_model.ps1 heavy     # pose_landmarker_heavy.task
# Download the hand model separately from the README URL:
#   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
#   → models/hand_landmarker.task

# 3. RTMPose (default body backend, GPU)
.\scripts\install_rtmpose.ps1                    # rtmlib + onnxruntime-gpu
#   Install ORDER matters: rtmlib pulls in the CPU onnxruntime, which shadows the
#   GPU build — the script installs rtmlib first, then force-reinstalls
#   onnxruntime-gpu. No system CUDA Toolkit is required: RTMPoseBodyBackend.start()
#   puts torch/lib's CUDA 12 / cuDNN 9 DLLs on PATH. Without them onnxruntime
#   silently falls back to CPU (~775 ms/frame vs ~34 ms).

# 4. Test
pytest -q                                        # 133 passing (no camera needed)
```

### (Optional) HMR2 (4D-Humans) body backend — robust to upper-body occlusion

MediaPipe Pose jitters its keypoints when an arm crosses in front of the torso.
Swapping in **HMR2.0** (Berkeley CVPR 2023, ViT-Huge + SMPL prior) lets the SMPL
kinematic prior fill in even the occluded joints naturally. GPU required
(RTX 4070 8 GB recommended, ~50 ms/frame). About 5x slower than MediaPipe in
exchange for much higher stability under occlusion.

> Note: HMR2 is monocular and ignores the depth sensor, regressing motion along
> the depth axis toward a mean — which compresses elbow flexion roughly 3.3x.
> That is why the default moved to RTMPose. If you switch back, raise
> `retarget.output_gain.r/l_elbow` again to compensate.

```powershell
.\scripts\install_hmr2.ps1
# step 1: verify shared deps (shared with install_hamer.ps1 — torch, chumpy, smplx ...)
# step 2: clone 4D-Humans + patch (renderer optional) + pip install --no-deps -e .
#         (skip detectron2 — use the MediaPipe Pose bbox to bypass ViTDet)
# step 3: auto-download the HMR2 checkpoint ~670 MB (~/.cache/4DHumans)
# step 4: import sanity check
# step 5: SMPL_NEUTRAL.pkl notice (MANUAL — license)
```

SMPL download (manual, license):
1. Register at https://smpl.is.tue.mpg.de/ (free, non-commercial only)
2. Download "SMPL for Python users" or SMPL v1.0/v1.1
3. Copy `basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl` from the zip to
   `~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl` (rename required)

Enable:
```yaml
# config/tracker.yaml
tracker:
  realsense:
    body_backend: hmr2        # rtmpose → hmr2
    hmr2_device: cuda
```

> install_hamer.ps1 and install_hmr2.ps1 share the same PyTorch/chumpy/smplx
> install. Even if you only use one of them, the dependency setup is identical.

### (Optional) HaMeR backend setup — much higher hand accuracy

Setting `tracker.realsense.hand_backend` to `hamer` uses HaMeR (Berkeley CVPR
2024, ViT-Huge + MANO) instead of MediaPipe Hand. Detection rate ~100 %, with a
large reduction in w_yaw/w_pitch jitter. GPU required (RTX 4070 8 GB
recommended, ~30 ms/hand). Latency is ~2x MediaPipe.

Total disk: ~13 GB (weights 5.7 GB tarball + extraction + torch). Time: ~15 min.

```powershell
.\scripts\install_hamer.ps1
# step 1: PyTorch + CUDA 12.1 (skip if already present)
# step 2: deps (gdown, pyrender, pytorch-lightning, scikit-image, yacs, timm, einops,
#               chumpy --no-build-isolation, smplx==0.1.28)
# step 3: clone HaMeR + patch (renderer optional) + pip install --no-deps -e .
#         (skip detectron2 — make the bbox from the wrist coords)
# step 4: HaMeR pretrained weights (~5.7 GB via UT Austin direct link)
# step 5: import sanity check
# step 6: MANO download notice (MANUAL)
```

The script is idempotent — already-installed steps are skipped automatically.

MANO download (manual, license):
1. Register at https://mano.is.tue.mpg.de/ (free, non-commercial only)
2. "Models & Code" → download `MANO_v1_2.zip`
3. Place only `MANO_RIGHT.pkl` under `third_party/hamer/_DATA/data/mano/`
   (HaMeR mirrors the left hand from the right, so only RIGHT is needed)

Enable:
```yaml
# config/tracker.yaml
tracker:
  realsense:
    hand_backend: hamer       # mediapipe → hamer
    hamer_device: cuda
```

> **Note**: some HaMeR imports break on Windows — pyrender's OpenGL.EGL
> (Linux-only) + detectron2 C++ build. install_hamer.ps1 works around this
> automatically: (a) wraps the renderer imports in `hamer/utils/__init__.py` in
> try/except (in-place patch), (b) installs HaMeR with `--no-deps` plus only the
> needed deps. The patch is applied only inside `third_party/hamer/` (gitignored).

## Running

### Teleop sim — numerical IK (default path)

```powershell
python -m app.sim_teleop --config .\config\ubp.yaml
```

Opens the MuJoCo robot window alongside a camera skeleton window. Options:

| option | meaning |
|--------|---------|
| `--ik analytic` | switch to analytic Yi-2012 IK (less accurate on bent arms) |
| `--model <path>` | override the robot model (to check a URDF/MJCF drop-in) |
| `--no-camera` | robot window only |
| `--no-filter` | disable the joint OneEuro (for comparing jitter) |

Camera window: `q`/ESC quits, `s` saves a snapshot. If the camera stops feeding
frames, the window banner and the console both show `CAMERA STALLED` and the
robot holds its last pose.

### Estimation only

```powershell
python -m app.viz_camera --config .\config\ubp.yaml --backend rtmpose
python -m app.debug_imu  --config .\config\ubp.yaml --duration 30   # verify IMU gravity
```

### Analytic IK pipeline (with safety + publisher)

```powershell
.\scripts\run.ps1 --config .\config\ubp.yaml --tracker realsense --publisher mujoco
```

Calibration instructions are printed to the console (drop both arms naturally,
relax the shoulders, hold ~0.5 s). After calibration the operator's motion is
mapped onto the sim robot.

### Raw-angle debug (bypass filter / publisher)

```powershell
python -m app.debug_retarget --config .\config\ubp.yaml --duration 30
```

Console columns: `r_elb` `l_elb` | `r_sp` `r_sr` | `r_shy` `r_wy` `r_wp` |
confidence + whether the hand was detected.

### Real robot (Dynamixel)

```powershell
.\scripts\run.ps1 --config .\config\ubp.yaml --tracker realsense --publisher dynamixel --deadman
```

`--deadman`: commands flow only while `space` is held; `esc` triggers E-stop.

### Mock JSONL replay (no camera)

```powershell
python -m tests.generate_fixtures                # once
python -m app.main --config .\config\default.yaml --tracker mock --publisher mock `
                  --replay .\tests\fixtures\arm_circle.jsonl --duration 5
```

## Main pipeline

```
RTMPose 2D → foreground depth + spike rejection → deproject (3D)
  → temporal median → segment-length gate → bone-length pinning
  → aligner (+IMU gravity) → keypoint OneEuro
  → numerical IK (elbow+wrist, warm-started) → target jump clamp
  → joint OneEuro → robot
```

1. **Tracker** — capture thread for RealSense color+depth(+accel). The body
   backend produces keypoints and the hand backend the hand landmarks, merged
   into a `SkeletonFrame`. `health()` exposes capture-thread liveness and the age
   of the most recent frame, so a stall is detectable.

2. **Depth lift** (`tracker/depth_lift.py`, RTMPose path) — removes, in five
   stages, the noise from a thin limb mixing background depth at a silhouette
   edge. Each stage catches a different failure: `foreground_depth` (background
   bleed) → `RobustDepthLifter` (holes / spikes) → `TemporalMedianFilter`
   (single-frame outliers) → `SegmentConsistencyGate` (the backward flip on a
   frontal reach, identified by a ballooning segment length) →
   `BoneLengthStabilizer` (residual z jitter).

3. **Aligner** — builds a torso basis from shoulders/head/hips. Uses the
   measured gravity when the IMU is enabled, otherwise the configured
   `gravity_up`. Rotates keypoints into the torso frame and carries the torso RPY
   plus **per-keypoint confidence**.

4a. **Numerical IK** (default) — DLS on the robot model driving elbow and wrist
    together. Targets are placed at the robot's own link lengths along the
    operator's directions, so they are always reachable and no arm-length
    calibration is needed. The elbow is seeded only near the straight-arm
    singularity.

4b. **Analytic retargeter** (fallback) — Yi 2012 Eq. 1–3, sh_yaw referenced to
    `cross(upper, world_down)`, and w_yaw/w_pitch from the hand MCPs. Outputs
    subtract `rest_offsets` then wrap to (-π, π]. An arm that could not be
    observed is **omitted** rather than snapped to zero.

5. **Filter / Limiter** — One Euro + mechanical joint limits + velocity clamp.
   Joints absent this frame are held at their previous value, so the joint set
   stays fixed and the unwrap/velocity clamp still work on the recovery frame.

6. **Safety** — deadman hotkey, confidence threshold + loss grace → safe_pose
   ramp, and a **watchdog on its own thread** that E-stops when the main loop
   stops feeding commands. E-stop zeroes the output.

7. **Publisher** — mock / udp / pybullet / mujoco / dynamixel. A 100 Hz
   interpolating loop (operator 30 Hz < robot command 100 Hz).

## Key config keys

`config/ubp.yaml` is an include orchestrator — the values live in the
purpose-split files, are deep-merged in the listed order, and the entrypoint's
own keys are applied last.

| key | file | meaning |
|-----|------|---------|
| `tracker.realsense.body_backend` | tracker.yaml | `rtmpose` (default) / `mediapipe` / `hmr2` |
| `tracker.realsense.enable_imu` | tracker.yaml | accelerometer gravity alignment (verify on hardware first) |
| `tracker.realsense.depth_lift.*` | tracker.yaml | the 5 depth stages (tuning knobs are in HANDOFF.md) |
| `tracker.pose.min_visibility` | tracker.yaml | hard keypoint reject threshold (default 0.3) |
| `tracker.pose.min_arm_confidence` | tracker.yaml | per-arm soft gate — a low arm is held (default 0.4, 0 = off) |
| `robot.model_path` / `arms.*` | robot.yaml | numerical-IK robot model — the URDF drop-in point |
| `robot.ik.max_target_step_m` | robot.yaml | per-frame endpoint travel cap (suppresses jumps) |
| `robot.ik.elbow_seed_below_rad` | robot.yaml | seed the elbow only below this angle (0 = off) |
| `robot.joint_limits` | joint_limit.yaml | ★ the only file to edit for a real robot |
| `retarget.output_gain` | retarget.yaml | per-joint scaling. Revisit when changing backend |
| `filter.one_euro.min_cutoff, beta` | filter.yaml | smaller = stronger smoothing (less responsive) |
| `safety.deadman.key` | safety.yaml | default `space` |
| `safety.watchdog_timeout_s` | safety.yaml | E-stop if no command arrives within this window (default 0.5 s) |

## Dependencies

- pyrealsense2, opencv-python, numpy, scipy, pyyaml, **mujoco** (the default IK path)
- mediapipe (fallback body backend + hand), pynput (deadman hotkey)
- pybullet (sim), dynamixel-sdk (real robot)
- (extras) `[rtmpose]` rtmlib + onnxruntime-gpu, `[hamer]`; HMR2 via install script
- (dev) pytest, ruff, mypy

## Tests / CI

```powershell
pytest -q          # 133 tests, no camera or GPU needed
ruff check .       # lint (E,F,W,I,B,UP,PTH)
mypy               # strict — core + tracker + publisher
```

`.github/workflows/ci.yml` runs all three on windows-latest. CI deliberately does
not install the camera stack (pyrealsense2/mediapipe/rtmlib): those imports are
all lazy and the heavy tests are gated with `pytest.importorskip`.

## Known limitations

- Single camera, assumes the operator faces it (accuracy drops from the side due
  to self-occlusion).
- MediaPipe Hand is detected in only ~70 % of frames (dropout under occlusion /
  fast motion).
- sh_yaw is underdetermined → returns 0 when the elbow is nearly straight or the
  upper arm is parallel to gravity.
- The analytic IK is a ~25° approximation on a bent arm — inherent to the paper's
  3-DOF formulation, not a model bug. Use the numerical IK when accurate bent-arm
  tracking matters.
- `publisher/udp_publisher.py` is a deliberate skeleton; the packet schema waits
  on the target robot platform.
- If the operator's calibration pose (arms hanging naturally) differs greatly
  from the robot zero, the rest_offset absorbs it — but moving during
  calibration makes the baseline unstable.

## Future work

- Integrate the numerical IK into `app/main.py` (the safety + publisher
  pipeline). It currently lives only in `app/sim_teleop.py`, so the two must be
  merged before running on real hardware.
- Apply the real robot URDF (only `config/robot.yaml` + `config/joint_limit.yaml`
  need editing).
- If hand accuracy is insufficient, swap in GPU-based hand mesh reconstruction
  such as HaMeR / WiLoR.
- Multi-camera (front + side) to resolve depth ambiguity.
- Lower-body controller (external balance) — currently out of scope.
