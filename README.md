# imitation_upper — RealSense D435i upper-body teleoperation controller

Captures the operator's upper-body pose with an Intel RealSense D435i +
MediaPipe Pose / Hand and maps it in real time to the joint angles of a 14-DOF
humanoid robot (UBP). It started from the analytical IK in Yi et al.
(Humanoids 2012) and adds the following:

- **Gravity-aligned aligner** — a torso frame that is invariant to the
  operator's torso tilt.
- **Decoupled shoulder pitch ↔ elbow** — a law-of-cosines elbow plus an
  atan2-based shoulder pitch, so MediaPipe elbow noise does not propagate into
  `sh_pitch`.
- **Auto rest-pose calibration** — runs the retargeter over the calibration
  frames and captures the per-joint circular mean as `rest_offsets`. The
  operator's anatomical asymmetry is mapped onto the robot zero automatically.
- **MediaPipe Hand keypoints → sh_yaw / w_yaw / w_pitch** — estimates 4 DOF that
  are unobservable from pose alone, using the finger-MCP positions of the 21
  hand landmarks (gravity-referenced axes, ±π wrap-aware).
- **4 publishers (sim + real robot)** — PyBullet (URDF), MuJoCo (MJCF),
  Dynamixel (UART/RS485 direct), and a Mock JSONL logger.
- **Safety layer** — dead-man hotkey, E-stop, watchdog, ramp-to-safe-pose.

Platform: Windows 11 + Python 3.11, MediaPipe Tasks API, OpenCV, PyRealSense2.

## Directory layout

```
imitation_upper/
├── config/
│   ├── ubp.yaml             # UBP 14-DOF robot (main operating config)
│   ├── default.yaml         # generic mock/replay
│   └── loose_visibility.yaml
├── models/
│   ├── ubp.urdf / .xml / .xacro     # robot model (URDF / MJCF / xacro)
│   └── (pose|hand)_landmarker_*.task  # fetched by the download script (gitignored)
├── core/
│   ├── types.py             # SkeletonFrame, JointCommand, KEYPOINT_NAMES
│   ├── aligner.py           # torso basis, gravity-aligned option
│   ├── retarget.py          # Yi 2012 IK + hand-based sh_yaw/w_yaw/w_pitch + auto calibration
│   ├── filter.py            # One Euro + joint limit / velocity clamp
│   └── safety.py            # deadman, E-stop, watchdog, ramp-to-safe
├── tracker/
│   ├── mock_tracker.py      # JSONL replay
│   └── realsense_tracker.py # D435i + MediaPipe Pose + Hand
├── publisher/
│   ├── mock_publisher.py    # JSONL logging
│   ├── udp_publisher.py     # UDP to an external robot
│   ├── pybullet_publisher.py
│   ├── mujoco_publisher.py
│   └── dynamixel_publisher.py
├── app/
│   ├── main.py              # main entrypoint
│   ├── debug_retarget.py    # bypasses filter/safety/publisher, prints raw angles
│   ├── tune_filter.py       # One Euro parameter sweep
│   ├── record.py / replay.py
│   └── check_camera.py / viz_*.py / test_sim_*.py
├── scripts/
│   ├── setup_env.ps1
│   ├── run.ps1
│   └── download_mediapipe_model.ps1
└── tests/                   # 52 unit + e2e mock tests
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

# 3. Test
pytest -q                                        # 52 passing
```

### (Optional) HMR2 (4D-Humans) body backend — robust to upper-body occlusion

The default MediaPipe Pose jitters its keypoints when an arm crosses in front of
the torso. Swapping in **HMR2.0** (Berkeley CVPR 2023, ViT-Huge + SMPL prior)
lets the SMPL kinematic prior fill in even the occluded joints naturally. GPU
required (RTX 4070 8 GB recommended, ~50 ms/frame). About 5x slower than
MediaPipe in exchange for much higher stability under occlusion.

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
# config/ubp.yaml
tracker:
  realsense:
    body_backend: hmr2        # mediapipe → hmr2
    hmr2_device: cuda
```

> install_hamer.ps1 and install_hmr2.ps1 share the same PyTorch/chumpy/smplx
> install. Even if you only use one of them, the dependency setup is identical.

### (Optional) HaMeR backend setup — much higher hand accuracy

Setting `tracker.realsense.hand_backend` in `config/ubp.yaml` to `hamer` uses
HaMeR (Berkeley CVPR 2024, ViT-Huge + MANO) instead of MediaPipe Hand.
Detection rate ~100 %, with a large reduction in w_yaw/w_pitch jitter. GPU
required (RTX 4070 8 GB recommended, ~30 ms/hand). Latency is ~2x MediaPipe.

Total disk: ~13 GB (weights 5.7 GB tarball + extraction + torch). Time: ~15 min.

```powershell
.\scripts\install_hamer.ps1
# step 1: PyTorch + CUDA 12.1 (skip if already present)
# step 2: deps (gdown, pyrender, pytorch-lightning, scikit-image, yacs, timm, einops,
#               chumpy --no-build-isolation, smplx==0.1.28)
# step 3: clone HaMeR + patch (renderer optional) + pip install --no-deps -e .
#         (skip detectron2 — make the bbox from the MediaPipe Pose wrist coords)
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
# config/ubp.yaml
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

### Simulation (MuJoCo, recommended)

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

1. **Tracker** — RealSense color+depth → MediaPipe Pose (33 landmarks) + Hand
   (2 × 21 landmarks). pose_world_landmarks (hip-centered, meters) +
   hand_world_landmarks (wrist-relative) → the keypoints dict of a
   `SkeletonFrame`.

2. **Aligner** — builds a torso basis from shoulders/head/hips. With
   `gravity_up` set it ignores operator tilt (gravity-aligned), otherwise it is
   body-relative. Result: every keypoint rotated into the torso frame plus the
   torso RPY.

3. **Retargeter** —
   - sh_pitch, sh_roll, elbow: Yi 2012 Eq. 1–3 (law-of-cosines elbow,
     decoupled atan2-based sh_pitch).
   - sh_yaw: rotation angle of the elbow-flex plane, with
     `cross(upper, world_down)` as the reference. Underdetermined → 0 when the
     arm is parallel to gravity.
   - w_yaw, w_pitch: palm direction from hand_middle_mcp / index_mcp / pinky_mcp,
     gravity-referenced. 0 when no hand is detected.
   - All outputs subtract `Calibration.rest_offsets` then wrap to (-π, π].

4. **Filter / Limiter** — One Euro smoother + mechanical joint limits +
   velocity clamp.

5. **Safety** — deadman hotkey state, confidence threshold, frame-loss grace
   period, watchdog, ramp to safe_pose on E-stop.

6. **Publisher** — mock / udp / pybullet / mujoco / dynamixel. A 100 Hz
   interpolating loop (operator 30 Hz < robot command 100 Hz).

## Key config keys (`config/ubp.yaml`)

| key | meaning |
|-----|---------|
| `tracker.realsense.use_world_landmarks` | true = MediaPipe pose_world_landmarks; false = RealSense depth deprojection |
| `tracker.realsense.gravity_up` | up direction for the gravity-aligned aligner. For MediaPipe pose_world it is `[0,-1,0]` |
| `tracker.realsense.hand_model_asset_path` | path to hand_landmarker.task. null disables the hand |
| `retarget.decouple_pitch_elbow` | true = atan2(z,x) only, isolating MediaPipe elbow noise |
| `retarget.rest_offsets` | null = auto calibration. A dict = manual override |
| `retarget.output_gain` | per-joint 1.0 (as-is) / 1.2 (amplify 20%) / 0.5 (half), etc. |
| `filter.one_euro.min_cutoff, beta` | smaller = stronger smoothing (less responsive), larger = faster motion response |
| `safety.deadman.key` | default `space` |

## Dependencies

- pyrealsense2, mediapipe, opencv-python, numpy, scipy, pyyaml
- pynput (deadman hotkey)
- pybullet, mujoco (simulation), dynamixel-sdk (real robot)
- (dev) pytest, ruff, mypy

## Tests / CI

`pytest -q` — 52 unit/e2e tests, ~10 s. `ruff check .` + `mypy --strict core`
recommended.

## Known limitations

- Single camera, assumes the operator faces it (accuracy drops from the side due
  to self-occlusion).
- MediaPipe Hand is detected in only ~70 % of frames (dropout under occlusion /
  fast motion).
- sh_yaw is underdetermined → returns 0 when the elbow is nearly straight or the
  upper arm is parallel to gravity.
- If the operator's calibration pose (arms hanging naturally) differs greatly
  from the robot zero, the rest_offset absorbs it — but moving during
  calibration makes the baseline unstable.

## Future work

- Numerical IK (mink / pink) — joint 14-DOF optimization integrating joint
  limits / smoothness.
- If hand accuracy is insufficient, swap in GPU-based hand mesh reconstruction
  such as HaMeR / WiLoR.
- Multi-camera (front + side) to resolve depth ambiguity.
- Lower-body controller (external balance) — currently out of scope.
