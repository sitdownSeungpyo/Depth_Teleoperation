# C++ Port — Resume Guide (HANDOFF)

Self-contained notes to **continue the C++ port on another PC**. Branch:
`feature/c++` (pushed to origin). The Python source stays at the repo root; the
C++ port lives under `cpp/`.

> Note: Claude Code's local "memory" is NOT in this repo (it lives in
> `~/.claude/...` on the original machine). Everything needed to resume is in
> this file.

## 1. What this is — the boundary decision

Port the teleop pipeline to C++. **Only the neural-net forward pass stays Python**
(a sidecar: RTMPose / MediaPipe / HMR2 / HaMeR → 2D pixel keypoints + score).
Everything portable is C++: RealSense capture, deproject, depth_lift, aligner,
retarget, numIK, filter, safety, publishers. The C++/Python seam is at the **2D
keypoint level** (narrowest Python surface).

Verification strategy: **golden tests**. `cpp/tools/golden/gen_golden.py` (run in
a Python venv) dumps the Python reference I/O to JSON; GoogleTest reads it and
asserts C++ == Python numerically. So the port is pinned to the reference without
the C++ build needing Python at runtime.

## 2. Status (commits on `feature/c++`)

| commit | content |
|---|---|
| 49d3257 | core scaffolding + group-A pure algorithm + golden tests |
| 694bad6 | verify fixes (foreground-depth test, MSVC /utf-8 /EHsc) |
| 526ddbd | numik + robot_model (MuJoCo C API) + golden test |
| fbd4da4 | publishers (base/mock/udp/dxl-conv/mujoco-headless) + golden tests |

Done + verified (**ctest 26/26 green**):
- [x] types, config (yaml-cpp), mathutil (numpy-faithful), filter (OneEuro +
      limiter), gravity, aligner, depth_lift, retarget, safety
- [x] numik (DLS IK) + robot_model (MuJoCo C API)
- [x] publishers: InterpolatingPublisherBase, MockPublisher, UdpPublisher,
      Dynamixel rad→unit, headless MuJoCoPublisher

Tolerances: 1e-9 for closed-form modules, 1e-6 for numik (iterative DLS).

## 3. Remaining work (next steps)

1. Remaining publishers: Dynamixel SDK driver (gated `ITU_BUILD_DYNAMIXEL`),
   PyBullet, MuJoCo GLFW viewer.
2. **Device layer**: librealsense2 capture (color/depth/IMU + deproject) + OpenCV
   viz. Wraps a `cv::Mat` depth into `DepthImageView` (already defined).
3. **Python sidecar + IPC**: the only Python that remains. Design the RGB→2D
   keypoint protocol (recommend shared-memory or local socket; sidecar process so
   a GPU/ML crash can't kill the C++ main). Sidecar reuses the existing
   `tracker/rtmpose_body_backend.py` etc.
4. **App main loop**: tracker → aligner → IK → filter → safety → publisher
   orchestration (mirror `app/sim_teleop.py` / `app/main.py`).

## 4. Prerequisites on a fresh PC

- **MSVC** (Visual Studio 2019 or 2022 Build Tools, C++ workload). Note the
  `vcvars64.bat` path, e.g.
  `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat`.
- **CMake ≥ 3.24 + Ninja** — easiest via `pip install cmake ninja` inside the venv.
- **vcpkg** — clone + bootstrap (pulls eigen3/yaml-cpp/nlohmann-json/gtest):
  ```powershell
  git clone https://github.com/microsoft/vcpkg $env:USERPROFILE\vcpkg
  & "$env:USERPROFILE\vcpkg\bootstrap-vcpkg.bat" -disableMetrics
  ```
- **MuJoCo SDK 3.9.0** (for `ITU_BUILD_NUMIK`) — must match the pip `mujoco`
  version used to generate golden data, so compiled models are identical:
  ```powershell
  $u="https://github.com/google-deepmind/mujoco/releases/download/3.9.0/mujoco-3.9.0-windows-x86_64.zip"
  Invoke-WebRequest $u -OutFile $env:TEMP\mj.zip
  Expand-Archive $env:TEMP\mj.zip $env:USERPROFILE\mujoco-3.9.0
  # -> include/ lib/ bin/ ; set MUJOCO_DIR to this folder
  ```
- **Python venv** for golden generation (only needs a few packages; the group-A
  + numik + publisher modules import numpy/yaml/mujoco only — NOT the full
  pyrealsense2/mediapipe stack). Python 3.10–3.12 is fine despite the repo's 3.11
  pin (the ported modules don't use 3.11-only features):
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\python -m pip install numpy pyyaml mujoco==3.9.0 cmake ninja
  ```

## 5. Build + test recipe

Golden data is **gitignored** (so it never drifts from the reference) — you MUST
regenerate it on each machine before tests, else the golden tests SKIP:

```powershell
.\.venv\Scripts\python cpp\tools\golden\gen_golden.py   # -> cpp/tests/golden/data/*.json
```

Configure + build + test. The Ninja+MSVC build must run inside the `vcvars64`
environment so `cl.exe` is found. **Critical ordering gotcha**: prepend the venv
Scripts to `PATH` *before* calling `vcvars64.bat`, because `cmd` expands `%PATH%`
at parse time — if you `set PATH=...;%PATH%` *after* vcvars, you wipe the MSVC
paths and CMake can't find the compiler.

```powershell
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$venv   = "<repo>\.venv\Scripts"
$vcpkg  = "$env:USERPROFILE\vcpkg"
$mjdir  = "$env:USERPROFILE\mujoco-3.9.0"
$cpp    = "<repo>\cpp"
$cfg = "cmake --preset ninja-release -DITU_BUILD_NUMIK=ON -DMUJOCO_DIR=`"$mjdir`""
$cmd = "set `"PATH=$venv;%PATH%`" && set `"VCPKG_ROOT=$vcpkg`" && `"$vcvars`" && " +
       "cd /d `"$cpp`" && $cfg && cmake --build --preset ninja-release && ctest --preset ninja-release"
cmd /c $cmd
```

Without MuJoCo, omit `-DITU_BUILD_NUMIK=ON` (and `MUJOCO_DIR`) — the pure core +
publishers still build and test with only the vcpkg base deps.

CMake feature flags: `ITU_BUILD_NUMIK` (MuJoCo numik + mujoco publisher),
`ITU_BUILD_DEVICE` (RealSense/OpenCV, not implemented yet), `ITU_BUILD_RTMPOSE`,
`ITU_BUILD_DYNAMIXEL` — all OFF by default so the core is cheap to build anywhere.

## 6. Gotchas already hit (don't re-debug these)

- `cmd` `%PATH%` parse-time expansion — set venv PATH *before* vcvars (see above).
- `<windows.h>` defines `min`/`max` macros that break `std::max` → `#define
  NOMINMAX` before including it (done in interpolating_publisher.cpp).
- yaml-cpp exports `yaml-cpp::yaml-cpp` (new) or `yaml-cpp` (old) — CMakeLists
  aliases it.
- MSVC `/utf-8` needed (source comments contain λ/≤/π…) + `/EHsc`.
- MuJoCo is a DLL — the test CMake copies `mujoco.dll` next to `itu_tests.exe`.
- numik golden uses a 1e-6 tol (LAPACK gesv vs Eigen partialPivLu in the DLS solve).

## 7. Layout

```
cpp/
├── CMakeLists.txt, CMakePresets.json (ninja-release/debug), vcpkg.json
├── include/itu/*.hpp     # public headers
├── src/*.cpp             # implementations
├── tests/                # GoogleTest + golden tests; golden/data/*.json (gitignored)
└── tools/golden/gen_golden.py   # run in venv to (re)generate golden data
```
