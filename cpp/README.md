# imitation_upper (C++) — RealSense D435i upper-body teleoperation

A C++ port of the Python implementation (`../`). **Boundary principle**:
everything portable goes to C++; only the neural-net forward pass
(RTMPose/MediaPipe/HMR2/HaMeR) stays in a Python sidecar.

```
C++ (this directory)                       Python sidecar (../python_sidecar)
  RealSense capture (librealsense)           RGB → 2D pixel keypoints + score
  depth_lift · aligner · retarget · numik      (RTMPose / MediaPipe / HMR2 / HaMeR)
  filter · safety · publishers
```

## Build (CMake + Ninja)

Required tools:
- CMake ≥ 3.24, **Ninja**
- MSVC (Visual Studio 2022 Build Tools) — Windows
- vcpkg (manifest mode)

```powershell
# bootstrap vcpkg (once)
git clone https://github.com/microsoft/vcpkg $env:USERPROFILE\vcpkg
& "$env:USERPROFILE\vcpkg\bootstrap-vcpkg.bat"
$env:VCPKG_ROOT = "$env:USERPROFILE\vcpkg"

# configure + build (Ninja)
cmake --preset ninja-release
cmake --build --preset ninja-release

# test
ctest --preset ninja-release --output-on-failure
```

## Dependency sources

| Library | Source | Use |
|---|---|---|
| Eigen3 | vcpkg `eigen3` | linear algebra (replaces numpy) |
| yaml-cpp | vcpkg `yaml-cpp` | config loading (replaces pyyaml) |
| nlohmann-json | vcpkg `nlohmann-json` | golden-test data I/O |
| GoogleTest | vcpkg `gtest` | unit / golden tests |
| OpenCV | vcpkg `opencv4` | image processing / visualization |
| librealsense2 | vcpkg `realsense2` | D435i color/depth/IMU |
| Bullet | vcpkg `bullet3` | sim publisher (optional) |
| ONNX Runtime | **manual** (vcpkg support is thin) — onnxruntime-gpu release | RTMPose C++ inference (later) |
| MuJoCo | **manual** (official release zip) | numIK backbone (later) |
| DynamixelSDK | **manual** (git submodule) | real-robot servos (later) |

MuJoCo / ONNXRuntime / DynamixelSDK are absent or thin in vcpkg, so they are
located via `cmake/Find*.cmake` + environment variables (`MUJOCO_DIR`,
`ONNXRUNTIME_DIR`). Those modules (numik / rtmpose / dynamixel) are behind
optional build flags, so the pure-algorithm core builds and tests with only the
vcpkg base packages above.

## Golden tests (C++ == Python numeric equality)

Dump the Python reference output to JSON, then have the C++ tests assert they
match.

```powershell
# 1) generate golden data in your 3.11 venv (independent of this machine)
python tools\golden\gen_golden.py        # -> tests/golden/data/*.json

# 2) C++ tests read that JSON and compare
ctest --preset ninja-release
```

## Progress

- [x] build scaffolding (CMake/Ninja/vcpkg)
- [x] core pure algorithm (types/config/filter/gravity/aligner/depth_lift/retarget/safety) + golden tests
- [ ] numIK + robot_model (MuJoCo C API)
- [ ] device + publishers + RTMPose ONNX + Python sidecar
- [ ] app main loop
