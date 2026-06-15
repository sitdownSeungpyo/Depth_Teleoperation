# imitation_upper (C++) — RealSense D435i 상반신 텔레오퍼레이션

Python 구현(`../`)의 C++ 포팅. **경계 원칙**: 포팅 가능한 모든 것은 C++로,
신경망 forward pass(RTMPose/MediaPipe/HMR2/HaMeR)만 Python 사이드카로 남긴다.

```
C++ (이 디렉토리)                          Python 사이드카 (../python_sidecar)
  RealSense capture (librealsense)           RGB → 2D pixel keypoints + score
  depth_lift · aligner · retarget · numik      (RTMPose / MediaPipe / HMR2 / HaMeR)
  filter · safety · publishers
```

## 빌드 (CMake + Ninja)

요구 도구:
- CMake ≥ 3.24, **Ninja**
- MSVC (Visual Studio 2022 Build Tools) — Windows
- vcpkg (의존성 매니페스트 모드)

```powershell
# vcpkg 부트스트랩 (한 번)
git clone https://github.com/microsoft/vcpkg $env:USERPROFILE\vcpkg
& "$env:USERPROFILE\vcpkg\bootstrap-vcpkg.bat"
$env:VCPKG_ROOT = "$env:USERPROFILE\vcpkg"

# 구성 + 빌드 (Ninja)
cmake --preset ninja-release
cmake --build --preset ninja-release

# 테스트
ctest --preset ninja-release --output-on-failure
```

## 의존성 출처

| 라이브러리 | 출처 | 용도 |
|---|---|---|
| Eigen3 | vcpkg `eigen3` | 선형대수 (numpy 대체) |
| yaml-cpp | vcpkg `yaml-cpp` | config 로드 (pyyaml 대체) |
| nlohmann-json | vcpkg `nlohmann-json` | 골든 테스트 데이터 I/O |
| GoogleTest | vcpkg `gtest` | 단위/골든 테스트 |
| OpenCV | vcpkg `opencv4` | 영상 처리/시각화 |
| librealsense2 | vcpkg `realsense2` | D435i color/depth/IMU |
| Bullet | vcpkg `bullet3` | sim publisher (옵션) |
| ONNX Runtime | **수동** (vcpkg 빈약) — onnxruntime-gpu 릴리스 | RTMPose C++ 추론 (후속) |
| MuJoCo | **수동** (공식 릴리스 zip) | numIK 백본 (후속) |
| DynamixelSDK | **수동** (git submodule) | 실 로봇 서보 (후속) |

MuJoCo/ONNXRuntime/DynamixelSDK는 vcpkg에 없거나 빈약해서 `cmake/Find*.cmake`
+ 환경변수(`MUJOCO_DIR`, `ONNXRUNTIME_DIR`)로 잡는다. 해당 모듈(numik/rtmpose/
dynamixel)은 옵션 빌드 플래그로 끌 수 있어, 순수 알고리즘 코어는 위 vcpkg 패키지만
으로 빌드/테스트된다.

## 골든 테스트 (C++ == Python 수치 일치)

Python 레퍼런스 출력을 JSON으로 덤프하고 C++ 테스트가 그것과 일치하는지 검증한다.

```powershell
# 1) 사용자의 3.11 venv 에서 골든 데이터 생성 (제 환경 의존 X)
python tools\golden\gen_golden.py        # -> tests/golden/data/*.json

# 2) C++ 테스트가 그 JSON 을 읽어 비교
ctest --preset ninja-release
```

## 진행 상태

- [x] 빌드 스캐폴딩 (CMake/Ninja/vcpkg)
- [ ] 코어 순수 알고리즘 (types/config/filter/gravity/aligner/depth_lift/retarget/safety)
- [ ] numIK + robot_model (MuJoCo C API)
- [ ] device + publishers + RTMPose ONNX + Python 사이드카
- [ ] app 메인 루프
