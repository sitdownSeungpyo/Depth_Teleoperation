cmake_minimum_required(VERSION 3.24)
project(imitation_upper_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Optional feature modules. The pure-algorithm core builds with only the
# vcpkg base deps (Eigen/yaml-cpp); device / IK / inference modules need
# external SDKs (MuJoCo, ONNXRuntime, librealsense, DynamixelSDK) and are
# off by default so the core stays cheap to build + test anywhere.
option(ITU_BUILD_NUMIK     "Numerical IK (requires MuJoCo)"          OFF)
option(ITU_BUILD_DEVICE    "RealSense capture + OpenCV viz"          OFF)
option(ITU_BUILD_RTMPOSE   "RTMPose ONNXRuntime backend"            OFF)
option(ITU_BUILD_DYNAMIXEL "Dynamixel publisher"                     OFF)
option(ITU_BUILD_TESTS     "Build unit + golden tests"               ON)

find_package(Eigen3 CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
# yaml-cpp's exported target name varies by version: newer exports
# `yaml-cpp::yaml-cpp`, older just `yaml-cpp`. Normalize to the namespaced one.
if(NOT TARGET yaml-cpp::yaml-cpp AND TARGET yaml-cpp)
  add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
endif()

# ---- core library: pure-algorithm port (no hardware deps) ----
add_library(itu_core
  src/config.cpp
  src/mathutil.cpp
  src/filter.cpp
  src/gravity.cpp
  src/aligner.cpp
  src/depth_lift.cpp
  src/retarget.cpp
  src/safety.cpp
)
target_include_directories(itu_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(itu_core PUBLIC Eigen3::Eigen yaml-cpp::yaml-cpp)

if(ITU_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
