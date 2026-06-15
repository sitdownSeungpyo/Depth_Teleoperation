cmake_minimum_required(VERSION 3.24)
project(imitation_upper_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
  # /utf-8: sources carry UTF-8 comments (λ, ≤, →, π ...); without this MSVC warns
  #         C4819 on non-UTF-8 locales (e.g. cp949). /EHsc: standard C++ unwind.
  add_compile_options(/utf-8 /EHsc)
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

# ---- numik library: numerical IK on the robot model (needs MuJoCo) ----
if(ITU_BUILD_NUMIK)
  if(NOT MUJOCO_DIR AND DEFINED ENV{MUJOCO_DIR})
    set(MUJOCO_DIR "$ENV{MUJOCO_DIR}")
  endif()
  if(NOT MUJOCO_DIR OR NOT EXISTS "${MUJOCO_DIR}/include/mujoco/mujoco.h")
    message(FATAL_ERROR
      "ITU_BUILD_NUMIK=ON needs a MuJoCo SDK. Set -DMUJOCO_DIR=<sdk> (with "
      "include/ lib/ bin/) or the MUJOCO_DIR env var. Got: '${MUJOCO_DIR}'")
  endif()
  add_library(mujoco_imported SHARED IMPORTED GLOBAL)
  set_target_properties(mujoco_imported PROPERTIES
    IMPORTED_IMPLIB "${MUJOCO_DIR}/lib/mujoco.lib"
    IMPORTED_LOCATION "${MUJOCO_DIR}/bin/mujoco.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_DIR}/include")
  add_library(itu_numik src/numik.cpp src/robot_model.cpp)
  target_include_directories(itu_numik PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(itu_numik PUBLIC itu_core Eigen3::Eigen mujoco_imported)
endif()

if(ITU_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
