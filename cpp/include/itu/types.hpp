// Core data types + naming constants shared across modules.
// Mirrors core/types.py (spec §3.1–§3.3).
#pragma once

#include <Eigen/Core>
#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace itu {

// Keypoints are stored name->vec3 / name->confidence. std::map (ordered) is used
// deliberately so iteration order is deterministic — important for golden tests
// that compare per-keypoint outputs against the Python reference.
using KeypointMap = std::map<std::string, Eigen::Vector3d>;
using ConfidenceMap = std::map<std::string, double>;
using JointMap = std::map<std::string, double>;

// Required skeleton keypoint names (spec §3.1).
inline constexpr std::array<const char*, 11> KEYPOINT_NAMES = {
    "head", "neck", "torso", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip"};

// One pose observation in the camera frame (spec §3.1).
struct SkeletonFrame {
  double timestamp = 0.0;
  KeypointMap keypoints;
  ConfidenceMap confidence;

  double mean_confidence() const {
    if (confidence.empty()) return 0.0;
    double s = 0.0;
    for (const auto& [name, c] : confidence) s += c;
    return s / static_cast<double>(confidence.size());
  }
};

// One joint setpoint frame sent to the robot (spec §3.2).
struct JointCommand {
  double timestamp = 0.0;
  JointMap positions;
  std::optional<JointMap> velocities;
  double source_frame_ts = 0.0;
};

}  // namespace itu
