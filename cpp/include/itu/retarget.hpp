// Kinematic Retargeter — analytic IK (Yi 2012 Eq.1-3) + hand-based wrist/shoulder
// yaw + auto rest-pose calibration. Mirrors core/retarget.py.
#pragma once

#include "itu/aligner.hpp"
#include "itu/types.hpp"

#include <Eigen/Core>
#include <array>
#include <deque>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace itu {

enum class Side { Left, Right };
inline const char* side_name(Side s) { return s == Side::Left ? "left" : "right"; }
inline const char* side_prefix(Side s) { return s == Side::Left ? "l" : "r"; }

class SingularConfigurationError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct RobotGeometry {
  double upper_arm_length;
  double lower_arm_length;
  std::array<double, 3> shoulder_offset;
};

struct Calibration {
  double operator_arm_length = 0.0;
  std::map<std::string, double> rest_offsets;

  // Average shoulder->wrist distance across both arms over held frames.
  static Calibration from_tpose_frames(const std::vector<AlignedFrame>& frames);
};

// Map one operator arm onto robot joint angles (radians).
JointMap retarget_arm(const AlignedFrame& aligned, Side side, const RobotGeometry& robot,
                      const Calibration& calibration, bool decouple_pitch_elbow = false);

// Both arms + torso yaw / head pitch, with per-joint passthrough defaults.
JointMap retarget_full_upper_body(const AlignedFrame& aligned, const RobotGeometry& robot,
                                  const Calibration& calibration,
                                  bool decouple_pitch_elbow = false);

double estimate_arm_length(const KeypointMap& keypoints);

// Buffers AlignedFrames and finalises a Calibration (incl. auto rest-offsets).
class CalibrationCollector {
 public:
  explicit CalibrationCollector(int target_frames = 30) : target_(target_frames) {}
  void push(const AlignedFrame& frame);
  bool ready() const { return static_cast<int>(buf_.size()) >= target_; }
  // robot present -> also compute per-joint circular-mean rest_offsets.
  Calibration finalise(const std::optional<RobotGeometry>& robot = std::nullopt,
                       bool decouple_pitch_elbow = false);

 private:
  int target_;
  std::deque<AlignedFrame> buf_;
};

}  // namespace itu
