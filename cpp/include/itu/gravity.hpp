// IMU gravity estimation — accelerometer samples -> torso-up vector.
// Mirrors tracker/gravity.py (pure / hardware-free, unit-testable).
#pragma once

#include <Eigen/Core>
#include <optional>

namespace itu {

class GravityEstimator {
 public:
  // lpf_alpha: EMA weight for a new sample (small = heavy smoothing).
  // warmup_frames: accepted samples before update() returns a value.
  // norm_tol: accept a sample only when ||a|-g|/g <= norm_tol.
  // axis_sign: ±1 escape hatch for accel convention.
  explicit GravityEstimator(double lpf_alpha = 0.02, int warmup_frames = 10,
                            double norm_tol = 0.30, double axis_sign = 1.0);

  bool ready() const { return up_.has_value() && accepted_ >= warmup_frames_; }

  // Current smoothed up vector regardless of warm-up (nullopt before 1st sample).
  std::optional<Eigen::Vector3d> up() const { return up_; }

  // (accepted, rejected) sample counts — diagnostics.
  std::pair<int, int> stats() const { return {accepted_, rejected_}; }

  // Ingest one accel sample (optical frame); returns up if ready else nullopt.
  std::optional<Eigen::Vector3d> update(const Eigen::Vector3d& accel_optical);

 private:
  std::optional<Eigen::Vector3d> ready_value() const {
    return ready() ? up_ : std::nullopt;
  }

  double lpf_alpha_;
  int warmup_frames_;
  double norm_tol_;
  double axis_sign_;
  std::optional<Eigen::Vector3d> up_;
  int accepted_ = 0;
  int rejected_ = 0;
};

// Angle (deg) between a measured up vector and the assumed level-up.
double tilt_degrees(const Eigen::Vector3d& up, const Eigen::Vector3d& level_up);

}  // namespace itu
