// Filter & Limiter — OneEuro (Casiez 2012) per joint/keypoint + clamps.
// Mirrors core/filter.py.
#pragma once

#include "itu/types.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace itu {

struct OneEuroParams {
  double min_cutoff = 1.0;
  double beta = 0.05;
  double d_cutoff = 1.0;
};

// Single-channel One Euro filter.
class OneEuroFilter {
 public:
  explicit OneEuroFilter(OneEuroParams params = {}) : p_(params) {}
  void reset();
  double update(double x, double t);

 private:
  static double alpha(double cutoff, double dt);
  OneEuroParams p_;
  std::optional<double> x_prev_;
  double dx_prev_ = 0.0;
  std::optional<double> t_prev_;
};

// OneEuro per (keypoint, coordinate). Rejected (conf≈0) keypoints pass through
// untouched and reset their filter state.
class KeypointSmoother {
 public:
  explicit KeypointSmoother(OneEuroParams params) : params_(params) {}
  SkeletonFrame smooth(const SkeletonFrame& frame);
  void reset() { filters_.clear(); }

 private:
  OneEuroParams params_;
  std::map<std::string, std::vector<OneEuroFilter>> filters_;
};

struct JointLimits {
  double soft_min;
  double soft_max;
  double max_velocity;
};

struct JointLimiterConfig {
  std::map<std::string, JointLimits> limits;
  double velocity_violation_factor = 5.0;
};

// Per-joint OneEuro + soft limits + per-step velocity clamp + NaN guard.
class FilterAndLimiter {
 public:
  FilterAndLimiter(OneEuroParams one_euro, JointLimiterConfig limiter)
      : one_euro_(one_euro), limiter_(std::move(limiter)) {}

  JointCommand operator()(const JointMap& raw_positions, double timestamp,
                          double source_frame_ts);

 private:
  OneEuroFilter& filter_for(const std::string& joint);
  OneEuroParams one_euro_;
  JointLimiterConfig limiter_;
  std::map<std::string, OneEuroFilter> filters_;
  std::optional<JointCommand> last_;
};

// Build a JointLimiterConfig from per-joint mechanical (min,max) bounds.
JointLimiterConfig default_limits_from_mechanical(
    const std::map<std::string, std::pair<double, double>>& mechanical, double factor,
    double max_velocity);

}  // namespace itu
