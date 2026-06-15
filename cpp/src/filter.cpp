#include "itu/filter.hpp"

#include <Eigen/Core>
#include <algorithm>  // std::max, std::min
#include <cmath>

namespace itu {

namespace {
constexpr double kPi = 3.14159265358979323846;

// Add ±2π to `target` so its principal-value distance to `prev` is ≤ π.
double unwrap(double target, double prev) {
  const double diff = target - prev;
  if (diff > kPi) return target - 2.0 * kPi;
  if (diff < -kPi) return target + 2.0 * kPi;
  return target;
}
}  // namespace

double OneEuroFilter::alpha(double cutoff, double dt) {
  const double tau = 1.0 / (2.0 * kPi * cutoff);
  return 1.0 / (1.0 + tau / dt);
}

void OneEuroFilter::reset() {
  x_prev_.reset();
  dx_prev_ = 0.0;
  t_prev_.reset();
}

double OneEuroFilter::update(double x, double t) {
  if (!t_prev_.has_value() || !x_prev_.has_value()) {
    t_prev_ = t;
    x_prev_ = x;
    return x;
  }
  const double dt = std::max(t - *t_prev_, 1e-6);
  const double dx = (x - *x_prev_) / dt;
  const double a_d = alpha(p_.d_cutoff, dt);
  const double dx_hat = a_d * dx + (1.0 - a_d) * dx_prev_;
  const double cutoff = p_.min_cutoff + p_.beta * std::abs(dx_hat);
  const double a = alpha(cutoff, dt);
  const double x_hat = a * x + (1.0 - a) * *x_prev_;
  x_prev_ = x_hat;
  dx_prev_ = dx_hat;
  t_prev_ = t;
  return x_hat;
}

SkeletonFrame KeypointSmoother::smooth(const SkeletonFrame& frame) {
  SkeletonFrame out;
  out.timestamp = frame.timestamp;
  out.confidence = frame.confidence;
  for (const auto& [name, pos] : frame.keypoints) {
    double conf = 0.0;
    if (auto it = frame.confidence.find(name); it != frame.confidence.end()) conf = it->second;
    if (conf < 1e-6) {
      // Rejected keypoint: pass through + reset filter (avoid jump on recovery).
      filters_.erase(name);
      out.keypoints[name] = pos;
      continue;
    }
    auto it = filters_.find(name);
    if (it == filters_.end()) {
      it = filters_.emplace(name, std::vector<OneEuroFilter>{OneEuroFilter(params_),
                                                             OneEuroFilter(params_),
                                                             OneEuroFilter(params_)})
               .first;
    }
    Eigen::Vector3d v;
    for (int i = 0; i < 3; ++i) v[i] = it->second[i].update(pos[i], frame.timestamp);
    out.keypoints[name] = v;
  }
  return out;
}

OneEuroFilter& FilterAndLimiter::filter_for(const std::string& joint) {
  auto it = filters_.find(joint);
  if (it == filters_.end()) it = filters_.emplace(joint, OneEuroFilter(one_euro_)).first;
  return it->second;
}

JointCommand FilterAndLimiter::operator()(const JointMap& raw_positions_in, double timestamp,
                                          double source_frame_ts) {
  JointMap raw_positions = raw_positions_in;

  bool any_nonfinite = false;
  for (const auto& [j, v] : raw_positions) {
    if (!std::isfinite(v)) {
      any_nonfinite = true;
      break;
    }
  }
  if (any_nonfinite) {
    if (last_.has_value()) return *last_;
    for (auto& [j, v] : raw_positions) v = 0.0;
  }

  const JointMap* prev_positions = last_.has_value() ? &last_->positions : nullptr;
  const double prev_ts = last_.has_value() ? last_->timestamp : timestamp;
  const double dt = std::max(timestamp - prev_ts, 1e-6);

  // Unwrap atan2-style discontinuities against the previous command.
  if (prev_positions != nullptr) {
    for (auto& [joint, target] : raw_positions) {
      auto it = prev_positions->find(joint);
      if (it != prev_positions->end()) target = unwrap(target, it->second);
    }
  }

  JointMap out_positions;
  for (const auto& [joint, target_in] : raw_positions) {
    double target = target_in;
    const JointLimits* limits = nullptr;
    if (auto it = limiter_.limits.find(joint); it != limiter_.limits.end())
      limits = &it->second;

    // 5x velocity violation -> drop this joint's update, hold its previous value.
    if (prev_positions != nullptr && limits != nullptr) {
      auto pit = prev_positions->find(joint);
      if (pit != prev_positions->end()) {
        if (std::abs(target - pit->second) >
            limits->max_velocity * dt * limiter_.velocity_violation_factor) {
          out_positions[joint] = pit->second;
          continue;
        }
      }
    }

    double smoothed = filter_for(joint).update(target, timestamp);

    if (limits != nullptr) {
      if (prev_positions != nullptr) {
        auto pit = prev_positions->find(joint);
        if (pit != prev_positions->end()) {
          const double delta_max = limits->max_velocity * dt;
          smoothed = std::max(pit->second - delta_max,
                              std::min(smoothed, pit->second + delta_max));
        }
      }
      smoothed = std::max(limits->soft_min, std::min(smoothed, limits->soft_max));
    }
    out_positions[joint] = smoothed;
  }

  JointCommand cmd;
  cmd.timestamp = timestamp;
  cmd.positions = std::move(out_positions);
  cmd.source_frame_ts = source_frame_ts;
  last_ = cmd;
  return cmd;
}

JointLimiterConfig default_limits_from_mechanical(
    const std::map<std::string, std::pair<double, double>>& mechanical, double factor,
    double max_velocity) {
  JointLimiterConfig cfg;
  for (const auto& [joint, lohi] : mechanical) {
    cfg.limits[joint] = JointLimits{lohi.first * factor, lohi.second * factor, max_velocity};
  }
  return cfg;
}

}  // namespace itu
