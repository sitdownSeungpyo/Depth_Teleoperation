#include "itu/gravity.hpp"

#include "itu/mathutil.hpp"

#include <algorithm>
#include <cmath>

namespace itu {

namespace {
constexpr double kEps = 1e-9;
constexpr double kG = 9.80665;  // standard gravity (m/s²)
constexpr double kPi = 3.14159265358979323846;
}  // namespace

GravityEstimator::GravityEstimator(double lpf_alpha, int warmup_frames, double norm_tol,
                                   double axis_sign)
    : lpf_alpha_(lpf_alpha),
      warmup_frames_(warmup_frames),
      norm_tol_(norm_tol),
      axis_sign_(axis_sign) {}

std::optional<Eigen::Vector3d> GravityEstimator::update(const Eigen::Vector3d& accel_optical) {
  const double n = accel_optical.norm();
  if (n < kEps) {
    ++rejected_;
    return ready_value();
  }
  // Outlier gate: only a roughly-1g magnitude is trustworthy as gravity.
  if (std::abs(n - kG) / kG > norm_tol_) {
    ++rejected_;
    return ready_value();
  }
  Eigen::Vector3d u = (accel_optical / n) * axis_sign_;
  if (!up_.has_value()) {
    up_ = u;
  } else {
    Eigen::Vector3d blended = lpf_alpha_ * u + (1.0 - lpf_alpha_) * (*up_);
    auto norm = mathutil::safe_normalize(blended, kEps);
    if (norm.has_value()) up_ = *norm;  // else keep previous
  }
  ++accepted_;
  return ready_value();
}

double tilt_degrees(const Eigen::Vector3d& up, const Eigen::Vector3d& level_up) {
  auto a = mathutil::safe_normalize(up, kEps);
  auto b = mathutil::safe_normalize(level_up, kEps);
  if (!a.has_value() || !b.has_value()) return 0.0;
  const double d = std::clamp(a->dot(*b), -1.0, 1.0);
  return std::acos(d) * 180.0 / kPi;
}

}  // namespace itu
