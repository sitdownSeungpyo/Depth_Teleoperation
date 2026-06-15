#include "itu/mathutil.hpp"

#include <Eigen/Dense>  // SVD + LU (determinant)
#include <algorithm>
#include <cmath>

namespace itu::mathutil {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

double median(std::vector<double> values) {
  const std::size_t n = values.size();
  if (n == 0) return 0.0;
  std::sort(values.begin(), values.end());
  if (n % 2 == 1) return values[n / 2];
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

double percentile(std::vector<double> values, double p) {
  const std::size_t n = values.size();
  if (n == 0) return 0.0;
  if (n == 1) return values[0];
  std::sort(values.begin(), values.end());
  const double rank = (p / 100.0) * static_cast<double>(n - 1);
  const double lo = std::floor(rank);
  const double frac = rank - lo;
  const auto i = static_cast<std::size_t>(lo);
  if (i + 1 >= n) return values[n - 1];
  return values[i] + frac * (values[i + 1] - values[i]);
}

double mean(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  double s = 0.0;
  for (double v : values) s += v;
  return s / static_cast<double>(values.size());
}

double circular_mean(const std::vector<double>& angles) {
  if (angles.empty()) return 0.0;
  double s = 0.0, c = 0.0;
  for (double a : angles) {
    s += std::sin(a);
    c += std::cos(a);
  }
  s /= static_cast<double>(angles.size());
  c /= static_cast<double>(angles.size());
  if (std::abs(s) < 1e-12 && std::abs(c) < 1e-12) return 0.0;
  return std::atan2(s, c);
}

Eigen::Vector3d median_vec3(const std::vector<Eigen::Vector3d>& points) {
  Eigen::Vector3d out = Eigen::Vector3d::Zero();
  if (points.empty()) return out;
  for (int k = 0; k < 3; ++k) {
    std::vector<double> col;
    col.reserve(points.size());
    for (const auto& p : points) col.push_back(p[k]);
    out[k] = median(std::move(col));
  }
  return out;
}

Eigen::Matrix3d closest_rotation(const Eigen::Matrix3d& m) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d u = svd.matrixU();
  const Eigen::Matrix3d& v = svd.matrixV();
  Eigen::Matrix3d r = u * v.transpose();
  if (r.determinant() < 0.0) {
    u.col(2) *= -1.0;
    r = u * v.transpose();
  }
  return r;
}

std::optional<Eigen::Vector3d> safe_normalize(const Eigen::Vector3d& v, double min_norm) {
  const double n = v.norm();
  if (n < min_norm) return std::nullopt;
  return v / n;
}

double safe_arccos(double x) {
  return std::acos(std::clamp(x, -1.0, 1.0));
}

double wrap_pi(double angle) {
  // Python: (angle + pi) % (2pi) - pi, with non-negative modulo for positive divisor.
  double m = std::fmod(angle + kPi, 2.0 * kPi);
  if (m < 0.0) m += 2.0 * kPi;
  return m - kPi;
}

}  // namespace itu::mathutil
