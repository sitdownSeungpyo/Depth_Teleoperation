// numpy-faithful numeric helpers. The golden tests pin these to NumPy's exact
// semantics (median averaging, percentile linear interpolation, circular mean,
// polar-decomposition rotation), so the C++ pipeline reproduces the Python
// reference bit-for-bit within tolerance.
#pragma once

#include <Eigen/Core>
#include <optional>
#include <vector>

namespace itu::mathutil {

// np.median over a flat list: sort; odd -> middle, even -> mean of two middles.
// Returns 0.0 for an empty input (callers guard emptiness explicitly).
double median(std::vector<double> values);

// np.percentile(values, p) with the default 'linear' interpolation method:
//   rank = p/100*(N-1); result = a[floor(rank)] + frac*(a[ceil]-a[floor]).
double percentile(std::vector<double> values, double p);

// np.mean.
double mean(const std::vector<double>& values);

// Circular mean of angles (radians): atan2(mean(sin), mean(cos)).
// Matches core.retarget._circular_mean, incl. the all-zero -> 0.0 guard.
double circular_mean(const std::vector<double>& angles);

// Per-coordinate median of a set of 3D points (np.median(stack, axis=0)).
Eigen::Vector3d median_vec3(const std::vector<Eigen::Vector3d>& points);

// Nearest rotation to m in Frobenius norm via SVD (R = U Vᵀ, det-flip guard).
// Matches core.aligner._closest_rotation.
Eigen::Matrix3d closest_rotation(const Eigen::Matrix3d& m);

// v normalized, or nullopt when ‖v‖ < min_norm. Matches _safe_normalize.
std::optional<Eigen::Vector3d> safe_normalize(const Eigen::Vector3d& v,
                                              double min_norm = 1e-6);

// arccos(clip(x, -1, 1)). Matches _safe_arccos.
double safe_arccos(double x);

// Wrap a scalar angle into (-π, π]. Matches _wrap_pi.
double wrap_pi(double angle);

}  // namespace itu::mathutil
