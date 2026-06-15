// Dynamixel helpers — radian↔unit conversion + servo identity. Mirrors the pure
// parts of publisher/dynamixel_publisher.py. The SDK-driven DynamixelPublisher
// (hardware) is gated behind ITU_BUILD_DYNAMIXEL and added later.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>

namespace itu {

inline constexpr int kDxlResolution = 4096;  // units per revolution
inline constexpr int kDxlCenter = 2048;      // unit at angle 0

namespace detail {
// Python's round() is banker's rounding (round half to even); replicate it so the
// unit conversion matches the reference at exact .5 ties.
inline double py_round(double x) {
  const double f = std::floor(x);
  const double diff = x - f;
  if (diff < 0.5) return f;
  if (diff > 0.5) return f + 1.0;
  return (std::fmod(f, 2.0) == 0.0) ? f : f + 1.0;  // tie -> even
}
}  // namespace detail

// Convert a radian angle to a DXL position unit, clamped to [0, 4095].
inline int angle_rad_to_dxl_unit(double angle_rad) {
  constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
  const int unit =
      static_cast<int>(detail::py_round(kDxlCenter + angle_rad * kDxlResolution / kTwoPi));
  return std::max(0, std::min(unit, kDxlResolution - 1));
}

struct ServoSpec {
  int id;
  std::string model;  // "MX-64", "MX-28", "XL430"
};

}  // namespace itu
