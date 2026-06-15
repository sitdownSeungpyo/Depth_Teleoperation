#include "itu/aligner.hpp"

#include "itu/mathutil.hpp"

#include <Eigen/Geometry>  // Vector3d::cross
#include <algorithm>       // std::clamp
#include <cmath>

namespace itu {

Eigen::Vector3d rpy_from_rotation(const Eigen::Matrix3d& r) {
  // R = R_y(yaw) · R_x(pitch) · R_z(roll); +x across shoulders, +y up, +z fwd.
  const double pitch = std::asin(-std::clamp(r(1, 2), -1.0, 1.0));
  const double cp = std::cos(pitch);
  double roll, yaw;
  if (std::abs(cp) < 1e-6) {
    // Gimbal lock at pitch = ±π/2 — yaw underdetermined; fold into roll.
    roll = std::atan2(-r(0, 1), r(0, 0));
    yaw = 0.0;
  } else {
    roll = std::atan2(r(1, 0), r(1, 1));
    yaw = std::atan2(r(0, 2), r(2, 2));
  }
  return Eigen::Vector3d(roll, pitch, yaw);
}

AlignedFrame align_to_torso(const SkeletonFrame& frame, std::optional<Eigen::Vector3d> gravity_up) {
  for (const char* name : {"left_shoulder", "right_shoulder", "head"}) {
    if (frame.keypoints.find(name) == frame.keypoints.end())
      throw AlignmentError(std::string("missing required keypoint: ") + name);
  }

  const Eigen::Vector3d ls = frame.keypoints.at("left_shoulder");
  const Eigen::Vector3d rs = frame.keypoints.at("right_shoulder");
  const Eigen::Vector3d head = frame.keypoints.at("head");

  Eigen::Vector3d v;
  if (gravity_up.has_value()) {
    v = *gravity_up;  // gravity-aligned: ignore operator body tilt entirely.
  } else {
    auto conf = [&](const char* n) {
      auto it = frame.confidence.find(n);
      return it == frame.confidence.end() ? 0.0 : it->second;
    };
    const bool hips_present = frame.keypoints.count("left_hip") &&
                              frame.keypoints.count("right_hip") &&
                              conf("left_hip") > 0.0 && conf("right_hip") > 0.0;
    if (hips_present) {
      const Eigen::Vector3d mid_hip =
          0.5 * (frame.keypoints.at("left_hip") + frame.keypoints.at("right_hip"));
      v = head - mid_hip;
    } else {
      // Seated-operator fallback: head minus shoulder midpoint.
      const Eigen::Vector3d mid_shoulder = 0.5 * (ls + rs);
      v = head - mid_shoulder;
    }
  }

  const Eigen::Vector3d u = rs - ls;       // +x: across shoulders, right
  const Eigen::Vector3d w = u.cross(v);    // +z: forward (out of the chest)

  if (u.norm() < 1e-6 || v.norm() < 1e-6 || w.norm() < 1e-6)
    throw AlignmentError("operator basis is rank-deficient (degenerate pose)");

  Eigen::Matrix3d m;
  m.col(0) = u / u.norm();
  m.col(1) = v / v.norm();
  m.col(2) = w / w.norm();
  const Eigen::Matrix3d r = mathutil::closest_rotation(m);

  const Eigen::Matrix3d rt = r.transpose();
  AlignedFrame out;
  out.rotation = r;
  out.rpy = rpy_from_rotation(r);
  for (const auto& [name, p] : frame.keypoints) out.keypoints[name] = rt * p;
  return out;
}

}  // namespace itu
