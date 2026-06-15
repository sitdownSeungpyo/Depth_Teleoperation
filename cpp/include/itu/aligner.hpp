// Frame Aligner — operator torso basis + camera->torso projection (spec §4.2).
// Mirrors core/aligner.py.
#pragma once

#include "itu/types.hpp"

#include <Eigen/Core>
#include <optional>
#include <stdexcept>

namespace itu {

// Raised when the operator basis cannot be constructed (rank-deficient).
class AlignmentError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct AlignedFrame {
  KeypointMap keypoints;
  Eigen::Matrix3d rotation;  // 3x3, camera -> torso
  Eigen::Vector3d rpy;       // (roll, pitch, yaw) of operator torso (radians)
};

// YXZ intrinsic Euler angles as (roll, pitch, yaw). Mirrors _rpy_from_rotation.
Eigen::Vector3d rpy_from_rotation(const Eigen::Matrix3d& r);

// Express a SkeletonFrame in the operator's torso frame.
// `gravity_up` (optional): fixed torso-up direction (gravity-aligned mode);
// when absent, up is computed from head - mid_hip (or head - mid_shoulder when
// hips are missing/zero-confidence). Throws AlignmentError on degenerate pose.
AlignedFrame align_to_torso(const SkeletonFrame& frame,
                            std::optional<Eigen::Vector3d> gravity_up = std::nullopt);

}  // namespace itu
