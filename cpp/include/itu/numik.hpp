// Numerical task-space IK (MuJoCo damped least squares). Mirrors core/numik.py.
// Drives an arm's elbow + wrist bodies to target positions via DLS on the actual
// robot model, so Jacobians come from mujoco (no hard-coded kinematics).
#pragma once

#include "itu/types.hpp"

#include <Eigen/Core>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct mjModel_;
struct mjData_;
typedef struct mjModel_ mjModel;
typedef struct mjData_ mjData;

namespace itu {

class ArmPositionIK {
 public:
  // joint_limits (optional): canonical/full joint name -> (lo, hi) in rad,
  // overriding the model's jnt_range.
  ArmPositionIK(const mjModel* model, std::vector<std::string> joint_names,
                const std::string& shoulder_body, const std::string& elbow_body,
                const std::string& wrist_body, double damping = 0.08, int max_iters = 16,
                double pos_tol = 2e-3, double step_clip = 0.35,
                const std::map<std::string, std::pair<double, double>>* joint_limits = nullptr);

  // Body origin position ("shoulder" | "elbow" | "wrist") from data->xpos.
  Eigen::Vector3d body_pos(const mjData* data, const std::string& which) const;

  // Iterate DLS from data->qpos (warm start) toward targets. Mutates data->qpos
  // for the arm joints and returns {joint_name: angle}.
  JointMap solve(mjData* data, const Eigen::Vector3d& elbow_target,
                 const Eigen::Vector3d& wrist_target) const;

  // Accessors used by RobotModel (elbow seed + warm-start clamp).
  const std::vector<std::string>& joint_names() const { return joint_names_; }
  const std::vector<int>& qadr() const { return qadr_; }
  const std::vector<double>& lo() const { return lo_; }
  const std::vector<double>& hi() const { return hi_; }

 private:
  const mjModel* model_;
  std::vector<std::string> joint_names_;
  std::vector<int> qadr_;
  std::vector<int> dof_;
  std::vector<double> lo_, hi_;
  int sb_, eb_, wb_;
  double damping_;
  int max_iters_;
  double pos_tol_, step_clip_;
};

}  // namespace itu
