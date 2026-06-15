// Robot-model adapter — loads a MuJoCo model, measures link lengths, turns
// operator keypoints into elbow/wrist IK targets. Mirrors core/robot_model.py.
#pragma once

#include "itu/numik.hpp"
#include "itu/types.hpp"

#include <Eigen/Core>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace itu {

struct ArmConfig {
  std::vector<std::string> joints;
  std::string shoulder_body;
  std::string elbow_body;
  std::string wrist_body;
};

struct IkParams {
  double damping = 0.08;
  int max_iters = 16;
  double pos_tol = 2e-3;
  double step_clip = 0.35;
  double max_target_step_m = 0.0;
};

struct RobotConfig {
  std::string model_path;
  Eigen::Matrix3d operator_to_robot_R = Eigen::Matrix3d::Identity();
  IkParams ik;
  std::map<std::string, std::pair<double, double>> joint_limits;  // canonical/full -> (lo,hi)
  std::map<std::string, ArmConfig> arms;                          // "right" / "left"
};

class RobotModel {
 public:
  explicit RobotModel(const RobotConfig& cfg);
  ~RobotModel();
  RobotModel(const RobotModel&) = delete;
  RobotModel& operator=(const RobotModel&) = delete;

  // {canonical_joint_name: angle} over all model joints.
  JointMap joint_qpos() const;

  // Solve one arm. Operator points are in the torso frame (meters). Targets are
  // placed at the robot's own link lengths along the operator directions.
  // Returns nullopt if a direction is degenerate.
  std::optional<JointMap> solve_arm(const std::string& side, const Eigen::Vector3d& op_shoulder,
                                    const Eigen::Vector3d& op_elbow,
                                    const Eigen::Vector3d& op_wrist);

  std::map<std::string, std::pair<double, double>> link_lengths() const;

  const mjModel* model() const { return model_; }
  mjData* data() { return data_; }

 private:
  struct Arm {
    ArmPositionIK ik;
    double upper_len;
    double lower_len;
  };

  Eigen::Vector3d limit_step(const std::string& key, const Eigen::Vector3d& target);

  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  Eigen::Matrix3d frame_R_;
  double max_target_step_ = 0.0;
  std::map<std::string, std::pair<double, double>> joint_limits_;
  std::map<std::string, Arm> arms_;
  std::map<std::string, Eigen::Vector3d> last_targets_;
};

}  // namespace itu
