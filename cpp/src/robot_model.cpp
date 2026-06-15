#include "itu/robot_model.hpp"

#include <mujoco/mujoco.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <stdexcept>

namespace itu {

namespace {
std::string canonical(const std::string& n) {
  const std::string suffix = "_joint";
  if (n.size() > suffix.size() && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0)
    return n.substr(0, n.size() - suffix.size());
  return n;
}

std::optional<Eigen::Vector3d> unit(const Eigen::Vector3d& v) {
  const double n = v.norm();
  if (n < 1e-9) return std::nullopt;
  return v / n;
}
}  // namespace

RobotModel::RobotModel(const RobotConfig& cfg)
    : frame_R_(cfg.operator_to_robot_R),
      max_target_step_(cfg.ik.max_target_step_m),
      joint_limits_(cfg.joint_limits) {
  if (!std::filesystem::exists(cfg.model_path))
    throw std::runtime_error("robot model not found: " + cfg.model_path);
  char error[1024] = {0};
  model_ = mj_loadXML(cfg.model_path.c_str(), nullptr, error, sizeof(error));
  if (model_ == nullptr) throw std::runtime_error(std::string("mj_loadXML failed: ") + error);
  data_ = mj_makeData(model_);
  mj_forward(model_, data_);  // rest pose for length measurement

  for (const auto& [side, c] : cfg.arms) {
    ArmPositionIK ik(model_, c.joints, c.shoulder_body, c.elbow_body, c.wrist_body, cfg.ik.damping,
                     cfg.ik.max_iters, cfg.ik.pos_tol, cfg.ik.step_clip, &joint_limits_);
    const Eigen::Vector3d sh = ik.body_pos(data_, "shoulder");
    const Eigen::Vector3d el = ik.body_pos(data_, "elbow");
    const Eigen::Vector3d wr = ik.body_pos(data_, "wrist");
    arms_.emplace(side, Arm{std::move(ik), (el - sh).norm(), (wr - el).norm()});
  }
}

RobotModel::~RobotModel() {
  if (data_ != nullptr) mj_deleteData(data_);
  if (model_ != nullptr) mj_deleteModel(model_);
}

JointMap RobotModel::joint_qpos() const {
  JointMap out;
  for (int j = 0; j < model_->njnt; ++j) {
    const char* jn = mj_id2name(model_, mjOBJ_JOINT, j);
    if (jn == nullptr) continue;
    out[canonical(jn)] = data_->qpos[model_->jnt_qposadr[j]];
  }
  return out;
}

std::optional<JointMap> RobotModel::solve_arm(const std::string& side,
                                              const Eigen::Vector3d& op_shoulder,
                                              const Eigen::Vector3d& op_elbow,
                                              const Eigen::Vector3d& op_wrist) {
  auto ait = arms_.find(side);
  if (ait == arms_.end()) throw std::invalid_argument("unknown arm side: " + side);
  Arm& arm = ait->second;

  auto du = unit(frame_R_ * (op_elbow - op_shoulder));
  auto df = unit(frame_R_ * (op_wrist - op_elbow));
  if (!du || !df) return std::nullopt;

  // Seed the elbow joint with the directly-observable flex angle (lifts the arm
  // off the straight-arm singularity so DLS converges).
  const double flex = std::acos(std::clamp(du->dot(*df), -1.0, 1.0));
  const int eadr = arm.ik.qadr().back();  // elbow joint listed last (base->tip)
  const double elo = arm.ik.lo().back();
  const double ehi = arm.ik.hi().back();
  data_->qpos[eadr] = std::clamp(flex, elo, ehi);

  const Eigen::Vector3d shoulder = arm.ik.body_pos(data_, "shoulder");
  const Eigen::Vector3d elbow_target = limit_step(side + "_elbow", shoulder + arm.upper_len * (*du));
  const Eigen::Vector3d wrist_target =
      limit_step(side + "_wrist", elbow_target + arm.lower_len * (*df));
  return arm.ik.solve(data_, elbow_target, wrist_target);
}

Eigen::Vector3d RobotModel::limit_step(const std::string& key, const Eigen::Vector3d& target_in) {
  Eigen::Vector3d target = target_in;
  if (max_target_step_ <= 0.0) return target;
  auto it = last_targets_.find(key);
  if (it != last_targets_.end()) {
    const Eigen::Vector3d delta = target - it->second;
    const double dist = delta.norm();
    if (dist > max_target_step_) target = it->second + delta * (max_target_step_ / dist);
  }
  last_targets_[key] = target;
  return target;
}

std::map<std::string, std::pair<double, double>> RobotModel::link_lengths() const {
  std::map<std::string, std::pair<double, double>> out;
  for (const auto& [side, arm] : arms_) out[side] = {arm.upper_len, arm.lower_len};
  return out;
}

}  // namespace itu
