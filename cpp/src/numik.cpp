#include "itu/numik.hpp"

#include <mujoco/mujoco.h>

#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace itu {

namespace {
// Strip a trailing "_joint" so a config keyed by canonical name still matches.
std::string canonical(const std::string& n) {
  const std::string suffix = "_joint";
  if (n.size() > suffix.size() && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0)
    return n.substr(0, n.size() - suffix.size());
  return n;
}
}  // namespace

ArmPositionIK::ArmPositionIK(const mjModel* model, std::vector<std::string> joint_names,
                             const std::string& shoulder_body, const std::string& elbow_body,
                             const std::string& wrist_body, double damping, int max_iters,
                             double pos_tol, double step_clip,
                             const std::map<std::string, std::pair<double, double>>* joint_limits)
    : model_(model),
      joint_names_(std::move(joint_names)),
      damping_(damping),
      max_iters_(max_iters),
      pos_tol_(pos_tol),
      step_clip_(step_clip) {
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<std::string> missing;
  for (const auto& n : joint_names_) {
    const int jid = mj_name2id(model, mjOBJ_JOINT, n.c_str());
    if (jid < 0) {
      missing.push_back(n);
      continue;
    }
    qadr_.push_back(model->jnt_qposadr[jid]);
    dof_.push_back(model->jnt_dofadr[jid]);
    if (model->jnt_limited[jid]) {
      lo_.push_back(model->jnt_range[2 * jid + 0]);
      hi_.push_back(model->jnt_range[2 * jid + 1]);
    } else {
      lo_.push_back(-inf);
      hi_.push_back(inf);
    }
  }
  if (!missing.empty()) {
    std::string msg = "joints not found in model:";
    for (const auto& m : missing) msg += " " + m;
    throw std::invalid_argument(msg);
  }
  // Dedicated joint-limit config overrides the model's jnt_range.
  if (joint_limits != nullptr) {
    for (std::size_t i = 0; i < joint_names_.size(); ++i) {
      const std::string& full = joint_names_[i];
      auto it = joint_limits->find(full);
      if (it == joint_limits->end()) it = joint_limits->find(canonical(full));
      if (it != joint_limits->end()) {
        lo_[i] = it->second.first;
        hi_[i] = it->second.second;
      }
    }
  }
  sb_ = mj_name2id(model, mjOBJ_BODY, shoulder_body.c_str());
  eb_ = mj_name2id(model, mjOBJ_BODY, elbow_body.c_str());
  wb_ = mj_name2id(model, mjOBJ_BODY, wrist_body.c_str());
  if (std::min({sb_, eb_, wb_}) < 0)
    throw std::invalid_argument("shoulder/elbow/wrist body not found in model");
}

Eigen::Vector3d ArmPositionIK::body_pos(const mjData* data, const std::string& which) const {
  int bid = sb_;
  if (which == "elbow") bid = eb_;
  else if (which == "wrist") bid = wb_;
  return Eigen::Vector3d(data->xpos[3 * bid + 0], data->xpos[3 * bid + 1], data->xpos[3 * bid + 2]);
}

JointMap ArmPositionIK::solve(mjData* data, const Eigen::Vector3d& elbow_target,
                              const Eigen::Vector3d& wrist_target) const {
  const int nv = model_->nv;
  const int n = static_cast<int>(joint_names_.size());
  std::vector<mjtNum> jp(static_cast<std::size_t>(3) * nv);
  const Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();

  for (int iter = 0; iter < max_iters_; ++iter) {
    mj_forward(model_, data);
    const Eigen::Vector3d pe(data->xpos[3 * eb_ + 0], data->xpos[3 * eb_ + 1],
                             data->xpos[3 * eb_ + 2]);
    const Eigen::Vector3d pw(data->xpos[3 * wb_ + 0], data->xpos[3 * wb_ + 1],
                             data->xpos[3 * wb_ + 2]);
    Eigen::Matrix<double, 6, 1> err;
    err.head<3>() = elbow_target - pe;
    err.tail<3>() = wrist_target - pw;
    if (err.norm() < pos_tol_) break;

    Eigen::MatrixXd jac(6, n);
    mj_jacBody(model_, data, jp.data(), nullptr, eb_);
    for (int r = 0; r < 3; ++r)
      for (int k = 0; k < n; ++k) jac(r, k) = jp[static_cast<std::size_t>(r) * nv + dof_[k]];
    mj_jacBody(model_, data, jp.data(), nullptr, wb_);
    for (int r = 0; r < 3; ++r)
      for (int k = 0; k < n; ++k) jac(3 + r, k) = jp[static_cast<std::size_t>(r) * nv + dof_[k]];

    // dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
    const Eigen::Matrix<double, 6, 6> a = jac * jac.transpose() + (damping_ * damping_) * I6;
    const Eigen::Matrix<double, 6, 1> y = a.partialPivLu().solve(err);
    Eigen::VectorXd dq = jac.transpose() * y;
    for (int k = 0; k < n; ++k) {
      const double d = std::clamp(dq[k], -step_clip_, step_clip_);
      const double q = std::clamp(data->qpos[qadr_[k]] + d, lo_[k], hi_[k]);
      data->qpos[qadr_[k]] = q;
    }
  }
  // Final command clamp — guarantee returned/applied angles are within [lo, hi].
  for (int k = 0; k < n; ++k)
    data->qpos[qadr_[k]] = std::clamp(data->qpos[qadr_[k]], lo_[k], hi_[k]);
  mj_forward(model_, data);

  JointMap out;
  for (int k = 0; k < n; ++k) out[joint_names_[k]] = data->qpos[qadr_[k]];
  return out;
}

}  // namespace itu
