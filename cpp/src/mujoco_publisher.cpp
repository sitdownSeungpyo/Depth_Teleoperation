#include "itu/mujoco_publisher.hpp"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace itu {

namespace {
std::string canonical(const std::string& n) {
  const std::string suffix = "_joint";
  if (n.size() > suffix.size() && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0)
    return n.substr(0, n.size() - suffix.size());
  return n;
}
}  // namespace

MuJoCoPublisher::MuJoCoPublisher(std::string model_path, int rate_hz, bool gui)
    : InterpolatingPublisherBase(rate_hz), model_path_(std::move(model_path)), gui_(gui) {}

MuJoCoPublisher::~MuJoCoPublisher() {
  stop_internal();
  if (data_ != nullptr) mj_deleteData(data_);
  if (model_ != nullptr) mj_deleteModel(model_);
}

void MuJoCoPublisher::load() {
  if (!std::filesystem::exists(model_path_))
    throw std::runtime_error("model not found: " + model_path_);
  char error[1024] = {0};
  model_ = mj_loadXML(model_path_.c_str(), nullptr, error, sizeof(error));
  if (model_ == nullptr) throw std::runtime_error(std::string("mj_loadXML failed: ") + error);
  data_ = mj_makeData(model_);

  // actuator index -> joint -> canonical name (via actuator_trnid).
  for (int a = 0; a < model_->nu; ++a) {
    const int jid = model_->actuator_trnid[2 * a + 0];
    const char* jname = mj_id2name(model_, mjOBJ_JOINT, jid);
    if (jname == nullptr) continue;
    actuator_idx_[canonical(jname)] = a;
  }

  // Steps per emit so the sim advances real-time per command.
  const double real_dt = 1.0 / rate_hz();
  const double ts = model_->opt.timestep;
  steps_per_emit_ = std::max(1, static_cast<int>(std::lround(real_dt / ts)));
}

void MuJoCoPublisher::start() {
  load();
  InterpolatingPublisherBase::start();
}

void MuJoCoPublisher::stop() { stop_internal(); }

void MuJoCoPublisher::emit(const JointCommand& command) {
  if (model_ == nullptr || data_ == nullptr) return;
  std::lock_guard<std::mutex> lk(sim_mtx_);
  for (const auto& [joint_name, target] : command.positions) {
    auto it = actuator_idx_.find(joint_name);
    if (it == actuator_idx_.end()) continue;
    data_->ctrl[it->second] = target;
  }
  for (int i = 0; i < steps_per_emit_; ++i) mj_step(model_, data_);
}

JointMap MuJoCoPublisher::current_joint_state() {
  JointMap out;
  if (model_ == nullptr || data_ == nullptr) return out;
  std::lock_guard<std::mutex> lk(sim_mtx_);
  for (int j = 0; j < model_->njnt; ++j) {
    const char* jname = mj_id2name(model_, mjOBJ_JOINT, j);
    if (jname == nullptr) continue;
    out[canonical(jname)] = data_->qpos[model_->jnt_qposadr[j]];
  }
  return out;
}

std::vector<std::string> MuJoCoPublisher::actuator_names() const {
  std::vector<std::string> names;
  for (const auto& [name, idx] : actuator_idx_) names.push_back(name);
  return names;
}

}  // namespace itu
