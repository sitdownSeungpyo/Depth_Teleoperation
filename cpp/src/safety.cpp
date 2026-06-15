#include "itu/safety.hpp"

#include <algorithm>  // std::max, std::min
#include <chrono>
#include <cmath>

namespace itu {

namespace {
constexpr double kPi = 3.14159265358979323846;

double perf_counter_seconds() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}
}  // namespace

SafetyLayer::SafetyLayer(Publisher* publisher, SafetyConfig config, HotkeyBackend* hotkey,
                         std::function<double()> clock)
    : pub_(publisher), cfg_(std::move(config)), hotkey_(hotkey),
      clock_(clock ? std::move(clock) : std::function<double()>(&perf_counter_seconds)) {
  last_update_ = clock_();
}

void SafetyLayer::start() {
  if (hotkey_ != nullptr) hotkey_->start();
  pub_->start();
  last_update_ = clock_();
}

void SafetyLayer::stop() {
  try {
    pub_->stop();
  } catch (...) {
    if (hotkey_ != nullptr) hotkey_->stop();
    throw;
  }
  if (hotkey_ != nullptr) hotkey_->stop();
}

void SafetyLayer::trigger_estop(const std::string& /*reason*/) { estopped_ = true; }

bool SafetyLayer::deadman_held() {
  if (hotkey_ == nullptr) return true;  // tests / headless default to allowing flow
  return hotkey_->is_pressed(cfg_.deadman_key);
}

void SafetyLayer::check_estop_key() {
  if (hotkey_ != nullptr && hotkey_->is_pressed(cfg_.estop_key)) trigger_estop("estop hotkey");
}

JointCommand SafetyLayer::maybe_ramp(const JointCommand& command, double mean_confidence,
                                     double now) {
  if (mean_confidence < cfg_.confidence_threshold) {
    if (!loss_.started_at.has_value()) {
      loss_.started_at = now;
    } else if (now - *loss_.started_at > cfg_.loss_grace_period_s && !loss_.ramping) {
      loss_.ramping = true;
      loss_.ramp_start = now;
      loss_.ramp_from = command.positions;
    }
  } else {
    loss_ = LossState{};
  }

  if (!loss_.ramping) return command;

  const double elapsed = now - loss_.ramp_start;
  const double u = std::min(1.0, elapsed / std::max(cfg_.ramp_to_safe_s, 1e-3));
  const double e = 0.5 - 0.5 * std::cos(kPi * u);  // cosine ease-in/out

  auto get_or = [](const std::map<std::string, double>& m, const std::string& k, double dflt) {
    auto it = m.find(k);
    return it != m.end() ? it->second : dflt;
  };

  JointMap positions;
  for (const auto& [joint, cur] : command.positions) {
    const double from = get_or(loss_.ramp_from, joint, cur);
    const double to = get_or(cfg_.safe_pose, joint, cur);
    positions[joint] = from * (1.0 - e) + to * e;
  }
  JointCommand out;
  out.timestamp = command.timestamp;
  out.positions = std::move(positions);
  out.source_frame_ts = command.source_frame_ts;
  return out;
}

void SafetyLayer::update(const JointCommand& command, double mean_confidence) {
  const double now = clock_();
  check_estop_key();

  if (estopped_) {
    JointCommand zero;
    zero.timestamp = command.timestamp;
    for (const auto& [j, v] : command.positions) zero.positions[j] = 0.0;
    zero.source_frame_ts = command.source_frame_ts;
    pub_->set_target(zero);
    last_update_ = now;
    last_cmd_ = zero;
    return;
  }

  if (!deadman_held()) {
    if (last_cmd_.has_value()) pub_->set_target(*last_cmd_);
    last_update_ = now;
    return;
  }

  JointCommand cmd = maybe_ramp(command, mean_confidence, now);
  pub_->set_target(cmd);
  last_update_ = now;
  last_cmd_ = cmd;
}

void SafetyLayer::watchdog_tick() {
  const double now = clock_();
  const double budget = cfg_.cycle_dt_s * cfg_.watchdog_factor;
  if (now - last_update_ > budget) trigger_estop("watchdog");
}

}  // namespace itu
