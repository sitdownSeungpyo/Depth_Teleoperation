// Safety Layer — deadman, E-stop, watchdog, ramp-to-safe. Mirrors core/safety.py.
#pragma once

#include "itu/publisher.hpp"
#include "itu/types.hpp"

#include <functional>
#include <map>
#include <optional>
#include <string>

namespace itu {

// Global hotkey backend interface (concrete Win32 impl lives in the device layer).
class HotkeyBackend {
 public:
  virtual ~HotkeyBackend() = default;
  virtual bool is_pressed(const std::string& key) = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
};

struct SafetyConfig {
  std::string deadman_key = "space";
  std::string estop_key = "esc";
  double confidence_threshold = 0.5;
  double loss_grace_period_s = 0.5;
  double ramp_to_safe_s = 1.5;
  int watchdog_factor = 3;  // cycles
  double cycle_dt_s = 1.0 / 30.0;
  std::map<std::string, double> safe_pose;
};

// Wraps a Publisher; can override or zero commands at any time.
class SafetyLayer {
 public:
  // hotkey may be null (headless / tests default to allowing command flow).
  // clock defaults to a monotonic perf-counter (seconds) when left empty.
  SafetyLayer(Publisher* publisher, SafetyConfig config, HotkeyBackend* hotkey = nullptr,
              std::function<double()> clock = {});

  void start();
  void stop();

  bool estopped() const { return estopped_; }
  void trigger_estop(const std::string& reason = "manual");
  void reset_estop() { estopped_ = false; }

  void update(const JointCommand& command, double mean_confidence = 1.0);
  void watchdog_tick();
  void note_alive() { last_update_ = clock_(); }

 private:
  struct LossState {
    std::optional<double> started_at;
    bool ramping = false;
    double ramp_start = 0.0;
    std::map<std::string, double> ramp_from;
  };

  bool deadman_held();
  void check_estop_key();
  JointCommand maybe_ramp(const JointCommand& command, double mean_confidence, double now);

  Publisher* pub_;
  SafetyConfig cfg_;
  HotkeyBackend* hotkey_;
  std::function<double()> clock_;
  bool estopped_ = false;
  double last_update_ = 0.0;
  LossState loss_;
  std::optional<JointCommand> last_cmd_;
};

}  // namespace itu
