// Interpolating publisher base — fixed-rate linear interpolation between the two
// most recent setpoints on a worker thread. Mirrors publisher/base.py.
#pragma once

#include "itu/publisher.hpp"
#include "itu/types.hpp"

#include <atomic>
#include <mutex>
#include <optional>
#include <thread>

namespace itu {

inline constexpr double kStaleInputThresholdS = 0.2;

// Monotonic seconds (matches Python time.perf_counter semantics for the loop).
double perf_now_seconds();

class InterpolatingPublisherBase : public Publisher {
 public:
  explicit InterpolatingPublisherBase(int rate_hz = 100);
  ~InterpolatingPublisherBase() override;

  void start() override;
  void stop() override;
  void set_target(const JointCommand& command) override;

  std::optional<JointCommand> current();
  int rate_hz() const { return rate_hz_; }

  // Pure interpolation at time `now` (seconds). Public so it can be golden-tested
  // independently of the worker thread.
  std::optional<JointCommand> interpolate(double now);

 protected:
  // Subclasses send the interpolated command elsewhere (UDP, mock log, sim).
  virtual void emit(const JointCommand& command) = 0;

  // Join the worker thread without virtual dispatch — safe to call from dtors.
  void stop_internal();

 private:
  void loop();

  double dt_;
  int rate_hz_;
  std::mutex mtx_;
  std::optional<JointCommand> prev_;
  std::optional<JointCommand> next_;
  std::optional<JointCommand> latest_emit_;
  bool prev_is_next_ = false;  // mirrors Python's `prev is nxt` identity check
  bool stale_warned_ = false;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  bool running_ = false;
};

}  // namespace itu
