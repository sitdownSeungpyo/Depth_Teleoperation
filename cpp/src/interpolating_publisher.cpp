#include "itu/interpolating_publisher.hpp"

#include <algorithm>
#include <chrono>

#ifdef _WIN32
#define NOMINMAX  // keep windows.h from defining min/max macros (breaks std::max)
#include <windows.h>
#pragma comment(lib, "winmm.lib")
#endif

namespace itu {

double perf_now_seconds() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

namespace {
void raise_windows_timer_resolution() {
#ifdef _WIN32
  timeBeginPeriod(1);  // best-effort: 1 ms scheduler tick for tighter sleeps
#endif
}
}  // namespace

InterpolatingPublisherBase::InterpolatingPublisherBase(int rate_hz)
    : dt_(1.0 / rate_hz), rate_hz_(rate_hz) {}

InterpolatingPublisherBase::~InterpolatingPublisherBase() { stop_internal(); }

void InterpolatingPublisherBase::start() {
  if (running_) return;
  raise_windows_timer_resolution();
  stop_ = false;
  thread_ = std::thread(&InterpolatingPublisherBase::loop, this);
  running_ = true;
}

void InterpolatingPublisherBase::stop_internal() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
  running_ = false;
}

void InterpolatingPublisherBase::stop() { stop_internal(); }

void InterpolatingPublisherBase::set_target(const JointCommand& command) {
  std::lock_guard<std::mutex> lk(mtx_);
  if (!next_.has_value()) {
    prev_ = command;
    prev_is_next_ = true;
  } else {
    prev_ = next_;
    prev_is_next_ = false;
  }
  next_ = command;
  stale_warned_ = false;
}

std::optional<JointCommand> InterpolatingPublisherBase::current() {
  std::lock_guard<std::mutex> lk(mtx_);
  return latest_emit_;
}

std::optional<JointCommand> InterpolatingPublisherBase::interpolate(double now) {
  std::optional<JointCommand> prev, nxt;
  bool stale_warned, prev_is_next;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    prev = prev_;
    nxt = next_;
    stale_warned = stale_warned_;
    prev_is_next = prev_is_next_;
  }
  if (!nxt.has_value()) return std::nullopt;

  auto hold = [&]() {
    JointCommand c;
    c.timestamp = now;
    c.positions = nxt->positions;
    c.source_frame_ts = nxt->source_frame_ts;
    return c;
  };

  if (!prev.has_value() || prev_is_next) return hold();

  const double span = std::max(nxt->timestamp - prev->timestamp, 1e-6);
  const double age = now - nxt->timestamp;
  if (age > kStaleInputThresholdS) {
    if (!stale_warned) {
      std::lock_guard<std::mutex> lk(mtx_);
      stale_warned_ = true;
    }
    return hold();
  }

  const double u = std::clamp((now - prev->timestamp) / span, 0.0, 1.0);
  JointCommand c;
  c.timestamp = now;
  c.source_frame_ts = nxt->source_frame_ts;
  for (const auto& [joint, value] : nxt->positions) {
    auto it = prev->positions.find(joint);
    c.positions[joint] =
        (it != prev->positions.end()) ? it->second * (1.0 - u) + value * u : value;
  }
  return c;
}

void InterpolatingPublisherBase::loop() {
  using namespace std::chrono;
  auto next_tick = steady_clock::now();
  const auto dt = duration_cast<steady_clock::duration>(duration<double>(dt_));
  while (!stop_.load()) {
    const double now = perf_now_seconds();
    auto cmd = interpolate(now);
    if (cmd.has_value()) {
      emit(*cmd);
      std::lock_guard<std::mutex> lk(mtx_);
      latest_emit_ = cmd;
    }
    next_tick += dt;
    const auto now_tp = steady_clock::now();
    if (next_tick > now_tp) {
      std::this_thread::sleep_until(next_tick);
    } else {
      next_tick = now_tp;  // behind schedule; reset baseline
    }
  }
}

}  // namespace itu
