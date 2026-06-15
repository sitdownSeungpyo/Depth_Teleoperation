#include "itu/depth_lift.hpp"

#include "itu/mathutil.hpp"

#include <algorithm>
#include <cmath>

namespace itu {

namespace {
constexpr double kEps = 1e-6;

// Push with a max-length cap (mirrors collections.deque(maxlen=...)).
void push_capped(std::deque<double>& d, double v, int maxlen) {
  d.push_back(v);
  while (static_cast<int>(d.size()) > maxlen) d.pop_front();
}
void push_capped(std::deque<Eigen::Vector3d>& d, const Eigen::Vector3d& v, int maxlen) {
  d.push_back(v);
  while (static_cast<int>(d.size()) > maxlen) d.pop_front();
}
double median_deque(const std::deque<double>& d) {
  return mathutil::median(std::vector<double>(d.begin(), d.end()));
}
}  // namespace

const std::vector<Segment>& arm_segments() {
  static const std::vector<Segment> kSegments = {
      {"left_shoulder", "left_elbow"},
      {"left_elbow", "left_wrist"},
      {"right_shoulder", "right_elbow"},
      {"right_elbow", "right_wrist"},
  };
  return kSegments;
}

double foreground_depth(const DepthImageView& depth, int px, int py, int window,
                        double foreground_percentile, double depth_scale, double depth_max_m) {
  const int h = depth.height, w = depth.width;
  const int r = window / 2;
  const int x0 = std::max(0, px - r), x1 = std::min(w, px + r + 1);
  const int y0 = std::max(0, py - r), y1 = std::min(h, py + r + 1);
  std::vector<double> valid;
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      const std::uint16_t raw = depth.at(y, x);
      if (raw == 0) continue;
      const double val = static_cast<double>(raw) * depth_scale;
      if (val > 0.0 && val <= depth_max_m) valid.push_back(val);
    }
  }
  if (valid.empty()) return 0.0;
  const double thresh = mathutil::percentile(valid, foreground_percentile);
  std::vector<double> near;
  for (double v : valid)
    if (v <= thresh) near.push_back(v);
  if (!near.empty()) return mathutil::median(std::move(near));
  return mathutil::median(std::move(valid));
}

RobustDepthLifter::RobustDepthLifter(double depth_scale, double depth_max_m, int window,
                                     double foreground_percentile, double max_jump_m,
                                     int max_stale_frames)
    : depth_scale_(depth_scale),
      depth_max_m_(depth_max_m),
      foreground_percentile_(foreground_percentile),
      max_jump_m_(max_jump_m),
      window_(window),
      max_stale_frames_(max_stale_frames) {}

std::optional<double> RobustDepthLifter::lift(const std::string& name, const DepthImageView& depth,
                                              int px, int py) {
  const double d = foreground_depth(depth, px, py, window_, foreground_percentile_, depth_scale_,
                                    depth_max_m_);
  auto pit = last_.find(name);
  const bool has_prev = pit != last_.end();
  const bool fresh = has_prev && (frame_ - pit->second.second) <= max_stale_frames_;

  if (d <= 0.0) {
    if (fresh) return pit->second.first;
    return std::nullopt;
  }

  if (fresh && std::abs(d - pit->second.first) > max_jump_m_) {
    auto pend = pending_.find(name);
    if (pend != pending_.end() && std::abs(d - pend->second) <= max_jump_m_) {
      last_[name] = {d, frame_};
      pending_.erase(name);
      return d;
    }
    pending_[name] = d;
    return pit->second.first;
  }

  last_[name] = {d, frame_};
  pending_.erase(name);
  return d;
}

BoneLengthStabilizer::BoneLengthStabilizer(std::vector<Segment> segments, int history,
                                           double ratio_min, double ratio_max)
    : segments_(std::move(segments)), ratio_min_(ratio_min), ratio_max_(ratio_max),
      history_(history) {
  for (const auto& seg : segments_) hist_[seg];  // ensure entry exists
}

KeypointMap BoneLengthStabilizer::operator()(const KeypointMap& keypoints) {
  KeypointMap out = keypoints;
  for (const auto& seg : segments_) {
    const auto& parent = seg.first;
    const auto& child = seg.second;
    auto pit = keypoints.find(parent);
    auto cit = keypoints.find(child);
    if (pit == keypoints.end() || cit == keypoints.end()) continue;
    if (pit->second.norm() < kEps || cit->second.norm() < kEps) continue;
    const Eigen::Vector3d vv = cit->second - pit->second;
    const double measured = vv.norm();
    if (measured < kEps) continue;
    push_capped(hist_[seg], measured, history_);
    const double ref = median_deque(hist_[seg]);
    const double ratio = std::min(std::max(ref / measured, ratio_min_), ratio_max_);
    out[child] = out[parent] + vv * ratio;  // chain via stabilized parent
  }
  return out;
}

TemporalMedianFilter::TemporalMedianFilter(int window) : w_(std::max(1, window)) {}

KeypointMap TemporalMedianFilter::operator()(const KeypointMap& keypoints) {
  KeypointMap out = keypoints;
  for (const auto& [name, pos] : keypoints) {
    if (pos.norm() < kEps) {
      hist_.erase(name);  // rejected — reset so no stale median
      continue;
    }
    push_capped(hist_[name], pos, w_);
    std::vector<Eigen::Vector3d> stack(hist_[name].begin(), hist_[name].end());
    out[name] = mathutil::median_vec3(stack);
  }
  return out;
}

SegmentConsistencyGate::SegmentConsistencyGate(std::vector<Segment> segments, int history,
                                               double ratio_tol, int confirm_frames,
                                               int min_history)
    : segments_(std::move(segments)),
      ratio_tol_(ratio_tol),
      confirm_(std::max(1, confirm_frames)),
      min_history_(std::max(1, min_history)),
      history_(history) {
  for (const auto& seg : segments_) len_hist_[seg];
}

KeypointMap SegmentConsistencyGate::operator()(const KeypointMap& keypoints) {
  KeypointMap out = keypoints;
  for (const auto& seg : segments_) {
    const auto& parent = seg.first;
    const auto& child = seg.second;
    auto pit = out.find(parent);          // stabilized parent (from out)
    auto cit = keypoints.find(child);     // original child
    if (pit == out.end() || cit == keypoints.end()) continue;
    const Eigen::Vector3d p = pit->second;
    const Eigen::Vector3d c = cit->second;
    if (p.norm() < kEps || c.norm() < kEps) continue;
    const Eigen::Vector3d vv = c - p;
    const double length = vv.norm();
    if (length < kEps) continue;

    auto& hist = len_hist_[seg];
    if (static_cast<int>(hist.size()) < min_history_) {  // warm-up: seed unconditionally
      push_capped(hist, length, history_);
      offset_[seg] = vv;
      pending_.erase(seg);
      out[child] = p + vv;
      continue;
    }
    const double ref = median_deque(hist);
    if (ref < kEps || std::abs(length - ref) / ref <= ratio_tol_) {
      push_capped(hist, length, history_);  // plausible -> accept
      offset_[seg] = vv;
      pending_.erase(seg);
      out[child] = p + vv;
      continue;
    }
    // Implausible length: hold unless several consecutive frames agree.
    ++rejected_;
    auto pend = pending_.find(seg);
    if (pend != pending_.end() && (vv - pend->second.first).norm() <= ratio_tol_ * ref) {
      const int count = pend->second.second + 1;
      if (count >= confirm_) {
        push_capped(hist, length, history_);
        offset_[seg] = vv;
        pending_.erase(seg);
        out[child] = p + vv;  // confirmed genuine change
        continue;
      }
      pending_[seg] = {vv, count};
    } else {
      pending_[seg] = {vv, 1};
    }
    auto oit = offset_.find(seg);
    out[child] = p + (oit != offset_.end() ? oit->second : vv);  // hold last good offset
  }
  return out;
}

}  // namespace itu
