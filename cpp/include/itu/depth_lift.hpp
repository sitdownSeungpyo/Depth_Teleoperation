// Robust depth lifting for 2D->3D keypoint estimation. Mirrors tracker/depth_lift.py.
// Core stays OpenCV-free: a uint16 depth frame is passed as a lightweight view
// (the device layer wraps a cv::Mat into this).
#pragma once

#include "itu/types.hpp"

#include <Eigen/Core>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace itu {

// Row-major uint16 depth image view (no ownership).
struct DepthImageView {
  const std::uint16_t* data = nullptr;
  int height = 0;
  int width = 0;
  std::uint16_t at(int y, int x) const { return data[y * width + x]; }
};

using Segment = std::pair<std::string, std::string>;

// Default upper-body arm chain.
const std::vector<Segment>& arm_segments();

// Foreground (body-surface) depth in metres at (px,py), or 0.0. Keeps the nearest
// `foreground_percentile`% of valid window depths and returns their median.
double foreground_depth(const DepthImageView& depth, int px, int py, int window,
                        double foreground_percentile, double depth_scale, double depth_max_m);

// Per-keypoint robust depth sampling with temporal hole-fill + spike rejection.
class RobustDepthLifter {
 public:
  RobustDepthLifter(double depth_scale, double depth_max_m, int window = 7,
                    double foreground_percentile = 40.0, double max_jump_m = 0.25,
                    int max_stale_frames = 5);
  void begin_frame() { ++frame_; }
  std::optional<double> lift(const std::string& name, const DepthImageView& depth, int px, int py);

 private:
  double depth_scale_, depth_max_m_, foreground_percentile_, max_jump_m_;
  int window_, max_stale_frames_;
  int frame_ = 0;
  std::map<std::string, std::pair<double, int>> last_;  // name -> (depth_m, frame_idx)
  std::map<std::string, double> pending_;               // name -> candidate
};

// Online running-median segment-length normalization for an arm chain.
class BoneLengthStabilizer {
 public:
  explicit BoneLengthStabilizer(std::vector<Segment> segments, int history = 60,
                                double ratio_min = 0.5, double ratio_max = 2.0);
  KeypointMap operator()(const KeypointMap& keypoints);

 private:
  std::vector<Segment> segments_;
  double ratio_min_, ratio_max_;
  int history_;
  std::map<Segment, std::deque<double>> hist_;
};

// Per-keypoint causal temporal median — kills lone-frame 3D spikes.
class TemporalMedianFilter {
 public:
  explicit TemporalMedianFilter(int window = 3);
  KeypointMap operator()(const KeypointMap& keypoints);

 private:
  int w_;
  std::map<std::string, std::deque<Eigen::Vector3d>> hist_;
};

// Reject implausible per-frame 3D depth by gating arm-segment LENGTH.
class SegmentConsistencyGate {
 public:
  explicit SegmentConsistencyGate(std::vector<Segment> segments, int history = 60,
                                  double ratio_tol = 0.35, int confirm_frames = 3,
                                  int min_history = 8);
  KeypointMap operator()(const KeypointMap& keypoints);
  int rejected() const { return rejected_; }

 private:
  std::vector<Segment> segments_;
  double ratio_tol_;
  int confirm_, min_history_, history_;
  std::map<Segment, std::deque<double>> len_hist_;
  std::map<Segment, Eigen::Vector3d> offset_;
  std::map<Segment, std::pair<Eigen::Vector3d, int>> pending_;
  int rejected_ = 0;
};

}  // namespace itu
