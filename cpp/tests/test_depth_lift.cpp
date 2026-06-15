#include "itu/depth_lift.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace itu;

namespace {
// Build a small depth frame (row-major uint16, raw units).
DepthImageView make_view(std::vector<std::uint16_t>& buf, int w, int h) {
  return DepthImageView{buf.data(), h, w};
}
}  // namespace

TEST(DepthLift, ForegroundDepthPicksNearCluster) {
  const int w = 5, h = 5;
  // Center 3x3 = near body (1000 units, 9 px = 36%), surround = far background
  // (3000, 16 px). The foreground percentile must sit below the near fraction to
  // isolate the body: at p=30 the threshold lands in the near cluster, so the
  // result is the body depth (1.0 m) — whereas a plain median would return the
  // far background (3.0 m). That contrast is exactly what foreground_depth fixes.
  std::vector<std::uint16_t> buf(w * h, 3000);
  for (int y = 1; y <= 3; ++y)
    for (int x = 1; x <= 3; ++x) buf[y * w + x] = 1000;
  DepthImageView view = make_view(buf, w, h);
  const double d = foreground_depth(view, 2, 2, 5, 30.0, 0.001, 4.0);
  EXPECT_NEAR(d, 1.0, 1e-9);  // 1000 * 0.001 m (near cluster, background rejected)
}

TEST(DepthLift, ForegroundDepthZeroWhenAllInvalid) {
  const int w = 3, h = 3;
  std::vector<std::uint16_t> buf(w * h, 0);
  DepthImageView view = make_view(buf, w, h);
  EXPECT_EQ(foreground_depth(view, 1, 1, 3, 40.0, 0.001, 4.0), 0.0);
}

TEST(DepthLift, RobustLifterHoleFillUsesLastValid) {
  RobustDepthLifter lifter(0.001, 4.0, /*window=*/3, 40.0, /*max_jump=*/0.25, /*stale=*/5);
  const int w = 3, h = 3;
  std::vector<std::uint16_t> good(w * h, 1000);
  std::vector<std::uint16_t> hole(w * h, 0);
  DepthImageView gv{good.data(), h, w};
  DepthImageView hv{hole.data(), h, w};

  lifter.begin_frame();
  auto d0 = lifter.lift("wrist", gv, 1, 1);
  ASSERT_TRUE(d0.has_value());
  EXPECT_NEAR(*d0, 1.0, 1e-9);

  lifter.begin_frame();
  auto d1 = lifter.lift("wrist", hv, 1, 1);  // hole -> reuse last valid
  ASSERT_TRUE(d1.has_value());
  EXPECT_NEAR(*d1, 1.0, 1e-9);
}

TEST(DepthLift, TemporalMedianRejectsLoneSpike) {
  TemporalMedianFilter tm(3);
  KeypointMap a{{"w", Eigen::Vector3d(1, 1, 1)}};
  KeypointMap b{{"w", Eigen::Vector3d(1, 1, 1.1)}};
  KeypointMap spike{{"w", Eigen::Vector3d(1, 1, 9.0)}};
  tm(a);
  tm(b);
  KeypointMap out = tm(spike);  // window {1.0, 1.1, 9.0} -> median z = 1.1
  EXPECT_NEAR(out["w"].z(), 1.1, 1e-9);
}
