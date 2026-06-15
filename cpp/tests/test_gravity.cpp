#include "itu/gravity.hpp"

#include "golden_common.hpp"

#include <gtest/gtest.h>

using namespace itu;

TEST(Gravity, EstimatorGolden) {
  LOAD_GOLDEN_OR_SKIP("gravity.json", data);
  const auto& p = data["params"];
  GravityEstimator est(p["lpf_alpha"].get<double>(), p["warmup_frames"].get<int>(),
                       p["norm_tol"].get<double>(), p["axis_sign"].get<double>());
  const auto& samples = data["samples"];
  const auto& expected = data["expected_up"];
  for (std::size_t i = 0; i < samples.size(); ++i) {
    auto up = est.update(golden::vec3(samples[i]));
    if (expected[i].is_null()) {
      EXPECT_FALSE(up.has_value()) << "sample " << i;
    } else {
      ASSERT_TRUE(up.has_value()) << "sample " << i;
      EXPECT_NEAR((*up - golden::vec3(expected[i])).norm(), 0.0, 1e-9) << "sample " << i;
    }
  }
}

TEST(Gravity, TiltDegrees) {
  EXPECT_NEAR(tilt_degrees({0, -1, 0}, {0, -1, 0}), 0.0, 1e-9);
  EXPECT_NEAR(tilt_degrees({1, 0, 0}, {0, -1, 0}), 90.0, 1e-6);
}
