#include "itu/mathutil.hpp"

#include "golden_common.hpp"

#include <gtest/gtest.h>

using namespace itu;

constexpr double kTol = 1e-9;

TEST(MathUtil, MedianGolden) {
  LOAD_GOLDEN_OR_SKIP("mathutil.json", data);
  for (const auto& c : data["median"]) {
    std::vector<double> in = c["input"].get<std::vector<double>>();
    EXPECT_NEAR(mathutil::median(in), c["expected"].get<double>(), kTol);
  }
}

TEST(MathUtil, PercentileGolden) {
  LOAD_GOLDEN_OR_SKIP("mathutil.json", data);
  for (const auto& c : data["percentile"]) {
    std::vector<double> in = c["input"].get<std::vector<double>>();
    EXPECT_NEAR(mathutil::percentile(in, c["p"].get<double>()), c["expected"].get<double>(), kTol);
  }
}

TEST(MathUtil, CircularMeanGolden) {
  LOAD_GOLDEN_OR_SKIP("mathutil.json", data);
  for (const auto& c : data["circular_mean"]) {
    std::vector<double> in = c["input"].get<std::vector<double>>();
    EXPECT_NEAR(mathutil::circular_mean(in), c["expected"].get<double>(), kTol);
  }
}

TEST(MathUtil, WrapPi) {
  EXPECT_NEAR(mathutil::wrap_pi(3.5), 3.5 - 2 * 3.14159265358979323846, 1e-12);
  EXPECT_NEAR(mathutil::wrap_pi(-3.5), -3.5 + 2 * 3.14159265358979323846, 1e-12);
  EXPECT_NEAR(mathutil::wrap_pi(0.0), 0.0, 1e-12);
}

TEST(MathUtil, ClosestRotationIsOrthonormal) {
  Eigen::Matrix3d m;
  m << 1.0, 0.1, 0.0, 0.0, 0.9, 0.2, 0.1, 0.0, 1.1;
  Eigen::Matrix3d r = mathutil::closest_rotation(m);
  EXPECT_NEAR((r * r.transpose() - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-9);
  EXPECT_NEAR(r.determinant(), 1.0, 1e-9);
}
