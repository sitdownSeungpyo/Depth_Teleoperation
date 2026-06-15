#include "itu/filter.hpp"

#include "golden_common.hpp"

#include <gtest/gtest.h>

using namespace itu;

TEST(Filter, OneEuroGolden) {
  LOAD_GOLDEN_OR_SKIP("one_euro.json", data);
  const auto& p = data["params"];
  OneEuroParams params{p["min_cutoff"].get<double>(), p["beta"].get<double>(),
                       p["d_cutoff"].get<double>()};
  OneEuroFilter f(params);
  const auto& samples = data["samples"];
  const auto& expected = data["expected"];
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const double y = f.update(samples[i][0].get<double>(), samples[i][1].get<double>());
    EXPECT_NEAR(y, expected[i].get<double>(), 1e-9) << "sample " << i;
  }
}

TEST(Filter, FilterAndLimiterClampsPosition) {
  JointLimiterConfig cfg;
  cfg.limits["j"] = JointLimits{-1.0, 1.0, 100.0};  // wide velocity, tight position
  FilterAndLimiter fl(OneEuroParams{10.0, 0.0, 1.0}, cfg);
  // First call seeds; second pushes way past the soft_max -> clamps to 1.0.
  fl({{"j", 0.0}}, 0.0, 0.0);
  JointCommand cmd = fl({{"j", 5.0}}, 0.1, 0.1);
  EXPECT_LE(cmd.positions["j"], 1.0 + 1e-9);
}

TEST(Filter, FilterAndLimiterHoldsOnNonFinite) {
  JointLimiterConfig cfg;
  FilterAndLimiter fl(OneEuroParams{}, cfg);
  fl({{"j", 0.3}}, 0.0, 0.0);
  JointCommand cmd = fl({{"j", std::nan("")}}, 0.1, 0.1);
  EXPECT_NEAR(cmd.positions["j"], 0.3, 1e-9);  // held previous
}
