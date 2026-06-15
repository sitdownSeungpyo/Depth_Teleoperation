#include "itu/dynamixel.hpp"
#include "itu/mock_publisher.hpp"

#include "golden_common.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

using namespace itu;

namespace {
JointCommand cmd_from_json(const golden::json& j) {
  JointCommand c;
  c.timestamp = j["timestamp"].get<double>();
  c.source_frame_ts = j["source_frame_ts"].get<double>();
  for (auto it = j["positions"].begin(); it != j["positions"].end(); ++it)
    c.positions[it.key()] = it.value().get<double>();
  return c;
}

void expect_cmd_eq(const JointCommand& got, const golden::json& exp, const std::string& ctx) {
  EXPECT_NEAR(got.timestamp, exp["timestamp"].get<double>(), 1e-12) << ctx << " ts";
  EXPECT_NEAR(got.source_frame_ts, exp["source_frame_ts"].get<double>(), 1e-12) << ctx << " sft";
  const auto& ep = exp["positions"];
  ASSERT_EQ(got.positions.size(), ep.size()) << ctx << " npos";
  for (auto it = ep.begin(); it != ep.end(); ++it) {
    ASSERT_TRUE(got.positions.count(it.key())) << ctx << " missing " << it.key();
    EXPECT_NEAR(got.positions.at(it.key()), it.value().get<double>(), 1e-12)
        << ctx << " pos " << it.key();
  }
}
}  // namespace

TEST(Publisher, InterpolateGolden) {
  LOAD_GOLDEN_OR_SKIP("publisher.json", data);
  MockPublisher mp;  // not started — interpolate() is pure
  mp.set_target(cmd_from_json(data["cmdA"]));
  mp.set_target(cmd_from_json(data["cmdB"]));
  const auto& nows = data["nows"];
  const auto& interp = data["interp"];
  for (std::size_t i = 0; i < nows.size(); ++i) {
    auto c = mp.interpolate(nows[i].get<double>());
    ASSERT_TRUE(c.has_value()) << "now " << i;
    expect_cmd_eq(*c, interp[i], "interp[" + std::to_string(i) + "]");
  }
}

TEST(Publisher, InterpolateSingleSetpointGolden) {
  LOAD_GOLDEN_OR_SKIP("publisher.json", data);
  MockPublisher mp;
  mp.set_target(cmd_from_json(data["single"]["cmd"]));
  auto c = mp.interpolate(data["single"]["now"].get<double>());
  ASSERT_TRUE(c.has_value());
  expect_cmd_eq(*c, data["single"]["expected"], "single");
}

TEST(Publisher, DynamixelUnitGolden) {
  LOAD_GOLDEN_OR_SKIP("publisher.json", data);
  for (const auto& d : data["dxl"])
    EXPECT_EQ(angle_rad_to_dxl_unit(d["angle"].get<double>()), d["unit"].get<int>())
        << "angle " << d["angle"].get<double>();
}

TEST(Publisher, MockThreadedEmitsAndRecords) {
  MockPublisher mp(/*rate_hz=*/100);
  JointCommand c;
  c.timestamp = perf_now_seconds();  // same clock as the loop, so not stale
  c.positions = {{"j1", 0.3}, {"j2", -0.2}};
  mp.set_target(c);
  mp.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(60));
  mp.stop();
  EXPECT_FALSE(mp.history().empty());
  auto cur = mp.current();
  ASSERT_TRUE(cur.has_value());
  EXPECT_NEAR(cur->positions.at("j1"), 0.3, 1e-9);
  EXPECT_NEAR(cur->positions.at("j2"), -0.2, 1e-9);
}
