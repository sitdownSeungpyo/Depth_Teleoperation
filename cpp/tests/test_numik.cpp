#include "itu/robot_model.hpp"

#include "golden_common.hpp"

#include <gtest/gtest.h>

using namespace itu;

namespace {
RobotConfig config_from_json(const golden::json& c) {
  RobotConfig cfg;
  cfg.model_path = c["model_path"].get<std::string>();
  const auto& R = c["operator_to_robot_R"];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) cfg.operator_to_robot_R(i, j) = R[i][j].get<double>();
  const auto& ik = c["ik"];
  cfg.ik.damping = ik["damping"].get<double>();
  cfg.ik.max_iters = ik["max_iters"].get<int>();
  cfg.ik.pos_tol = ik["pos_tol"].get<double>();
  cfg.ik.step_clip = ik["step_clip"].get<double>();
  cfg.ik.max_target_step_m = ik["max_target_step_m"].get<double>();
  for (auto it = c["joint_limits"].begin(); it != c["joint_limits"].end(); ++it)
    cfg.joint_limits[it.key()] = {it.value()[0].get<double>(), it.value()[1].get<double>()};
  for (auto it = c["arms"].begin(); it != c["arms"].end(); ++it) {
    ArmConfig a;
    a.joints = it.value()["joints"].get<std::vector<std::string>>();
    a.shoulder_body = it.value()["shoulder_body"].get<std::string>();
    a.elbow_body = it.value()["elbow_body"].get<std::string>();
    a.wrist_body = it.value()["wrist_body"].get<std::string>();
    cfg.arms[it.key()] = std::move(a);
  }
  return cfg;
}
}  // namespace

// Iterative DLS solved through the same MuJoCo lib version on both sides, so the
// only cross-implementation difference is the 6x6 linear solve (LAPACK gesv vs
// Eigen partialPivLu); 1e-6 comfortably covers that.
constexpr double kTol = 1e-6;

TEST(NumIK, SolveArmSequenceGolden) {
  LOAD_GOLDEN_OR_SKIP("numik.json", data);
  RobotConfig cfg = config_from_json(data["config"]);
  RobotModel rm(cfg);

  // Link lengths measured from the model must match the Python reference.
  for (auto it = data["link_lengths"].begin(); it != data["link_lengths"].end(); ++it) {
    auto ll = rm.link_lengths().at(it.key());
    EXPECT_NEAR(ll.first, it.value()[0].get<double>(), 1e-9) << it.key() << " upper";
    EXPECT_NEAR(ll.second, it.value()[1].get<double>(), 1e-9) << it.key() << " lower";
  }

  // Replay the call sequence in order (warm-start makes order significant).
  int i = 0;
  for (const auto& call : data["calls"]) {
    const std::string side = call["side"].get<std::string>();
    auto sol = rm.solve_arm(side, golden::vec3(call["shoulder"]), golden::vec3(call["elbow"]),
                            golden::vec3(call["wrist"]));
    if (call["expected"].is_null()) {
      EXPECT_FALSE(sol.has_value()) << "call " << i;
    } else {
      ASSERT_TRUE(sol.has_value()) << "call " << i;
      const auto& expected = call["expected"];
      ASSERT_EQ(sol->size(), expected.size()) << "call " << i;
      for (auto it = expected.begin(); it != expected.end(); ++it) {
        ASSERT_TRUE(sol->count(it.key())) << "call " << i << " missing " << it.key();
        EXPECT_NEAR((*sol)[it.key()], it.value().get<double>(), kTol)
            << "call " << i << " joint " << it.key();
      }
    }
    ++i;
  }
}
