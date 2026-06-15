#include "itu/retarget.hpp"

#include "golden_common.hpp"

#include <gtest/gtest.h>

using namespace itu;

TEST(Retarget, FullUpperBodyGolden) {
  LOAD_GOLDEN_OR_SKIP("retarget.json", data);
  for (const auto& c : data) {
    AlignedFrame aligned;
    for (auto it = c["aligned_keypoints"].begin(); it != c["aligned_keypoints"].end(); ++it)
      aligned.keypoints[it.key()] = golden::vec3(it.value());
    aligned.rotation = Eigen::Matrix3d::Identity();
    aligned.rpy = Eigen::Vector3d(c["rpy"][0].get<double>(), c["rpy"][1].get<double>(),
                                  c["rpy"][2].get<double>());

    RobotGeometry robot{c["robot"]["upper"].get<double>(), c["robot"]["lower"].get<double>(),
                        {0.0, 0.0, 0.0}};
    Calibration cal;
    cal.operator_arm_length = c["calibration"]["operator_arm_length"].get<double>();

    JointMap out = retarget_full_upper_body(aligned, robot, cal, c["decouple"].get<bool>());

    const auto& expected = c["expected"];
    ASSERT_EQ(out.size(), expected.size()) << c["name"];
    for (auto it = expected.begin(); it != expected.end(); ++it) {
      ASSERT_TRUE(out.count(it.key())) << c["name"] << " missing joint " << it.key();
      EXPECT_NEAR(out[it.key()], it.value().get<double>(), 1e-9)
          << c["name"] << " joint " << it.key();
    }
  }
}

TEST(Retarget, MissingArmKeypointThrows) {
  AlignedFrame aligned;
  aligned.keypoints["right_shoulder"] = {-0.2, -0.3, 0.0};
  aligned.keypoints["right_elbow"] = {0, 0, 0};  // zero-vector -> rejected
  aligned.keypoints["right_wrist"] = {-0.7, -0.3, 0.0};
  RobotGeometry robot{0.25, 0.25, {0, 0, 0}};
  Calibration cal;
  cal.operator_arm_length = 0.5;
  EXPECT_THROW(retarget_arm(aligned, Side::Right, robot, cal), SingularConfigurationError);
}
