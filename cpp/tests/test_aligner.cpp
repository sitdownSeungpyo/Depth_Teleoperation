#include "itu/aligner.hpp"

#include "golden_common.hpp"

#include <gtest/gtest.h>

using namespace itu;

TEST(Aligner, AlignToTorsoGolden) {
  LOAD_GOLDEN_OR_SKIP("aligner.json", data);
  for (const auto& c : data) {
    SkeletonFrame frame;
    frame.timestamp = 0.0;
    for (auto it = c["keypoints"].begin(); it != c["keypoints"].end(); ++it)
      frame.keypoints[it.key()] = golden::vec3(it.value());
    for (auto it = c["confidence"].begin(); it != c["confidence"].end(); ++it)
      frame.confidence[it.key()] = it.value().get<double>();

    std::optional<Eigen::Vector3d> grav;
    if (!c["gravity_up"].is_null()) grav = golden::vec3(c["gravity_up"]);

    AlignedFrame aligned = align_to_torso(frame, grav);

    // rotation
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        EXPECT_NEAR(aligned.rotation(i, j), c["expected_rotation"][i][j].get<double>(), 1e-9)
            << c["name"] << " R(" << i << "," << j << ")";
    // rpy
    for (int i = 0; i < 3; ++i)
      EXPECT_NEAR(aligned.rpy[i], c["expected_rpy"][i].get<double>(), 1e-9)
          << c["name"] << " rpy[" << i << "]";
    // keypoints
    for (auto it = c["expected_keypoints"].begin(); it != c["expected_keypoints"].end(); ++it) {
      ASSERT_TRUE(aligned.keypoints.count(it.key())) << it.key();
      EXPECT_NEAR((aligned.keypoints[it.key()] - golden::vec3(it.value())).norm(), 0.0, 1e-9)
          << c["name"] << " kp " << it.key();
    }
  }
}

TEST(Aligner, DegeneratePoseThrows) {
  SkeletonFrame frame;
  frame.keypoints["left_shoulder"] = {0, 0, 0};
  frame.keypoints["right_shoulder"] = {0, 0, 0};  // coincident -> u == 0
  frame.keypoints["head"] = {0, -1, 0};
  EXPECT_THROW(align_to_torso(frame, Eigen::Vector3d(0, -1, 0)), AlignmentError);
}
