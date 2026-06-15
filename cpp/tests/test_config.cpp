#include "itu/config.hpp"

#include <gtest/gtest.h>

using namespace itu;

TEST(Config, DeepMergeNestedMapsMergeRecursively) {
  YAML::Node base = YAML::Load("a:\n  x: 1\n  y: 2\nb: 5");
  YAML::Node over = YAML::Load("a:\n  y: 9\n  z: 3\nc: 7");
  YAML::Node out = deep_merge(base, over);
  EXPECT_EQ(out["a"]["x"].as<int>(), 1);  // kept from base
  EXPECT_EQ(out["a"]["y"].as<int>(), 9);  // overridden
  EXPECT_EQ(out["a"]["z"].as<int>(), 3);  // added
  EXPECT_EQ(out["b"].as<int>(), 5);
  EXPECT_EQ(out["c"].as<int>(), 7);
}

TEST(Config, DeepMergeScalarReplacesMap) {
  YAML::Node base = YAML::Load("a:\n  x: 1");
  YAML::Node over = YAML::Load("a: 42");
  YAML::Node out = deep_merge(base, over);
  EXPECT_EQ(out["a"].as<int>(), 42);
}

TEST(Config, DeepMergeSequenceReplaces) {
  YAML::Node base = YAML::Load("a: [1, 2, 3]");
  YAML::Node over = YAML::Load("a: [9]");
  YAML::Node out = deep_merge(base, over);
  ASSERT_TRUE(out["a"].IsSequence());
  EXPECT_EQ(out["a"].size(), 1u);
  EXPECT_EQ(out["a"][0].as<int>(), 9);
}
