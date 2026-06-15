// Shared helpers for golden tests: locate + load the generated JSON data.
#pragma once

#include <Eigen/Core>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace golden {

using json = nlohmann::json;

// Loads cpp/tests/golden/data/<name>. Skips the test (via a thrown message the
// caller turns into GTEST_SKIP) when the file is absent — i.e. the user hasn't
// run gen_golden.py yet.
inline bool load(const std::string& name, json& out) {
  const std::string path = std::string(GOLDEN_DATA_DIR) + "/" + name;
  std::ifstream f(path);
  if (!f.is_open()) return false;
  f >> out;
  return true;
}

inline Eigen::Vector3d vec3(const json& a) {
  return Eigen::Vector3d(a[0].get<double>(), a[1].get<double>(), a[2].get<double>());
}

}  // namespace golden

// Skip the test with a clear message when golden data hasn't been generated.
#define LOAD_GOLDEN_OR_SKIP(name, var)                                                \
  golden::json var;                                                                   \
  if (!golden::load(name, var)) {                                                     \
    GTEST_SKIP() << "missing golden data '" << name                                   \
                 << "'; run: python cpp/tools/golden/gen_golden.py";                  \
  }
