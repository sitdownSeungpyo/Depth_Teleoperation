find_package(GTest CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(itu_tests
  test_mathutil.cpp
  test_config.cpp
  test_filter.cpp
  test_gravity.cpp
  test_aligner.cpp
  test_retarget.cpp
  test_depth_lift.cpp
)
target_link_libraries(itu_tests PRIVATE
  itu_core
  GTest::gtest
  GTest::gtest_main
  nlohmann_json::nlohmann_json
)
# Golden data is generated next to the sources by tools/golden/gen_golden.py.
target_compile_definitions(itu_tests PRIVATE
  GOLDEN_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden/data")

include(GoogleTest)
gtest_discover_tests(itu_tests)
