// MockPublisher — records emitted commands in memory + optional JSONL log.
// Mirrors publisher/mock_publisher.py.
#pragma once

#include "itu/interpolating_publisher.hpp"
#include "itu/types.hpp"

#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace itu {

class MockPublisher : public InterpolatingPublisherBase {
 public:
  // history: optional cap on retained commands (nullopt = unbounded).
  // log_path: optional JSONL file (truncated on construction); one object/line.
  explicit MockPublisher(int rate_hz = 100, std::optional<int> history = std::nullopt,
                         std::optional<std::string> log_path = std::nullopt);
  ~MockPublisher() override;

  std::vector<JointCommand> history() const;

 protected:
  void emit(const JointCommand& command) override;

 private:
  mutable std::mutex hist_mtx_;
  std::deque<JointCommand> history_;
  std::optional<int> history_cap_;
  std::optional<std::string> log_path_;
};

}  // namespace itu
