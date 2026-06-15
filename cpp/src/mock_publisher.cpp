#include "itu/mock_publisher.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace itu {

namespace {
std::string to_jsonl(const JointCommand& c) {
  std::ostringstream os;
  os << std::setprecision(17);
  os << "{\"timestamp\": " << c.timestamp << ", \"source_frame_ts\": " << c.source_frame_ts
     << ", \"positions\": {";
  bool first = true;
  for (const auto& [joint, value] : c.positions) {
    if (!first) os << ", ";
    first = false;
    os << '"' << joint << "\": " << value;
  }
  os << "}}";
  return os.str();
}
}  // namespace

MockPublisher::MockPublisher(int rate_hz, std::optional<int> history,
                             std::optional<std::string> log_path)
    : InterpolatingPublisherBase(rate_hz), history_cap_(history), log_path_(std::move(log_path)) {
  if (log_path_.has_value()) {
    std::filesystem::path p(*log_path_);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    std::ofstream(*log_path_, std::ios::trunc).close();  // truncate
  }
}

MockPublisher::~MockPublisher() { stop_internal(); }

std::vector<JointCommand> MockPublisher::history() const {
  std::lock_guard<std::mutex> lk(hist_mtx_);
  return std::vector<JointCommand>(history_.begin(), history_.end());
}

void MockPublisher::emit(const JointCommand& command) {
  std::lock_guard<std::mutex> lk(hist_mtx_);
  history_.push_back(command);
  if (history_cap_.has_value())
    while (static_cast<int>(history_.size()) > *history_cap_) history_.pop_front();
  if (log_path_.has_value()) {
    std::ofstream f(*log_path_, std::ios::app);
    if (f) f << to_jsonl(command) << "\n";
  }
}

}  // namespace itu
