// MuJoCoPublisher — loads an MJCF/URDF model and applies JointCommand -> data.ctrl
// + mj_step. Mirrors publisher/mujoco_publisher.py (headless; the GLFW viewer
// belongs to the later device/app phase, so `gui` is accepted but a no-op here).
#pragma once

#include "itu/interpolating_publisher.hpp"
#include "itu/types.hpp"

#include <map>
#include <mutex>
#include <string>
#include <vector>

struct mjModel_;
struct mjData_;
typedef struct mjModel_ mjModel;
typedef struct mjData_ mjData;

namespace itu {

class MuJoCoPublisher : public InterpolatingPublisherBase {
 public:
  MuJoCoPublisher(std::string model_path, int rate_hz = 100, bool gui = false);
  ~MuJoCoPublisher() override;

  void start() override;  // load model, then start the interpolation thread
  void stop() override;

  // Current MuJoCo joint positions {canonical_name: qpos} (tests/diagnostics).
  JointMap current_joint_state();
  std::vector<std::string> actuator_names() const;

 protected:
  void emit(const JointCommand& command) override;

 private:
  void load();

  std::string model_path_;
  bool gui_;
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  std::map<std::string, int> actuator_idx_;  // canonical joint name -> actuator index
  mutable std::mutex sim_mtx_;
  int steps_per_emit_ = 1;
};

}  // namespace itu
