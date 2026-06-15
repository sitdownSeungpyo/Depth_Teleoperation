// Abstract publisher interface (concrete mock/udp/mujoco/dynamixel come later).
// Mirrors the Publisher Protocol in publisher/base.py.
#pragma once

#include "itu/types.hpp"

namespace itu {

class Publisher {
 public:
  virtual ~Publisher() = default;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual void set_target(const JointCommand& command) = 0;
};

}  // namespace itu
