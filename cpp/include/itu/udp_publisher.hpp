// UdpPublisher — opens a UDP socket and sends a placeholder packet (skeleton:
// real packet schema is TBD per target platform). Mirrors publisher/udp_publisher.py.
#pragma once

#include "itu/interpolating_publisher.hpp"
#include "itu/types.hpp"

#include <cstdint>
#include <string>

namespace itu {

class UdpPublisher : public InterpolatingPublisherBase {
 public:
  UdpPublisher(std::string host, int port, int rate_hz = 100);
  ~UdpPublisher() override;
  void stop() override;

 protected:
  void emit(const JointCommand& command) override;

 private:
  std::string host_;
  int port_;
  std::intptr_t sock_ = -1;  // SOCKET (Windows) / fd (POSIX), -1 = invalid
};

}  // namespace itu
