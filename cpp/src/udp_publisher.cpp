#include "itu/udp_publisher.hpp"

#include <sstream>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace itu {

namespace {
#ifdef _WIN32
struct WsaGuard {
  WsaGuard() {
    WSADATA d;
    WSAStartup(MAKEWORD(2, 2), &d);
  }
  ~WsaGuard() { WSACleanup(); }
};
// One init for the process; refcounted by the OS anyway.
WsaGuard g_wsa;
void close_sock(std::intptr_t s) { closesocket(static_cast<SOCKET>(s)); }
#else
void close_sock(std::intptr_t s) { ::close(static_cast<int>(s)); }
#endif
}  // namespace

UdpPublisher::UdpPublisher(std::string host, int port, int rate_hz)
    : InterpolatingPublisherBase(rate_hz), host_(std::move(host)), port_(port) {
  sock_ = static_cast<std::intptr_t>(::socket(AF_INET, SOCK_DGRAM, 0));
}

UdpPublisher::~UdpPublisher() { stop(); }

void UdpPublisher::stop() {
  stop_internal();
  if (sock_ != -1) {
    close_sock(sock_);
    sock_ = -1;
  }
}

void UdpPublisher::emit(const JointCommand& command) {
  if (sock_ == -1) return;
  // Placeholder payload — schema to be defined once the robot platform is fixed.
  std::ostringstream os;
  os << "PLACEHOLDER:{";
  bool first = true;
  for (const auto& [joint, value] : command.positions) {
    if (!first) os << ", ";
    first = false;
    os << '\'' << joint << "': " << value;
  }
  os << "}";
  const std::string payload = os.str();

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<unsigned short>(port_));
  inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);
  ::sendto(static_cast<
#ifdef _WIN32
               SOCKET
#else
               int
#endif
               >(sock_),
           payload.data(), static_cast<int>(payload.size()), 0,
           reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
}

}  // namespace itu
