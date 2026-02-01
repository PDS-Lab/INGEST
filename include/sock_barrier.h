#pragma once

#include <arpa/inet.h>
#include <cerrno>
#include <string>
#include <unistd.h>

struct SockBarrier {
  int sockfd;
  SockBarrier(const std::string &addr) {
    // parse addr:port
    auto pos = addr.find(':');
    if (pos == std::string::npos) {
      throw std::invalid_argument("invalid address");
    }
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
      perror("socket");
      throw std::runtime_error("socket failed");
    }
    sockaddr_in tmp{};
    tmp.sin_family = AF_INET;
    tmp.sin_port = htons(std::stoi(addr.substr(pos + 1)));
    if (inet_pton(AF_INET, addr.substr(0, pos).c_str(), &tmp.sin_addr) != 1) {
      throw std::invalid_argument("invalid ip address");
    }
    if (connect(sockfd, (sockaddr *)&tmp, sizeof(tmp)) < 0) {
      perror("connect");
      throw std::runtime_error("connect failed");
    }
  }
  ~SockBarrier() {
    if (sockfd >= 0)
      close(sockfd);
  }
  void wait() {
    char buf = 1;
    while (true) {
      auto n = ::write(sockfd, &buf, 1);
      if (n == 1)
        break;
      if (n < 0 && errno == EINTR)
        continue;
      throw std::runtime_error("SockBarrier write failed");
    }
    while (true) {
      auto n = ::read(sockfd, &buf, 1);
      if (n == 1)
        break;
      if (n == 0)
        throw std::runtime_error("SockBarrier read: peer closed");
      if (n < 0 && errno == EINTR)
        continue;
      throw std::runtime_error("SockBarrier read failed");
    }
  }
};
