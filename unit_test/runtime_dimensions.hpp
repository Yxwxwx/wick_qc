#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace wickqc::test {
// CTest sets these at process launch. The same binary runs different sizes.
inline std::size_t Dimension(const char* name) {
  const char* text = std::getenv(name);
  if (text == nullptr) {
    throw std::invalid_argument(
        std::string("Missing runtime test dimension: ") + name);
  }
  std::size_t consumed = 0;
  const std::string value(text);
  const auto extent = std::stoul(value, &consumed);
  if (consumed != value.size() || extent < 2 || extent > 8) {
    throw std::invalid_argument(
        std::string("Expected a test dimension in [2,8]: ") + name);
  }
  return extent;
}
} // namespace wickqc::test
