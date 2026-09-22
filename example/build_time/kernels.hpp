#pragma once

#include "runtime/numeric.hpp"

#include <span>
#include <string_view>

namespace wickqc::generated {
const runtime::NumericKernel& Kernel_example_ccsd();
}

namespace wickqc::example {
struct NamedKernel {
  std::string_view name;
  const runtime::NumericKernel* kernel;
};
// Generated at build time from every ICNEVPT2Generator::Equations() block.
[[nodiscard]] std::span<const NamedKernel> ICNEVPT2Kernels();
[[nodiscard]] std::span<const NamedKernel> SCNEVPT2Kernels();
} // namespace wickqc::example
