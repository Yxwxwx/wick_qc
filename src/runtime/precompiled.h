#pragma once

#include "method/spatial_method.h"
#include "runtime/numeric_kernel.h"

#include <span>

namespace wickqc::runtime {

// This registry links only numeric kernels selected at CMake configure time.
// No method generator or Wick code is needed by a precompiled-only consumer.
[[nodiscard]] const NumericKernel* FindPrecompiled(method::SpatialMethod requested);
[[nodiscard]] std::span<const method::SpatialMethod> PrecompiledMethods();

} // namespace wickqc::runtime
