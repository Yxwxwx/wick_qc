#pragma once

#include "runtime/ndarray_executor.h"

#include <string>
#include <string_view>

namespace wickqc::codegen {

// Uses the executor's validated lowering and lifetime schedule. The emitted
// translation unit depends only on wickqc_numeric, and supports runtime sizes.
class CPPEmitter {
 public:
  [[nodiscard]] static std::string Render(const runtime::NDArrayExecutor& program, std::string_view function_name);
};

} // namespace wickqc::codegen
