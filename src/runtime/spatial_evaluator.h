#pragma once

#include "method/spatial_method.h"
#include "runtime/ndarray_executor.h"
#include "runtime/numeric_kernel.h"
#include "runtime/tensor_binding.h"

#include <complex>
#include <cstdint>
#include <variant>
#include <vector>

namespace wickqc::runtime {

enum class GenerationPolicy : std::uint8_t { kPreferPrecompiled, kPrecompiledOnly, kRuntimeOnly };

// Construct once, then evaluate repeatedly with new amplitudes and/or shapes.
// A fallback construction performs Wick expansion and lowering once, locally;
// Evaluate never performs symbolic work. No global cache or mutable state.
class SpatialEvaluator {
 public:
  explicit SpatialEvaluator(method::SpatialMethod method,
                            GenerationPolicy policy = GenerationPolicy::kPreferPrecompiled);
  [[nodiscard]] bool IsPrecompiled() const noexcept;
  [[nodiscard]] const std::vector<TensorBinding>& Inputs() const;
  [[nodiscard]] const std::vector<TensorBinding>& Outputs() const;
  [[nodiscard]] TensorMap<double> Evaluate(const TensorMap<double>& inputs, const Dimensions& dimensions) const;
  [[nodiscard]] TensorMap<std::complex<double>> Evaluate(const TensorMap<std::complex<double>>& inputs,
                                                         const Dimensions& dimensions) const;

 private:
  std::variant<const NumericKernel*, NDArrayExecutor> implementation_;
};

} // namespace wickqc::runtime
