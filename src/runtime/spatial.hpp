#pragma once

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <variant>
#include <vector>
#include "equation/graph.hpp"
#include "method/spatial.hpp"
#include "method/specification.hpp"
#include "runtime/executor.hpp"
#include "runtime/numeric.hpp"

namespace wickqc::runtime {

enum class GenerationPolicy : std::uint8_t {
  kPreferPrecompiled,
  kPrecompiledOnly,
  kRuntimeOnly
};

// Construct once, then evaluate repeatedly with new amplitudes and/or shapes.
// A fallback construction performs Wick expansion and lowering once, locally;
// Evaluate never performs symbolic work. No global cache or mutable state.
// A null lookup enables standalone use. Pass the optional generated registry
// explicitly to reuse build-time kernels; returned kernels must outlive this
// object.
class SpatialEvaluator {
 public:
  using KernelLookup = const NumericKernel* (*)(method::SpatialMethod);

  explicit SpatialEvaluator(
      method::SpatialMethod method,
      GenerationPolicy policy = GenerationPolicy::kPreferPrecompiled,
      KernelLookup lookup = nullptr);
  [[nodiscard]] bool IsPrecompiled() const noexcept;
  [[nodiscard]] const std::vector<TensorBinding>& Inputs() const;
  [[nodiscard]] const std::vector<TensorBinding>& Outputs() const;
  [[nodiscard]] TensorMap<double> Evaluate(
      const TensorMap<double>& inputs,
      const Dimensions& dimensions) const;
  [[nodiscard]] TensorMap<std::complex<double>> Evaluate(
      const TensorMap<std::complex<double>>& inputs,
      const Dimensions& dimensions) const;

 private:
  std::variant<const NumericKernel*, NDArrayExecutor> implementation_;
};

inline SpatialEvaluator::SpatialEvaluator(
    method::SpatialMethod method,
    GenerationPolicy policy,
    KernelLookup lookup) {
  if (policy != GenerationPolicy::kPreferPrecompiled &&
      policy != GenerationPolicy::kPrecompiledOnly &&
      policy != GenerationPolicy::kRuntimeOnly) {
    throw std::invalid_argument("Invalid spatial generation policy");
  }
  if (method.convention != method::IntegralConvention::kChemist &&
      method.convention != method::IntegralConvention::kPhysicist) {
    throw std::invalid_argument("Invalid spatial integral convention");
  }
  if (method.maximum_excitation_rank < 0 ||
      (method.family == method::SpatialFamily::kCC &&
       method.maximum_excitation_rank != 0)) {
    throw std::invalid_argument(
        "An excitation bound is supported only for MP and must be nonnegative");
  }
  if (policy != GenerationPolicy::kRuntimeOnly) {
    if (const auto* kernel = lookup == nullptr ? nullptr : lookup(method)) {
      implementation_ = kernel;
      return;
    }
    if (policy == GenerationPolicy::kPrecompiledOnly) {
      throw std::invalid_argument(
          "Requested method is absent from the supplied kernel lookup; pass the generated registry and select the method with WICKQC_PRECOMPILE_MP/CC and WICKQC_PRECOMPILE_CONVENTIONS");
    }
  }
  equation::ContractionGraph graph;
  switch (method.family) {
    case method::SpatialFamily::kMP:
      graph =
          method::SpatialMPGenerator(
              method.order, method.convention, method.maximum_excitation_rank)
              .Equations();
      break;
    case method::SpatialFamily::kCC:
      graph = method::SpatialCCGenerator(method.order, method.convention)
                  .Equations();
      break;
    default:
      throw std::invalid_argument("Invalid spatial method family");
  }
  implementation_ = NDArrayExecutor::Compile(graph.Simplify());
}

inline bool SpatialEvaluator::IsPrecompiled() const noexcept {
  return std::holds_alternative<const NumericKernel*>(implementation_);
}
inline const std::vector<TensorBinding>& SpatialEvaluator::Inputs() const {
  if (IsPrecompiled()) {
    return std::get<const NumericKernel*>(implementation_)->Inputs();
  }
  return std::get<NDArrayExecutor>(implementation_).Inputs();
}
inline const std::vector<TensorBinding>& SpatialEvaluator::Outputs() const {
  if (IsPrecompiled()) {
    return std::get<const NumericKernel*>(implementation_)->Outputs();
  }
  return std::get<NDArrayExecutor>(implementation_).Outputs();
}
inline TensorMap<double> SpatialEvaluator::Evaluate(
    const TensorMap<double>& inputs,
    const Dimensions& dimensions) const {
  if (IsPrecompiled()) {
    return std::get<const NumericKernel*>(implementation_)
        ->Evaluate(inputs, dimensions);
  }
  return std::get<NDArrayExecutor>(implementation_)
      .Evaluate(inputs, dimensions);
}
inline TensorMap<std::complex<double>> SpatialEvaluator::Evaluate(
    const TensorMap<std::complex<double>>& inputs,
    const Dimensions& dimensions) const {
  if (IsPrecompiled()) {
    return std::get<const NumericKernel*>(implementation_)
        ->Evaluate(inputs, dimensions);
  }
  return std::get<NDArrayExecutor>(implementation_)
      .Evaluate(inputs, dimensions);
}
} // namespace wickqc::runtime
