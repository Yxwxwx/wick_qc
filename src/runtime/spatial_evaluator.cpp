#include "runtime/spatial_evaluator.h"

#include "equation/graph.h"
#include "method/integral_convention.h"
#include "method/spatial_cc.h"
#include "method/spatial_method.h"
#include "method/spatial_mp.h"
#include "runtime/ndarray_executor.h"
#include "runtime/numeric_kernel.h"
#include "runtime/precompiled.h"
#include "runtime/tensor_binding.h"

#include <complex>
#include <stdexcept>
#include <variant>
#include <vector>

namespace wickqc::runtime {
SpatialEvaluator::SpatialEvaluator(
    method::SpatialMethod method,
    GenerationPolicy policy) {
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
    if (const auto* kernel = FindPrecompiled(method)) {
      implementation_ = kernel;
      return;
    }
    if (policy == GenerationPolicy::kPrecompiledOnly) {
      throw std::invalid_argument(
          "Requested method was not precompiled; select it with WICKQC_PRECOMPILE_MP/CC and WICKQC_PRECOMPILE_CONVENTIONS");
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

bool SpatialEvaluator::IsPrecompiled() const noexcept {
  return std::holds_alternative<const NumericKernel*>(implementation_);
}
const std::vector<TensorBinding>& SpatialEvaluator::Inputs() const {
  if (IsPrecompiled()) {
    return std::get<const NumericKernel*>(implementation_)->Inputs();
  }
  return std::get<NDArrayExecutor>(implementation_).Inputs();
}
const std::vector<TensorBinding>& SpatialEvaluator::Outputs() const {
  if (IsPrecompiled()) {
    return std::get<const NumericKernel*>(implementation_)->Outputs();
  }
  return std::get<NDArrayExecutor>(implementation_).Outputs();
}
TensorMap<double> SpatialEvaluator::Evaluate(
    const TensorMap<double>& inputs,
    const Dimensions& dimensions) const {
  if (IsPrecompiled()) {
    return std::get<const NumericKernel*>(implementation_)
        ->Evaluate(inputs, dimensions);
  }
  return std::get<NDArrayExecutor>(implementation_)
      .Evaluate(inputs, dimensions);
}
TensorMap<std::complex<double>> SpatialEvaluator::Evaluate(
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
