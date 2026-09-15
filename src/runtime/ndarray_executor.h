#pragma once

#include "equation/graph.h"
#include "runtime/tensor_binding.h"

#include <complex> // IWYU pragma: keep (explicit instantiations below)
#include <cstdint>
#include <string>
#include <vector>

namespace wickqc::codegen {
class CppEmitter;
}

namespace wickqc::runtime {

// Compile once after Wick expansion, optionally from graph.Simplify().
// Evaluation performs no symbolic algebra or string einsum parsing. Each call
// owns its outputs and intermediates; caller inputs (including views) stay intact.
// double and complex<double> are supported by every GEMM backend.
class NdArrayExecutor {
 public:
  [[nodiscard]] static NdArrayExecutor Compile(const equation::ContractionGraph& graph);
  [[nodiscard]] const std::vector<TensorBinding>& Inputs() const noexcept;
  [[nodiscard]] const std::vector<TensorBinding>& Outputs() const noexcept;

  template <typename T>
  [[nodiscard]] TensorMap<T> Evaluate(const TensorMap<T>& inputs, const Dimensions& dimensions) const;

 private:
  friend class codegen::CppEmitter;
  enum class Source : std::uint8_t { kInput, kValue, kOnes, kDelta };
  struct Input {
    TensorBinding binding;
    Source source = Source::kInput;
  };
  struct Contraction {
    double coefficient = 1.0;
    std::vector<Input> operands;
    std::vector<std::vector<int>> indices;
    std::vector<int> output_indices;
  };
  struct Transform {
    double coefficient = 1.0;
    std::vector<int> axes;
  };
  struct Assignment {
    TensorBinding output;
    std::vector<Contraction> terms;
    std::vector<Transform> transforms;
    std::vector<std::string> release;
  };
  std::vector<TensorBinding> inputs_;
  std::vector<TensorBinding> outputs_;
  std::vector<Assignment> assignments_;
};

extern template TensorMap<double> NdArrayExecutor::Evaluate(const TensorMap<double>&, const Dimensions&) const;
extern template TensorMap<std::complex<double>> NdArrayExecutor::Evaluate(const TensorMap<std::complex<double>>&,
                                                                          const Dimensions&) const;

} // namespace wickqc::runtime
