#pragma once

#include "backend/ndarray.hpp"
#include "equation/graph.h"

#include <complex>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace wickqc::runtime {

using Dimensions = std::map<symbolic::IndexDomain, std::size_t>;
template <typename T = double>
using TensorMap = std::map<std::string, NDArray<T>>;

// Names follow the NumPy emitter (e.g. vIIEE, tEEII, E1); axes retain the
// method's orbital and spin domains. General domains require an explicit size.
struct TensorBinding {
  std::string name;
  std::vector<symbolic::IndexDomain> domains;
  [[nodiscard]] std::vector<std::size_t> Shape(const Dimensions& dimensions) const;
  bool operator==(const TensorBinding&) const = default;
};

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
  enum class Source { kInput, kValue, kOnes, kDelta };
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
