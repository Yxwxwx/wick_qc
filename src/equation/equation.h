#pragma once

#include "symbolic/wick.h"

#include <vector>

namespace wickqc::equation {

struct TensorEquationTerm {
  double coefficient = 1.0;
  symbolic::Tensor output;
  std::vector<symbolic::Tensor> inputs;
  std::vector<symbolic::Index> reduction_indices;
};

class TensorEquation {
 public:
  [[nodiscard]] static TensorEquation FromExpression(const symbolic::Expression& expression,
                                                     const symbolic::Tensor& output);

  [[nodiscard]] const std::vector<TensorEquationTerm>& Terms() const noexcept;

 private:
  explicit TensorEquation(std::vector<TensorEquationTerm> terms);

  std::vector<TensorEquationTerm> terms_;
};

} // namespace wickqc::equation
