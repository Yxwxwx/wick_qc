#include "equation/equation.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>
#include "symbolic/wick.h"

namespace wickqc::equation {
namespace {

bool ContainsIndex(
    const std::vector<symbolic::Index>& indices,
    const symbolic::Index& target) {
  return std::ranges::find(indices, target) != indices.end();
}

void ValidateTerm(const symbolic::Term& term, const symbolic::Tensor& output) {
  if (output.kind != symbolic::TensorKind::kGeneric) {
    throw std::invalid_argument("Equation output must be a coefficient tensor");
  }
  std::vector<symbolic::Index> output_indices;
  for (const auto& index : output.indices) {
    if (ContainsIndex(output_indices, index)) {
      throw std::invalid_argument("Repeated output index '" + index.name + "'");
    }
    if (ContainsIndex(term.summed_indices, index)) {
      throw std::invalid_argument(
          "Output index '" + index.name + "' is also reduced");
    }
    output_indices.push_back(index);
  }
  std::vector<symbolic::Index> input_indices;
  for (const auto& tensor : term.tensors) {
    if (tensor.IsFermionOperator()) {
      throw std::invalid_argument(
          "Tensor equations require a fully expanded Wick expression");
    }
    input_indices.insert(
        input_indices.end(), tensor.indices.begin(), tensor.indices.end());
  }

  for (const auto& index : input_indices) {
    if (!ContainsIndex(term.summed_indices, index) &&
        !ContainsIndex(output.indices, index)) {
      throw std::invalid_argument(
          "Free input index '" + index.name +
          "' is absent from the output tensor");
    }
  }
  for (const auto& index : term.summed_indices) {
    if (!ContainsIndex(input_indices, index)) {
      throw std::invalid_argument(
          "Reduction index '" + index.name + "' has no tensor occurrence");
    }
  }
}

} // namespace

TensorEquation::TensorEquation(std::vector<TensorEquationTerm> terms)
    : terms_(std::move(terms)) {}

const std::vector<TensorEquationTerm>& TensorEquation::Terms() const noexcept {
  return terms_;
}

TensorEquation TensorEquation::FromExpression(
    const symbolic::Expression& expression,
    const symbolic::Tensor& output) {
  std::vector<TensorEquationTerm> terms;
  terms.reserve(expression.Terms().size());
  for (const auto& term : expression.Terms()) {
    ValidateTerm(term, output);
    terms.push_back(
        {term.coefficient, output, term.tensors, term.summed_indices});
  }
  return TensorEquation(std::move(terms));
}

} // namespace wickqc::equation
