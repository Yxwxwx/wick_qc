#pragma once

#include "equation/graph.h"
#include "symbolic/wick.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace wickqc::method {

class ICNEVPT2Generator {
 public:
  ICNEVPT2Generator();

  [[nodiscard]] std::string GenerateNumpy() const;
  // Each block contains the RHS vectors and effective Hamiltonian matrices,
  // before orbital restrictions and the final linear solve.
  [[nodiscard]] std::vector<std::pair<std::string, equation::ContractionGraph>> Equations() const;

 private:
  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] symbolic::Tensor ParseTensor(std::string_view text) const;
  [[nodiscard]] symbolic::Expression BuildCommutator(std::string_view bra, std::string_view ket) const;
  [[nodiscard]] symbolic::Expression BuildRhs(std::string_view bra, std::string_view ket) const;
  [[nodiscard]] std::string RenderEquation(const symbolic::Expression& expression,
                                           const symbolic::Tensor& output) const;
  [[nodiscard]] static std::string Allocate(const symbolic::Tensor& tensor);
  [[nodiscard]] static std::string Restrict(const symbolic::Tensor& tensor, bool restrict_active, bool strict);
  [[nodiscard]] static std::string Indent(std::string_view text);

  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  std::vector<std::pair<std::string, std::string>> subspaces_;
  symbolic::Expression zeroth_order_hamiltonian_;
};

} // namespace wickqc::method
