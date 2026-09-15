#pragma once

#include "equation/graph.h"
#include "symbolic/wick.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace wickqc::method {

// Strongly contracted, spin-free spatial NEVPT2 in its eight outer subspaces.
class ScNevpt2Generator {
 public:
  ScNevpt2Generator();

  [[nodiscard]] std::vector<std::pair<std::string, equation::ContractionGraph>> Equations(bool sum_outer = false) const;
  [[nodiscard]] std::string GenerateNumpy(bool optimize = false) const;

 private:
  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] symbolic::Tensor ParseTensor(std::string_view text) const;

  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  symbolic::Expression active_hamiltonian_;
  std::vector<std::pair<std::string, std::string>> subspaces_;
};

} // namespace wickqc::method
