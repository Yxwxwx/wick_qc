#pragma once

#include "symbolic/wick.h"

#include <string>
#include <string_view>

namespace wickqc::method {

// Spin-orbital CCSD, with a normal-ordered Hamiltonian and factorial BCH
// weights.
class CcsdGenerator {
 public:
  explicit CcsdGenerator(bool antisymmetrized_integrals = true);

  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] symbolic::Expression Energy(int order = 2) const;
  [[nodiscard]] symbolic::Expression Singles(int order = 4) const;
  [[nodiscard]] symbolic::Expression Doubles(int order = 4) const;
  [[nodiscard]] std::string GenerateNumpy() const;
  [[nodiscard]] const symbolic::Expression& Hamiltonian() const;

 private:
  [[nodiscard]] symbolic::Expression SimilarityTransform(int order, int rank) const;
  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  symbolic::Expression h_;
  symbolic::Expression t_;
};

} // namespace wickqc::method
