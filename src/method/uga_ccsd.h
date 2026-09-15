#pragma once

#include "method/integral_convention.h"
#include "symbolic/wick.h"

#include <string>
#include <string_view>

namespace wickqc::method {

// Closed-shell unitary-group CCSD with pair-symmetric spatial amplitudes.
class UgaCcsdGenerator {
 public:
  explicit UgaCcsdGenerator(IntegralConvention convention = IntegralConvention::kPhysicist);

  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] const symbolic::Expression& Hamiltonian() const;
  [[nodiscard]] symbolic::Expression Energy(int order = 2) const;
  [[nodiscard]] symbolic::Expression Singles(int order = 4) const;
  [[nodiscard]] symbolic::Expression Doubles(int order = 4) const;
  [[nodiscard]] std::string GenerateNumpy() const;

 private:
  [[nodiscard]] symbolic::Expression Projected(int order, int rank) const;
  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  symbolic::Expression h_;
  symbolic::Expression singles_;
  symbolic::Expression doubles_;
  symbolic::Expression cluster_;
};

} // namespace wickqc::method
