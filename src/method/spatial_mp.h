#pragma once

#include "equation/graph.h"
#include "method/integral_convention.h"

#include <string>
#include <string_view>

namespace wickqc::method {

// Canonical closed-shell MP2--MP4 with intermediate-normalized first- and
// second-order wavefunctions. The spin-free projected linear systems retain
// their overlap metric; these residuals are not denominator-divided updates.
class SpatialMpGenerator {
 public:
  explicit SpatialMpGenerator(int order = 4, IntegralConvention convention = IntegralConvention::kPhysicist);
  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] equation::ContractionGraph Equations() const;
  [[nodiscard]] std::string GenerateNumpy(bool optimize = false) const;

 private:
  [[nodiscard]] symbolic::Expression Wavefunction(int order, int rank) const;
  int order_;
  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  symbolic::Expression fock_;
  symbolic::Expression perturbation_;
};

} // namespace wickqc::method
