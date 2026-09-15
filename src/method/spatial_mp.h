#pragma once

#include "equation/graph.h"
#include "method/integral_convention.h"
#include "symbolic/wick.h"

#include <string>
#include <string_view>

namespace wickqc::method {

// Canonical closed-shell MPn with intermediate-normalized wavefunctions. The spin-free projected linear systems retain
// their overlap metric; these residuals are not denominator-divided updates.
class SpatialMpGenerator {
 public:
  explicit SpatialMpGenerator(int order = 4,
                              IntegralConvention convention = IntegralConvention::kPhysicist,
                              int maximum_excitation_rank = 0);
  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] equation::ContractionGraph Equations() const;
  [[nodiscard]] std::string GenerateNumpy(bool optimize = false) const;

 private:
  [[nodiscard]] symbolic::Expression Wavefunction(int order, int rank) const;
  [[nodiscard]] int WavefunctionRank(int order) const;
  int order_;
  // Zero retains every rank (through 2 * wavefunction order). A physical
  // electron/hole rank bound can be supplied for runtime generation.
  int maximum_excitation_rank_;
  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  symbolic::Expression fock_;
  symbolic::Expression perturbation_;
};

} // namespace wickqc::method
