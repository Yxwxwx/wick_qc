#pragma once

#include "equation/graph.h"
#include "method/integral_convention.h"
#include "symbolic/wick.h"

#include <string>
#include <string_view>
#include <vector>

namespace wickqc::method {

// Closed-shell spin-free CC with pair-symmetric spatial amplitudes T_1...T_n.
// Residuals use the covariant E1...E1 projectors of the supplied UGA fixture.
class SpatialCcGenerator {
 public:
  explicit SpatialCcGenerator(int excitation_rank = 2, IntegralConvention convention = IntegralConvention::kPhysicist);
  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] const symbolic::Expression& Hamiltonian() const;
  [[nodiscard]] symbolic::Expression Projected(int rank, int bch_order = 4) const;
  [[nodiscard]] equation::ContractionGraph Equations() const;
  [[nodiscard]] std::string GenerateNumpy(bool optimize = false) const;

 private:
  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  symbolic::Expression hamiltonian_;
  std::vector<symbolic::Expression> cluster_ranks_;
};

} // namespace wickqc::method
