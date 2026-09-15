#pragma once

#include "method/integral_convention.h"

#include <cstdint>

namespace wickqc::method {

enum class SpatialFamily : std::uint8_t { kMp, kCc };

struct SpatialMethod {
  SpatialFamily family = SpatialFamily::kMp;
  // Perturbation order for MP; maximum excitation rank for CC.
  int order = 2;
  IntegralConvention convention = IntegralConvention::kChemist;
  // Optional MP excitation bound for a runtime calculation. Zero retains
  // every excitation through the required wavefunction order.
  int maximum_excitation_rank = 0;
  bool operator==(const SpatialMethod&) const = default;
};

} // namespace wickqc::method
