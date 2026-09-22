#pragma once

#include <cstdint>

namespace wickqc::method {

// Unantisymmetrized spatial two-electron integrals:
// chemist v[p,q,r,s] = (pq|rs); physicist v[p,q,r,s] = (pr|qs).
enum class IntegralConvention : std::uint8_t { kPhysicist, kChemist };

enum class SpatialFamily : std::uint8_t { kMP, kCC };

enum class NEVPT2Method : std::uint8_t { kSC, kIC };

struct SpatialMethod {
  SpatialFamily family = SpatialFamily::kMP;
  // Perturbation order for MP; maximum excitation rank for CC.
  int order = 2;
  IntegralConvention convention = IntegralConvention::kChemist;
  // Optional MP excitation bound for a runtime calculation. Zero retains
  // every excitation through the required wavefunction order.
  int maximum_excitation_rank = 0;
  bool operator==(const SpatialMethod&) const = default;
};

} // namespace wickqc::method
