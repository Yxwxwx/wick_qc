#pragma once

#include <cstdint>

namespace wickqc::method {

// Unantisymmetrized spatial two-electron integrals:
// chemist v[p,q,r,s] = (pq|rs); physicist v[p,q,r,s] = (pr|qs).
enum class IntegralConvention : std::uint8_t { kPhysicist, kChemist };

} // namespace wickqc::method
