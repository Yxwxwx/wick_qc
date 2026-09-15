#pragma once

namespace wickqc::method {

// Unantisymmetrized spatial two-electron integrals:
// chemist v[p,q,r,s] = (pq|rs); physicist v[p,q,r,s] = (pr|qs).
enum class IntegralConvention { kPhysicist, kChemist };

} // namespace wickqc::method
