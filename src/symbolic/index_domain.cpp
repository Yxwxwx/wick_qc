#include "symbolic/index_domain.h"

#include <bit>
#include <compare>
#include <cstdint>

namespace wickqc::symbolic {

std::strong_ordering IndexDomain::operator<=>(
    const IndexDomain& other) const noexcept {
  const auto packed = static_cast<std::uint8_t>(orbital_spaces | (spins << 4));
  const auto other_packed =
      static_cast<std::uint8_t>(other.orbital_spaces | (other.spins << 4));
  return packed <=> other_packed;
}

bool IndexDomain::IsConcrete() const noexcept {
  return std::popcount(orbital_spaces) <= 1 && std::popcount(spins) <= 1;
}

bool IndexDomain::IsCompatibleWith(const IndexDomain& other) const noexcept {
  const bool orbital_compatible = orbital_spaces == 0 ||
      other.orbital_spaces == 0 || (orbital_spaces & other.orbital_spaces) != 0;
  const bool spin_compatible =
      spins == 0 || other.spins == 0 || (spins & other.spins) != 0;
  return orbital_compatible && spin_compatible;
}

} // namespace wickqc::symbolic
