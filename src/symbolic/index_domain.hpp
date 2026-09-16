#pragma once

#include <bit>
#include <compare>
#include <cstdint>

namespace wickqc::symbolic {

enum class OrbitalSpace : std::uint8_t {
  kGeneral = 0,
  kInactive = 1,
  kActive = 2,
  kSingle = 4,
  kExternal = 8,
};

enum class Spin : std::uint8_t {
  kNone = 0,
  kAlpha = 1,
  kBeta = 2,
};

struct IndexDomain {
  std::uint8_t orbital_spaces = 0;
  std::uint8_t spins = 0;

  [[nodiscard]] std::strong_ordering operator<=>(
      const IndexDomain& other) const noexcept;
  bool operator==(const IndexDomain&) const = default;
  [[nodiscard]] bool IsConcrete() const noexcept;
  [[nodiscard]] bool IsCompatibleWith(const IndexDomain& other) const noexcept;
};

inline std::strong_ordering IndexDomain::operator<=>(
    const IndexDomain& other) const noexcept {
  const auto packed = static_cast<std::uint8_t>(orbital_spaces | (spins << 4));
  const auto other_packed =
      static_cast<std::uint8_t>(other.orbital_spaces | (other.spins << 4));
  return packed <=> other_packed;
}

inline bool IndexDomain::IsConcrete() const noexcept {
  return std::popcount(orbital_spaces) <= 1 && std::popcount(spins) <= 1;
}

inline bool IndexDomain::IsCompatibleWith(
    const IndexDomain& other) const noexcept {
  const bool orbital_compatible = orbital_spaces == 0 ||
      other.orbital_spaces == 0 || (orbital_spaces & other.orbital_spaces) != 0;
  const bool spin_compatible =
      spins == 0 || other.spins == 0 || (spins & other.spins) != 0;
  return orbital_compatible && spin_compatible;
}

} // namespace wickqc::symbolic
