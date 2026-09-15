#pragma once

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

  [[nodiscard]] std::strong_ordering operator<=>(const IndexDomain& other) const noexcept;
  bool operator==(const IndexDomain&) const = default;
  [[nodiscard]] bool IsConcrete() const noexcept;
  [[nodiscard]] bool IsCompatibleWith(const IndexDomain& other) const noexcept;
};

} // namespace wickqc::symbolic
