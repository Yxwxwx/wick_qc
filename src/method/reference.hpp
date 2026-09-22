#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/numeric.hpp"

namespace wickqc::method {
// Preserve the public enum's existing int representation.
// NOLINTNEXTLINE(performance-enum-size)
enum class ReferenceKind { kRHF, kCASSCF };

// Real, spin-free spatial orbitals ordered core/active/external. ncore includes
// the frozen prefix. All tensors refer to the same final MO basis. E1--E4 use
// block2's spin-free normal-ordered density convention, not raw PySCF RDMs.
// reference_energy is the selected state's total energy including nuclei and
// all frozen/core constants; a method adds its correlation energy exactly once.
struct SpatialReference {
  ReferenceKind kind = ReferenceKind::kRHF;
  std::size_t nmo = 0, ncore = 0, nactive = 0, frozen = 0;
  std::array<std::size_t, 2> electrons{}; // In the supplied orbital space.
  std::size_t root = 0, root_count = 1;
  std::vector<std::size_t> orbital_ids;
  std::string orbital_identity;
  std::string energy_source;
  std::string fock_source;
  // RHF: canonical; CASSCF: canonicalize(cas_natorb=False), core/virtual only.
  NDArray<double> orbital_energies, occupations, fock_mo;
  std::array<NDArray<double>, 4> rdms;
  double reference_energy = 0;

  void Validate() const;
  [[nodiscard]] runtime::Dimensions Dimensions() const;
  [[nodiscard]] std::size_t TensorBytes() const;
};

namespace reference_detail {
inline void Tensor(
    const NDArray<double>& tensor,
    const NDArray<double>::Shape& shape,
    const std::string& name) {
  if (tensor.shape() != shape) {
    throw std::invalid_argument("Reference shape mismatch: " + name);
  }
  for (std::size_t i = 0; i < tensor.Size(); ++i) {
    if (!std::isfinite(tensor.data()[tensor.LinearOffset(i)])) {
      throw std::invalid_argument("Non-finite reference tensor: " + name);
    }
  }
}
} // namespace reference_detail

inline void SpatialReference::Validate() const {
  if (nmo == 0 || ncore > nmo || nactive > nmo - ncore || frozen > ncore) {
    throw std::invalid_argument("Invalid reference orbital partition");
  }
  if (kind != ReferenceKind::kRHF && kind != ReferenceKind::kCASSCF) {
    throw std::invalid_argument("Unsupported spatial reference kind");
  }
  if ((kind == ReferenceKind::kRHF && nactive != 0) ||
      (kind == ReferenceKind::kCASSCF && nactive == 0)) {
    throw std::invalid_argument("Reference kind/active-space mismatch");
  }
  if (root_count == 0 || root >= root_count ||
      (kind == ReferenceKind::kRHF && (root != 0 || root_count != 1))) {
    throw std::invalid_argument("Invalid reference root");
  }
  for (auto n : electrons) {
    if (n < ncore || n - ncore > nactive) {
      throw std::invalid_argument("Reference electron/partition mismatch");
    }
  }
  if (orbital_identity.empty() || energy_source.empty() ||
      fock_source.empty() || orbital_ids.size() != nmo ||
      std::set<std::size_t>(orbital_ids.begin(), orbital_ids.end()).size() !=
          nmo) {
    throw std::invalid_argument("Missing/duplicate reference orbital identity");
  }
  if (!std::isfinite(reference_energy)) {
    throw std::invalid_argument("Non-finite reference energy");
  }
  reference_detail::Tensor(orbital_energies, {nmo}, "orbital_energies");
  reference_detail::Tensor(occupations, {nmo}, "occupations");
  reference_detail::Tensor(fock_mo, {nmo, nmo}, "fock_mo");
  for (std::size_t i = 0; i < nmo; ++i) {
    const auto occupation = occupations.At({i});
    if (occupation < -1e-10 || occupation > 2 + 1e-10 ||
        (i < ncore && std::abs(occupation - 2) > 1e-10) ||
        (i >= ncore + nactive && std::abs(occupation) > 1e-10)) {
      throw std::invalid_argument(
          "Reference occupations disagree with partition");
    }
    for (std::size_t j = 0; j < nmo; ++j) {
      const double value = fock_mo.At({i, j});
      if (std::abs(value - fock_mo.At({j, i})) >
          1e-10 * (1 + std::abs(value))) {
        throw std::invalid_argument("Reference Fock is not symmetric");
      }
      if (kind == ReferenceKind::kRHF &&
          std::abs(value - (i == j ? orbital_energies.At({i}) : 0)) > 1e-8) {
        throw std::invalid_argument(
            "RHF reference requires canonical Fock/energies");
      }
    }
  }
  double occupied = 0;
  for (std::size_t i = 0; i < nmo; ++i) {
    occupied += occupations.At({i});
  }
  if (std::abs(
          occupied - static_cast<double>(electrons[0]) -
          static_cast<double>(electrons[1])) > 1e-8) {
    throw std::invalid_argument("Reference occupations/electrons disagree");
  }
  if (kind == ReferenceKind::kCASSCF) {
    for (std::size_t i = 0; i < rdms.size(); ++i) {
      reference_detail::Tensor(
          rdms[i],
          NDArray<double>::Shape(2 * (i + 1), nactive),
          "E" + std::to_string(i + 1));
    }
    for (std::size_t a = 0; a < nactive; ++a) {
      if (std::abs(rdms[0].At({a, a}) - occupations.At({ncore + a})) > 1e-8) {
        throw std::invalid_argument("E1/active occupations disagree");
      }
    }
  }
}

inline runtime::Dimensions SpatialReference::Dimensions() const {
  Validate();
  return {
      {{1, 0}, ncore - frozen},
      {{2, 0}, nactive},
      {{8, 0}, nmo - ncore - nactive}};
}

inline std::size_t SpatialReference::TensorBytes() const {
  std::size_t size = 0;
  const auto add = [&size](const NDArray<double>& tensor) {
    if (tensor.Size() >
        std::numeric_limits<std::size_t>::max() / sizeof(double) - size) {
      throw std::overflow_error("Reference tensor payload exceeds size_t");
    }
    size += tensor.Size();
  };
  add(orbital_energies);
  add(occupations);
  add(fock_mo);
  if (kind == ReferenceKind::kCASSCF) {
    for (const auto& rdm : rdms) {
      add(rdm);
    }
  }
  return size * sizeof(double);
}
} // namespace wickqc::method
