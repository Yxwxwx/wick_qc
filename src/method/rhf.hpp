#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>
#include "backend/ndarray.hpp"
#include "method/specification.hpp"
#include "runtime/numeric.hpp"

namespace wickqc::method {

// Closed-shell canonical spatial MO data, occupied orbitals first. All arrays
// have runtime sizes. v[p,q,r,s]=(pq|rs); f is the MO Fock matrix, not h_core.
struct RHFData {
  std::size_t occupied = 0;
  NDArray<double> orbital_energies;
  NDArray<double> fock;
  NDArray<double> chemist_integrals;

  [[nodiscard]] runtime::Dimensions Dimensions() const;
  // Required amplitude names/shapes come from the kernel's Inputs(). They are
  // never implicitly initialized or solved here. Integral blocks remain views.
  [[nodiscard]] runtime::TensorMap<double> Bind(
      std::span<const runtime::TensorBinding> bindings,
      const runtime::TensorMap<double>& amplitudes,
      IntegralConvention convention = IntegralConvention::kChemist) const;
};

inline runtime::Dimensions RHFData::Dimensions() const {
  if (orbital_energies.Rank() != 1 || occupied == 0 ||
      occupied >= orbital_energies.Size()) {
    throw std::invalid_argument(
        "RHF data requires a 1D orbital-energy array and nonempty occupied/virtual spaces");
  }
  const auto nmo = orbital_energies.Size();
  if (fock.shape() != NDArray<double>::Shape{nmo, nmo} ||
      chemist_integrals.shape() != NDArray<double>::Shape{nmo, nmo, nmo, nmo}) {
    throw std::invalid_argument(
        "RHF Fock/integral dimensions do not match orbital energies");
  }
  return {{{1, 0}, occupied}, {{8, 0}, nmo - occupied}};
}

inline runtime::TensorMap<double> RHFData::Bind(
    std::span<const runtime::TensorBinding> bindings,
    const runtime::TensorMap<double>& amplitudes,
    IntegralConvention convention) const {
  const auto dimensions = Dimensions();
  if (convention != IntegralConvention::kChemist &&
      convention != IntegralConvention::kPhysicist) {
    throw std::invalid_argument("Invalid RHF integral convention");
  }
  const auto eri = convention == IntegralConvention::kChemist
      ? chemist_integrals
      : chemist_integrals.TransposeView({0, 2, 1, 3});
  runtime::TensorMap<double> inputs;
  for (const auto& binding : bindings) {
    const NDArray<double>* source = nullptr;
    const auto rank = binding.domains.size();
    if (binding.name.starts_with("eps") && rank == 1) {
      source = &orbital_energies;
    } else if (binding.name.starts_with("f") && rank == 2) {
      source = &fock;
    } else if (binding.name.starts_with("v") && rank == 4) {
      source = &eri;
    }
    if (source) {
      std::vector<NDArraySlice> slices;
      for (const auto domain : binding.domains) {
        if (domain.spins != 0 ||
            (domain.orbital_spaces != 1 && domain.orbital_spaces != 8)) {
          throw std::invalid_argument(
              "RHF tensor '" + binding.name +
              "' requires spin-free inactive/external axes");
        }
        const bool inactive = domain.orbital_spaces == 1;
        slices.push_back(
            NDArraySlice::Range(
                inactive ? 0 : occupied,
                inactive ? occupied : orbital_energies.Size()));
      }
      inputs.emplace(binding.name, source->Slice(slices));
    } else {
      const auto found = amplitudes.find(binding.name);
      if (found == amplitudes.end()) {
        throw std::invalid_argument(
            "Missing amplitude tensor '" + binding.name + "'");
      }
      if (found->second.shape() != binding.Shape(dimensions)) {
        throw std::invalid_argument(
            "Shape mismatch for amplitude tensor '" + binding.name + "'");
      }
      inputs.emplace(binding.name, found->second);
    }
  }
  return inputs;
}
} // namespace wickqc::method
