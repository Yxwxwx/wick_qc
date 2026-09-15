#pragma once

#include "backend/ndarray.hpp"
#include "method/integral_convention.h"
#include "runtime/tensor_binding.h"

#include <cstddef>
#include <span>

namespace wickqc::method {

// Closed-shell canonical spatial MO data, occupied orbitals first. All arrays
// have runtime sizes. v[p,q,r,s]=(pq|rs); f is the MO Fock matrix, not h_core.
struct RhfData {
  std::size_t occupied = 0;
  NDArray<double> orbital_energies;
  NDArray<double> fock;
  NDArray<double> chemist_integrals;

  [[nodiscard]] runtime::Dimensions Dimensions() const;
  // Required amplitude names/shapes come from the kernel's Inputs(). They are
  // never implicitly initialized or solved here. Integral blocks remain views.
  [[nodiscard]] runtime::TensorMap<double> Bind(std::span<const runtime::TensorBinding> bindings,
                                                const runtime::TensorMap<double>& amplitudes,
                                                IntegralConvention convention = IntegralConvention::kChemist) const;
};

} // namespace wickqc::method
