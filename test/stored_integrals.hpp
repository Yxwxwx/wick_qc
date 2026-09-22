#pragma once

#include "ao2mo/reference.hpp"

// Test-only dense reference path. The fixture supplies AO integrals and MO
// coefficients; every extent and integral value is read at runtime.
namespace wickqc::test {
using Array = NDArray<double>;

inline Array ReadTensor(
    const std::filesystem::path& path,
    const char* name,
    const Array::Shape& shape,
    std::size_t budget = 512ULL << 20) {
  auto file = ao2mo::h5::OpenFile(path, H5F_ACC_RDONLY);
  return ao2mo::reference_detail::Tensor(file, name, shape, budget);
}

struct StoredIntegrals {
  Array eri, h1e, fock;

  StoredIntegrals(
      const std::filesystem::path& path,
      const ao2mo::ReferenceInput& input) {
    const auto& coefficients =
        *input.integrals.coefficients.at(input.coefficient_family);
    const auto na = coefficients.nao, nm = coefficients.nmo;
    const Array c({na, nm}, coefficients.alpha);
    auto ao = ReadTensor(path, "ao/eri", {na, na, na, na});
    eri = Array::Einsum("up,uvwx->pvwx", {c, ao});
    ao = {};
    eri = Array::Einsum("vq,pvwx->pqwx", {c, eri});
    eri = Array::Einsum("wr,pqwx->pqrx", {c, eri});
    eri = Array::Einsum("xs,pqrx->pqrs", {c, eri});
    h1e = Array::Einsum(
        "up,uv,vq->pq", {c, ReadTensor(path, "ao/h1e", {na, na}), c});
    fock = Array::Einsum(
        "up,uv,vq->pq", {c, ReadTensor(path, "ao/fock", {na, na}), c});
    for (std::size_t p = 0; p < nm; ++p) {
      for (std::size_t q = 0; q < nm; ++q) {
        if (std::abs(h1e.At({p, q}) - input.integrals.h1e[p * nm + q]) >
                1e-10 ||
            std::abs(fock.At({p, q}) - input.reference.fock_mo.At({p, q})) >
                1e-10) {
          throw std::runtime_error(
              "AO one-electron transform disagrees with fixture");
        }
      }
    }
  }

  method::RHFData RHF(const ao2mo::ReferenceInput& input) const {
    const auto& p = input.integrals.partition;
    const auto retained = NDArraySlice::Range(p.frozen, p.nmo);
    method::RHFData data;
    data.occupied = p.ncore - p.frozen;
    data.orbital_energies = input.reference.orbital_energies.Slice({retained});
    data.fock = fock.Slice({retained, retained});
    data.chemist_integrals =
        eri.Slice({retained, retained, retained, retained});
    return data;
  }

  runtime::TensorMap<double> NEVPT2(
      const ao2mo::ReferenceInput& input,
      std::span<const runtime::TensorBinding> bindings) const {
    const auto& p = input.integrals.partition;
    auto h1eff = h1e.Clone();
    // Inactive-core dressing includes the frozen prefix as in DressCore.
    for (std::size_t x = 0; x < p.nmo; ++x) {
      for (std::size_t y = 0; y < p.nmo; ++y) {
        for (std::size_t i = 0; i < p.ncore; ++i) {
          h1eff.At({x, y}) += 2 * eri.At({x, y, i, i}) - eri.At({x, i, i, y});
        }
      }
    }
    const auto physicist = eri.TransposeView({0, 2, 1, 3});
    runtime::TensorMap<double> result;
    for (const auto& binding : bindings) {
      const auto axes = ao2mo::reference_detail::Axes(binding);
      const auto& name = binding.name;
      if (name == "w" + axes) {
        result.emplace(
            name, ao2mo::reference_detail::Slice(physicist, p, axes));
      } else if (name == "h" + axes || name == "f" + axes) {
        result.emplace(name, ao2mo::reference_detail::Slice(h1eff, p, axes));
      } else if (name == "orbe" + axes) {
        result.emplace(
            name,
            ao2mo::reference_detail::Slice(
                input.reference.orbital_energies, p, axes));
      } else if (name == "E" + std::to_string(axes.size() / 2)) {
        result.emplace(name, input.reference.rdms.at(axes.size() / 2 - 1));
      } else {
        throw std::invalid_argument("Unknown fixture binding: " + name);
      }
    }
    runtime::numeric_detail::ValidateInputs(
        bindings, result, input.reference.Dimensions());
    return result;
  }
};
} // namespace wickqc::test
