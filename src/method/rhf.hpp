#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
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
  // Alternative to the dense tensor: only the blocks requested by a kernel.
  runtime::TensorMap<double> integral_blocks;
  std::optional<IntegralConvention> block_convention;

  [[nodiscard]] runtime::Dimensions Dimensions() const;
  // Required amplitude names/shapes come from the kernel's Inputs(). They are
  // never implicitly initialized or solved here. Integral blocks remain views.
  [[nodiscard]] runtime::TensorMap<double> Bind(
      std::span<const runtime::TensorBinding> bindings,
      const runtime::TensorMap<double>& amplitudes,
      IntegralConvention convention = IntegralConvention::kChemist) const;
};

// Excitation axes are (external, inactive) and (external, external,
// inactive, inactive). Doubles obey t(abij)=t(baji), without
// antisymmetrization.
struct RHFAmplitudes {
  NDArray<double> singles, doubles;
};

struct MP2Result {
  double correlation_energy = 0, total_energy = 0, residual_norm = 0;
  RHFAmplitudes amplitudes;
  std::optional<std::size_t> estimated_peak_bytes;
};

struct CCSDOptions {
  std::size_t max_iterations = 100;
  double energy_tolerance = 1e-10;
  double residual_tolerance = 1e-8;
  runtime::MemoryBudget memory{};
};

struct CCSDIteration {
  std::size_t iteration = 0; // Number of completed Jacobi updates.
  double correlation_energy = 0, energy_change = 0, residual_norm = 0;
};

struct CCSDResult {
  double correlation_energy = 0, total_energy = 0, residual_norm = 0;
  std::size_t iterations = 0;
  bool converged = false;
  RHFAmplitudes amplitudes;
  RHFAmplitudes
      residuals; // Physical residuals after inverting the spin metric.
  std::vector<CCSDIteration> history; // Includes the initial evaluation (0).
  std::optional<std::size_t> estimated_peak_bytes;
};

inline runtime::Dimensions RHFData::Dimensions() const {
  if (orbital_energies.Rank() != 1 || occupied == 0 ||
      occupied >= orbital_energies.Size()) {
    throw std::invalid_argument(
        "RHF data requires a 1D orbital-energy array and nonempty occupied/virtual spaces");
  }
  const auto nmo = orbital_energies.Size();
  if (fock.shape() != NDArray<double>::Shape{nmo, nmo} ||
      (!block_convention &&
       chemist_integrals.shape() !=
           NDArray<double>::Shape{nmo, nmo, nmo, nmo})) {
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
  if (block_convention && convention != *block_convention) {
    throw std::invalid_argument("RHF integral block convention mismatch");
  }
  const auto eri =
      !block_convention && convention == IntegralConvention::kPhysicist
      ? chemist_integrals.TransposeView({0, 2, 1, 3})
      : chemist_integrals;
  runtime::TensorMap<double> inputs;
  for (const auto& binding : bindings) {
    const NDArray<double>* source = nullptr;
    const auto rank = binding.domains.size();
    std::string axes;
    for (const auto domain : binding.domains) {
      if (domain.spins != 0 ||
          (domain.orbital_spaces != 1 && domain.orbital_spaces != 8)) {
        throw std::invalid_argument(
            "RHF tensor '" + binding.name +
            "' requires spin-free inactive/external axes");
      }
      axes += domain.orbital_spaces == 1 ? 'I' : 'E';
    }
    if (binding.name == "eps" + axes && rank == 1) {
      source = &orbital_energies;
    } else if (binding.name == "f" + axes && rank == 2) {
      source = &fock;
    } else if (binding.name == "v" + axes && rank == 4) {
      if (block_convention) {
        const auto block = integral_blocks.find(binding.name);
        if (block == integral_blocks.end() ||
            block->second.shape() != binding.Shape(dimensions)) {
          throw std::invalid_argument(
              "Missing or wrong-shaped RHF integral block '" + binding.name +
              "'");
        }
        inputs.emplace(binding.name, block->second);
        continue;
      }
      source = &eri;
    }
    if (source) {
      std::vector<NDArraySlice> slices;
      for (const auto domain : binding.domains) {
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

namespace rhf_detail {
template <typename Kernel>
inline std::optional<std::size_t> CheckMemory(
    const RHFData& data,
    const Kernel& kernel,
    const runtime::MemoryBudget& budget,
    std::size_t history_bytes,
    const std::optional<RHFAmplitudes>& initial = std::nullopt) {
  using runtime::memory_detail::Add;
  using runtime::memory_detail::Multiply;
  const auto dimensions = data.Dimensions();
  const auto workspace =
      runtime::numeric_detail::WorkspaceBytes(kernel, dimensions, budget);
  if (!workspace) {
    return std::nullopt;
  }
  std::vector<NDArray<double>> resident{
      data.orbital_energies, data.fock, data.chemist_integrals};
  for (const auto& [name, block] : data.integral_blocks) {
    resident.push_back(block);
  }
  if (initial) {
    resident.push_back(initial->singles);
    resident.push_back(initial->doubles);
  }
  const auto singles =
      Multiply(data.occupied, data.orbital_energies.Size() - data.occupied);
  const auto amplitudes =
      Multiply(Add(singles, Multiply(singles, singles)), sizeof(double));
  // Initial/current/updated amplitudes and physical residuals, including copies
  // during spin-metric conversion. Kernel outputs are covered by workspace.
  const auto bytes =
      Add(NDArray<double>::StorageBytes(resident),
          Add(*workspace, Add(Multiply(5, amplitudes), history_bytes)));
  return budget.Check(bytes, "RHF method");
}
inline void Finite(const NDArray<double>& array, const std::string& name) {
  for (std::size_t i = 0; i < array.Size(); ++i) {
    if (!std::isfinite(array.data()[array.LinearOffset(i)])) {
      throw std::invalid_argument("Non-finite RHF tensor: " + name);
    }
  }
}

inline void Validate(const RHFData& data, double reference_energy) {
  (void)data.Dimensions();
  if (!std::isfinite(reference_energy)) {
    throw std::invalid_argument("Non-finite RHF reference energy");
  }
  Finite(data.orbital_energies, "orbital_energies");
  Finite(data.fock, "fock");
  for (std::size_t i = 0; i < data.orbital_energies.Size(); ++i) {
    for (std::size_t j = 0; j < data.orbital_energies.Size(); ++j) {
      if (std::abs(
              data.fock.At({i, j}) -
              (i == j ? data.orbital_energies.At({i}) : 0)) > 1e-8) {
        throw std::invalid_argument(
            "MP2/CCSD requires canonical RHF Fock and orbital energies");
      }
    }
  }
}

inline void CheckAmplitudes(
    const RHFData& data,
    const RHFAmplitudes& amplitudes,
    const char* name = "amplitudes") {
  const auto ni = data.occupied, ne = data.orbital_energies.Size() - ni;
  if (amplitudes.singles.shape() != NDArray<double>::Shape{ne, ni} ||
      amplitudes.doubles.shape() != NDArray<double>::Shape{ne, ne, ni, ni}) {
    throw std::invalid_argument("RHF singles/doubles amplitude shape mismatch");
  }
  Finite(amplitudes.singles, "singles");
  Finite(amplitudes.doubles, "doubles");
  for (std::size_t a = 0; a < ne; ++a) {
    for (std::size_t b = 0; b < ne; ++b) {
      for (std::size_t i = 0; i < ni; ++i) {
        for (std::size_t j = 0; j < ni; ++j) {
          const auto value = amplitudes.doubles.At({a, b, i, j});
          if (std::abs(value - amplitudes.doubles.At({b, a, j, i})) >
              1e-10 * (1 + std::abs(value))) {
            std::ostringstream message;
            message.precision(17);
            message << "RHF " << name << " violate t(abij)=t(baji) at " << a
                    << ',' << b << ',' << i << ',' << j << ": " << value
                    << " vs " << amplitudes.doubles.At({b, a, j, i});
            throw std::invalid_argument(message.str());
          }
        }
      }
    }
  }
}

inline double Gap(const RHFData& data, std::size_t a, std::size_t i) {
  const auto gap = data.orbital_energies.At({i}) -
      data.orbital_energies.At({data.occupied + a});
  if (!std::isfinite(gap) || gap >= -1e-10) {
    throw std::invalid_argument(
        "Expected negative occupied-virtual orbital-energy gaps");
  }
  return gap;
}

// The equations are defined on pair-symmetric amplitudes. Preserve that
// manifold exactly between iterations (as compressed CC amplitude storage
// does); otherwise roundoff in redundant entries can grow off the manifold.
// Call only after CheckAmplitudes, so this cannot hide a broken equation/input.
inline void SymmetrizeDoubles(NDArray<double>& doubles) {
  const auto ne = doubles.shape()[0], ni = doubles.shape()[2];
  for (std::size_t a = 0; a < ne; ++a) {
    for (std::size_t i = 0; i < ni; ++i) {
      for (std::size_t b = 0; b < ne; ++b) {
        for (std::size_t j = 0; j < ni; ++j) {
          if (a * ni + i < b * ni + j) {
            const auto value = std::midpoint(
                doubles.At({a, b, i, j}), doubles.At({b, a, j, i}));
            doubles.At({a, b, i, j}) = doubles.At({b, a, j, i}) = value;
          }
        }
      }
    }
  }
}

template <typename Kernel>
void CheckKernel(const Kernel& kernel, bool mp2) {
  const std::vector<runtime::TensorBinding> expected{
      {mp2 ? "energy2" : "energy", {}},
      {mp2 ? "residual1_rank1" : "residual1", {{8, 0}, {1, 0}}},
      {mp2 ? "residual1_rank2" : "residual2",
       {{8, 0}, {8, 0}, {1, 0}, {1, 0}}}};
  if (kernel.Outputs().size() != expected.size()) {
    throw std::invalid_argument(
        "RHF driver requires an MP2 or CCSD kernel, not a higher-rank kernel");
  }
  for (const auto& binding : expected) {
    if (std::find(kernel.Outputs().begin(), kernel.Outputs().end(), binding) ==
        kernel.Outputs().end()) {
      throw std::invalid_argument(
          "Wrong kernel output contract for RHF driver: " + binding.name);
    }
  }
}

inline double Norm(const RHFAmplitudes& amplitudes) {
  const auto norm =
      std::hypot(amplitudes.singles.Norm(), amplitudes.doubles.Norm());
  if (!std::isfinite(norm)) {
    throw std::runtime_error("Non-finite RHF residual norm");
  }
  return norm;
}
} // namespace rhf_detail

// Include this block in an AO2MO preparation contract if the chosen energy
// kernel does not already request it. No full pppp tensor is needed.
inline runtime::TensorBinding MP2IntegralBinding(
    IntegralConvention convention) {
  if (convention == IntegralConvention::kChemist) {
    return {"vEIEI", {{8, 0}, {1, 0}, {8, 0}, {1, 0}}};
  }
  if (convention == IntegralConvention::kPhysicist) {
    return {"vEEII", {{8, 0}, {8, 0}, {1, 0}, {1, 0}}};
  }
  throw std::invalid_argument("Invalid RHF integral convention");
}

inline RHFAmplitudes MP2Amplitudes(
    const RHFData& data,
    IntegralConvention convention) {
  rhf_detail::Validate(data, 0);
  const auto ni = data.occupied, ne = data.orbital_energies.Size() - ni;
  RHFAmplitudes amplitudes{
      NDArray<double>({ne, ni}), NDArray<double>({ne, ne, ni, ni})};
  const auto binding = MP2IntegralBinding(convention);
  const auto inputs = data.Bind(std::array{binding}, {}, convention);
  const auto& eri = inputs.at(binding.name);
  rhf_detail::Finite(eri, binding.name);
  for (std::size_t a = 0; a < ne; ++a) {
    for (std::size_t b = 0; b < ne; ++b) {
      for (std::size_t i = 0; i < ni; ++i) {
        for (std::size_t j = 0; j < ni; ++j) {
          const auto integral = convention == IntegralConvention::kChemist
              ? eri.At({a, i, b, j})
              : eri.At({a, b, i, j});
          amplitudes.doubles.At({a, b, i, j}) = integral /
              (rhf_detail::Gap(data, a, i) + rhf_detail::Gap(data, b, j));
        }
      }
    }
  }
  rhf_detail::CheckAmplitudes(data, amplitudes);
  rhf_detail::SymmetrizeDoubles(amplitudes.doubles);
  return amplitudes;
}

// Invert the spin-free projection metric: r1=2 R1 and
// r2(abij)=4 R2(abij)-2 R2(abji). Do not divide covariant r2 by a gap directly.
inline RHFAmplitudes CCSDResiduals(
    const NDArray<double>& r1,
    const NDArray<double>& r2) {
  if (r1.Rank() != 2 ||
      r2.shape() !=
          NDArray<double>::Shape{
              r1.shape()[0], r1.shape()[0], r1.shape()[1], r1.shape()[1]}) {
    throw std::invalid_argument("CCSD covariant residual shape mismatch");
  }
  RHFAmplitudes result{r1 * 0.5, NDArray<double>(r2.shape())};
  for (std::size_t a = 0; a < r1.shape()[0]; ++a) {
    for (std::size_t b = 0; b < r1.shape()[0]; ++b) {
      for (std::size_t i = 0; i < r1.shape()[1]; ++i) {
        for (std::size_t j = 0; j < r1.shape()[1]; ++j) {
          result.doubles.At({a, b, i, j}) =
              r2.At({a, b, i, j}) / 3 + r2.At({a, b, j, i}) / 6;
        }
      }
    }
  }
  rhf_detail::Finite(result.singles, "singles residual");
  rhf_detail::Finite(result.doubles, "doubles residual");
  return result;
}

// One undamped Jacobi step, with physical (metric-inverted) residuals.
// Inputs are never mutated, including when they are NDArray views.
inline RHFAmplitudes CCSDStep(
    const RHFData& data,
    const RHFAmplitudes& amplitudes,
    const RHFAmplitudes& residuals) {
  (void)data.Dimensions();
  rhf_detail::CheckAmplitudes(data, amplitudes);
  rhf_detail::CheckAmplitudes(data, residuals, "physical residuals");
  const auto ni = data.occupied, ne = data.orbital_energies.Size() - ni;
  RHFAmplitudes next{amplitudes.singles.Clone(), amplitudes.doubles.Clone()};
  for (std::size_t a = 0; a < ne; ++a) {
    for (std::size_t i = 0; i < ni; ++i) {
      next.singles.At({a, i}) +=
          residuals.singles.At({a, i}) / rhf_detail::Gap(data, a, i);
      for (std::size_t b = 0; b < ne; ++b) {
        for (std::size_t j = 0; j < ni; ++j) {
          next.doubles.At({a, b, i, j}) += residuals.doubles.At({a, b, i, j}) /
              (rhf_detail::Gap(data, a, i) + rhf_detail::Gap(data, b, j));
        }
      }
    }
  }
  rhf_detail::CheckAmplitudes(data, next, "updated amplitudes");
  rhf_detail::SymmetrizeDoubles(next.doubles);
  return next;
}

// Kernel can be NumericKernel, SpatialEvaluator or NDArrayExecutor. The
// caller constructs it once; no symbolic work happens in these drivers.
template <typename Kernel>
MP2Result SolveMP2(
    const RHFData& data,
    const Kernel& kernel,
    double reference_energy,
    IntegralConvention convention = IntegralConvention::kChemist,
    const runtime::MemoryBudget& memory = {}) {
  rhf_detail::Validate(data, reference_energy);
  rhf_detail::CheckKernel(kernel, true);
  MP2Result result;
  result.estimated_peak_bytes =
      rhf_detail::CheckMemory(data, kernel, memory, 0);
  result.amplitudes = MP2Amplitudes(data, convention);
  const auto values = kernel.Evaluate(
      data.Bind(
          kernel.Inputs(),
          {{"u1EI", result.amplitudes.singles},
           {"u1EEII", result.amplitudes.doubles}},
          convention),
      data.Dimensions());
  result.correlation_energy = values.at("energy2").Item();
  result.total_energy = reference_energy + result.correlation_energy;
  result.residual_norm = rhf_detail::Norm(CCSDResiduals(
      values.at("residual1_rank1"), values.at("residual1_rank2")));
  if (!std::isfinite(result.total_energy)) {
    throw std::runtime_error("Non-finite MP2 energy");
  }
  return result;
}

template <typename Kernel>
CCSDResult SolveCCSD(
    const RHFData& data,
    const Kernel& kernel,
    double reference_energy,
    IntegralConvention convention = IntegralConvention::kChemist,
    const CCSDOptions& options = {},
    const std::optional<RHFAmplitudes>& initial = std::nullopt) {
  rhf_detail::Validate(data, reference_energy);
  rhf_detail::CheckKernel(kernel, false);
  if (!std::isfinite(options.energy_tolerance) ||
      options.energy_tolerance <= 0 ||
      !std::isfinite(options.residual_tolerance) ||
      options.residual_tolerance <= 0) {
    throw std::invalid_argument(
        "CCSD convergence tolerances must be positive and finite");
  }
  CCSDResult result;
  const auto history_bytes = runtime::memory_detail::Multiply(
      runtime::memory_detail::Add(options.max_iterations, 1),
      3 * sizeof(CCSDIteration));
  result.estimated_peak_bytes = rhf_detail::CheckMemory(
      data, kernel, options.memory, history_bytes, initial);
  if (initial) {
    rhf_detail::CheckAmplitudes(data, *initial);
    result.amplitudes = {initial->singles.Clone(), initial->doubles.Clone()};
    rhf_detail::SymmetrizeDoubles(result.amplitudes.doubles);
  } else {
    result.amplitudes = MP2Amplitudes(data, convention);
    for (std::size_t a = 0; a < result.amplitudes.singles.shape()[0]; ++a) {
      for (std::size_t i = 0; i < data.occupied; ++i) {
        result.amplitudes.singles.At({a, i}) =
            data.fock.At({data.occupied + a, i}) / rhf_detail::Gap(data, a, i);
      }
    }
  }
  // ponytail: undamped Jacobi keeps the validated single-step semantics;
  // add bounded DIIS history if representative cases require acceleration.
  for (std::size_t iteration = 0;; ++iteration) {
    const double previous = result.correlation_energy;
    const auto values = kernel.Evaluate(
        data.Bind(
            kernel.Inputs(),
            {{"tEI", result.amplitudes.singles},
             {"tEEII", result.amplitudes.doubles}},
            convention),
        data.Dimensions());
    result.correlation_energy = values.at("energy").Item();
    result.total_energy = reference_energy + result.correlation_energy;
    if (!std::isfinite(result.total_energy)) {
      throw std::runtime_error("Non-finite CCSD energy");
    }
    result.residuals =
        CCSDResiduals(values.at("residual1"), values.at("residual2"));
    result.residual_norm = rhf_detail::Norm(result.residuals);
    result.iterations = iteration;
    const auto change =
        iteration == 0 ? 0 : result.correlation_energy - previous;
    result.history.push_back(
        {iteration, result.correlation_energy, change, result.residual_norm});
    result.converged = iteration > 0 &&
        std::abs(change) < options.energy_tolerance &&
        result.residual_norm < options.residual_tolerance;
    if (result.converged || iteration == options.max_iterations) {
      return result;
    }
    try {
      result.amplitudes = CCSDStep(data, result.amplitudes, result.residuals);
    } catch (const std::invalid_argument& error) {
      throw std::invalid_argument(
          "CCSD update " + std::to_string(iteration + 1) + ": " + error.what());
    }
  }
}
} // namespace wickqc::method
