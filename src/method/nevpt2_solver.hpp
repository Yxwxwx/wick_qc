#pragma once

#include "method/reference.hpp"

#if defined(WICKQC_LAPACK_MKL) || defined(WICKQC_LAPACK_OPENBLAS) || \
    defined(WICKQC_LAPACK_NETLIB) || defined(WICKQC_LAPACK_EIGEN)
#include "backend/lapack.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace wickqc::method {

inline constexpr std::array<std::string_view, 8>
    kSCNEVPT2Subspaces{"ijrs", "rsi", "ijr", "rs", "ij", "ir", "r", "i"};
inline constexpr std::array<std::string_view, 13> kICNEVPT2Subspaces{
    "ijrs_plus",
    "ijrs_minus",
    "rsiap_plus",
    "rsiap_minus",
    "ijrap_plus",
    "ijrap_minus",
    "rsabpq_plus",
    "rsabpq_minus",
    "ijabpq_plus",
    "ijabpq_minus",
    "irabpq",
    "rabcpqg",
    "iabcpqg"};

struct NEVPT2Result {
  double correlation_energy = 0, total_energy = 0;
  std::map<std::string, double> subspace_energies;
  // SC only; IC reports energies summed over its +/- components.
  std::map<std::string, double> subspace_norms;
  std::optional<std::size_t> estimated_peak_bytes;
  double contraction_seconds = 0, assembly_seconds = 0, solve_seconds = 0;
};

struct SCNEVPT2BlockResult {
  double correlation_energy = 0, norm = 0;
  std::size_t retained = 0;
};

namespace nevpt2_solver_detail {
inline std::size_t Product(std::span<const std::size_t> shape) {
  std::size_t result = 1;
  for (const auto n : shape) {
    if (n && result > std::numeric_limits<std::size_t>::max() / n) {
      throw std::overflow_error("NEVPT2 tensor dimensions exceed size_t");
    }
    result *= n;
  }
  return result;
}

inline std::vector<std::size_t> Indices(
    std::size_t linear,
    std::span<const std::size_t> shape) {
  std::vector<std::size_t> result(shape.size());
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    result[axis] = linear % shape[axis];
    linear /= shape[axis];
  }
  return result;
}

inline NDArray<double>::Shape OuterShape(
    std::string_view name,
    std::size_t core,
    std::size_t external) {
  NDArray<double>::Shape shape;
  for (const auto index : name) {
    shape.push_back(index == 'i' || index == 'j' ? core : external);
  }
  return shape;
}

inline bool Ordered(
    std::string_view name,
    std::span<const std::size_t> indices,
    bool strict) {
  for (const auto pair : {"ij", "rs"}) {
    const auto axis = name.find(pair);
    if (axis != std::string_view::npos &&
        (strict ? indices[axis] >= indices[axis + 1]
                : indices[axis] > indices[axis + 1])) {
      return false;
    }
  }
  return true;
}

inline const NDArray<double>& Tensor(
    const runtime::TensorMap<double>& tensors,
    const std::string& name,
    const NDArray<double>::Shape& shape) {
  const auto found = tensors.find(name);
  if (found == tensors.end()) {
    throw std::invalid_argument("Missing NEVPT2 tensor '" + name + "'");
  }
  reference_detail::Tensor(found->second, shape, name);
  return found->second;
}

inline double Element(const NDArray<double>& tensor, std::size_t linear) {
  return tensor.data()[tensor.LinearOffset(linear)];
}

inline void Finish(NEVPT2Result& result, double reference) {
  for (const auto& [name, energy] : result.subspace_energies) {
    result.correlation_energy += energy;
  }
  result.total_energy = reference + result.correlation_energy;
  if (!std::isfinite(result.total_energy)) {
    throw std::runtime_error("Non-finite NEVPT2 energy");
  }
}

inline runtime::Dimensions Dimensions(const SpatialReference& reference) {
  const auto dimensions = reference.Dimensions();
  if (reference.kind != ReferenceKind::kCASSCF) {
    throw std::invalid_argument("NEVPT2 requires a spatial CASSCF reference");
  }
  return dimensions;
}

inline std::size_t ResidentBytes(
    const runtime::TensorMap<double>& inputs,
    const SpatialReference& reference) {
  std::vector<NDArray<double>> tensors{
      reference.orbital_energies, reference.occupations, reference.fock_mo};
  tensors.insert(tensors.end(), reference.rdms.begin(), reference.rdms.end());
  for (const auto& [name, tensor] : inputs) {
    tensors.push_back(tensor);
  }
  return NDArray<double>::StorageBytes(tensors);
}

template <typename Kernel>
inline std::optional<std::size_t> CheckMemory(
    const Kernel& kernel,
    const runtime::Dimensions& dimensions,
    std::size_t resident,
    const runtime::MemoryBudget& budget,
    bool ic) {
  using runtime::memory_detail::Add;
  using runtime::memory_detail::Multiply;
  const auto workspace =
      runtime::numeric_detail::WorkspaceBytes(kernel, dimensions, budget);
  if (!workspace) {
    return std::nullopt;
  }
  std::size_t bytes = Add(resident, *workspace);
  if (ic) {
    // Raw, restricted, and row-major matrix/RHS copies plus solution/singular
    // arrays. Use unrestricted dimensions, which bound every pair selection.
    for (const auto& output : kernel.Outputs()) {
      const auto size = Product(output.Shape(dimensions));
      const auto copies = output.name.starts_with("hexp") ? 4U : 6U;
      bytes = Add(bytes, Multiply(Multiply(copies, size), sizeof(double)));
    }
  }
  return budget.Check(bytes, ic ? "IC-NEVPT2 subspace" : "SC-NEVPT2 subspace");
}
} // namespace nevpt2_solver_detail

// norm and hexp are the unsummed SC generator outputs. No metric or denominator
// threshold is added beyond block2's abs(norm) > 1e-14 selection.
[[nodiscard]] inline SCNEVPT2BlockResult SCNEVPT2Block(
    std::string_view name,
    const runtime::TensorMap<double>& tensors,
    const NDArray<double>& core_energies,
    const NDArray<double>& external_energies) {
  using nevpt2_solver_detail::Element;
  using nevpt2_solver_detail::Indices;
  using nevpt2_solver_detail::Ordered;
  using nevpt2_solver_detail::OuterShape;
  using nevpt2_solver_detail::Tensor;
  if (std::ranges::find(kSCNEVPT2Subspaces, name) == kSCNEVPT2Subspaces.end()) {
    throw std::invalid_argument(
        "Unknown SC-NEVPT2 subspace: " + std::string(name));
  }
  reference_detail::Tensor(core_energies, {core_energies.Size()}, "orbeI");
  reference_detail::Tensor(
      external_energies, {external_energies.Size()}, "orbeE");
  const auto shape =
      OuterShape(name, core_energies.Size(), external_energies.Size());
  const auto& norms = Tensor(tensors, "norm", shape);
  const auto& hexp = Tensor(tensors, "hexp", shape);
  SCNEVPT2BlockResult result;
  for (std::size_t linear = 0; linear < norms.Size(); ++linear) {
    const auto indices = Indices(linear, shape);
    const auto norm = Element(norms, linear);
    if (!Ordered(name, indices, false) || std::abs(norm) <= 1e-14) {
      continue;
    }
    double denominator = 0;
    for (std::size_t axis = 0; axis < name.size(); ++axis) {
      denominator += name[axis] == 'i' || name[axis] == 'j'
          ? -Element(core_energies, indices[axis])
          : Element(external_energies, indices[axis]);
    }
    denominator += Element(hexp, linear) / norm;
    if (!std::isfinite(denominator) || denominator == 0) {
      throw std::runtime_error(
          "Invalid SC-NEVPT2 denominator in " + std::string(name));
    }
    const double energy = -norm / denominator;
    if (!std::isfinite(energy)) {
      throw std::runtime_error(
          "Invalid SC-NEVPT2 denominator in " + std::string(name));
    }
    result.correlation_energy += energy;
    result.norm += norm;
    ++result.retained;
  }
  if (!std::isfinite(result.correlation_energy) ||
      !std::isfinite(result.norm)) {
    throw std::runtime_error(
        "Non-finite SC-NEVPT2 sum in " + std::string(name));
  }
  return result;
}

// The optional observer can stream diagnostics to disk before a block's
// tensors are released. Normal execution retains only the scalar summaries.
template <typename Kernel>
[[nodiscard]] NEVPT2Result SolveSCNEVPT2(
    const std::map<std::string, Kernel>& kernels,
    const runtime::TensorMap<double>& inputs,
    const SpatialReference& reference,
    const std::function<void(
        std::string_view,
        const runtime::TensorMap<double>&,
        const SCNEVPT2BlockResult&)>& observe = {},
    const runtime::MemoryBudget& memory = {}) {
  using nevpt2_solver_detail::CheckMemory;
  using nevpt2_solver_detail::Dimensions;
  using nevpt2_solver_detail::Finish;
  using nevpt2_solver_detail::ResidentBytes;
  const auto dimensions = Dimensions(reference);
  const auto core = reference.orbital_energies.Slice(
      {NDArraySlice::Range(reference.frozen, reference.ncore)});
  const auto external = reference.orbital_energies.Slice({NDArraySlice::Range(
      reference.ncore + reference.nactive, reference.nmo)});
  if (kernels.size() != kSCNEVPT2Subspaces.size()) {
    throw std::invalid_argument("SC-NEVPT2 requires all eight block kernels");
  }
  NEVPT2Result result;
  const auto resident = ResidentBytes(inputs, reference);
  result.estimated_peak_bytes = 0;
  for (const auto& [name, kernel] : kernels) {
    const auto peak = CheckMemory(kernel, dimensions, resident, memory, false);
    result.estimated_peak_bytes = peak && result.estimated_peak_bytes
        ? std::optional(std::max(*result.estimated_peak_bytes, *peak))
        : std::nullopt;
  }
  for (const auto name : kSCNEVPT2Subspaces) {
    const auto& kernel = kernels.at(std::string(name));
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    const auto tensors = kernel.Evaluate(inputs, dimensions);
    result.contraction_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();
    start = Clock::now();
    const auto block = SCNEVPT2Block(name, tensors, core, external);
    result.solve_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();
    result.subspace_energies.emplace(name, block.correlation_energy);
    result.subspace_norms.emplace(name, block.norm);
    if (observe) {
      observe(name, tensors, block);
    }
  }
  Finish(result, reference.reference_energy);
  return result;
}

struct ICNEVPT2System {
  // Batch, bra, ket; rhs is batch, ket. The two ir components are interleaved
  // inside each active tuple: row = active_tuple * 2 + component.
  NDArray<double> hamiltonian, rhs;
};

namespace nevpt2_solver_detail {
struct ICSubspace {
  std::string_view outer;
  std::size_t active_rank;
  bool restricted = false, strict = false, coupled = false;
};

inline ICSubspace ICDescription(std::string_view name) {
  if (std::ranges::find(kICNEVPT2Subspaces, name) == kICNEVPT2Subspaces.end()) {
    throw std::invalid_argument(
        "Unknown IC-NEVPT2 subspace: " + std::string(name));
  }
  const bool restricted = name.ends_with("_plus") || name.ends_with("_minus");
  const bool strict = name.ends_with("_minus");
  if (name.starts_with("ijrs")) {
    return {"ijrs", 0, restricted, strict};
  }
  if (name.starts_with("rsiap")) {
    return {"rsi", 1, restricted, strict};
  }
  if (name.starts_with("ijrap")) {
    return {"ijr", 1, restricted, strict};
  }
  if (name.starts_with("rsabpq")) {
    return {"rs", 2, restricted, strict};
  }
  if (name.starts_with("ijabpq")) {
    return {"ij", 2, restricted, strict};
  }
  if (name == "irabpq") {
    return {"ir", 2, false, false, true};
  }
  return {name == "rabcpqg" ? "r" : "i", 3};
}
} // namespace nevpt2_solver_detail

// Restrict outer/active pairs in C order, matching the full IC/FIC equations.
// Triple active tuples are unrestricted. No orthogonalization or symmetrization
// is performed; in particular H12 and H21 are taken from their own equations.
[[nodiscard]] inline ICNEVPT2System AssembleICNEVPT2(
    std::string_view name,
    const runtime::TensorMap<double>& tensors,
    const runtime::Dimensions& dimensions) {
  using nevpt2_solver_detail::Element;
  using nevpt2_solver_detail::ICDescription;
  using nevpt2_solver_detail::Indices;
  using nevpt2_solver_detail::Ordered;
  using nevpt2_solver_detail::OuterShape;
  using nevpt2_solver_detail::Product;
  using nevpt2_solver_detail::Tensor;
  const auto description = ICDescription(name);
  const auto outer_shape = OuterShape(
      description.outer, dimensions.at({1, 0}), dimensions.at({8, 0}));
  const NDArray<double>::Shape active_shape(
      description.active_rank, dimensions.at({2, 0}));
  const auto active_size = Product(active_shape);
  std::vector<std::size_t> outer, active;
  for (std::size_t i = 0; i < Product(outer_shape); ++i) {
    if (Ordered(
            description.outer, Indices(i, outer_shape), description.strict)) {
      outer.push_back(i);
    }
  }
  for (std::size_t i = 0; i < active_size; ++i) {
    const auto indices = Indices(i, active_shape);
    if (description.restricted && description.active_rank == 2 &&
        (description.strict ? indices[0] >= indices[1]
                            : indices[0] > indices[1])) {
      continue;
    }
    active.push_back(i);
  }
  auto rhs_shape = outer_shape;
  rhs_shape.insert(rhs_shape.end(), active_shape.begin(), active_shape.end());
  auto h_shape = rhs_shape;
  h_shape.insert(h_shape.end(), active_shape.begin(), active_shape.end());
  (void)Product(h_shape);
  const std::size_t components = description.coupled ? 2 : 1;
  std::array<const NDArray<double>*, 2> rhs{};
  std::array<const NDArray<double>*, 4> h{};
  for (std::size_t c = 0; c < components; ++c) {
    rhs[c] = &Tensor(
        tensors,
        description.coupled ? "rheq" + std::to_string(c + 1) : "rheq",
        rhs_shape);
    for (std::size_t d = 0; d < components; ++d) {
      h[c * components + d] = &Tensor(
          tensors,
          description.coupled
              ? "hexp" + std::to_string(c + 1) + std::to_string(d + 1)
              : "hexp",
          h_shape);
    }
  }
  const auto width = Product(std::array{active.size(), components});
  (void)Product(std::array{outer.size(), width, width, sizeof(double)});
  ICNEVPT2System system{
      NDArray<double>({outer.size(), width, width}),
      NDArray<double>({outer.size(), width})};
  for (std::size_t b = 0; b < outer.size(); ++b) {
    for (std::size_t a = 0; a < active.size(); ++a) {
      for (std::size_t c = 0; c < components; ++c) {
        const auto row = a * components + c;
        system.rhs.At({b, row}) =
            Element(*rhs[c], outer[b] * active_size + active[a]);
        for (std::size_t k = 0; k < active.size(); ++k) {
          for (std::size_t d = 0; d < components; ++d) {
            system.hamiltonian.At({b, row, k * components + d}) = Element(
                *h[c * components + d],
                (outer[b] * active_size + active[a]) * active_size + active[k]);
          }
        }
      }
    }
  }
  return system;
}

#if defined(WICKQC_LAPACK_MKL) || defined(WICKQC_LAPACK_OPENBLAS) || \
    defined(WICKQC_LAPACK_NETLIB) || defined(WICKQC_LAPACK_EIGEN)
struct ICNEVPT2BlockResult {
  double correlation_energy = 0;
  NDArray<double> amplitudes, singular_values;
  std::vector<std::size_t> ranks;
};

[[nodiscard]] inline ICNEVPT2BlockResult SolveICNEVPT2Block(
    const ICNEVPT2System& system) {
  if (system.rhs.Rank() != 2) {
    throw std::invalid_argument(
        "IC-NEVPT2 RHS must have batch and active axes");
  }
  const auto batch = system.rhs.shape()[0], width = system.rhs.shape()[1];
  reference_detail::Tensor(system.rhs, {batch, width}, "IC RHS");
  reference_detail::Tensor(
      system.hamiltonian, {batch, width, width}, "IC Hamiltonian");
  ICNEVPT2BlockResult result{
      0,
      NDArray<double>({batch, width}),
      NDArray<double>({batch, width}),
      std::vector<std::size_t>(batch)};
  if (batch == 0 || width == 0) {
    return result;
  }
  std::vector<double> matrix(width * width), rhs(width);
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t a = 0; a < width; ++a) {
      rhs[a] = system.rhs.At({b, a});
      for (std::size_t k = 0; k < width; ++k) {
        matrix[a * width + k] = system.hamiltonian.At({b, a, k});
      }
    }
    // NumPy lstsq(rcond=None): epsilon * max(rows, columns). SVD failures are
    // propagated, never replaced with a potentially different ordinary solve.
    const auto solution = lapack::LeastSquares(matrix, width, width, rhs);
    result.ranks[b] = solution.rank;
    for (std::size_t a = 0; a < width; ++a) {
      if (!std::isfinite(solution.solution[a]) ||
          !std::isfinite(solution.singular_values[a])) {
        throw std::runtime_error("Non-finite IC-NEVPT2 least-squares solution");
      }
      result.amplitudes.At({b, a}) = solution.solution[a];
      result.singular_values.At({b, a}) = solution.singular_values[a];
      result.correlation_energy -= solution.solution[a] * rhs[a];
    }
  }
  if (!std::isfinite(result.correlation_energy)) {
    throw std::runtime_error("Non-finite IC-NEVPT2 energy");
  }
  return result;
}

template <typename Kernel>
[[nodiscard]] NEVPT2Result SolveICNEVPT2(
    const std::map<std::string, Kernel>& kernels,
    const runtime::TensorMap<double>& inputs,
    const SpatialReference& reference,
    const std::function<void(
        std::string_view,
        const runtime::TensorMap<double>&,
        const ICNEVPT2System&,
        const ICNEVPT2BlockResult&)>& observe = {},
    const runtime::MemoryBudget& memory = {}) {
  using nevpt2_solver_detail::CheckMemory;
  using nevpt2_solver_detail::Dimensions;
  using nevpt2_solver_detail::Finish;
  using nevpt2_solver_detail::ICDescription;
  using nevpt2_solver_detail::ResidentBytes;
  const auto dimensions = Dimensions(reference);
  if (kernels.size() != kICNEVPT2Subspaces.size()) {
    throw std::invalid_argument(
        "IC-NEVPT2 requires all thirteen block kernels");
  }
  NEVPT2Result result;
  const auto resident = ResidentBytes(inputs, reference);
  result.estimated_peak_bytes = 0;
  for (const auto& [name, kernel] : kernels) {
    const auto peak = CheckMemory(kernel, dimensions, resident, memory, true);
    result.estimated_peak_bytes = peak && result.estimated_peak_bytes
        ? std::optional(std::max(*result.estimated_peak_bytes, *peak))
        : std::nullopt;
  }
  for (const auto name : kICNEVPT2Subspaces) {
    const auto& kernel = kernels.at(std::string(name));
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    const auto tensors = kernel.Evaluate(inputs, dimensions);
    result.contraction_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();
    start = Clock::now();
    const auto system = AssembleICNEVPT2(name, tensors, dimensions);
    result.assembly_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();
    start = Clock::now();
    const auto block = SolveICNEVPT2Block(system);
    result.solve_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();
    result.subspace_energies[std::string(ICDescription(name).outer)] +=
        block.correlation_energy;
    if (observe) {
      observe(name, tensors, system, block);
    }
  }
  Finish(result, reference.reference_energy);
  return result;
}
#endif
} // namespace wickqc::method
