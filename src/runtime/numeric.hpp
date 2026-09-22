#pragma once

#include <complex>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "backend/ndarray.hpp"
#include "symbolic/index_domain.hpp"

namespace wickqc::runtime {

using Dimensions = std::map<symbolic::IndexDomain, std::size_t>;
template <typename T = double>
using TensorMap = std::map<std::string, NDArray<T>>;

// Names follow the NumPy emitter (e.g. vIIEE, tEEII, E1); axes retain the
// method's orbital and spin domains. General domains require an explicit size.
struct TensorBinding {
  std::string name;
  std::vector<symbolic::IndexDomain> domains;
  [[nodiscard]] std::vector<std::size_t> Shape(
      const Dimensions& dimensions) const;
  bool operator==(const TensorBinding&) const = default;
};

namespace memory_detail {
inline std::size_t Add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b) {
    throw std::overflow_error("Tensor memory estimate exceeds size_t");
  }
  return a + b;
}
inline std::size_t Multiply(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::overflow_error("Tensor memory estimate exceeds size_t");
  }
  return a * b;
}
} // namespace memory_detail

// Conservative owned-tensor payload preflight, not a process RSS limit.
// reserved_bytes accounts for other caller-owned live data. Allocator/metadata
// overhead and third-party library internal workspaces require RSS measurement.
struct MemoryBudget {
  std::size_t limit_bytes = std::numeric_limits<std::size_t>::max();
  std::size_t reserved_bytes = 0;

  [[nodiscard]] std::size_t Check(
      std::size_t managed_bytes,
      std::string_view operation) const {
    const auto required = memory_detail::Add(reserved_bytes, managed_bytes);
    if (required > limit_bytes) {
      throw std::runtime_error(
          std::string(operation) +
          " requires a conservative tensor payload of " +
          std::to_string(required) + " bytes; memory budget is " +
          std::to_string(limit_bytes));
    }
    return required;
  }
};

// One monomial in the runtime dimensions. Generated kernels carry only this
// compact bound, not symbolic equations or a second execution plan.
struct WorkspaceTerm {
  std::size_t count;
  std::vector<symbolic::IndexDomain> domains;
};

inline std::size_t WorkspaceElements(
    std::span<const WorkspaceTerm> terms,
    const Dimensions& dimensions) {
  std::size_t total = 0;
  for (const auto& term : terms) {
    std::size_t size = term.count;
    bool empty = false;
    for (const auto domain : term.domains) {
      empty |= dimensions.at(domain) == 0;
    }
    if (empty) {
      continue;
    }
    for (const auto domain : term.domains) {
      size = memory_detail::Multiply(size, dimensions.at(domain));
    }
    total = memory_detail::Add(total, size);
  }
  return total;
}

// A compiled numeric function and its input/output contract. No symbolic
// objects, orbital sizes, or amplitudes are stored in this descriptor.
struct NumericKernel {
  std::vector<TensorBinding> inputs;
  std::vector<TensorBinding> outputs;
  TensorMap<double> (*real)(const TensorMap<double>&, const Dimensions&);
  TensorMap<std::complex<double>> (
      *complex)(const TensorMap<std::complex<double>>&, const Dimensions&);
  // Absent in legacy/user-written descriptors. Such kernels remain executable
  // but cannot satisfy an explicitly bounded full-method calculation.
  std::optional<std::vector<WorkspaceTerm>> workspace = std::nullopt;

  [[nodiscard]] const std::vector<TensorBinding>& Inputs() const noexcept {
    return inputs;
  }
  [[nodiscard]] const std::vector<TensorBinding>& Outputs() const noexcept {
    return outputs;
  }
  [[nodiscard]] std::optional<std::size_t> WorkspaceElements(
      const Dimensions& dimensions) const {
    if (!workspace) {
      return std::nullopt;
    }
    return runtime::WorkspaceElements(*workspace, dimensions);
  }
  [[nodiscard]] TensorMap<double> Evaluate(
      const TensorMap<double>& inputs,
      const Dimensions& dimensions) const {
    return real(inputs, dimensions);
  }
  [[nodiscard]] TensorMap<std::complex<double>> Evaluate(
      const TensorMap<std::complex<double>>& inputs,
      const Dimensions& dimensions) const {
    return complex(inputs, dimensions);
  }
};

inline std::vector<std::size_t> TensorBinding::Shape(
    const Dimensions& dimensions) const {
  std::vector<std::size_t> shape;
  for (const auto domain : domains) {
    const auto found = dimensions.find(domain);
    if (found == dimensions.end()) {
      throw std::invalid_argument(
          "Missing dimension for tensor '" + name + "', orbital mask " +
          std::to_string(domain.orbital_spaces) + ", spin mask " +
          std::to_string(domain.spins));
    }
    shape.push_back(found->second);
  }
  return shape;
}

namespace numeric_detail {
template <typename Kernel>
inline std::optional<std::size_t> WorkspaceBytes(
    const Kernel& kernel,
    const Dimensions& dimensions,
    const MemoryBudget& budget) {
  std::optional<std::size_t> elements;
  if constexpr (requires { kernel.WorkspaceElements(dimensions); }) {
    elements = kernel.WorkspaceElements(dimensions);
  }
  if (!elements) {
    if (budget.limit_bytes != std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(
          "A bounded method calculation requires kernel workspace metadata");
    }
    return std::nullopt;
  }
  return memory_detail::Multiply(*elements, sizeof(double));
}
template <typename T>
inline void ValidateInputs(
    std::span<const TensorBinding> bindings,
    const TensorMap<T>& inputs,
    const Dimensions& dimensions) {
  for (const auto& binding : bindings) {
    const auto found = inputs.find(binding.name);
    if (found == inputs.end()) {
      throw std::invalid_argument(
          "Missing input tensor '" + binding.name + "'");
    }
    if (found->second.shape() != binding.Shape(dimensions)) {
      throw std::invalid_argument(
          "Shape mismatch for input tensor '" + binding.name + "'");
    }
  }
}
} // namespace numeric_detail

} // namespace wickqc::runtime
