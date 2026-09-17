#pragma once

#include <complex>
#include <cstddef>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
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

// A compiled numeric function and its input/output contract. No symbolic
// objects, orbital sizes, or amplitudes are stored in this descriptor.
struct NumericKernel {
  std::vector<TensorBinding> inputs;
  std::vector<TensorBinding> outputs;
  TensorMap<double> (*real)(const TensorMap<double>&, const Dimensions&);
  TensorMap<std::complex<double>> (
      *complex)(const TensorMap<std::complex<double>>&, const Dimensions&);

  [[nodiscard]] const std::vector<TensorBinding>& Inputs() const noexcept {
    return inputs;
  }
  [[nodiscard]] const std::vector<TensorBinding>& Outputs() const noexcept {
    return outputs;
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
