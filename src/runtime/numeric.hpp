#pragma once

#include <complex>
#include <cstddef>
#include <map>
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

  [[nodiscard]] const std::vector<TensorBinding>& Inputs() const noexcept;
  [[nodiscard]] const std::vector<TensorBinding>& Outputs() const noexcept;
  [[nodiscard]] TensorMap<double> Evaluate(
      const TensorMap<double>& inputs,
      const Dimensions& dimensions) const;
  [[nodiscard]] TensorMap<std::complex<double>> Evaluate(
      const TensorMap<std::complex<double>>& inputs,
      const Dimensions& dimensions) const;
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

inline const std::vector<TensorBinding>& NumericKernel::Inputs()
    const noexcept {
  return inputs;
}
inline const std::vector<TensorBinding>& NumericKernel::Outputs()
    const noexcept {
  return outputs;
}
inline TensorMap<double> NumericKernel::Evaluate(
    const TensorMap<double>& inputs,
    const Dimensions& dimensions) const {
  return real(inputs, dimensions);
}
inline TensorMap<std::complex<double>> NumericKernel::Evaluate(
    const TensorMap<std::complex<double>>& inputs,
    const Dimensions& dimensions) const {
  return complex(inputs, dimensions);
}

} // namespace wickqc::runtime
