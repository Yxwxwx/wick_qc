#pragma once

#include "runtime/tensor_binding.h"

#include <complex>
#include <vector>

namespace wickqc::runtime {

// A compiled numeric function and its input/output contract. No symbolic
// objects, orbital sizes, or amplitudes are stored in this descriptor.
struct NumericKernel {
  std::vector<TensorBinding> inputs;
  std::vector<TensorBinding> outputs;
  TensorMap<double> (*real)(const TensorMap<double>&, const Dimensions&);
  TensorMap<std::complex<double>> (*complex)(const TensorMap<std::complex<double>>&, const Dimensions&);

  [[nodiscard]] const std::vector<TensorBinding>& Inputs() const noexcept;
  [[nodiscard]] const std::vector<TensorBinding>& Outputs() const noexcept;
  [[nodiscard]] TensorMap<double> Evaluate(const TensorMap<double>& inputs, const Dimensions& dimensions) const;
  [[nodiscard]] TensorMap<std::complex<double>> Evaluate(const TensorMap<std::complex<double>>& inputs,
                                                         const Dimensions& dimensions) const;
};

} // namespace wickqc::runtime
