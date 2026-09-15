#include "runtime/numeric_kernel.h"
#include <complex>
#include <vector>
#include "runtime/tensor_binding.h"

namespace wickqc::runtime {

const std::vector<TensorBinding>& NumericKernel::Inputs() const noexcept {
  return inputs;
}
const std::vector<TensorBinding>& NumericKernel::Outputs() const noexcept {
  return outputs;
}
TensorMap<double> NumericKernel::Evaluate(
    const TensorMap<double>& inputs,
    const Dimensions& dimensions) const {
  return real(inputs, dimensions);
}
TensorMap<std::complex<double>> NumericKernel::Evaluate(
    const TensorMap<std::complex<double>>& inputs,
    const Dimensions& dimensions) const {
  return complex(inputs, dimensions);
}

} // namespace wickqc::runtime
