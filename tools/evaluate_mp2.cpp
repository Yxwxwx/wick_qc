#include "backend/ndarray.hpp"
#include "method/spatial_mp.h"
#include "runtime/ndarray_executor.h"
#include "runtime/tensor_binding.h"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// A small canonical RHF model with pair-symmetric spatial integrals. The
// first-order amplitudes are known from orbital-energy denominators, so both
// the MP2 energy and the generated covariant residuals can be checked directly.
int main() {
  try {
    using wickqc::NDArray;
    using wickqc::runtime::NdArrayExecutor;
    using wickqc::runtime::TensorMap;
    using Array = NDArray<double>;
    const wickqc::runtime::Dimensions dimensions{{{1, 0}, 2}, {{8, 0}, 3}};
    const Array occupied({2}, std::vector<double>{-1.0, -0.7});
    const Array external({3}, std::vector<double>{0.5, 0.8, 1.2});
    Array integrals({2, 2, 3, 3});
    Array doubles({3, 3, 2, 2});
    double reference_energy = 0;
    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t j = 0; j < 2; ++j) {
        for (std::size_t a = 0; a < 3; ++a) {
          for (std::size_t b = 0; b < 3; ++b) {
            const double integral =
                0.02 * static_cast<double>(1 + i + j + a + b);
            const double denominator = occupied.At({i}) + occupied.At({j}) -
                external.At({a}) - external.At({b});
            integrals.At({i, j, a, b}) = integral;
            doubles.At({a, b, i, j}) = integral / denominator;
            reference_energy += integral * integral / denominator;
          }
        }
      }
    }
    const TensorMap<double> inputs{
        {"epsI", occupied},
        {"epsE", external},
        {"vIIEE", integrals},
        {"vEEII", integrals.TransposeView({2, 3, 0, 1})},
        {"u1EI", Array::Zeros({3, 2})},
        {"u1EEII", doubles}};
    const auto graph =
        wickqc::method::SpatialMpGenerator(2).Equations().Simplify();
    const auto executable = NdArrayExecutor::Compile(graph);
    const auto result = executable.Evaluate(inputs, dimensions);
    const double energy = result.at("energy2").Item();
    const double singles_norm = result.at("residual1_rank1").Norm();
    const double doubles_norm = result.at("residual1_rank2").Norm();
    std::cout
        << std::setprecision(15)
        << "Canonical spatial MP2 model (2 occupied, 3 virtual orbitals)\n"
        << "E2 = " << energy << '\n'
        << "Residual norms: " << singles_norm << ", " << doubles_norm << '\n';
    if (std::abs(energy - reference_energy) > 1e-12 || singles_norm > 1e-12 ||
        doubles_norm > 1e-12) {
      throw std::runtime_error(
          "MP2 model disagrees with the direct denominator formula");
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
