#include "backend/ndarray.hpp"
#include "build_time/kernels.h"
#include "method/ic_nevpt2.h"
#include "method/spatial_cc.h"
#include "runtime/ndarray_executor.h"
#include "runtime/numeric_kernel.h"
#include "runtime/tensor_binding.h"
#include "runtime_dimensions.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using wickqc::runtime::Dimensions;
using wickqc::runtime::TensorBinding;
using wickqc::runtime::TensorMap;

Dimensions RuntimeDimensions() {
  return {
      {{1, 0}, wickqc::test::Dimension("WICKQC_TEST_OCCUPIED")},
      {{2, 0}, wickqc::test::Dimension("WICKQC_TEST_ACTIVE")},
      {{8, 0}, wickqc::test::Dimension("WICKQC_TEST_VIRTUAL")}};
}

// Spin-free RDM of an active closed-shell determinant. Each permutation cycle
// carries two spin choices; fermionic parity fixes its sign.
double Density(const std::vector<std::size_t>& indices, std::size_t filled) {
  const auto rank = indices.size() / 2;
  std::vector<std::size_t> permutation(rank);
  std::iota(permutation.begin(), permutation.end(), 0);
  double result = 0;
  do {
    bool allowed = true;
    int sign = 1, cycles = 0;
    std::vector<bool> visited(rank, false);
    for (std::size_t i = 0; i < rank; ++i) {
      allowed = allowed && indices[i] < filled &&
          indices[i] == indices[rank + permutation[i]];
      for (std::size_t j = i + 1; j < rank; ++j) {
        if (permutation[i] > permutation[j]) {
          sign = -sign;
        }
      }
      if (!visited[i]) {
        ++cycles;
        auto j = i;
        do {
          visited[j] = true;
          j = permutation[j];
        } while (!visited[j]);
      }
    }
    if (allowed) {
      result += static_cast<double>(sign) * std::pow(2.0, cycles);
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  return result;
}

template <typename T>
TensorMap<T> Inputs(
    std::span<const TensorBinding> bindings,
    const Dimensions& dimensions,
    bool chemist) {
  using Array = wickqc::NDArray<T>;
  TensorMap<T> inputs;
  const auto ni = dimensions.at({1, 0}), na = dimensions.at({2, 0});
  const auto total = ni + na + dimensions.at({8, 0});
  for (const auto& binding : bindings) {
    const auto shape = binding.Shape(dimensions);
    Array array(shape);
    for (std::size_t linear = 0; linear < array.Size(); ++linear) {
      std::vector<std::size_t> index(shape.size());
      auto remaining = linear;
      for (std::size_t i = shape.size(); i-- > 0;) {
        index[i] = remaining % shape[i];
        remaining /= shape[i];
      }
      if (binding.name.starts_with("E")) {
        array.data()[linear] = T(Density(index, na - 1));
        continue;
      }
      for (std::size_t i = 0; i < index.size(); ++i) {
        const auto space = binding.domains[i].orbital_spaces;
        index[i] += space == 2 ? ni : (space == 8 ? ni + na : 0);
      }
      std::vector<std::size_t> keys;
      const bool integral =
          binding.name.starts_with("v") || binding.name.starts_with("w");
      if (integral && index.size() == 4) {
        if (!chemist) {
          std::swap(index[1], index[2]);
        }
        for (std::size_t i = 0; i < 4; i += 2) {
          const auto [lo, hi] = std::minmax(index[i], index[i + 1]);
          keys.push_back(lo * total + hi);
        }
        std::sort(keys.begin(), keys.end());
      } else if (binding.name.starts_with("t") && index.size() == 4) {
        keys = {index[0] * total + index[2], index[1] * total + index[3]};
        std::sort(keys.begin(), keys.end());
      } else {
        keys = index;
        std::sort(keys.begin(), keys.end());
      }
      double key = 1;
      for (const auto value : keys) {
        key = 1.13 * key + static_cast<double>(value + 1);
      }
      const double real = 0.03 * std::sin(key);
      if constexpr (std::is_same_v<T, std::complex<double>>) {
        array.data()[linear] = {real, 0.02 * std::cos(key)};
      } else {
        array.data()[linear] = real;
      }
    }
    inputs.emplace(binding.name, std::move(array));
  }
  return inputs;
}

template <typename T>
void Compare(
    const wickqc::runtime::NumericKernel& kernel,
    const wickqc::runtime::NDArrayExecutor& reference,
    const Dimensions& dimensions,
    bool chemist) {
  const auto actual = kernel.Evaluate(
      Inputs<T>(kernel.Inputs(), dimensions, chemist), dimensions);
  const auto expected = reference.Evaluate(
      Inputs<T>(reference.Inputs(), dimensions, chemist), dimensions);
  for (const auto& binding : kernel.Outputs()) {
    SCOPED_TRACE(binding.name);
    const auto& a = actual.at(binding.name);
    const auto& b = expected.at(binding.name);
    ASSERT_EQ(a.shape(), b.shape());
    double worst = 0;
    for (std::size_t i = 0; i < a.Size(); ++i) {
      ASSERT_TRUE(std::isfinite(std::abs(a.data()[i])));
      const auto error = std::abs(a.data()[i] - b.data()[i]);
      worst = std::max(worst, error / (1e-11 + 1e-10 * std::abs(b.data()[i])));
    }
    EXPECT_LE(worst, 1.0);
  }
}

TEST(Codegen, SpatialCCSDMatchesUnoptimizedEquations) {
  const auto dimensions = RuntimeDimensions();
  const auto reference = wickqc::runtime::NDArrayExecutor::Compile(
      wickqc::method::SpatialCCGenerator(
          2, wickqc::method::IntegralConvention::kChemist)
          .Equations());
  const auto& kernel = wickqc::generated::Kernel_example_ccsd();
  Compare<double>(kernel, reference, dimensions, true);
  Compare<std::complex<double>>(kernel, reference, dimensions, true);
  EXPECT_THROW(
      kernel.Evaluate(TensorMap<double>{}, dimensions), std::invalid_argument);
}

TEST(Codegen, AllICNEVPT2BlocksMatchUnoptimizedEquations) {
  const auto dimensions = RuntimeDimensions();
  const auto blocks = wickqc::method::ICNEVPT2Generator().Equations();
  const auto kernels = wickqc::example::ICNEVPT2Kernels();
  ASSERT_EQ(blocks.size(), 13U);
  ASSERT_EQ(kernels.size(), blocks.size());
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    SCOPED_TRACE(blocks[i].first);
    ASSERT_EQ(kernels[i].name, blocks[i].first);
    const auto reference =
        wickqc::runtime::NDArrayExecutor::Compile(blocks[i].second);
    Compare<double>(*kernels[i].kernel, reference, dimensions, false);
    Compare<std::complex<double>>(
        *kernels[i].kernel, reference, dimensions, false);
  }
}
} // namespace
