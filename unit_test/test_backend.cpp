#include "runtime_dimensions.hpp"
#include "wick.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace {
template <typename T>
T Value(double real, double imaginary) {
  if constexpr (std::is_same_v<T, std::complex<double>>) {
    return {real, imaginary};
  } else {
    return real;
  }
}

template <typename T>
void CheckBackend() {
  using Array = wickqc::NDArray<T>;
  const auto m = 5 * wickqc::test::Dimension("WICKQC_TEST_OCCUPIED") + 1;
  const auto n = 4 * wickqc::test::Dimension("WICKQC_TEST_VIRTUAL") + 3;
  const auto k = 3 * wickqc::test::Dimension("WICKQC_TEST_ACTIVE") + 1;
  Array a({m, k}), b({n, k});
#if defined(WICKQC_USE_TBLIS)
  const auto leading_dimension = n;
#else
  const auto leading_dimension =
      n + wickqc::test::Dimension("WICKQC_TEST_ACTIVE");
#endif
  const auto sentinel = Value<T>(17.0, -3.0);
  auto c = Array::Full({m, leading_dimension}, sentinel);
  for (std::size_t i = 0; i < a.Size(); ++i) {
    a.data()[i] = Value<T>(
        0.01 * static_cast<double>(i % 17), -0.02 * static_cast<double>(i % 5));
  }
  for (std::size_t i = 0; i < b.Size(); ++i) {
    b.data()[i] = Value<T>(
        0.03 * static_cast<double>(i % 13), 0.01 * static_cast<double>(i % 7));
  }
  const T alpha = Value<T>(0.7, 0.3);
#if defined(WICKQC_USE_TBLIS)
  const std::vector<int> ai{0, 2}, bi{1, 2}, ci{0, 1};
  wickqc::backend::TblisContract(
      a.data(),
      wickqc::backend::TBLISMetadata(a.shape(), a.strides(), ai),
      b.data(),
      wickqc::backend::TBLISMetadata(b.shape(), b.strides(), bi),
      c.data(),
      wickqc::backend::TBLISMetadata(c.shape(), c.strides(), ci),
      alpha,
      T{});
#else
  wickqc::blas::Gemm(
      m, n, k, leading_dimension, alpha, a.data(), b.data(), c.data());
#endif
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      T expected{};
      for (std::size_t p = 0; p < k; ++p) {
        expected += a.At({i, p}) * b.At({j, p});
      }
      EXPECT_LE(
          std::abs(c.At({i, j}) - alpha * expected),
          1e-12 + 1e-12 * std::abs(expected));
    }
    for (std::size_t j = n; j < leading_dimension; ++j) {
      EXPECT_EQ(c.At({i, j}), sentinel);
    }
  }
}

TEST(Backend, RealContractionMatchesIndependentLoops) {
  CheckBackend<double>();
}
TEST(Backend, ComplexContractionDoesNotConjugate) {
  CheckBackend<std::complex<double>>();
}

TEST(Backend, StridedBatchesAndAlphaBeta) {
  using Array = wickqc::NDArray<double>;
  const auto batch = wickqc::test::Dimension("WICKQC_TEST_ACTIVE");
  const auto m = wickqc::test::Dimension("WICKQC_TEST_OCCUPIED");
  const auto n = wickqc::test::Dimension("WICKQC_TEST_VIRTUAL");
  auto a = Array::Random({batch, m, n}, 31).Slice(":,:,::-1");
  auto b = Array::Random({batch, m, n}, 71).Slice(":,:,::-1");
  auto c = Array::Full({batch, m, m}, 2.0);
  Array::TensordotInto(a, b, c, {2}, {2}, {0}, {0}, 0.5, -0.25);
  for (std::size_t q = 0; q < batch; ++q) {
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < m; ++j) {
        double expected = -0.5;
        for (std::size_t k = 0; k < n; ++k) {
          expected += 0.5 * a.At({q, i, k}) * b.At({q, j, k});
        }
        EXPECT_NEAR(c.At({q, i, j}), expected, 1e-12);
      }
    }
  }
  EXPECT_THROW(
      (void)Array::Tensordot(a, b, {0}, {0}, {0}, {0}), std::invalid_argument);
}
} // namespace
