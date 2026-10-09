#include "backend/ndarray.hpp"
#include "runtime_dimensions.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {
template <typename T>
T Value(double real, double imaginary) {
  if constexpr (
      std::is_same_v<T, std::complex<double>> ||
      std::is_same_v<T, std::complex<long double>>) {
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

#if !defined(WICKQC_USE_TBLIS)
template <typename T>
void CheckAxpy() {
  using Real = decltype(std::abs(T{}));
  for (std::size_t count = 0; count <= 65; ++count) {
    for (std::size_t offset = 0; offset < 3; ++offset) {
      std::vector<T> input(count + 4), actual(count + 4, Value<T>(0.75, -0.2));
      for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = Value<T>(std::sin(double(i) + 0.2), std::cos(double(i)));
      }
      auto expected = actual;
      const T weight = Value<T>(-0.3, 0.2);
      for (std::size_t i = 0; i < count; ++i) {
        expected[i + offset] += weight * input[i + offset];
      }
      wickqc::blas::Axpy(
          count, weight, input.data() + offset, actual.data() + offset);
      for (std::size_t i = 0; i < actual.size(); ++i) {
        ASSERT_LE(
            std::abs(actual[i] - expected[i]),
            32 * std::numeric_limits<Real>::epsilon() *
                (1 + std::abs(expected[i])));
      }
    }
  }
  const T nan(std::numeric_limits<Real>::quiet_NaN());
  T output = Value<T>(2, -1);
  wickqc::blas::Axpy(1, T{}, &nan, &output);
  EXPECT_EQ(output, Value<T>(2, -1));
}

TEST(Backend, AxpyUnalignedSpansAndAllTails) {
  CheckAxpy<double>();
  CheckAxpy<std::complex<double>>();
  CheckAxpy<long double>();
  CheckAxpy<std::complex<long double>>();
}

template <typename T, bool TransposeB>
void CheckStridedGemm() {
  using Real = decltype(std::abs(T{}));
  const auto sentinel = Value<T>(17, -3);
  for (std::size_t m = 0; m <= 9; ++m) {
    for (std::size_t n = 0; n <= 9; ++n) {
      for (std::size_t k : {0, 1, 2, 3, 4, 7, 8, 9, 17}) {
        const auto lda = k + 2;
        const auto ldb = (TransposeB ? k : n) + 3;
        const auto ldc = n + 4;
        std::vector<T> a(1 + m * lda), b(1 + (TransposeB ? n : k) * ldb);
        for (std::size_t i = 0; i < a.size(); ++i) {
          a[i] = Value<T>(0.01 * double(i % 17), -0.02 * double(i % 5));
        }
        for (std::size_t i = 0; i < b.size(); ++i) {
          b[i] = Value<T>(0.03 * double(i % 13), 0.01 * double(i % 7));
        }
        const T alpha = Value<T>(0.7, 0.3);
        for (const T beta : {T{}, Value<T>(-0.25, 0.15)}) {
          std::vector<T> actual(1 + (m + 1) * ldc, sentinel);
          for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
              actual[1 + i * ldc + j] = beta == T{}
                  ? T(std::numeric_limits<Real>::quiet_NaN())
                  : Value<T>(0.4, -0.2);
            }
          }
          auto expected = actual;
          for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
              T sum{};
              for (std::size_t p = 0; p < k; ++p) {
                sum += a[1 + i * lda + p] *
                    b[1 + (TransposeB ? j * ldb + p : p * ldb + j)];
              }
              auto& value = expected[1 + i * ldc + j];
              value = beta == T{} ? alpha * sum : alpha * sum + beta * value;
            }
          }
          wickqc::blas::Gemm<T, TransposeB>(
              m,
              n,
              k,
              ldc,
              alpha,
              a.data() + 1,
              b.data() + 1,
              actual.data() + 1,
              beta,
              lda,
              ldb);
          for (std::size_t i = 0; i < actual.size(); ++i) {
            ASSERT_LE(
                std::abs(actual[i] - expected[i]),
                64 * std::numeric_limits<Real>::epsilon() *
                    (1 + std::abs(expected[i])))
                << "m=" << m << " n=" << n << " k=" << k
                << " transpose=" << TransposeB << " index=" << i
                << " type_bytes=" << sizeof(T);
          }
        }
      }
    }
  }
  const T nan(std::numeric_limits<Real>::quiet_NaN());
  T output = nan;
  wickqc::blas::Gemm(1, 1, 1, 1, T{}, &nan, &nan, &output);
  EXPECT_EQ(output, T{});
  output = Value<T>(2, -1);
  wickqc::blas::Gemm(1, 1, 1, 1, T{}, &nan, &nan, &output, Value<T>(0.5, 0.2));
  EXPECT_EQ(output, Value<T>(0.5, 0.2) * Value<T>(2, -1));
}

TEST(Backend, TransposeStridesAccumulationAndAllTileTails) {
  CheckStridedGemm<double, true>();
  CheckStridedGemm<double, false>();
  CheckStridedGemm<std::complex<double>, true>();
  CheckStridedGemm<std::complex<double>, false>();
  CheckStridedGemm<long double, true>();
  CheckStridedGemm<long double, false>();
  CheckStridedGemm<std::complex<long double>, true>();
  CheckStridedGemm<std::complex<long double>, false>();
}
#endif

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
