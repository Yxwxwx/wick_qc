#include "backend/lapack.hpp"
#include "runtime_dimensions.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using wickqc::lapack::LeastSquares;

TEST(Lapack, OverdeterminedLeastSquaresPreservesInputs) {
  const auto n = wickqc::test::Dimension("WICKQC_TEST_ACTIVE");
  std::vector<double> matrix(2 * n * n), rhs(2 * n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto scale = static_cast<double>(i + 1);
    matrix[i * n + i] = matrix[(n + i) * n + i] = scale;
    rhs[i] = scale * scale + 1;
    rhs[n + i] = scale * scale - 1;
  }
  const auto original_matrix = matrix;
  const auto original_rhs = rhs;
  const auto result = LeastSquares(matrix, 2 * n, n, rhs);
  EXPECT_EQ(result.rank, n);
  EXPECT_EQ(matrix, original_matrix);
  EXPECT_EQ(rhs, original_rhs);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(result.solution[i], static_cast<double>(i + 1), 1e-12);
    EXPECT_NEAR(
        result.singular_values[i],
        std::sqrt(2.) * static_cast<double>(n - i),
        1e-12);
  }
}

TEST(Lapack, RankDeficientMinimumNormSolution) {
  const auto n = wickqc::test::Dimension("WICKQC_TEST_ACTIVE");
  std::vector<double> matrix(n * n), rhs(n);
  for (std::size_t i = 0; i < n; ++i) {
    matrix[i * n] = matrix[i * n + 1] = static_cast<double>(i + 1);
    rhs[i] = 3 * static_cast<double>(i + 1);
  }
  const auto result = LeastSquares(matrix, n, n, rhs);
  EXPECT_EQ(result.rank, 1U);
  EXPECT_NEAR(result.solution[0], 1.5, 1e-12);
  EXPECT_NEAR(result.solution[1], 1.5, 1e-12);
  for (std::size_t i = 2; i < n; ++i) {
    EXPECT_NEAR(result.solution[i], 0, 1e-12);
  }
}

TEST(Lapack, UnderdeterminedMinimumNormSolution) {
  const auto n = wickqc::test::Dimension("WICKQC_TEST_ACTIVE");
  std::vector<double> matrix(n * 2 * n), rhs(n);
  for (std::size_t i = 0; i < n; ++i) {
    matrix[i * 2 * n + i] = matrix[i * 2 * n + n + i] = 1;
    rhs[i] = 2 * static_cast<double>(i + 1);
  }
  const auto result = LeastSquares(matrix, n, 2 * n, rhs);
  EXPECT_EQ(result.rank, n);
  ASSERT_EQ(result.solution.size(), 2 * n);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(result.solution[i], static_cast<double>(i + 1), 1e-12);
    EXPECT_NEAR(result.solution[n + i], static_cast<double>(i + 1), 1e-12);
  }
}

TEST(Lapack, CutoffControlsNumericalRank) {
  const auto n = wickqc::test::Dimension("WICKQC_TEST_ACTIVE");
  std::vector<double> matrix(n * n), rhs(n, 1.);
  for (std::size_t i = 0; i < n; ++i) {
    matrix[i * n + i] = 1;
  }
  matrix.back() = 1e-8;
  const auto full = LeastSquares(matrix, n, n, rhs);
  const auto truncated = LeastSquares(matrix, n, n, rhs, 1e-6);
  EXPECT_EQ(full.rank, n);
  EXPECT_NEAR(full.solution.back(), 1e8, 1e-6);
  EXPECT_EQ(truncated.rank, n - 1);
  EXPECT_NEAR(truncated.solution.back(), 0, 1e-12);
}

TEST(Lapack, ZeroMatrixAndInvalidInputs) {
  const auto n = wickqc::test::Dimension("WICKQC_TEST_ACTIVE");
  std::vector<double> matrix(n * n), rhs(n, 1.);
  const auto result = LeastSquares(matrix, n, n, rhs);
  EXPECT_EQ(result.rank, 0U);
  for (double value : result.solution) {
    EXPECT_NEAR(value, 0, 1e-12);
  }
  EXPECT_THROW((void)LeastSquares(matrix, 0, n, rhs), std::invalid_argument);
  EXPECT_THROW(
      (void)LeastSquares(matrix, n + 1, n, rhs), std::invalid_argument);
  EXPECT_THROW(
      (void)LeastSquares(matrix, n, n, rhs, -1.), std::invalid_argument);
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
      (void)LeastSquares(matrix, n, n, rhs, nan), std::invalid_argument);
  rhs[0] = nan;
  EXPECT_THROW((void)LeastSquares(matrix, n, n, rhs), std::invalid_argument);
}

TEST(Lapack, CutoffEqualityAndLapackFallback) {
  const auto n = wickqc::test::Dimension("WICKQC_TEST_ACTIVE");
  std::vector<double> matrix(n * n), rhs(n, 1.);
  for (std::size_t i = 0; i < n; ++i) {
    matrix[i * n + i] = 2.;
  }
  matrix.back() = 1.;
  // Singular values exactly at a cutoff in (0, 1) are discarded.
  const auto boundary = LeastSquares(matrix, n, n, rhs, 0.5);
  EXPECT_EQ(boundary.rank, n - 1);
  EXPECT_DOUBLE_EQ(boundary.solution.back(), 0.);
  // DGELSD treats cutoffs outside (0, 1) as machine precision, not rank zero.
  for (const double cutoff : {1., 2.}) {
    const auto fallback = LeastSquares(matrix, n, n, rhs, cutoff);
    EXPECT_EQ(fallback.rank, n);
    EXPECT_NEAR(fallback.solution.back(), 1., 1e-12);
  }
  matrix.back() = std::numeric_limits<double>::epsilon() * 0.25;
  const auto zero_cutoff = LeastSquares(matrix, n, n, rhs, 0.);
  EXPECT_EQ(zero_cutoff.rank, n - 1);
  EXPECT_DOUBLE_EQ(zero_cutoff.solution.back(), 0.);
  for (std::size_t i = 0; i + 1 < n; ++i) {
    EXPECT_NEAR(zero_cutoff.solution[i], 0.5, 1e-12);
  }
}
} // namespace
