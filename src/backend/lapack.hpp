#pragma once

#if defined(WICKQC_LAPACK_MKL)
#include <mkl_lapacke.h>
#elif defined(WICKQC_LAPACK_OPENBLAS)
#include <lapacke.h>
#else
#error "A LAPACK backend must be selected for wickqc_lapack"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace wickqc::lapack {

struct LeastSquaresResult {
  std::vector<double> solution;
  std::vector<double> singular_values;
  std::size_t rank = 0;
};

// Minimum-norm solution of min ||A x - b||_2, using an SVD. A is row-major
// (rows x columns), b has rows entries; inputs are not modified. Singular
// values <= relative_cutoff * largest are discarded. The default cutoff is
// machine epsilon * max(rows, columns), matching NumPy lstsq(rcond=None).
[[nodiscard]] LeastSquaresResult LeastSquares(
    std::span<const double> matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const double> rhs,
    std::optional<double> relative_cutoff = std::nullopt);

inline LeastSquaresResult LeastSquares(
    std::span<const double> matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const double> rhs,
    std::optional<double> relative_cutoff) {
  const auto limit =
      static_cast<std::size_t>(std::numeric_limits<lapack_int>::max());
  if (rows == 0 || columns == 0 || rows > limit || columns > limit ||
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::invalid_argument(
        "LeastSquares matrix dimensions must be positive and fit the LAPACK integer range");
  }
  if (matrix.size() != rows * columns || rhs.size() != rows) {
    throw std::invalid_argument(
        "LeastSquares requires rows * columns matrix entries and rows RHS entries");
  }
  const double cutoff = relative_cutoff.value_or(
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max(rows, columns)));
  if (!std::isfinite(cutoff) || cutoff < 0) {
    throw std::invalid_argument(
        "LeastSquares relative cutoff must be finite and nonnegative");
  }
  const auto finite = [](double value) { return std::isfinite(value); };
  if (!std::ranges::all_of(matrix, finite) ||
      !std::ranges::all_of(rhs, finite)) {
    throw std::invalid_argument(
        "LeastSquares matrix and RHS must contain finite values");
  }
  std::vector<double> work_matrix(matrix.begin(), matrix.end());
  LeastSquaresResult result;
  result.solution.resize(std::max(rows, columns));
  std::ranges::copy(rhs, result.solution.begin());
  result.singular_values.resize(std::min(rows, columns));
  lapack_int rank = 0;
  const auto status = LAPACKE_dgelsd(
      LAPACK_ROW_MAJOR,
      static_cast<lapack_int>(rows),
      static_cast<lapack_int>(columns),
      1,
      work_matrix.data(),
      static_cast<lapack_int>(columns),
      result.solution.data(),
      1,
      result.singular_values.data(),
      cutoff,
      &rank);
  if (status != 0) {
    throw std::runtime_error(
        "LAPACK DGELSD failed with status " + std::to_string(status));
  }
  result.solution.resize(columns);
  result.rank = static_cast<std::size_t>(rank);
  return result;
}

} // namespace wickqc::lapack
