#pragma once

#if defined(WICKQC_LAPACK_MKL)
#include <mkl_lapacke.h>
#elif defined(WICKQC_LAPACK_OPENBLAS) || defined(WICKQC_LAPACK_NETLIB)
#include <lapacke.h>
#elif defined(WICKQC_LAPACK_EIGEN)
#if defined(EIGEN_USE_BLAS) || defined(EIGEN_USE_LAPACKE) ||       \
    defined(EIGEN_USE_LAPACKE_STRICT) || defined(EIGEN_USE_MKL) || \
    defined(EIGEN_USE_MKL_ALL) || defined(EIGEN_USE_MKL_VML)
#error \
    "The EIGEN backend requires Eigen without external BLAS/LAPACK delegation"
#endif
#include <Eigen/Core>
#include <Eigen/SVD>
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
// values <= relative_cutoff * largest are discarded for cutoffs in (0, 1).
// As in DGELSD, zero or >= 1 selects LAPACK's machine-precision cutoff.
// The default is machine epsilon * max(rows, columns), as in lstsq(rcond=None).
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
#if defined(WICKQC_LAPACK_EIGEN)
  const auto limit =
      static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max());
#else
  const auto limit =
      static_cast<std::size_t>(std::numeric_limits<lapack_int>::max());
#endif
  if (rows == 0 || columns == 0 || rows > limit || columns > limit ||
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::invalid_argument(
        "LeastSquares matrix dimensions must be positive and fit the backend integer range");
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
  LeastSquaresResult result;
#if defined(WICKQC_LAPACK_EIGEN)
  using Matrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const auto m = static_cast<Eigen::Index>(rows);
  const auto n = static_cast<Eigen::Index>(columns);
  const Eigen::Map<const Matrix> coefficients(matrix.data(), m, n);
  const Eigen::Map<const Eigen::VectorXd> right_hand_side(rhs.data(), m);
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      coefficients, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (svd.info() != Eigen::Success) {
    throw std::runtime_error("Eigen JacobiSVD failed to converge");
  }
  const auto& singular = svd.singularValues();
  result.singular_values.assign(
      singular.data(), singular.data() + singular.size());
  Eigen::VectorXd projected = svd.matrixU().adjoint() * right_hand_side;
  // DGELSD's DLALSD uses DLAMCH('E') for rcond <= 0 or >= 1. For binary64
  // round-to-nearest this is half of C++ epsilon (LAPACK INSTALL/dlamch.f).
  const double effective_cutoff = cutoff == 0 || cutoff >= 1
      ? std::numeric_limits<double>::epsilon() * 0.5
      : cutoff;
  // Eigen's rank()/solve() keep equality at the threshold; DGELSD discards it.
  const double threshold = effective_cutoff * singular[0];
  for (Eigen::Index i = 0; i < singular.size(); ++i) {
    if (singular[i] > threshold) {
      projected[i] /= singular[i];
      ++result.rank;
    } else {
      projected[i] = 0;
    }
  }
  result.solution.resize(columns);
  Eigen::Map<Eigen::VectorXd>(result.solution.data(), n).noalias() =
      svd.matrixV() * projected;
#else
  std::vector<double> work_matrix(matrix.begin(), matrix.end());
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
#endif
  return result;
}

} // namespace wickqc::lapack
