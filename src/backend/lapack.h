#pragma once

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
[[nodiscard]] LeastSquaresResult LeastSquares(std::span<const double> matrix,
                                              std::size_t rows,
                                              std::size_t columns,
                                              std::span<const double> rhs,
                                              std::optional<double> relative_cutoff = std::nullopt);

} // namespace wickqc::lapack
