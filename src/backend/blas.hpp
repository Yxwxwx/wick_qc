#pragma once

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>

#if defined(WICKQC_USE_MKL)
#include <mkl_cblas.h>
#elif defined(WICKQC_USE_CBLAS) || defined(WICKQC_USE_OPENBLAS) || \
    defined(WICKQC_USE_BLIS) || defined(WICKQC_USE_NETLIB)
#include <cblas.h>
#elif defined(WICKQC_USE_EIGEN)
#if defined(EIGEN_USE_BLAS) || defined(EIGEN_USE_LAPACKE) ||       \
    defined(EIGEN_USE_LAPACKE_STRICT) || defined(EIGEN_USE_MKL) || \
    defined(EIGEN_USE_MKL_ALL) || defined(EIGEN_USE_MKL_VML)
#error \
    "The EIGEN backend requires Eigen without external BLAS/LAPACK delegation"
#endif
#include <Eigen/Core>
#endif

namespace wickqc::blas {

// Contiguous y += alpha * x; independent lanes expose SIMD without ISA flags.
template <typename T>
inline void Axpy(std::size_t count, T alpha, const T* x, T* y) {
  if (alpha == T{})
    return;
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    y[i] += alpha * x[i];
    y[i + 1] += alpha * x[i + 1];
    y[i + 2] += alpha * x[i + 2];
    y[i + 3] += alpha * x[i + 3];
    y[i + 4] += alpha * x[i + 4];
    y[i + 5] += alpha * x[i + 5];
    y[i + 6] += alpha * x[i + 6];
    y[i + 7] += alpha * x[i + 7];
  }
  if (i + 4 <= count) {
    y[i] += alpha * x[i];
    y[i + 1] += alpha * x[i + 1];
    y[i + 2] += alpha * x[i + 2];
    y[i + 3] += alpha * x[i + 3];
    i += 4;
  }
  if (i + 2 <= count) {
    y[i] += alpha * x[i];
    y[i + 1] += alpha * x[i + 1];
    i += 2;
  }
  if (i < count)
    y[i] += alpha * x[i];
}

namespace detail {

// Constant tile sizes expose the original 4/2/1 register blocking to compilers.
template <typename T, bool TransposeB, std::size_t Rows, std::size_t Cols>
inline void NativeGemmTile(
    std::size_t nk,
    std::size_t lda,
    std::size_t ldb,
    std::size_t ldc,
    T alpha,
    const T* a,
    const T* b,
    T beta,
    T* c) {
  T values[Rows * Cols]{};
  for (std::size_t k = 0; k < nk; ++k) {
    for (std::size_t j = 0; j < Cols; ++j) {
      const T right = b[TransposeB ? j * ldb + k : k * ldb + j];
      for (std::size_t i = 0; i < Rows; ++i) {
        values[j * Rows + i] += a[i * lda + k] * right;
      }
    }
  }
  for (std::size_t i = 0; i < Rows; ++i) {
    for (std::size_t j = 0; j < Cols; ++j) {
      const T value = alpha * values[j * Rows + i];
      c[i * ldc + j] = beta == T{} ? value : value + beta * c[i * ldc + j];
    }
  }
}

template <typename T, bool TransposeB, std::size_t Rows>
inline void NativeGemmRows(
    std::size_t nj,
    std::size_t nk,
    std::size_t lda,
    std::size_t ldb,
    std::size_t ldc,
    T alpha,
    const T* a,
    const T* b,
    T beta,
    T* c) {
  const std::size_t right_stride = TransposeB ? ldb : 1;
  std::size_t j = 0;
  for (; j + 4 <= nj; j += 4) {
    NativeGemmTile<T, TransposeB, Rows, 4>(
        nk, lda, ldb, ldc, alpha, a, b + j * right_stride, beta, c + j);
  }
  if (j + 2 <= nj) {
    NativeGemmTile<T, TransposeB, Rows, 2>(
        nk, lda, ldb, ldc, alpha, a, b + j * right_stride, beta, c + j);
    j += 2;
  }
  if (j < nj) {
    NativeGemmTile<T, TransposeB, Rows, 1>(
        nk, lda, ldb, ldc, alpha, a, b + j * right_stride, beta, c + j);
  }
}

template <typename T, bool TransposeB>
inline void NativeGemm(
    std::size_t ni,
    std::size_t nj,
    std::size_t nk,
    std::size_t lda,
    std::size_t ldb,
    std::size_t ldc,
    T alpha,
    const T* a,
    const T* b,
    T beta,
    T* c) {
  if constexpr (!TransposeB) {
    // Wide sparse rows benefit from AXPY; short rows keep register blocking.
    if (nj >= 8) {
      for (std::size_t i = 0; i < ni; ++i) {
        for (std::size_t j = 0; j < nj; ++j) {
          c[i * ldc + j] = beta == T{} ? T{} : beta * c[i * ldc + j];
        }
        for (std::size_t k = 0; k < nk; ++k) {
          const T weight = a[i * lda + k];
          if (weight != T{})
            Axpy(nj, alpha * weight, b + k * ldb, c + i * ldc);
        }
      }
      return;
    }
  }
  std::size_t i = 0;
  for (; i + 4 <= ni; i += 4) {
    NativeGemmRows<T, TransposeB, 4>(
        nj, nk, lda, ldb, ldc, alpha, a + i * lda, b, beta, c + i * ldc);
  }
  if (i + 2 <= ni) {
    NativeGemmRows<T, TransposeB, 2>(
        nj, nk, lda, ldb, ldc, alpha, a + i * lda, b, beta, c + i * ldc);
    i += 2;
  }
  if (i < ni) {
    NativeGemmRows<T, TransposeB, 1>(
        nj, nk, lda, ldb, ldc, alpha, a + i * lda, b, beta, c + i * ldc);
  }
}
} // namespace detail

// Row-major C = f * A * B^T + beta * C; TransposeB=false uses B directly.
// The transpose never conjugates. Inputs and output must not overlap.
template <typename T, bool TransposeB = true>
inline void Gemm(
    std::size_t ni,
    std::size_t nj,
    std::size_t nk,
    std::size_t ldn,
    T f,
    const T* xa,
    const T* xb,
    T* xc,
    T beta = T{},
    std::size_t lda = 0,
    std::size_t ldb = 0) {
  lda = lda == 0 ? nk : lda;
  ldb = ldb == 0 ? (TransposeB ? nk : nj) : ldb;
  assert(lda >= nk && ldb >= (TransposeB ? nk : nj) && ldn >= nj);
  if (ni == 0 || nj == 0) {
    return;
  }
  if (nk == 0 || f == T{}) {
    for (std::size_t i = 0; i < ni; ++i) {
      for (std::size_t j = 0; j < nj; ++j) {
        xc[i * ldn + j] = beta == T{} ? T{} : beta * xc[i * ldn + j];
      }
    }
    return;
  }

#if defined(WICKQC_USE_CBLAS) || defined(WICKQC_USE_MKL) ||     \
    defined(WICKQC_USE_OPENBLAS) || defined(WICKQC_USE_BLIS) || \
    defined(WICKQC_USE_NETLIB)
  if constexpr (
      std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) {
    [[maybe_unused]] const auto limit =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    assert(
        ni <= limit && nj <= limit && nk <= limit && ldn <= limit &&
        lda <= limit && ldb <= limit);
    const auto trans = TransposeB ? CblasTrans : CblasNoTrans;
    if constexpr (std::is_same_v<T, double>) {
      cblas_dgemm(
          CblasRowMajor,
          CblasNoTrans,
          trans,
          static_cast<int>(ni),
          static_cast<int>(nj),
          static_cast<int>(nk),
          f,
          xa,
          static_cast<int>(lda),
          xb,
          static_cast<int>(ldb),
          beta,
          xc,
          static_cast<int>(ldn));
    } else {
      // Installed BLIS complex kernels can read old C even when beta is zero.
      if (beta == T{}) {
        for (std::size_t i = 0; i < ni; ++i) {
          std::fill_n(xc + i * ldn, nj, T{});
        }
      }
      cblas_zgemm(
          CblasRowMajor,
          CblasNoTrans,
          trans,
          static_cast<int>(ni),
          static_cast<int>(nj),
          static_cast<int>(nk),
          &f,
          xa,
          static_cast<int>(lda),
          xb,
          static_cast<int>(ldb),
          &beta,
          xc,
          static_cast<int>(ldn));
    }
    return;
  }
#elif defined(WICKQC_USE_EIGEN)
  if constexpr (
      std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) {
    [[maybe_unused]] const auto limit =
        static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max());
    assert(
        ni <= limit && nj <= limit && nk <= limit && ldn <= limit &&
        lda <= limit && ldb <= limit);
    using Matrix =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Map =
        Eigen::Map<const Matrix, Eigen::Unaligned, Eigen::OuterStride<>>;
    const auto m = static_cast<Eigen::Index>(ni);
    const auto n = static_cast<Eigen::Index>(nj);
    const auto k = static_cast<Eigen::Index>(nk);
    const Map a(xa, m, k, Eigen::OuterStride<>(static_cast<Eigen::Index>(lda)));
    const Map b(
        xb,
        TransposeB ? n : k,
        TransposeB ? k : n,
        Eigen::OuterStride<>(static_cast<Eigen::Index>(ldb)));
    Eigen::Map<Matrix, Eigen::Unaligned, Eigen::OuterStride<>> c(
        xc, m, n, Eigen::OuterStride<>(static_cast<Eigen::Index>(ldn)));
    const auto store = [&](const auto& product) {
      if (beta == T{}) {
        c.noalias() = f * product;
      } else {
        c *= beta;
        c.noalias() += f * product;
      }
    };
    if constexpr (TransposeB) {
      store(a * b.transpose());
    } else {
      store(a * b);
    }
    return;
  }
#elif defined(WICKQC_USE_SIMPLE)
  for (std::size_t i = 0; i < ni; ++i) {
    for (std::size_t j = 0; j < nj; ++j) {
      T value{};
      for (std::size_t k = 0; k < nk; ++k) {
        value += xa[i * lda + k] * xb[TransposeB ? j * ldb + k : k * ldb + j];
      }
      xc[i * ldn + j] =
          beta == T{} ? f * value : f * value + beta * xc[i * ldn + j];
    }
  }
  return;
#elif defined(WICKQC_USE_TBLIS) && !defined(WICKQC_USE_NATIVE)
#error "TBLIS uses TblisContract instead of the GEMM adapter"
#endif
  // Extended precision remains native even when a double CBLAS backend is
  // selected.
  detail::NativeGemm<T, TransposeB>(
      ni, nj, nk, lda, ldb, ldn, f, xa, xb, beta, xc);
}

} // namespace wickqc::blas
