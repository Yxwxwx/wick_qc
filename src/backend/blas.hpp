#pragma once

#include <cassert>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>

#if defined(WICKQC_USE_MKL)
#include <mkl.h>

#elif defined(WICKQC_USE_OPENBLAS) || defined(WICKQC_USE_BLIS)
#include <cblas.h>
#endif

namespace wickqc::blas {

template <typename T>
inline void Gemm(
    std::size_t ni,
    std::size_t nj,
    std::size_t nk,
    std::size_t ldn,
    T f,
    const T* __restrict__ xa,
    const T* __restrict__ xb,
    T* __restrict__ xc) noexcept {
#if defined(WICKQC_USE_MKL) || defined(WICKQC_USE_OPENBLAS) || \
    defined(WICKQC_USE_BLIS)

  static_assert(
      std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>);

  assert(ni <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
  assert(nj <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
  assert(nk <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
  assert(ldn <= static_cast<std::size_t>(std::numeric_limits<int>::max()));

  const int m = static_cast<int>(ni);
  const int n = static_cast<int>(nj);
  const int k = static_cast<int>(nk);
  const int ldc = static_cast<int>(ldn);

  if constexpr (std::is_same_v<T, double>) {
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        m,
        n,
        k,
        f,
        xa,
        k,
        xb,
        k,
        0.0,
        xc,
        ldc);
  } else {
    const std::complex<double> beta{};

    cblas_zgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        m,
        n,
        k,
        &f,
        xa,
        k,
        xb,
        k,
        &beta,
        xc,
        ldc);
  }

#elif defined(WICKQC_USE_NATIVE)

  constexpr std::size_t ki = 4, kj = 4;
  constexpr std::size_t kii = 4, kiii = 2, kjj = 4, kjjj = 2;
  std::size_t xni = ni / ki * ki;
  if (ni >= ki) {
    std::size_t xnj = nj / kj * kj;
    for (std::size_t xj = 0; xj < xnj; xj += kj) {
      for (std::size_t xi = 0; xi < xni; xi += ki) {
        const T* __restrict__ za = &xa[xi * nk];
        const T* __restrict__ zb = &xb[xj * nk];
        T* __restrict__ zc = &xc[xi * ldn + xj];
        T t[ki * kj] = {0};
        for (std::size_t k = 0; k < nk; k++)
#pragma unroll kj
          for (int j = 0; j < kj; j++)
#pragma unroll ki
            for (int i = 0; i < ki; i++)
              t[j * ki + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll ki
        for (int i = 0; i < ki; i++)
#pragma unroll kj
          for (int j = 0; j < kj; j++)
            zc[i * ldn + j] = f * t[j * ki + i];
      }
    }
    if (kj > kjj && ((nj - xnj) & kjj)) {
      const std::size_t xj = xnj;
      for (std::size_t xi = 0; xi < xni; xi += ki) {
        const T* __restrict__ za = &xa[xi * nk];
        const T* __restrict__ zb = &xb[xj * nk];
        T* __restrict__ zc = &xc[xi * ldn + xj];
        T t[ki * kjj] = {0};
        for (std::size_t k = 0; k < nk; k++)
#pragma unroll kjj
          for (int j = 0; j < kjj; j++)
#pragma unroll ki
            for (int i = 0; i < ki; i++)
              t[j * ki + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll ki
        for (int i = 0; i < ki; i++)
#pragma unroll kjj
          for (int j = 0; j < kjj; j++)
            zc[i * ldn + j] = f * t[j * ki + i];
      }
      xnj += kjj;
    }
    if (kj > kjjj && ((nj - xnj) & kjjj)) {
      const std::size_t xj = xnj;
      for (std::size_t xi = 0; xi < xni; xi += ki) {
        const T* __restrict__ za = &xa[xi * nk];
        const T* __restrict__ zb = &xb[xj * nk];
        T* __restrict__ zc = &xc[xi * ldn + xj];
        T t[ki * kjjj] = {0};
        for (std::size_t k = 0; k < nk; k++)
#pragma unroll kjjj
          for (int j = 0; j < kjjj; j++)
#pragma unroll ki
            for (int i = 0; i < ki; i++)
              t[j * ki + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll ki
        for (int i = 0; i < ki; i++)
#pragma unroll kjjj
          for (int j = 0; j < kjjj; j++)
            zc[i * ldn + j] = f * t[j * ki + i];
      }
      xnj += kjjj;
    }
    if ((nj - xnj) & 1) {
      const std::size_t xj = xnj;
      for (std::size_t xi = 0; xi < xni; xi += ki) {
        const T* __restrict__ za = &xa[xi * nk];
        const T* __restrict__ zb = &xb[xj * nk];
        T* __restrict__ zc = &xc[xi * ldn + xj];
        T t[ki] = {0};
        for (std::size_t k = 0; k < nk; k++)
#pragma unroll ki
          for (int i = 0; i < ki; i++)
            t[0 + i] += za[i * nk + k] * zb[0 * nk + k];
#pragma unroll ki
        for (int i = 0; i < ki; i++)
          zc[i * ldn + 0] = f * t[0 + i];
      }
      xnj += 1;
    }
  }
  if (ki > kii && ((ni - xni) & kii)) {
    std::size_t xnj = nj / kj * kj;
    for (std::size_t xj = 0; xj < xnj; xj += kj) {
      const std::size_t xi = xni;
      const T* __restrict__ za = &xa[xi * nk];
      const T* __restrict__ zb = &xb[xj * nk];
      T* __restrict__ zc = &xc[xi * ldn + xj];
      T t[kii * kj] = {0};
      for (std::size_t k = 0; k < nk; k++)
#pragma unroll kj
        for (int j = 0; j < kj; j++)
#pragma unroll kii
          for (int i = 0; i < kii; i++)
            t[j * kii + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll kii
      for (int i = 0; i < kii; i++)
#pragma unroll kj
        for (int j = 0; j < kj; j++)
          zc[i * ldn + j] = f * t[j * kii + i];
    }
    if (kj > kjj && ((nj - xnj) & kjj)) {
      const std::size_t xi = xni;
      const std::size_t xj = xnj;
      const T* __restrict__ za = &xa[xi * nk];
      const T* __restrict__ zb = &xb[xj * nk];
      T* __restrict__ zc = &xc[xi * ldn + xj];
      T t[kii * kjj] = {0};
      for (std::size_t k = 0; k < nk; k++)
#pragma unroll kjj
        for (int j = 0; j < kjj; j++)
#pragma unroll kii
          for (int i = 0; i < kii; i++)
            t[j * kii + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll kii
      for (int i = 0; i < kii; i++)
#pragma unroll kjj
        for (int j = 0; j < kjj; j++)
          zc[i * ldn + j] = f * t[j * kii + i];
      xnj += kjj;
    }
    if (kj > kjjj && ((nj - xnj) & kjjj)) {
      const std::size_t xi = xni;
      const std::size_t xj = xnj;
      const T* __restrict__ za = &xa[xi * nk];
      const T* __restrict__ zb = &xb[xj * nk];
      T* __restrict__ zc = &xc[xi * ldn + xj];
      T t[kii * kjjj] = {0};
      for (std::size_t k = 0; k < nk; k++)
#pragma unroll kjjj
        for (int j = 0; j < kjjj; j++)
#pragma unroll kii
          for (int i = 0; i < kii; i++)
            t[j * kii + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll kii
      for (int i = 0; i < kii; i++)
#pragma unroll kjjj
        for (int j = 0; j < kjjj; j++)
          zc[i * ldn + j] = f * t[j * kii + i];
      xnj += kjjj;
    }
    if ((nj - xnj) & 1) {
      const std::size_t xi = xni;
      const std::size_t xj = xnj;
      const T* __restrict__ za = &xa[xi * nk];
      const T* __restrict__ zb = &xb[xj * nk];
      T* __restrict__ zc = &xc[xi * ldn + xj];
      T t[kii] = {0};
      for (std::size_t k = 0; k < nk; k++)
#pragma unroll kii
        for (int i = 0; i < kii; i++)
          t[0 + i] += za[i * nk + k] * zb[0 * nk + k];
#pragma unroll kii
      for (int i = 0; i < kii; i++)
        zc[i * ldn + 0] = f * t[0 + i];
      xnj += 1;
    }
    xni += kii;
  }
  if (ki > kiii && ((ni - xni) & kiii)) {
    std::size_t xnj = nj / kj * kj;
    for (std::size_t xj = 0; xj < xnj; xj += kj) {
      const std::size_t xi = xni;
      const T* __restrict__ za = &xa[xi * nk];
      const T* __restrict__ zb = &xb[xj * nk];
      T* __restrict__ zc = &xc[xi * ldn + xj];
      T t[kiii * kj] = {0};
      for (std::size_t k = 0; k < nk; k++)
#pragma unroll kj
        for (int j = 0; j < kj; j++)
#pragma unroll kiii
          for (int i = 0; i < kiii; i++)
            t[j * kiii + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll kiii
      for (int i = 0; i < kiii; i++)
#pragma unroll kj
        for (int j = 0; j < kj; j++)
          zc[i * ldn + j] = f * t[j * kiii + i];
    }
    if (kj > kjj && ((nj - xnj) & kjj)) {
      const std::size_t xi = xni;
      const std::size_t xj = xnj;
      const T* __restrict__ za = &xa[xi * nk];
      const T* __restrict__ zb = &xb[xj * nk];
      T* __restrict__ zc = &xc[xi * ldn + xj];
      T t[kiii * kjj] = {0};
      for (std::size_t k = 0; k < nk; k++)
#pragma unroll kjj
        for (int j = 0; j < kjj; j++)
#pragma unroll kiii
          for (int i = 0; i < kiii; i++)
            t[j * kiii + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll kiii
      for (int i = 0; i < kiii; i++)
#pragma unroll kjj
        for (int j = 0; j < kjj; j++)
          zc[i * ldn + j] = f * t[j * kiii + i];
      xnj += kjj;
    }
    if (kj > kjjj && ((nj - xnj) & kjjj)) {
      const std::size_t xi = xni;
      const std::size_t xj = xnj;
      const T* __restrict__ za = &xa[xi * nk];
      const T* __restrict__ zb = &xb[xj * nk];
      T* __restrict__ zc = &xc[xi * ldn + xj];
      T t[kiii * kjjj] = {0};
      for (std::size_t k = 0; k < nk; k++)
#pragma unroll kjjj
        for (int j = 0; j < kjjj; j++)
#pragma unroll kiii
          for (int i = 0; i < kiii; i++)
            t[j * kiii + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll kiii
      for (int i = 0; i < kiii; i++)
#pragma unroll kjjj
        for (int j = 0; j < kjjj; j++)
          zc[i * ldn + j] = f * t[j * kiii + i];
      xnj += kjjj;
    }
    if ((nj - xnj) & 1) {
      const std::size_t xi = xni;
      const std::size_t xj = xnj;
      const T* __restrict__ za = &xa[xi * nk];
      const T* __restrict__ zb = &xb[xj * nk];
      T* __restrict__ zc = &xc[xi * ldn + xj];
      T t[kiii] = {0};
      for (std::size_t k = 0; k < nk; k++)
#pragma unroll kiii
        for (int i = 0; i < kiii; i++)
          t[0 + i] += za[i * nk + k] * zb[0 * nk + k];
#pragma unroll kiii
      for (int i = 0; i < kiii; i++)
        zc[i * ldn + 0] = f * t[0 + i];
      xnj += 1;
    }
    xni += kiii;
  }
  if ((ni - xni) & 1) {
    std::size_t xnj = nj / kj * kj;
    for (std::size_t xj = 0; xj < xnj; xj += kj) {
      const std::size_t xi = xni;
      const T* __restrict__ za = &xa[xi * nk];
      const T* __restrict__ zb = &xb[xj * nk];
      T* __restrict__ zc = &xc[xi * ldn + xj];
      T t[1 * kj] = {0};
      for (std::size_t k = 0; k < nk; k++)
#pragma unroll kj
        for (int j = 0; j < kj; j++)
          t[j * 1 + 0] += za[0 * nk + k] * zb[j * nk + k];
#pragma unroll kj
      for (int j = 0; j < kj; j++)
        zc[0 * ldn + j] = f * t[j * 1 + 0];
    }
    if (kj > kjj && ((nj - xnj) & kjj)) {
      const std::size_t xi = xni;
      const std::size_t xj = xnj;
      const T* __restrict__ za = &xa[xi * nk];
      const T* __restrict__ zb = &xb[xj * nk];
      T* __restrict__ zc = &xc[xi * ldn + xj];
      T t[1 * kjj] = {0};
      for (std::size_t k = 0; k < nk; k++)
#pragma unroll kjj
        for (int j = 0; j < kjj; j++)
          t[j * 1 + 0] += za[0 * nk + k] * zb[j * nk + k];
#pragma unroll kjj
      for (int j = 0; j < kjj; j++)
        zc[0 * ldn + j] = f * t[j * 1 + 0];
      xnj += kjj;
    }
    if (kj > kjjj && ((nj - xnj) & kjjj)) {
      const std::size_t xi = xni;
      const std::size_t xj = xnj;
      const T* __restrict__ za = &xa[xi * nk];
      const T* __restrict__ zb = &xb[xj * nk];
      T* __restrict__ zc = &xc[xi * ldn + xj];
      T t[1 * kjjj] = {0};
      for (std::size_t k = 0; k < nk; k++)
#pragma unroll kjjj
        for (int j = 0; j < kjjj; j++)
          t[j * 1 + 0] += za[0 * nk + k] * zb[j * nk + k];
#pragma unroll kjjj
      for (int j = 0; j < kjjj; j++)
        zc[0 * ldn + j] = f * t[j * 1 + 0];
      xnj += kjjj;
    }
    if ((nj - xnj) & 1) {
      const std::size_t xi = xni;
      const std::size_t xj = xnj;
      const T* __restrict__ za = &xa[xi * nk];
      const T* __restrict__ zb = &xb[xj * nk];
      T* __restrict__ zc = &xc[xi * ldn + xj];
      T t[1] = {0};
      for (std::size_t k = 0; k < nk; k++)
        t[0 + 0] += za[0 * nk + k] * zb[0 * nk + k];
      zc[0 * ldn + 0] = f * t[0 + 0];
      xnj += 1;
    }
    xni += 1;
  }

#elif defined(WICKQC_USE_SIMPLE)

  for (size_t i = 0; i < ni; i++)
    for (size_t j = 0; j < nj; j++) {
      T x = 0.0;
      for (size_t k = 0; k < nk; k++)
        x += xa[i * nk + k] * xb[j * nk + k];
      xc[i * ldn + j] = f * x;
    }

#else
#error "No BLAS backend selected."
#endif
}

} // namespace wickqc::blas