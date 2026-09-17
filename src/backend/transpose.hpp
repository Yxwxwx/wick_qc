#pragma once

#if defined(WICKQC_TRANSPOSE_HPTT)
#include <hptt.h>
#endif

#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace wickqc::backend {

struct TransposeOptions {
  // Payload bytes, not read+write traffic. No unmeasured crossover is assumed.
  // nullopt explicitly keeps native execution, even in an HPTT-enabled build.
  std::optional<std::size_t> hptt_min_bytes =
#if defined(WICKQC_HPTT_MIN_BYTES)
      WICKQC_HPTT_MIN_BYTES;
#else
      std::nullopt;
#endif
  int num_threads = 1;
};

namespace transpose_detail {

template <typename T>
inline constexpr bool kSupportedScalar = std::is_same_v<T, float> ||
    std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> ||
    std::is_same_v<T, std::complex<double>>;

// Physical row-major input axes, with singleton axes removed. The permutation
// maps each output axis to an input axis, exactly as NDArray::TransposeView.
struct DenseLayout {
  std::vector<int> shape;
  std::vector<int> permutation;
  std::size_t elements = 1;
};

inline std::optional<DenseLayout> DescribeDenseLayout(
    std::span<const std::size_t> shape,
    std::span<const std::ptrdiff_t> source_strides,
    std::span<const int> permutation,
    std::span<const std::ptrdiff_t> destination_strides) {
  const auto rank = shape.size();
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (rank == 0 || rank > limit || source_strides.size() != rank ||
      permutation.size() != rank || destination_strides.size() != rank) {
    return std::nullopt;
  }
  DenseLayout layout;
  // HPTT uses int not only for extents but also for products/leading
  // dimensions.
  for (const auto extent : shape) {
    if (extent == 0 || extent > limit / layout.elements) {
      return std::nullopt;
    }
    layout.elements *= extent;
  }
  std::vector<bool> seen(rank, false);
  std::ptrdiff_t expected = 1;
  for (std::size_t out = rank; out-- > 0;) {
    const int in = permutation[out];
    if (in < 0 || static_cast<std::size_t>(in) >= rank || seen[in]) {
      return std::nullopt;
    }
    seen[in] = true;
    if (shape[in] > 1 && destination_strides[out] != expected) {
      return std::nullopt;
    }
    expected *= static_cast<std::ptrdiff_t>(shape[in]);
  }

  // A permuted dense view still occupies one dense interval. Recover its
  // physical axis order; reject holes, negative strides and broadcasting.
  std::vector<int> axes;
  for (std::size_t axis = 0; axis < rank; ++axis) {
    if (shape[axis] > 1) {
      axes.push_back(static_cast<int>(axis));
    }
  }
  std::stable_sort(axes.begin(), axes.end(), [&](int a, int b) {
    return source_strides[a] > source_strides[b];
  });
  expected = 1;
  for (auto axis = axes.rbegin(); axis != axes.rend(); ++axis) {
    if (source_strides[*axis] != expected) {
      return std::nullopt;
    }
    expected *= static_cast<std::ptrdiff_t>(shape[*axis]);
  }
  std::vector<int> physical_axis(rank, -1);
  for (const int axis : axes) {
    physical_axis[axis] = static_cast<int>(layout.shape.size());
    layout.shape.push_back(static_cast<int>(shape[axis]));
  }
  for (const int axis : permutation) {
    if (shape[axis] > 1) {
      layout.permutation.push_back(physical_axis[axis]);
    }
  }
  return layout;
}

} // namespace transpose_detail

// This gate is deliberately separate from layout recognition and execution.
// Calibrate the byte crossover with planning included, for the chosen type,
// permutation family and thread count. Until then auto-selection stays native.
template <typename T>
inline bool ShouldUseHptt(
    const transpose_detail::DenseLayout& layout,
    const TransposeOptions& options) {
  if (!transpose_detail::kSupportedScalar<T> || !options.hptt_min_bytes ||
      options.num_threads < 1 || layout.shape.size() < 3 ||
      layout.permutation.size() != layout.shape.size()) {
    return false;
  }
  // Identity, a flattened matrix transpose, and preserved contiguous inner
  // blocks stay native. Count runs of consecutive physical input axes.
  std::size_t runs = 1;
  for (std::size_t i = 1; i < layout.permutation.size(); ++i) {
    runs += layout.permutation[i] != layout.permutation[i - 1] + 1;
  }
  if (runs <= 2 ||
      layout.permutation.back() == static_cast<int>(layout.shape.size()) - 1) {
    return false;
  }
  const auto minimum = *options.hptt_min_bytes;
  const auto minimum_elements =
      minimum / sizeof(T) + (minimum % sizeof(T) != 0);
  if (layout.elements < minimum_elements) {
    return false;
  }
#if defined(WICKQC_TRANSPOSE_HPTT) && defined(_OPENMP)
  // Do not add nested parallelism to a caller's existing OpenMP region.
  return !omp_in_parallel();
#elif defined(WICKQC_TRANSPOSE_HPTT)
  return true;
#else
  return false;
#endif
}

// Called only for non-overlapping, validated arrays. Returning false leaves
// both buffers untouched so NDArray can use its existing native copy loop.
template <typename T>
inline bool TryHpttTranspose(
    [[maybe_unused]] const T* source,
    std::span<const std::size_t> shape,
    std::span<const std::ptrdiff_t> source_strides,
    std::span<const int> permutation,
    [[maybe_unused]] T* destination,
    std::span<const std::ptrdiff_t> destination_strides,
    [[maybe_unused]] T alpha,
    [[maybe_unused]] T beta,
    const TransposeOptions& options) {
  if (!options.hptt_min_bytes) {
    return false;
  }
  const auto layout = transpose_detail::DescribeDenseLayout(
      shape, source_strides, permutation, destination_strides);
  if (!layout || !ShouldUseHptt<T>(*layout, options)) {
    return false;
  }
#if defined(WICKQC_TRANSPOSE_HPTT)
  if constexpr (transpose_detail::kSupportedScalar<T>) {
    auto plan = hptt::create_plan(
        layout->permutation.data(),
        static_cast<int>(layout->shape.size()),
        alpha,
        source,
        layout->shape.data(),
        nullptr,
        beta,
        destination,
        nullptr,
        hptt::ESTIMATE,
        options.num_threads,
        nullptr,
        true);
    // Native Copy evaluates beta * destination even for beta == 0. Retain
    // that behavior (including NaNs) instead of HPTT's zero-beta shortcut.
    // Cached stores suit materialized tensors that are consumed immediately.
    if (options.num_threads > 1) {
      plan->template execute_expert<false, true, false>();
    } else {
      plan->template execute_expert<false, false, false>();
    }
    return true;
  }
#endif
  return false;
}

} // namespace wickqc::backend
