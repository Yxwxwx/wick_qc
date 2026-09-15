#pragma once

#include <tblis/frame/3t/mult.h>
#include <tblis/frame/base/basic_types.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace wickqc::backend {

// Metadata owns only descriptors. Tensor data stays in the NDArray view,
// including negative strides and broadcast (zero) strides; no matrix packing.
struct TBLISMetadata {
  std::vector<tblis::len_type> lengths;
  std::vector<tblis::stride_type> strides;
  std::vector<tblis::label_type> indices;

  TBLISMetadata(
      std::span<const std::size_t> shape,
      std::span<const std::ptrdiff_t> tensor_strides,
      std::span<const int> tensor_indices) {
    if (shape.size() != tensor_strides.size() ||
        shape.size() != tensor_indices.size() ||
        shape.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("invalid TBLIS tensor metadata ranks");
    }
    lengths.reserve(shape.size());
    strides.reserve(shape.size());
    indices.reserve(shape.size());
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      if (!std::in_range<tblis::len_type>(shape[axis]) ||
          !std::in_range<tblis::stride_type>(tensor_strides[axis])) {
        throw std::overflow_error(
            "tensor extent or stride exceeds TBLIS range");
      }
      // The planner supplies dense local IDs, independent of einsum spelling.
      // Use the full installed label_type range, without narrowing collisions.
      const auto label = static_cast<std::intmax_t>(
                             std::numeric_limits<tblis::label_type>::lowest()) +
          tensor_indices[axis];
      if (tensor_indices[axis] < 0 ||
          label > std::numeric_limits<tblis::label_type>::max()) {
        throw std::invalid_argument("contraction exceeds TBLIS label capacity");
      }
      lengths.push_back(static_cast<tblis::len_type>(shape[axis]));
      strides.push_back(static_cast<tblis::stride_type>(tensor_strides[axis]));
      indices.push_back(static_cast<tblis::label_type>(label));
    }
  }
};

// C = alpha * contract(A, B) + beta * C, with no implicit conjugation.
// Inputs and output must not overlap. Empty contractions are handled by
// NDArray.
template <typename T>
inline void TblisContract(
    const T* lhs,
    const TBLISMetadata& lhs_metadata,
    const T* rhs,
    const TBLISMetadata& rhs_metadata,
    T* output,
    const TBLISMetadata& output_metadata,
    T alpha,
    T beta) {
  static_assert(
      std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>);
  const tblis::tblis_tensor a(
      alpha,
      lhs,
      static_cast<int>(lhs_metadata.lengths.size()),
      lhs_metadata.lengths.data(),
      lhs_metadata.strides.data());
  const tblis::tblis_tensor b(
      T{1},
      rhs,
      static_cast<int>(rhs_metadata.lengths.size()),
      rhs_metadata.lengths.data(),
      rhs_metadata.strides.data());
  tblis::tblis_tensor c(
      beta,
      output,
      static_cast<int>(output_metadata.lengths.size()),
      output_metadata.lengths.data(),
      output_metadata.strides.data());
  tblis::tblis_tensor_mult(
      nullptr,
      nullptr,
      &a,
      lhs_metadata.indices.data(),
      &b,
      rhs_metadata.indices.data(),
      &c,
      output_metadata.indices.data());
}

} // namespace wickqc::backend
