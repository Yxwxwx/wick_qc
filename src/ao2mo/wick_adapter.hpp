#pragma once

#include <ao2mo.hpp>

// Optional adapter: only clients including wick_adapter.hpp require wick_qc.
#include "backend/ndarray.hpp"

namespace ao2mo {
// Transform directly into a contiguous NDArray owned by the caller. The
// returned block keeps the array alive; do not resize it while views exist.
template <typename T>
MemoryBuffer<T> WickBuffer(std::shared_ptr<wickqc::NDArray<T>> array) {
  if (!array || !array->IsContiguous()) {
    throw std::invalid_argument("wick_qc output must be a contiguous NDArray");
  }
  return {{array->data(), array->Size()}, std::move(array)};
}

// The current wick_qc public constructor copies a vector into owned storage;
// it has no const/conjugated external-owner constructor. Make that copy and
// its peak explicit instead of returning a mutable dangling borrowed pointer.
template <typename T>
wickqc::NDArray<T> ToWick(const BlockView<T>& view, std::size_t memory_budget) {
  const auto size = CheckedProduct(
      {view.shape[0], view.shape[1], view.shape[2], view.shape[3], sizeof(T)});
  if (size > memory_budget) {
    throw std::runtime_error("wick_qc owned output exceeds budget");
  }
  auto values = view.Materialize(memory_budget - size);
  return wickqc::NDArray<T>(
      {view.shape[0], view.shape[1], view.shape[2], view.shape[3]}, values);
}
} // namespace ao2mo
