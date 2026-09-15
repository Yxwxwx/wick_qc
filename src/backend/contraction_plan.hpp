#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace wickqc::backend {

// A binary contraction after einsum has resolved diagonals and unary sums.
// Indices describe the original tensor axes; permutations describe only the
// matrix layout needed by the BLAS executor. Output order is batch, lhs, rhs.
struct ContractionPlan {
  std::vector<int> lhs_indices;
  std::vector<int> rhs_indices;
  std::vector<int> output_indices;
  std::vector<std::size_t> output_shape;
  std::vector<int> lhs_permutation;
  std::vector<int> rhs_permutation;
  std::size_t batches = 1;
  std::size_t m = 1;
  std::size_t n = 1;
  std::size_t k = 1;
};

inline ContractionPlan PlanContraction(
    std::span<const std::size_t> lhs_shape,
    std::span<const std::size_t> rhs_shape,
    std::span<const int> lhs_contract,
    std::span<const int> rhs_contract,
    std::span<const int> lhs_batch,
    std::span<const int> rhs_batch) {
  if (lhs_shape.size() + rhs_shape.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("tensor rank exceeds contraction index range");
  }
  if (lhs_contract.size() != rhs_contract.size() ||
      lhs_batch.size() != rhs_batch.size()) {
    throw std::invalid_argument("contracted or batch axis count mismatch");
  }
  ContractionPlan plan;
  plan.lhs_indices.assign(lhs_shape.size(), -1);
  plan.rhs_indices.assign(rhs_shape.size(), -1);
  int next_index = 0;
  const auto pair_axes = [&](int lhs, int rhs) {
    if (lhs < 0 || rhs < 0 ||
        static_cast<std::size_t>(lhs) >= lhs_shape.size() ||
        static_cast<std::size_t>(rhs) >= rhs_shape.size()) {
      throw std::invalid_argument("contraction axis out of range");
    }
    if (plan.lhs_indices[lhs] != -1 || plan.rhs_indices[rhs] != -1) {
      throw std::invalid_argument("duplicate or overlapping contraction axes");
    }
    if (lhs_shape[lhs] != rhs_shape[rhs]) {
      throw std::invalid_argument(
          "contracted or batch dimensions do not match");
    }
    plan.lhs_indices[lhs] = plan.rhs_indices[rhs] = next_index++;
  };
  for (std::size_t i = 0; i < lhs_contract.size(); ++i) {
    pair_axes(lhs_contract[i], rhs_contract[i]);
    plan.k *= lhs_shape[lhs_contract[i]];
  }

  std::vector<std::pair<int, int>> batches;
  batches.reserve(lhs_batch.size());
  for (std::size_t i = 0; i < lhs_batch.size(); ++i) {
    batches.emplace_back(lhs_batch[i], rhs_batch[i]);
  }
  std::sort(batches.begin(), batches.end());
  for (const auto& [lhs, rhs] : batches) {
    pair_axes(lhs, rhs);
    plan.output_indices.push_back(plan.lhs_indices[lhs]);
    plan.output_shape.push_back(lhs_shape[lhs]);
    plan.lhs_permutation.push_back(lhs);
    plan.rhs_permutation.push_back(rhs);
    plan.batches *= lhs_shape[lhs];
  }
  const auto add_free_axes = [&](std::span<const std::size_t> shape,
                                 std::vector<int>& indices,
                                 std::vector<int>& permutation,
                                 std::size_t& extent) {
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      if (indices[axis] == -1) {
        indices[axis] = next_index++;
        plan.output_indices.push_back(indices[axis]);
        plan.output_shape.push_back(shape[axis]);
        permutation.push_back(static_cast<int>(axis));
        extent *= shape[axis];
      }
    }
  };
  add_free_axes(lhs_shape, plan.lhs_indices, plan.lhs_permutation, plan.m);
  add_free_axes(rhs_shape, plan.rhs_indices, plan.rhs_permutation, plan.n);
  plan.lhs_permutation.insert(
      plan.lhs_permutation.end(), lhs_contract.begin(), lhs_contract.end());
  plan.rhs_permutation.insert(
      plan.rhs_permutation.end(), rhs_contract.begin(), rhs_contract.end());
  return plan;
}

} // namespace wickqc::backend
