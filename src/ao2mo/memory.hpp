#pragma once

#include "types.hpp"

namespace ao2mo::memory_detail {
template <typename T>
std::size_t VectorBytes(const std::vector<T>& values) {
  return CheckedProduct({values.capacity(), sizeof(T)});
}
inline std::size_t StringBytes(const std::string& value) {
  return CheckedAdd(
      value.capacity(), 1); // Conservatively includes SSO storage.
}
// Node links/alignment allowance for the supported libstdc++/libc++ containers.
// Allocator bookkeeping is external to payload reservations.
template <typename Map>
inline constexpr std::size_t kNodeBytes =
    sizeof(typename Map::value_type) + 4 * sizeof(void*);

inline std::size_t NumericBytes(const Basis& basis) {
  return CheckedAdd(
      CheckedAdd(VectorBytes(basis.atm), VectorBytes(basis.bas)),
      VectorBytes(basis.env));
}
template <typename T>
std::size_t NumericBytes(const Coefficients<T>& coefficients) {
  return CheckedAdd(
      VectorBytes(coefficients.alpha), VectorBytes(coefficients.beta));
}
template <typename T>
std::size_t RequestBytes(const Request<T>& request) {
  auto bytes = CheckedAdd(sizeof(request), StringBytes(request.name));
  for (const auto& selection : request.indices) {
    bytes = CheckedAdd(bytes, VectorBytes(selection.columns));
  }
  return bytes;
}
template <typename T>
std::size_t RequestBytes(const std::vector<Request<T>>& requests) {
  auto bytes = VectorBytes(requests);
  for (const auto& request : requests) {
    bytes = CheckedAdd(bytes, RequestBytes(request) - sizeof(request));
  }
  return bytes;
}
inline std::size_t OptionsBytes(const Options& options) {
  auto bytes = CheckedAdd(
      sizeof(options),
      CheckedAdd(
          StringBytes(options.output_path.native()),
          StringBytes(options.scratch_directory.native())));
  for (const auto& [name, value] : options.provenance) {
    bytes = CheckedAdd(
        bytes,
        CheckedAdd(
            kNodeBytes<decltype(options.provenance)>,
            CheckedAdd(StringBytes(name), StringBytes(value))));
  }
  return bytes;
}
inline std::size_t PlanBytes(const Plan& plan) {
  auto bytes = CheckedAdd(
      sizeof(plan),
      CheckedAdd(VectorBytes(plan.groups), VectorBytes(plan.batches)));
  for (const auto& group : plan.groups) {
    bytes = CheckedAdd(
        bytes,
        CheckedAdd(
            VectorBytes(group.requests), (group.reversed.capacity() + 7) / 8));
  }
  for (const auto& batch : plan.batches) {
    bytes = CheckedAdd(bytes, VectorBytes(batch));
  }
  return bytes;
}
inline std::size_t PlanBound(std::size_t requests) {
  // At most one group/batch per request, doubling vector growth, and a whole
  // machine word per group's vector<bool> even for a single reversal flag.
  return CheckedAdd(
      sizeof(Plan),
      CheckedProduct(
          {2,
           requests,
           sizeof(HalfGroup) + sizeof(std::vector<std::size_t>) +
               3 * sizeof(std::size_t)}));
}
template <typename T>
std::size_t MetadataReserve(
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options) {
  // Covers caller selections, a copied audit partner plus transient vectors,
  // result/plan copies, planner sets, dataset handles and batch descriptors.
  auto bytes = CheckedProduct(
      {4, CheckedAdd(RequestBytes(requests), OptionsBytes(options))});
  bytes = CheckedAdd(bytes, CheckedProduct({4, PlanBound(requests.size())}));
  bytes = CheckedAdd(
      bytes,
      CheckedProduct({2, basis.bas.size() / 8 + 1, sizeof(std::size_t)}));
  bytes = CheckedAdd(bytes, StringBytes(basis.source_identity));
  bytes = CheckedAdd(
      bytes,
      CheckedProduct(
          {static_cast<std::size_t>(options.threads),
           sizeof(std::vector<double>)}));
  for (const auto& request : requests) {
    bytes = CheckedAdd(
        bytes,
        sizeof(Block<T>) + sizeof(AuditDiagnostic) + 12 * sizeof(void*) +
            5 * sizeof(std::vector<T>));
    bytes = CheckedAdd(bytes, CheckedProduct({2, StringBytes(request.name)}));
    bytes = CheckedAdd(bytes, StringBytes(options.output_path.native()));
    for (const auto& selection : request.indices) {
      if (selection.coefficients) {
        const auto& c = *selection.coefficients;
        bytes = CheckedAdd(
            bytes,
            CheckedAdd(
                sizeof(c) + 8 * sizeof(void*),
                CheckedAdd(
                    StringBytes(c.source_label),
                    StringBytes(c.source_identity))));
      }
    }
  }
  return bytes;
}
} // namespace ao2mo::memory_detail
