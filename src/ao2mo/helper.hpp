#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>

#include "hdf5_store.hpp"
#include "memory.hpp"
#include "transform.hpp"
#include "types.hpp"

namespace ao2mo {
// Preserve the public enum's existing int representation.
// NOLINTNEXTLINE(performance-enum-size)
enum class Profile { kScalarNevpt2, kScalarMrci, kSpinorDense, kSpinorNevpt2 };
struct Partition {
  // Original MO counts; ncore includes the scalar frozen prefix.
  std::size_t ncore = 0, ncas = 0, nmo = 0, frozen = 0;
  void Validate(bool spinor) const;
  std::vector<std::size_t> Indices(char space) const;
};
struct Dressing {
  // bare or frozen-folded. The latter requires explicit upstream frozen IDs
  // and a labelled electronic constant; helper-level frozen must then be 0.
  std::string state = "bare";
  std::string operator_label;
  std::vector<std::size_t> upstream_frozen;
  // Input MO column -> original orbital identity; required for folded input.
  // Bare input defaults to 0..nmo-1 when this is empty.
  std::vector<std::size_t> orbital_ids;
  double upstream_frozen_electronic = 0;
  double nuclear_repulsion = 0;
  void Validate(const Partition& partition) const;
};

template <typename T>
struct OneElectron {
  std::vector<T> h1e, h1eff; // Retained MO order; h1e includes frozen dressing.
  double frozen_electronic = 0, inactive_electronic = 0;
  double constant_imaginary_residual = 0, nuclear_repulsion = 0;
  Dressing dressing;
  Statistics statistics; // Core trace transforms only; add to the ERI report.
  std::size_t planned_peak_bytes = 0;
};

template <typename T>
std::vector<Request<T>> ProfileRequests(
    Profile profile,
    std::shared_ptr<const Coefficients<T>> coefficients,
    const Partition& partition);

// One bounded J/K request pair per core orbital. Plans include the resident
// one-electron matrices and count every additional integral pass.
template <typename T>
std::vector<Plan> MakeCorePlans(
    const Basis& basis,
    std::shared_ptr<const Coefficients<T>> c,
    const Partition& partition,
    const Options& options,
    const Dressing& dressing = {});

template <typename T>
OneElectron<T> DressCore(
    const Basis& basis,
    std::shared_ptr<const Coefficients<T>> c,
    const Partition& partition,
    std::span<const T> h0,
    const Dressing& dressing,
    const Options& options);
// Attach once to a complete ERI file. A write failure retains the caller's
// file, marked incomplete when HDF5 can still write; an existing attachment
// is rejected before any mutation. The CLI removes its own failed output.
template <typename T>
void WriteHelperMetadata(
    const Result<T>& result,
    const Partition& p,
    const OneElectron<T>& h);

// View of an existing in-memory block, or a disk handle carrying the same
// index map. Conjugation is explicit. owner keeps storage alive.
template <typename T>
struct BlockView {
  std::shared_ptr<const Result<T>> owner;
  std::size_t block = 0, offset = 0;
  std::array<std::size_t, 4> shape{}, strides{};
  bool conjugate = false;
  const T* Data() const; // Throws for disk or conjugated views.
  std::vector<T> Materialize(std::size_t memory_budget) const;
};

template <typename T>
BlockView<T> GetBlock(
    std::shared_ptr<const Result<T>> owner,
    const Partition& partition,
    std::string key,
    Ordering ordering);

namespace helper_detail {
inline char Normalize(char label) {
  switch (label) {
    case 'c':
    case 'I':
      return 'c';
    case 'a':
    case 'A':
      return 'a';
    case 'v':
    case 'E':
      return 'v';
    case 'p':
    case 'P':
      return 'p';
    default:
      throw std::invalid_argument("unknown orbital space label");
  }
}
inline std::string Normalize(std::string key) {
  if (key.size() != 4) {
    throw std::invalid_argument("ERI key must contain four labels");
  }
  for (char& c : key) {
    c = Normalize(c);
  }
  return key;
}
template <typename T>
void ValidateH1(std::span<const T> h0, std::size_t n) {
  if (h0.size() != CheckedProduct({n, n})) {
    throw std::invalid_argument("h1e must match input MO dimensions");
  }
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      const auto v = h0[p * n + q];
      if (!std::isfinite(std::abs(v)) ||
          std::abs(v - detail::Conjugate(h0[q * n + p])) >
              1e-12 + 1e-10 * std::abs(v)) {
        throw std::invalid_argument("h1e must be finite and Hermitian");
      }
    }
  }
}
inline constexpr std::array<std::array<int, 4>, 8> kPermutations{
    {{0, 1, 2, 3},
     {2, 3, 0, 1},
     {1, 0, 3, 2},
     {3, 2, 1, 0},
     {1, 0, 2, 3},
     {0, 1, 3, 2},
     {2, 3, 1, 0},
     {3, 2, 0, 1}}};

template <typename T>
std::vector<Request<T>> CoreRequests(
    std::shared_ptr<const Coefficients<T>> c,
    std::size_t i) {
  std::vector<std::size_t> all(c->nmo);
  std::iota(all.begin(), all.end(), 0);
  Request<T> coulomb, exchange;
  coulomb.name = "J";
  exchange.name = "K";
  coulomb.indices = {Selection<T>{c, all}, {c, all}, {c, {i}}, {c, {i}}};
  exchange.indices = {Selection<T>{c, all}, {c, {i}}, {c, {i}}, {c, all}};
  return {std::move(coulomb), std::move(exchange)};
}
inline std::size_t MetadataBytes(const Partition& p, const Dressing& d) {
  const auto payload = CheckedAdd(
      sizeof(Dressing),
      CheckedAdd(
          CheckedAdd(
              memory_detail::StringBytes(d.state),
              memory_detail::StringBytes(d.operator_label)),
          CheckedAdd(
              memory_detail::VectorBytes(d.upstream_frozen),
              memory_detail::VectorBytes(d.orbital_ids))));
  // Caller/result dressing, validation set, and serialized original/frozen
  // maps.
  return CheckedAdd(
      CheckedProduct({3, payload}),
      CheckedAdd(
          4096,
          CheckedProduct(
              {CheckedAdd(p.nmo, d.upstream_frozen.size()),
               4 * sizeof(std::size_t) +
                   memory_detail::kNodeBytes<std::set<std::size_t>>})));
}
inline std::size_t CoreMetadataBytes(
    const Partition& p,
    const Dressing& d,
    const Options& options) {
  // Inspect retains one plan per core orbital; execution uses only one at a
  // time. Reserve the same bound so inspect and execution have one contract.
  return CheckedAdd(
      MetadataBytes(p, d),
      CheckedAdd(
          CheckedProduct({p.ncore, memory_detail::PlanBound(2)}),
          CheckedProduct({2, memory_detail::OptionsBytes(options)})));
}
template <typename T>
Options CoreOptions(Options options, const Partition& p, const Dressing& d) {
  const auto reserved = CheckedAdd(
      CheckedProduct({5, p.nmo, p.nmo, sizeof(T)}),
      CoreMetadataBytes(p, d, options));
  if (reserved >= options.memory_bytes) {
    throw std::runtime_error("insufficient memory for h1e/core matrices");
  }
  options.audit = AuditMode::kRaw;
  options.output = Output::kMemory;
  options.output_path.clear();
  options.memory_bytes -= reserved;
  return options;
}
template <typename T>
std::size_t ReferencedBytes(const Basis& basis, const Coefficients<T>& c) {
  return CheckedAdd(
      memory_detail::NumericBytes(c), memory_detail::NumericBytes(basis));
}
template <typename T>
std::size_t ResidentBytes(const Partition& p) {
  return CheckedProduct(
      {sizeof(T),
       CheckedAdd(
           CheckedProduct({p.nmo, p.nmo}),
           CheckedProduct({2, p.nmo - p.frozen, p.nmo - p.frozen}))});
}
template <typename T>
std::size_t MetadataPeak(
    const Plan& plan,
    const Partition& p,
    const Dressing& d) {
  const auto nc = p.ncore - p.frozen, na = p.ncas,
             nv = p.nmo - p.ncore - p.ncas;
  const auto block = std::max(
      {CheckedProduct({na, na}),
       CheckedProduct({na, nc}),
       CheckedProduct({nv, nc}),
       CheckedProduct({nv, na})});
  return CheckedAdd(
      CheckedAdd(
          CheckedAdd(plan.metadata_bytes, MetadataBytes(p, d)),
          CheckedAdd(plan.coefficient_bytes, plan.basis_bytes)),
      CheckedAdd(
          ResidentBytes<T>(p),
          CheckedAdd(2ULL << 20, CheckedProduct({block, sizeof(T)}))));
}
template <typename T>
Statistics HelperStatistics(
    const Result<T>& result,
    const Partition& p,
    const OneElectron<T>& h) {
  auto total = result.statistics;
  total.metadata_reserve_bytes =
      CheckedAdd(total.metadata_reserve_bytes, MetadataBytes(p, h.dressing));
  total.managed_numeric_peak_bytes = CheckedAdd(
      total.managed_numeric_peak_bytes,
      CheckedProduct(
          {CheckedAdd(h.h1e.capacity(), h.h1eff.capacity()), sizeof(T)}));
  total.Add(h.statistics);
  return total;
}
} // namespace helper_detail

inline void Partition::Validate(bool spinor) const {
  if (ncore > nmo || ncas > nmo - ncore || frozen > ncore) {
    throw std::invalid_argument("invalid core/active/virtual/frozen partition");
  }
  if (spinor && frozen) {
    throw std::invalid_argument(
        "spinor helper-level frozen is unsupported; fold it upstream explicitly");
  }
}
inline std::vector<std::size_t> Partition::Indices(char space) const {
  Validate(false);
  std::size_t start, end;
  switch (helper_detail::Normalize(space)) {
    case 'c':
      start = frozen;
      end = ncore;
      break;
    case 'a':
      start = ncore;
      end = ncore + ncas;
      break;
    case 'v':
      start = ncore + ncas;
      end = nmo;
      break;
    default:
      start = frozen;
      end = nmo;
      break;
  }
  std::vector<std::size_t> indices(end - start);
  std::iota(indices.begin(), indices.end(), start);
  return indices;
}

template <typename T>
std::vector<Request<T>> ProfileRequests(
    Profile profile,
    // Retain the existing shared-ownership API.
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::shared_ptr<const Coefficients<T>> coefficients,
    const Partition& partition) {
  constexpr bool spinor = std::is_same_v<T, Complex>;
  partition.Validate(spinor);
  if (!coefficients || coefficients->nmo != partition.nmo) {
    throw std::invalid_argument("partition/coefficient dimension mismatch");
  }
  if (spinor !=
      (profile == Profile::kSpinorDense || profile == Profile::kSpinorNevpt2)) {
    throw std::invalid_argument("profile/coefficient dtype mismatch");
  }
  std::vector<std::string> keys;
  if (profile == Profile::kScalarNevpt2) {
    keys = {"ppaa", "papa", "pacv", "cvcv"};
  } else if (profile == Profile::kSpinorDense) {
    keys = {"pppp"};
  } else if (profile == Profile::kSpinorNevpt2) {
    keys = {
        "AAAA",
        "EAAA",
        "EAIA",
        "EAAI",
        "AAIA",
        "EEIA",
        "EAII",
        "EEAA",
        "AAII",
        "EEII"};
  } else {
    // Public key ABI of installed pyblock2 0.5.4rc16. Permutation handling
    // below is independent of its implementation.
    keys = {"vvca", "aaca", "cvca", "cvaa", "cvcv", "caac", "avaa",
            "vacc", "aacc", "avva", "avcv", "accc", "vaca", "vvaa",
            "aaaa", "vvvv", "vccc", "vvva", "vvvc", "cccc", "vvcc"};
  }
  std::vector<Request<T>> requests;
  requests.reserve(keys.size());
  for (const auto& key : keys) {
    Request<T> request;
    request.name =
        (profile == Profile::kSpinorNevpt2 ? "blocks/phys/" : "blocks/chem/") +
        key;
    request.rank_four_output = true;
    auto chem = helper_detail::Normalize(key);
    if (profile == Profile::kSpinorNevpt2) {
      std::swap(chem[1], chem[2]);
      request.ordering = Ordering::kPhysicist;
    }
    for (int i = 0; i < 4; ++i) {
      request.indices[i] = {coefficients, partition.Indices(chem[i])};
    }
    requests.push_back(std::move(request));
  }
  return requests;
}

template <typename T>
std::vector<Plan> MakeCorePlans(
    const Basis& basis,
    // Retain the existing shared-ownership API.
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::shared_ptr<const Coefficients<T>> c,
    const Partition& partition,
    const Options& options,
    const Dressing& dressing) {
  partition.Validate(std::is_same_v<T, Complex>);
  if (!c || c->nmo != partition.nmo) {
    throw std::invalid_argument("partition/coefficient dimension mismatch");
  }
  const auto metadata =
      helper_detail::CoreMetadataBytes(partition, dressing, options);
  const auto local =
      helper_detail::CoreOptions<T>(options, partition, dressing);
  if (helper_detail::ReferencedBytes(basis, *c) > local.memory_bytes) {
    throw std::runtime_error(
        "core matrices and referenced input exceed budget");
  }
  const auto square_bytes = CheckedProduct({c->nmo, c->nmo, sizeof(T)});
  std::vector<Plan> plans;
  plans.reserve(partition.ncore);
  for (std::size_t i = 0; i < partition.ncore; ++i) {
    auto plan = MakePlan(basis, helper_detail::CoreRequests(c, i), local);
    plan.peak_bytes = CheckedAdd(
        plan.peak_bytes,
        CheckedAdd(metadata, CheckedProduct({5, square_bytes})));
    plan.metadata_bytes = CheckedAdd(plan.metadata_bytes, metadata);
    plan.coefficient_bytes =
        CheckedAdd(plan.coefficient_bytes, square_bytes); // caller h0
    plan.buffer_bytes =
        CheckedAdd(plan.buffer_bytes, CheckedProduct({4, square_bytes}));
    plans.push_back(std::move(plan));
  }
  return plans;
}

inline void Dressing::Validate(const Partition& partition) const {
  if (operator_label.empty()) {
    throw std::invalid_argument("h1e requires its actual operator label");
  }
  if (state != "bare" && state != "frozen-folded") {
    throw std::invalid_argument("unknown or already core-dressed h1e state");
  }
  if ((state == "bare" &&
       (!upstream_frozen.empty() || upstream_frozen_electronic != 0)) ||
      (state == "frozen-folded" &&
       (partition.frozen || upstream_frozen.empty() ||
        orbital_ids.size() != partition.nmo))) {
    throw std::invalid_argument("inconsistent frozen dressing metadata");
  }
  if (!std::isfinite(upstream_frozen_electronic) ||
      !std::isfinite(nuclear_repulsion)) {
    throw std::invalid_argument("non-finite electronic/nuclear constant");
  }
  if (!orbital_ids.empty() && orbital_ids.size() != partition.nmo) {
    throw std::invalid_argument(
        "orbital identity map must match input MO columns");
  }
  std::set<std::size_t> ids(upstream_frozen.begin(), upstream_frozen.end());
  if (ids.size() != upstream_frozen.size()) {
    throw std::invalid_argument("duplicate upstream frozen identity");
  }
  for (auto id : orbital_ids) {
    if (!ids.insert(id).second) {
      throw std::invalid_argument(
          "duplicate or already folded orbital identity");
    }
  }
}

template <typename T>
OneElectron<T> DressCore(
    const Basis& basis,
    // Retain the existing shared-ownership API.
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::shared_ptr<const Coefficients<T>> c,
    const Partition& partition,
    std::span<const T> h0,
    const Dressing& dressing,
    const Options& options) {
  constexpr bool spinor = std::is_same_v<T, Complex>;
  partition.Validate(spinor);
  if (!c || c->nmo != partition.nmo ||
      h0.size() != CheckedProduct({c->nmo, c->nmo})) {
    throw std::invalid_argument(
        "h1e, partition and coefficients must have the same MO order");
  }
  const auto n = c->nmo, retained = n - partition.frozen;
  const auto square = CheckedProduct({n, n});
  // h0 is caller-owned and resident. Accumulate traces in O(nMO^2), retaining
  // only one core orbital's pp ii and p i i p outputs at a time.
  const auto metadata =
      helper_detail::CoreMetadataBytes(partition, dressing, options);
  const auto reserved =
      CheckedAdd(CheckedProduct({5, square, sizeof(T)}), metadata);
  const auto local =
      helper_detail::CoreOptions<T>(options, partition, dressing);
  const auto referenced = helper_detail::ReferencedBytes(basis, *c);
  if (referenced > local.memory_bytes) {
    throw std::runtime_error(
        "core matrices and referenced input exceed budget");
  }
  dressing.Validate(partition);
  std::vector<T> delta(square), frozen_delta(square);
  helper_detail::ValidateH1(h0, n);
  OneElectron<T> result;
  result.planned_peak_bytes = CheckedAdd(reserved, referenced);
  result.statistics.metadata_reserve_bytes = metadata;
  result.statistics.caller_referenced_bytes = referenced;
  const auto trace_capacity = CheckedProduct(
      {CheckedAdd(delta.capacity(), frozen_delta.capacity()), sizeof(T)});
  for (std::size_t i = 0; i < partition.ncore; ++i) {
    const auto transformed =
        Transform(basis, helper_detail::CoreRequests(c, i), local);
    result.statistics.Add(transformed.statistics);
    result.statistics.metadata_reserve_bytes = std::max(
        result.statistics.metadata_reserve_bytes,
        CheckedAdd(metadata, transformed.statistics.metadata_reserve_bytes));
    result.statistics.managed_numeric_peak_bytes = std::max(
        result.statistics.managed_numeric_peak_bytes,
        CheckedAdd(
            transformed.statistics.managed_numeric_peak_bytes, trace_capacity));
    result.planned_peak_bytes = std::max(
        result.planned_peak_bytes,
        CheckedAdd(transformed.plan.peak_bytes, reserved));
    for (std::size_t pq = 0; pq < square; ++pq) {
      const auto correction =
          (spinor ? 1.0 : 2.0) * transformed.blocks[0].values[pq] -
          transformed.blocks[1].values[pq];
      delta[pq] += correction;
      if (i < partition.frozen) {
        frozen_delta[pq] += correction;
      }
    }
  }
  result.dressing = dressing;
  result.nuclear_repulsion = dressing.nuclear_repulsion;
  result.h1e.resize(CheckedProduct({retained, retained}));
  result.h1eff.resize(result.h1e.size());
  result.statistics.managed_numeric_peak_bytes = std::max(
      result.statistics.managed_numeric_peak_bytes,
      CheckedAdd(
          trace_capacity,
          CheckedProduct(
              {CheckedAdd(result.h1e.capacity(), result.h1eff.capacity()),
               sizeof(T)})));
  result.statistics.caller_referenced_bytes = CheckedAdd(
      result.statistics.caller_referenced_bytes,
      CheckedProduct({h0.size(), sizeof(T)}));
  for (std::size_t p = partition.frozen; p < n; ++p) {
    for (std::size_t q = partition.frozen; q < n; ++q) {
      const auto target =
          (p - partition.frozen) * retained + q - partition.frozen;
      result.h1e[target] = h0[p * n + q] + frozen_delta[p * n + q];
      result.h1eff[target] = h0[p * n + q] + delta[p * n + q];
    }
  }
  T core{}, frozen{};
  for (std::size_t i = 0; i < partition.ncore; ++i) {
    core += (spinor ? 1.0 : 2.0) * (h0[i * n + i] + 0.5 * delta[i * n + i]);
  }
  for (std::size_t i = 0; i < partition.frozen; ++i) {
    frozen +=
        (spinor ? 1.0 : 2.0) * (h0[i * n + i] + 0.5 * frozen_delta[i * n + i]);
  }
  result.constant_imaginary_residual =
      std::max(std::abs(std::imag(core)), std::abs(std::imag(frozen)));
  if (!std::isfinite(std::abs(core)) || !std::isfinite(std::abs(frozen)) ||
      result.constant_imaginary_residual > 1e-10) {
    throw std::runtime_error(
        "electronic core constant is not finite and real within tolerance");
  }
  result.inactive_electronic =
      std::real(core) + dressing.upstream_frozen_electronic;
  result.frozen_electronic =
      std::real(frozen) + dressing.upstream_frozen_electronic;
  return result;
}

template <typename T>
void WriteHelperMetadata(
    const Result<T>& result,
    const Partition& p,
    const OneElectron<T>& h) {
  std::lock_guard lock(h5::ExecutionMutex());
  if (result.blocks.empty() || result.blocks.front().file.empty()) {
    throw std::invalid_argument("helper metadata requires a disk result");
  }
  p.Validate(std::is_same_v<T, Complex>);
  h.dressing.Validate(p);
  const auto n = p.nmo - p.frozen;
  helper_detail::ValidateH1<T>(h.h1e, n);
  helper_detail::ValidateH1<T>(h.h1eff, n);
  for (const auto& block : result.blocks) {
    if (block.file != result.blocks.front().file) {
      throw std::invalid_argument(
          "helper blocks must belong to one output file");
    }
  }
  auto file = h5::OpenFile(result.blocks.front().file, H5F_ACC_RDWR);
  // Reject a second/partial attachment before changing the existing result.
  for (const char* name :
       {"one_electron", "constants", "partition", "diagnostics"}) {
    if (h5::Check(H5Lexists(file, name, H5P_DEFAULT))) {
      throw std::invalid_argument("helper metadata already exists");
    }
  }
  {
    h5::Handle marker(H5Aopen(file, "complete", H5P_DEFAULT), H5Aclose);
    h5::ScalarAttribute(marker);
    int complete = 0;
    h5::Check(H5Aread(marker, H5T_NATIVE_INT, &complete));
    if (complete != 1) {
      throw std::invalid_argument(
          "helper metadata requires a complete ERI result");
    }
    marker.Close();
  }
  try {
    h5::Complete(file, 0);
    h5::Check(H5Fflush(file, H5F_SCOPE_GLOBAL));
    h5::Check(H5Adelete(file, "schema"));
    h5::StringAttribute(file, "schema", "ao2mo.helper.v1");
    auto matrix = [&](const std::string& name,
                      std::size_t rows,
                      std::size_t cols,
                      const T* data) {
      auto dataset = h5::Matrix<T>(file, name, {rows, cols});
      h5::Slab(dataset, true, {0, 0}, {rows, cols}, const_cast<T*>(data));
      dataset.Close();
    };
    matrix("one_electron/h1e", n, n, h.h1e.data());
    matrix("one_electron/h1eff", n, n, h.h1eff.data());
    h5::Handle one(H5Gopen2(file, "one_electron", H5P_DEFAULT), H5Gclose);
    h5::StringAttribute(
        one,
        "h1e_dressing_state",
        p.frozen ? "frozen-folded" : h.dressing.state);
    h5::StringAttribute(one, "h1eff_dressing_state", "all-inactive-folded");
    h5::StringAttribute(one, "operator_label", h.dressing.operator_label);
    for (const std::string key : {"AA", "AI", "EI", "EA"}) {
      const auto left = p.Indices(key[0]), right = p.Indices(key[1]);
      std::vector<T> block(CheckedProduct({left.size(), right.size()}));
      for (std::size_t i = 0; i < left.size(); ++i) {
        for (std::size_t j = 0; j < right.size(); ++j) {
          block[i * right.size() + j] =
              h.h1eff[(left[i] - p.frozen) * n + right[j] - p.frozen];
        }
      }
      matrix(
          "one_electron/h1eff_blocks/" + key,
          left.size(),
          right.size(),
          block.data());
    }
    for (const auto& [name, value] :
         std::vector<std::pair<std::string, double>>{
             {"frozen_electronic", h.frozen_electronic},
             {"all_inactive_electronic", h.inactive_electronic},
             {"nuclear_repulsion", h.nuclear_repulsion},
             {"constant_imaginary_residual", h.constant_imaginary_residual}}) {
      auto dataset = h5::Matrix<double>(file, "constants/" + name, {1, 1});
      auto data = value;
      h5::Slab(dataset, true, {0, 0}, {1, 1}, &data);
      dataset.Close();
    }
    h5::Handle partition(
        H5Gcreate2(file, "partition", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        H5Gclose);
    h5::SizeAttribute(partition, "ncore", p.ncore - p.frozen);
    h5::SizeAttribute(partition, "ncas", p.ncas);
    h5::SizeAttribute(partition, "nmo", n);
    h5::SizeAttribute(partition, "frozen_prefix", p.frozen);
    h5::SizeAttribute(partition, "input_nmo", p.nmo);
    auto input_ids = h.dressing.orbital_ids;
    if (input_ids.empty()) {
      input_ids.resize(p.nmo);
      std::iota(input_ids.begin(), input_ids.end(), 0);
    }
    h5::IndexVector(partition, "input_indices", input_ids);
    h5::IndexVector(
        partition, "retained_indices", std::span(input_ids).subspan(p.frozen));
    auto frozen_ids = h.dressing.upstream_frozen;
    frozen_ids.insert(
        frozen_ids.end(), input_ids.begin(), input_ids.begin() + p.frozen);
    h5::IndexVector(partition, "frozen_indices", frozen_ids);
    h5::IndexVector(
        partition, "upstream_frozen_indices", h.dressing.upstream_frozen);
    h5::StringAttribute(
        partition,
        "counting_unit",
        std::is_same_v<T, Complex> ? "spinor" : "spatial_orbital");
    h5::Handle diagnostics(
        H5Gcreate2(file, "diagnostics", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        H5Gclose);
    const auto total = helper_detail::HelperStatistics(result, p, h);
    h5::SizeAttribute(
        diagnostics,
        "managed_numeric_peak_bytes",
        total.managed_numeric_peak_bytes);
    h5::SizeAttribute(
        diagnostics, "caller_referenced_bytes", total.caller_referenced_bytes);
    h5::SizeAttribute(
        diagnostics, "metadata_reserve_bytes", total.metadata_reserve_bytes);
    h5::SizeAttribute(diagnostics, "ao_passes", total.ao_passes);
    h5::SizeAttribute(diagnostics, "core_ao_passes", h.statistics.ao_passes);
    h5::SizeAttribute(
        diagnostics, "core_shell_calls", h.statistics.shell_calls);
    h5::SizeAttribute(diagnostics, "half_reuses", total.half_reuses);
    h5::SizeAttribute(
        diagnostics, "eri_planned_peak_bytes", result.plan.peak_bytes);
    h5::SizeAttribute(
        diagnostics, "core_planned_peak_bytes", h.planned_peak_bytes);
    h5::SizeAttribute(
        diagnostics,
        "planned_peak_bytes",
        std::max(
            {h.planned_peak_bytes,
             CheckedAdd(
                 result.plan.peak_bytes,
                 CheckedAdd(
                     helper_detail::ResidentBytes<T>(p),
                     helper_detail::MetadataBytes(p, h.dressing))),
             helper_detail::MetadataPeak<T>(result.plan, p, h.dressing)}));
    h5::StringAttribute(diagnostics, "plan", Describe(result.plan));
    diagnostics.Close();
    partition.Close();
    one.Close();
    h5::Complete(file, 1);
    h5::Check(H5Fflush(file, H5F_SCOPE_GLOBAL));
    file.Close();
  } catch (...) {
    // This API extends the caller's existing ERI file, so retain it. A failed
    // final flush/close must not leave a successful completion marker when
    // HDF5 is still able to write; updates are not database transactions.
    try {
      h5::Complete(file, 0);
      h5::Check(H5Fflush(file, H5F_SCOPE_GLOBAL));
    } catch (...) { // NOLINT(bugprone-empty-catch)
      // Preserve the original write error if clearing the completion flag
      // fails.
    }
    throw;
  }
}

template <typename T>
const T* BlockView<T>::Data() const {
  if (!owner || conjugate || !owner->blocks.at(block).file.empty()) {
    throw std::logic_error(
        "disk/conjugated view requires explicit materialization");
  }
  const auto* data = owner->blocks.at(block).Values().data();
  return data ? data + offset : nullptr;
}

template <typename T>
std::vector<T> BlockView<T>::Materialize(std::size_t memory_budget) const {
  std::lock_guard lock(h5::ExecutionMutex());
  if (!owner) {
    throw std::invalid_argument("view has no owner");
  }
  const auto& source = owner->blocks.at(block);
  const auto size = CheckedProduct({shape[0], shape[1], shape[2], shape[3]});
  auto required = CheckedProduct({size, sizeof(T)});
  if (!source.file.empty()) {
    required = CheckedAdd(required, 2ULL << 20);
  }
  if (required > memory_budget) {
    throw std::runtime_error("explicit block load exceeds memory budget");
  }
  std::vector<T> values(size);
  h5::Handle file, dataset;
  if (!source.file.empty()) {
    file = h5::OpenFile(source.file, H5F_ACC_RDONLY);
    dataset = h5::OpenDataset(file, source.name);
  }
  std::size_t index = 0;
  for (std::size_t p = 0; p < shape[0]; ++p) {
    for (std::size_t q = 0; q < shape[1]; ++q) {
      for (std::size_t r = 0; r < shape[2]; ++r) {
        for (std::size_t s = 0; s < shape[3]; ++s) {
          const auto source_index = offset + p * strides[0] + q * strides[1] +
              r * strides[2] + s * strides[3];
          T value;
          if (source.file.empty()) {
            const auto memory = source.Values();
            if (source_index >= memory.size()) {
              throw std::out_of_range("block view exceeds its memory owner");
            }
            value = memory[source_index];
          } else {
            h5::Slab(
                dataset,
                false,
                {source_index / source.pair_shape[1],
                 source_index % source.pair_shape[1]},
                {1, 1},
                &value);
          }
          values[index++] = conjugate ? detail::Conjugate(value) : value;
        }
      }
    }
  }
  return values;
}

template <typename T>
BlockView<T> GetBlock(
    std::shared_ptr<const Result<T>> owner,
    const Partition& partition,
    std::string key,
    Ordering ordering) {
  constexpr bool spinor = std::is_same_v<T, Complex>;
  if (!owner) {
    throw std::invalid_argument("getter requires a result owner");
  }
  partition.Validate(spinor);
  key = helper_detail::Normalize(key);
  if (ordering == Ordering::kPhysicist) {
    std::swap(key[1], key[2]);
  }
  for (std::size_t block_index = 0; block_index < owner->blocks.size();
       ++block_index) {
    const auto& block = owner->blocks[block_index];
    if (block.layout != MoLayout::kDense) {
      throw std::invalid_argument("helper getters require dense blocks");
    }
    auto stored = helper_detail::Normalize(
        block.name.substr(block.name.find_last_of('/') + 1));
    std::array<std::size_t, 4> strides{0, 0, block.shape[3], 1};
    strides[1] = CheckedProduct({block.shape[2], strides[2]});
    strides[0] = CheckedProduct({block.shape[1], strides[1]});
    if (block.ordering == Ordering::kPhysicist) {
      std::swap(stored[1], stored[2]);
      std::swap(strides[1], strides[2]);
    }
    for (int iperm = 0; iperm < (spinor ? 4 : 8); ++iperm) {
      const auto& perm = helper_detail::kPermutations[iperm];
      BlockView<T> view;
      view.block = block_index;
      view.conjugate = spinor && (iperm == 2 || iperm == 3);
      bool matches = true;
      for (int axis = 0; axis < 4; ++axis) {
        const auto stored_axis = perm[axis];
        if (stored[stored_axis] != key[axis] && stored[stored_axis] != 'p') {
          matches = false;
          break;
        }
        const auto wanted = partition.Indices(key[axis]);
        const auto available = partition.Indices(stored[stored_axis]);
        view.shape[axis] = wanted.size();
        view.strides[axis] = strides[stored_axis];
        if (!wanted.empty() && !available.empty()) {
          view.offset +=
              (wanted.front() - available.front()) * strides[stored_axis];
        }
      }
      if (matches) {
        view.owner = std::move(owner);
        if (ordering == Ordering::kPhysicist) {
          std::swap(view.shape[1], view.shape[2]);
          std::swap(view.strides[1], view.strides[2]);
        }
        return view;
      }
    }
  }
  throw std::out_of_range(
      "requested block is not covered by stored blocks and legal symmetry");
}

} // namespace ao2mo
