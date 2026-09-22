#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ao2mo {
using Complex = std::complex<double>;

namespace detail {
// std::conj(double) returns a complex value; keep the scalar type unchanged.
template <typename T>
T Conjugate(T value) {
  if constexpr (std::is_same_v<T, Complex>) {
    return std::conj(value);
  } else {
    return value;
  }
}
} // namespace detail

std::size_t CheckedAdd(std::size_t a, std::size_t b);
// Preserve the public enums' existing int representation.
// NOLINTBEGIN(performance-enum-size)
enum class AoSymmetry { kS1, kS4 };
enum class MoLayout { kDense, kS4 };
enum class Ordering { kChemist, kPhysicist };
enum class Workspace { kIncore, kOutcore, kAuto };
enum class Output { kMemory, kHdf5 };
enum class AuditMode { kRaw, kAudit, kProjectRoundoff };
// NOLINTEND(performance-enum-size)

// libcint/PySCF ATM_SLOTS=6, BAS_SLOTS=8. Only real spherical int2e.
struct Basis {
  std::vector<int> atm, bas;
  std::vector<double> env;
  std::string source_identity; // Optional upstream label, separate from content
                               // fingerprint.
  std::vector<std::size_t> AoOffsets() const;
};

// Row-major (AO, MO); real coefficients have alpha only. Spinors have both
// alpha and beta on the same real spherical AO basis. No j-spinor conversion
// is performed in the numerical library.
template <typename T>
struct Coefficients {
  static_assert(
      std::is_same_v<T, double> || std::is_same_v<T, Complex>,
      "AO2MO supports double or std::complex<double> coefficients");
  std::size_t nao = 0, nmo = 0;
  std::vector<T> alpha, beta;
  std::string source_label, source_identity;
};

template <typename T>
struct Selection {
  std::shared_ptr<const Coefficients<T>> coefficients;
  std::vector<std::size_t> columns;
  bool operator==(const Selection&) const = default;
};

template <typename T>
struct Request {
  std::string name = "eri_mo";
  // Always in chemist order, even when output ordering is physicist.
  std::array<Selection<T>, 4> indices;
  MoLayout layout = MoLayout::kDense;
  Ordering ordering = Ordering::kChemist;
  bool rank_four_output = false; // Helper schema; numerical layout unchanged.
};

struct Options {
  AoSymmetry ao_symmetry = AoSymmetry::kS1;
  Workspace workspace = Workspace::kAuto;
  Output output = Output::kMemory;
  std::size_t memory_bytes = 512ULL << 20;
  std::size_t io_tile_bytes = 1ULL << 20;
  std::filesystem::path output_path;
  std::filesystem::path scratch_directory =
      std::filesystem::temp_directory_path();
  int threads = 1;
  AuditMode audit = AuditMode::kRaw;
  bool first_pair_only =
      false; // Independent audit paths bypass pair canonicalization.
  bool libcint_optimizer = false; // Explicit, budgeted libcint 6.1.3 optimizer.
  std::map<std::string, std::string>
      provenance; // Upstream declarations; never used as cache keys.
};

struct HalfGroup {
  std::vector<std::size_t> requests;
  std::vector<bool> reversed;
  std::size_t pairs = 0, half_bytes = 0;
  double pass1_flops = 0, pass2_flops = 0;
};

struct Plan {
  Workspace workspace;
  std::vector<HalfGroup> groups;
  std::vector<std::vector<std::size_t>> batches;
  std::size_t ao_passes = 0;
  std::size_t coefficient_bytes = 0, output_bytes = 0, scratch_bytes = 0;
  std::size_t basis_bytes = 0; // Included in buffer_bytes; caller-owned.
  std::size_t optimizer_bytes = 0; // libcint payload, included in buffer_bytes.
  std::size_t buffer_bytes = 0, io_cache_bytes = 0, peak_bytes = 0;
  std::size_t metadata_bytes =
      0; // Conservative container/index storage reservation.
  std::size_t cache_doubles = 0, max_shell = 0;
  std::size_t max_m = 0, max_pair = 0;
  std::size_t audit_ao_passes = 0;
};

struct Statistics {
  std::size_t ao_passes = 0, shell_calls = 0, half_reuses = 0;
  std::size_t zero_shell_calls = 0, half_transforms = 0;
  // Logical tensor bytes passed to HDF5, excluding metadata and filesystem
  // overhead.
  std::size_t temp_read_bytes = 0, temp_write_bytes = 0, final_write_bytes = 0;
  std::size_t audit_read_bytes = 0, audit_write_bytes = 0;
  // Destination payload copied/packed by the loops timed in transpose_seconds.
  // Counts each native move once; not read+write bus traffic or HDF5 staging.
  std::size_t transpose_bytes = 0;
  double pass1_seconds = 0, pass2_seconds = 0;
  double ao_seconds = 0, temp_io_seconds = 0, final_io_seconds = 0,
         audit_io_seconds = 0;
  double transpose_seconds = 0, io_seconds = 0;
  double audit_seconds = 0;
  // AO/MO numeric vector capacity observed at allocation boundaries, excluding
  // caller data, metadata tables, allocator overhead and external libraries.
  std::size_t managed_numeric_peak_bytes = 0, caller_referenced_bytes = 0;
  std::size_t reused_workspace_bytes = 0;
  std::size_t metadata_reserve_bytes = 0;
  void Add(const Statistics& other) {
    ao_passes += other.ao_passes;
    shell_calls += other.shell_calls;
    half_reuses += other.half_reuses;
    zero_shell_calls += other.zero_shell_calls;
    half_transforms += other.half_transforms;
    temp_read_bytes = CheckedAdd(temp_read_bytes, other.temp_read_bytes);
    temp_write_bytes = CheckedAdd(temp_write_bytes, other.temp_write_bytes);
    final_write_bytes = CheckedAdd(final_write_bytes, other.final_write_bytes);
    audit_read_bytes = CheckedAdd(audit_read_bytes, other.audit_read_bytes);
    audit_write_bytes = CheckedAdd(audit_write_bytes, other.audit_write_bytes);
    transpose_bytes = CheckedAdd(transpose_bytes, other.transpose_bytes);
    ao_seconds += other.ao_seconds;
    temp_io_seconds += other.temp_io_seconds;
    final_io_seconds += other.final_io_seconds;
    audit_io_seconds += other.audit_io_seconds;
    pass1_seconds += other.pass1_seconds;
    pass2_seconds += other.pass2_seconds;
    transpose_seconds += other.transpose_seconds;
    io_seconds += other.io_seconds;
    audit_seconds += other.audit_seconds;
    managed_numeric_peak_bytes =
        std::max(managed_numeric_peak_bytes, other.managed_numeric_peak_bytes);
    caller_referenced_bytes =
        std::max(caller_referenced_bytes, other.caller_referenced_bytes);
    reused_workspace_bytes =
        std::max(reused_workspace_bytes, other.reused_workspace_bytes);
    metadata_reserve_bytes =
        std::max(metadata_reserve_bytes, other.metadata_reserve_bytes);
  }
};
struct AuditDiagnostic {
  std::string name;
  std::array<double, 3> raw_residuals{};
  double roundoff_gate = 0, correction = 0;
};

// Exact-size contiguous output storage. owner is optional; without it the
// caller keeps storage alive through execution and every returned view.
// A failed execution may leave partially written values in this buffer.
template <typename T>
struct MemoryBuffer {
  std::span<T> values;
  std::shared_ptr<void> owner;
};

template <typename T>
struct Block {
  std::string name;
  std::array<std::size_t, 4> shape{};
  MoLayout layout = MoLayout::kDense;
  Ordering ordering = Ordering::kChemist;
  std::vector<T> values; // Empty for a disk handle; loading is explicit.
  MemoryBuffer<T> external; // Used instead of values for a caller-owned sink.
  std::filesystem::path file;
  std::array<std::size_t, 2> pair_shape{};
  std::span<const T> Values() const {
    if (!file.empty()) {
      throw std::logic_error("disk block requires explicit reading");
    }
    return values.empty() ? std::span<const T>(external.values)
                          : std::span<const T>(values);
  }
  std::span<T> MutableValues() {
    if (!file.empty()) {
      throw std::logic_error("disk block requires explicit reading");
    }
    return values.empty() ? external.values : std::span<T>(values);
  }
};

template <typename T>
struct Result {
  Plan plan;
  Statistics statistics;
  std::vector<Block<T>> blocks;
  std::vector<AuditDiagnostic> audit;
};

// Each call scopes reuse to one immutable basis/operator/options tuple.
// Ordered selections and coefficient object identity are compared exactly.
template <typename T>
Plan MakePlan(
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options);
template <typename T>
class PreparedTransform;
template <typename T>
Result<T> Transform(
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options,
    std::type_identity_t<std::span<const MemoryBuffer<T>>> buffers = {});

std::size_t CheckedProduct(std::initializer_list<std::size_t> factors);
std::string Describe(const Plan& plan);
std::map<std::string, std::string> RuntimeInfo();
std::string Fingerprint(const Basis& basis);
template <typename T>
std::string Fingerprint(const Coefficients<T>& coefficients);

} // namespace ao2mo
