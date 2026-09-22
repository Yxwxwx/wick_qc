#pragma once

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

#include "hdf5_store.hpp"
#include "memory.hpp"
#include "types.hpp"

namespace ao2mo::detail {
template <typename T>
Plan AuditPlan(
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options);
template <typename T>
Result<T> AuditTransform(
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options,
    std::span<const MemoryBuffer<T>> buffers);
namespace audit_detail {
inline constexpr std::array<std::array<int, 4>, 3> kPartners{
    {{2, 3, 0, 1}, {1, 0, 3, 2}, {3, 2, 1, 0}}};
class ScratchDirectory {
 public:
  explicit ScratchDirectory(
      const std::filesystem::path& parent,
      std::size_t bytes) {
    if (std::filesystem::space(parent).available < bytes) {
      throw std::runtime_error("insufficient audit scratch space");
    }
    auto name = (parent / "ao2mo-audit-XXXXXX").string();
    if (!mkdtemp(name.data())) {
      throw std::runtime_error("cannot create unique audit scratch directory");
    }
    path = name;
  }
  ~ScratchDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
  std::filesystem::path path;
};

template <typename T>
Request<T> Partner(const Request<T>& request, int partner) {
  auto result = request;
  for (int i = 0; i < 4; ++i) {
    result.indices[i] = request.indices[kPartners[partner][i]];
  }
  result.name = "partner";
  result.ordering = Ordering::kChemist;
  result.layout = MoLayout::kDense;
  result.rank_four_output = false;
  return result;
}
inline Options RawOptions(Options options) {
  options.audit = AuditMode::kRaw;
  options.first_pair_only = true;
  return options;
}
template <typename T>
std::size_t BlockBytes(const Request<T>& r) {
  const auto a = r.indices[0].columns.size(), b = r.indices[1].columns.size();
  const auto c = r.indices[2].columns.size(), d = r.indices[3].columns.size();
  if (r.layout == MoLayout::kS4) {
    return CheckedProduct(
               {a, CheckedAdd(a, 1), c, CheckedAdd(c, 1), sizeof(T)}) /
        4;
  }
  return CheckedProduct({a, b, c, d, sizeof(T)});
}
template <typename T>
std::size_t OutsideBytes(
    const Request<T>& r,
    const Plan& raw,
    const Options& options) {
  const bool disk = raw.workspace == Workspace::kOutcore;
  const bool project = options.audit == AuditMode::kProjectRoundoff;
  std::size_t bytes =
      options.output == Output::kMemory ? raw.output_bytes : (2ULL << 20);
  // All original request families remain caller-resident while one block's
  // independent partner runs. Its own plan includes only that block's families.
  std::set<const Coefficients<T>*> families;
  std::size_t current_coefficients = 0;
  for (const auto& selection : r.indices) {
    if (families.insert(selection.coefficients.get()).second) {
      const auto& c = *selection.coefficients;
      current_coefficients =
          CheckedAdd(current_coefficients, memory_detail::NumericBytes(c));
    }
  }
  bytes = CheckedAdd(bytes, raw.coefficient_bytes - current_coefficients);
  bytes = CheckedAdd(bytes, raw.metadata_bytes);
  // Projected memory results need an accumulator; incore disk results need
  // a raw copy, since explicit incore must not create audit scratch files.
  if (project && (options.output == Output::kMemory || !disk)) {
    bytes = CheckedAdd(bytes, BlockBytes(r));
  }
  std::size_t max_dimension = 0;
  for (const auto& s : r.indices) {
    max_dimension = std::max(max_dimension, s.columns.size());
  }
  const auto pairs = std::max(
      CheckedProduct(
          {r.indices[0].columns.size(), r.indices[1].columns.size()}),
      CheckedProduct(
          {r.indices[2].columns.size(), r.indices[3].columns.size()}));
  // A prepared execution shares the target's common arrays with every partner.
  // Charge the excess over the smaller partner plan; no second workspace lives
  // alongside it. The ordinary non-prepared path uses this same safe estimate.
  const auto extra = CheckedAdd(
      CheckedProduct(
          {r.indices[0].coefficients->nao, raw.max_m - max_dimension}),
      CheckedProduct(
          {raw.max_pair - pairs,
           CheckedAdd(2, CheckedProduct({raw.max_shell, raw.max_shell}))}));
  bytes = CheckedAdd(bytes, CheckedProduct({extra, sizeof(T)}));
  const auto tile = std::min(
      CheckedProduct({max_dimension, max_dimension}),
      options.io_tile_bytes / sizeof(T));
  bytes = CheckedAdd(
      bytes, CheckedProduct({tile, 3 * sizeof(T) + 2 * sizeof(hsize_t)}));
  if (disk) {
    bytes = CheckedAdd(bytes, 4ULL << 20); // raw-copy and partner reader caches
  }
  return bytes;
}
template <typename T>
Options PartnerOptions(
    const Request<T>& r,
    const Plan& raw,
    const Options& options) {
  auto partner = RawOptions(options);
  partner.workspace = raw.workspace;
  partner.output =
      raw.workspace == Workspace::kOutcore ? Output::kHdf5 : Output::kMemory;
  const auto outside = OutsideBytes(r, raw, options);
  if (outside >= options.memory_bytes) {
    throw std::runtime_error("audit buffers and resident output exceed budget");
  }
  partner.memory_bytes -= outside;
  // Reserve the same path length used by the unique execution directory.
  partner.output_path =
      options.scratch_directory / "ao2mo-audit-XXXXXX" / "partner.h5";
  return partner;
}

inline std::array<std::size_t, 2> Unpack(std::size_t index) {
  // Integer binary search avoids floating-point packed-index rounding.
  std::size_t lo = 0, hi = index + 1;
  while (lo + 1 < hi) {
    auto mid = lo + (hi - lo) / 2;
    const bool fits = mid % 2 == 0 ? mid / 2 <= index / (mid + 1)
                                   : (mid + 1) / 2 <= index / mid;
    if (fits) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return {lo, index - lo * (lo + 1) / 2};
}
template <typename T>
std::array<std::size_t, 4> Coordinates(
    const Block<T>& block,
    std::size_t row,
    std::size_t col) {
  if (block.layout == MoLayout::kS4) {
    const auto pq = Unpack(row), rs = Unpack(col);
    return {pq[0], pq[1], rs[0], rs[1]};
  }
  std::array<std::size_t, 4> coordinates{
      row / block.shape[1],
      row % block.shape[1],
      col / block.shape[3],
      col % block.shape[3]};
  if (block.ordering == Ordering::kPhysicist) {
    std::swap(coordinates[1], coordinates[2]);
  }
  return coordinates;
}

template <typename T>
void PartnerTile(
    const Block<T>& target,
    const Block<T>& partner,
    hid_t dataset,
    int iperm,
    std::size_t row,
    std::size_t first_column,
    std::size_t count,
    std::vector<T>& values,
    std::vector<hsize_t>& coordinates,
    std::vector<T>& scratch) {
  const bool whole_pair = dataset >= 0 && target.layout == MoLayout::kDense &&
      target.ordering == Ordering::kChemist && first_column == 0 &&
      count == target.pair_shape[1];
  if (whole_pair && count) {
    // A full (r,s) row maps to one partner row/column. Read it as a regular
    // hyperslab, then transpose its pair indices in the existing audit tile.
    const auto original = Coordinates(target, row, 0);
    if (iperm == 0) {
      const auto column = original[0] * partner.shape[3] + original[1];
      h5::Slab(dataset, false, {0, column}, {count, 1}, values.data());
    } else {
      const auto pair = original[1] * target.shape[0] + original[0];
      if (iperm == 1) {
        h5::Slab(dataset, false, {pair, 0}, {1, count}, scratch.data());
      } else {
        h5::Slab(dataset, false, {0, pair}, {count, 1}, scratch.data());
      }
      for (std::size_t r = 0; r < target.shape[2]; ++r) {
        for (std::size_t s = 0; s < target.shape[3]; ++s) {
          values[r * target.shape[3] + s] = scratch[s * target.shape[2] + r];
        }
      }
    }
  } else {
    for (std::size_t col = 0; col < count; ++col) {
      auto original = Coordinates(target, row, first_column + col);
      const auto& p = kPartners[iperm];
      const auto i = original[p[0]] * partner.shape[1] + original[p[1]];
      const auto j = original[p[2]] * partner.shape[3] + original[p[3]];
      if (dataset < 0) {
        values[col] = partner.Values()[i * partner.pair_shape[1] + j];
      } else {
        coordinates[2 * col] = i;
        coordinates[2 * col + 1] = j;
      }
    }
    if (dataset >= 0 && count) {
      h5::Handle space(H5Dget_space(dataset), H5Sclose);
      h5::Check(
          H5Sselect_elements(space, H5S_SELECT_SET, count, coordinates.data()));
      hsize_t dims[1] = {count};
      h5::Handle memory(H5Screate_simple(1, dims, nullptr), H5Sclose);
      h5::Check(H5Dread(
          dataset, h5::Type<T>(), memory, space, H5P_DEFAULT, values.data()));
    }
  }
  if (iperm >= 1) {
    for (std::size_t col = 0; col < count; ++col) {
      values[col] = detail::Conjugate(values[col]);
    }
  }
}

} // namespace audit_detail

template <typename T>
Plan AuditPlan(
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options) {
  try {
    auto raw_options = audit_detail::RawOptions(options);
    auto raw = MakePlan(basis, requests, raw_options);
    auto result = raw;
    for (const auto& request : requests) {
      if (!audit_detail::BlockBytes(request)) {
        continue;
      }
      const auto partner_options =
          audit_detail::PartnerOptions(request, raw, options);
      for (int p = 0; p < 3; ++p) {
        const auto plan = MakePlan(
            basis,
            std::vector<Request<T>>{audit_detail::Partner(request, p)},
            partner_options);
        result.peak_bytes = std::max(
            result.peak_bytes,
            CheckedAdd(
                plan.peak_bytes,
                audit_detail::OutsideBytes(request, raw, options)));
        result.metadata_bytes = std::max(
            result.metadata_bytes,
            CheckedAdd(raw.metadata_bytes, plan.metadata_bytes));
        result.audit_ao_passes += plan.ao_passes;
        if (raw.workspace == Workspace::kOutcore) {
          result.scratch_bytes = std::max(
              result.scratch_bytes,
              CheckedAdd(
                  plan.scratch_bytes,
                  CheckedAdd(
                      plan.output_bytes,
                      options.audit == AuditMode::kProjectRoundoff
                          ? audit_detail::BlockBytes(request)
                          : 0)));
        }
      }
    }
    return result;
  } catch (const std::runtime_error&) {
    if (options.workspace != Workspace::kAuto) {
      throw;
    }
    auto outcore = options;
    outcore.workspace = Workspace::kOutcore;
    return AuditPlan(basis, requests, outcore);
  }
}

template <typename T, typename Execute>
Result<T> AuditExecute(
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options,
    std::span<const MemoryBuffer<T>> buffers,
    const Plan& plan,
    Execute&& execute) {
  auto raw_options = audit_detail::RawOptions(options);
  raw_options.workspace = plan.workspace;
  auto result = execute(requests, raw_options, buffers);
  const auto raw_plan = result.plan;
  result.plan = plan;
  result.statistics.metadata_reserve_bytes = plan.metadata_bytes;
  const auto audit_start = std::chrono::steady_clock::now();
  const bool project = options.audit == AuditMode::kProjectRoundoff;
  const bool disk = plan.workspace == Workspace::kOutcore;
  auto transfer = [&](hid_t dataset,
                      bool write,
                      std::array<hsize_t, 2> offset,
                      std::array<hsize_t, 2> shape,
                      T* data) {
    const auto start = std::chrono::steady_clock::now();
    h5::Slab(dataset, write, offset, shape, data);
    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    result.statistics.audit_io_seconds += elapsed;
    result.statistics.io_seconds += elapsed;
    auto& bytes = write ? result.statistics.audit_write_bytes
                        : result.statistics.audit_read_bytes;
    bytes = CheckedAdd(bytes, CheckedProduct({shape[0], shape[1], sizeof(T)}));
  };
  try {
    h5::Handle output;
    if (options.output == Output::kHdf5) {
      output = h5::OpenFile(options.output_path, H5F_ACC_RDWR);
      h5::Complete(output, 0);
    }
    std::unique_ptr<audit_detail::ScratchDirectory> scratch;
    if (disk) {
      scratch = std::make_unique<audit_detail::ScratchDirectory>(
          options.scratch_directory, plan.scratch_bytes);
    }
    for (std::size_t bi = 0; bi < requests.size(); ++bi) {
      const auto& request = requests[bi];
      auto& block = result.blocks[bi];
      const auto rows = block.pair_shape[0], cols = block.pair_shape[1];
      AuditDiagnostic diagnostic;
      diagnostic.name = block.name;
      if (!rows || !cols) {
        result.audit.push_back(diagnostic);
        continue;
      }
      h5::Handle dataset, raw_file, raw_dataset;
      if (!block.file.empty()) {
        dataset = h5::OpenDataset(output, block.name);
      }
      std::vector<T> raw_copy, accumulator;
      const auto tile = std::min(cols, options.io_tile_bytes / sizeof(T));
      std::vector<T> raw_row(tile), partner_row(tile), sum_row(tile);
      std::vector<hsize_t> coordinates(CheckedProduct({tile, 2}));
      if (project) {
        if (block.file.empty()) {
          accumulator.resize(block.Values().size());
        } else if (!disk) {
          raw_copy.resize(CheckedProduct({rows, cols}));
        } else {
          raw_file = {
              H5Fcreate(
                  (scratch->path / "raw.h5").c_str(),
                  H5F_ACC_EXCL,
                  H5P_DEFAULT,
                  h5::FileAccess()),
              H5Fclose};
          raw_dataset = h5::Matrix<T>(raw_file, "raw", {rows, cols});
        }
      }
      double scale = 0;
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; col += tile) {
          const auto count = std::min(tile, cols - col);
          const auto offset = row * cols + col;
          if (block.file.empty()) {
            std::copy_n(block.Values().data() + offset, count, raw_row.data());
          } else {
            transfer(dataset, false, {row, col}, {1, count}, raw_row.data());
          }
          for (std::size_t j = 0; j < count; ++j) {
            scale = std::max(scale, std::abs(raw_row[j]));
          }
          if (project) {
            if (!raw_copy.empty()) {
              std::copy_n(raw_row.data(), count, raw_copy.data() + offset);
            }
            if (raw_dataset >= 0) {
              transfer(
                  raw_dataset, true, {row, col}, {1, count}, raw_row.data());
            }
            for (std::size_t j = 0; j < count; ++j) {
              sum_row[j] = 0.25 * raw_row[j];
            }
            if (block.file.empty()) {
              std::copy_n(sum_row.data(), count, accumulator.data() + offset);
            } else {
              transfer(dataset, true, {row, col}, {1, count}, sum_row.data());
            }
          }
        }
      }
      // Same fixed operation-count policy as local socutils. Complex oracle
      // uses j-spinor AO dimension=2*nao_sph; this is an empirical gate only.
      const double operations = 4.0 *
          (std::is_same_v<T, Complex> ? 16.0 : 2.0) * basis.AoOffsets().back();
      const auto epsk = operations * std::numeric_limits<double>::epsilon();
      // Negated ordered comparisons also reject NaN.
      // NOLINTNEXTLINE(readability-simplify-boolean-expr)
      if (!(epsk > 0 && epsk < 1)) {
        throw std::runtime_error("invalid roundoff operation count");
      }
      diagnostic.roundoff_gate = 2 * epsk / (1 - epsk) * std::max(1.0, scale);
      for (int iperm = 0; iperm < 3; ++iperm) {
        auto partner_options =
            audit_detail::PartnerOptions(request, raw_plan, options);
        if (disk) {
          partner_options.output_path = scratch->path / "partner.h5";
        }
        auto partner = execute(
            std::vector<Request<T>>{audit_detail::Partner(request, iperm)},
            partner_options,
            std::span<const MemoryBuffer<T>>{});
        auto partner_statistics = partner.statistics;
        if (disk) {
          // These raw subcall outputs are temporary audit partners.
          partner_statistics.temp_write_bytes = CheckedAdd(
              partner_statistics.temp_write_bytes,
              partner_statistics.final_write_bytes);
          partner_statistics.final_write_bytes = 0;
          partner_statistics.temp_io_seconds +=
              partner_statistics.final_io_seconds;
          partner_statistics.final_io_seconds = 0;
        }
        std::size_t outside_capacity = 0;
        for (const auto* buffer :
             {&raw_copy, &accumulator, &raw_row, &partner_row, &sum_row}) {
          outside_capacity = CheckedAdd(
              outside_capacity,
              CheckedProduct({buffer->capacity(), sizeof(T)}));
        }
        outside_capacity = CheckedAdd(
            outside_capacity,
            CheckedProduct({coordinates.capacity(), sizeof(hsize_t)}));
        for (const auto& resident : result.blocks) {
          outside_capacity = CheckedAdd(
              outside_capacity,
              CheckedProduct({resident.values.capacity(), sizeof(T)}));
        }
        partner_statistics.managed_numeric_peak_bytes = CheckedAdd(
            partner_statistics.managed_numeric_peak_bytes, outside_capacity);
        result.statistics.Add(partner_statistics);
        h5::Handle partner_file, partner_dataset;
        if (disk) {
          partner_file =
              h5::OpenFile(partner_options.output_path, H5F_ACC_RDONLY);
          partner_dataset = h5::OpenDataset(partner_file, "partner");
        }
        for (std::size_t row = 0; row < rows; ++row) {
          for (std::size_t col = 0; col < cols; col += tile) {
            const auto count = std::min(tile, cols - col);
            const auto offset = row * cols + col;
            if (block.file.empty()) {
              std::copy_n(
                  block.Values().data() + offset, count, raw_row.data());
            } else if (!raw_copy.empty()) {
              std::copy_n(raw_copy.data() + offset, count, raw_row.data());
            } else {
              transfer(
                  project ? static_cast<hid_t>(raw_dataset)
                          : static_cast<hid_t>(dataset),
                  false,
                  {row, col},
                  {1, count},
                  raw_row.data());
            }
            const auto partner_read_start = std::chrono::steady_clock::now();
            audit_detail::PartnerTile(
                block,
                partner.blocks[0],
                partner_dataset,
                iperm,
                row,
                col,
                count,
                partner_row,
                coordinates,
                sum_row);
            if (disk) {
              const auto elapsed =
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - partner_read_start)
                      .count();
              result.statistics.audit_io_seconds += elapsed;
              result.statistics.io_seconds += elapsed;
              result.statistics.audit_read_bytes = CheckedAdd(
                  result.statistics.audit_read_bytes,
                  CheckedProduct({count, sizeof(T)}));
            }
            for (std::size_t j = 0; j < count; ++j) {
              diagnostic.raw_residuals[iperm] = std::max(
                  diagnostic.raw_residuals[iperm],
                  std::abs(raw_row[j] - partner_row[j]));
            }
            if (project) {
              if (block.file.empty()) {
                std::copy_n(accumulator.data() + offset, count, sum_row.data());
              } else {
                transfer(
                    dataset, false, {row, col}, {1, count}, sum_row.data());
              }
              for (std::size_t j = 0; j < count; ++j) {
                sum_row[j] += 0.25 * partner_row[j];
                if (iperm == 2) {
                  diagnostic.correction = std::max(
                      diagnostic.correction, std::abs(sum_row[j] - raw_row[j]));
                }
              }
              if (block.file.empty()) {
                std::copy_n(sum_row.data(), count, accumulator.data() + offset);
              } else {
                transfer(dataset, true, {row, col}, {1, count}, sum_row.data());
              }
            }
          }
        }
        partner_dataset.Close();
        partner_file.Close();
        if (disk) {
          std::filesystem::remove(partner_options.output_path);
        }
      }
      const auto residual = *std::max_element(
          diagnostic.raw_residuals.begin(), diagnostic.raw_residuals.end());
      if (residual > diagnostic.roundoff_gate) {
        std::ostringstream message;
        message.precision(17);
        message << "independent raw symmetry residual exceeds roundoff gate: "
                << block.name << " residual=" << residual
                << " gate=" << diagnostic.roundoff_gate;
        throw std::runtime_error(message.str());
      }
      if (project && block.file.empty()) {
        if (block.external.values.empty()) {
          block.values.swap(accumulator);
        } else {
          std::copy(
              accumulator.begin(),
              accumulator.end(),
              block.MutableValues().begin());
        }
      }
      result.audit.push_back(diagnostic);
      dataset.Close();
      raw_dataset.Close();
      raw_file.Close();
      if (disk && project && !block.file.empty()) {
        std::filesystem::remove(scratch->path / "raw.h5");
      }
    }
    if (output >= 0) {
      const auto flush_start = std::chrono::steady_clock::now();
      h5::Check(H5Adelete(output, "mode"));
      h5::StringAttribute(
          output, "mode", project ? "project-roundoff" : "audit");
      for (std::size_t i = 0; i < result.audit.size(); ++i) {
        auto marker =
            h5::Matrix<double>(output, "audit/" + std::to_string(i), {0, 0});
        const auto& diagnostic = result.audit[i];
        h5::StringAttribute(marker, "block", diagnostic.name);
        h5::StringAttribute(
            marker,
            "source",
            "three fresh raw parameter-permuted transforms; first pair direction fixed");
        for (int p = 0; p < 3; ++p) {
          h5::DoubleAttribute(
              marker,
              ("raw_residual" + std::to_string(p)).c_str(),
              diagnostic.raw_residuals[p]);
        }
        h5::DoubleAttribute(marker, "roundoff_gate", diagnostic.roundoff_gate);
        h5::DoubleAttribute(
            marker, "maximum_correction", diagnostic.correction);
        marker.Close();
      }
      h5::Complete(output, 1);
      h5::Check(H5Fflush(output, H5F_SCOPE_GLOBAL));
      output.Close();
      const auto elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - flush_start)
                               .count();
      result.statistics.audit_io_seconds += elapsed;
      result.statistics.io_seconds += elapsed;
    }
  } catch (...) {
    if (options.output == Output::kHdf5) {
      std::error_code error;
      std::filesystem::remove(options.output_path, error);
    }
    throw;
  }
  result.statistics.audit_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - audit_start)
          .count();
  return result;
}

template <typename T>
Result<T> AuditTransform(
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options,
    std::span<const MemoryBuffer<T>> buffers) {
  const auto plan = AuditPlan(basis, requests, options);
  auto execute = [&](const auto& current,
                     const Options& raw,
                     std::span<const MemoryBuffer<T>> sink) {
    return Transform(basis, current, raw, sink);
  };
  return AuditExecute(basis, requests, options, buffers, plan, execute);
}
} // namespace ao2mo::detail
