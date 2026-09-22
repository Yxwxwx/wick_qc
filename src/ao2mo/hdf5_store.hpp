#pragma once

#include <hdf5.h>

#include <array>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "types.hpp"

namespace ao2mo::h5 {
// HDF5 native types describe the C buffer type, independently of the on-disk
// dtype.
// NOLINTNEXTLINE(google-runtime-int): H5T_NATIVE_LLONG ABI.
using NativeIndex = long long;
// ponytail: serialize library executions for non-thread-safe HDF5 builds;
// a dedicated I/O queue is warranted only if concurrent host throughput
// matters.
inline std::recursive_mutex& ExecutionMutex() {
  static std::recursive_mutex mutex;
  return mutex;
}
inline hid_t Check(hid_t id) {
  if (id < 0) {
    throw std::runtime_error("HDF5 operation failed");
  }
  return id;
}
class Handle {
 public:
  Handle() = default;
  Handle(hid_t id, herr_t (*close)(hid_t)) : id_(Check(id)), close_(close) {}
  ~Handle() {
    if (id_ >= 0) {
      close_(id_);
    }
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept
      : id_(std::exchange(other.id_, -1)), close_(other.close_) {}
  Handle& operator=(Handle&& other) noexcept {
    if (this != &other) {
      if (id_ >= 0) {
        close_(id_);
      }
      id_ = std::exchange(other.id_, -1);
      close_ = other.close_;
    }
    return *this;
  }
  // The RAII adapter intentionally interoperates with the HDF5 C handle API.
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator hid_t() const {
    return id_;
  }
  // Explicit success paths check close errors; unwinding remains noexcept.
  // Keep a failed handle so its destructor can retry releasing it.
  void Close() {
    if (id_ >= 0) {
      Check(close_(id_));
      id_ = -1;
    }
  }

 private:
  hid_t id_ = -1;
  herr_t (*close_)(hid_t) = nullptr;
};

inline Handle FileAccess() {
  Handle access(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
  H5AC_cache_config_t config{};
  config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
  Check(H5Pget_mdc_config(access, &config));
  config.set_initial_size = 1;
  config.initial_size = config.max_size = 2ULL << 20;
  config.min_size = 1ULL << 20;
  Check(H5Pset_mdc_config(access, &config));
  return access;
}
inline Handle OpenFile(const std::filesystem::path& path, unsigned flags) {
  return {H5Fopen(path.c_str(), flags, FileAccess()), H5Fclose};
}
inline Handle OpenDataset(hid_t file, const std::string& name) {
  Handle access(H5Pcreate(H5P_DATASET_ACCESS), H5Pclose);
  Check(H5Pset_chunk_cache(access, 1, 0, 0));
  return {H5Dopen2(file, name.c_str(), access), H5Dclose};
}
inline void ScalarAttribute(hid_t attribute) {
  Handle space(H5Aget_space(attribute), H5Sclose);
  if (H5Sget_simple_extent_npoints(space) != 1) {
    throw std::invalid_argument("expected scalar HDF5 attribute");
  }
}

template <typename T>
Handle Type();
template <>
inline Handle Type<double>() {
  return {H5Tcopy(H5T_NATIVE_DOUBLE), H5Tclose};
}
template <>
inline Handle Type<int>() {
  return {H5Tcopy(H5T_NATIVE_INT), H5Tclose};
}
template <>
inline Handle Type<NativeIndex>() {
  return {H5Tcopy(H5T_NATIVE_LLONG), H5Tclose};
}
template <>
inline Handle Type<Complex>() {
  static_assert(sizeof(Complex) == 2 * sizeof(double));
  Handle type(H5Tcreate(H5T_COMPOUND, sizeof(Complex)), H5Tclose);
  Check(H5Tinsert(type, "r", 0, H5T_NATIVE_DOUBLE));
  Check(H5Tinsert(type, "i", sizeof(double), H5T_NATIVE_DOUBLE));
  return type;
}
template <typename T>
Handle FileType();
template <>
inline Handle FileType<double>() {
  return {H5Tcopy(H5T_IEEE_F64LE), H5Tclose};
}
template <>
inline Handle FileType<int>() {
  return {H5Tcopy(H5T_STD_I32LE), H5Tclose};
}
template <>
inline Handle FileType<NativeIndex>() {
  return {H5Tcopy(H5T_STD_I64LE), H5Tclose};
}
template <>
inline Handle FileType<Complex>() {
  Handle type(H5Tcreate(H5T_COMPOUND, 16), H5Tclose);
  Check(H5Tinsert(type, "r", 0, H5T_IEEE_F64LE));
  Check(H5Tinsert(type, "i", 8, H5T_IEEE_F64LE));
  return type;
}

inline std::vector<hsize_t> Shape(hid_t dataset) {
  Handle space(H5Dget_space(dataset), H5Sclose);
  int rank = H5Sget_simple_extent_ndims(space);
  Check(rank);
  std::vector<hsize_t> shape(rank);
  Check(H5Sget_simple_extent_dims(space, shape.data(), nullptr));
  return shape;
}
inline void StringAttribute(
    hid_t object,
    const char* name,
    const std::string& value) {
  Handle space(H5Screate(H5S_SCALAR), H5Sclose);
  Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
  Check(H5Tset_size(type, value.size() + 1));
  Handle attr(
      H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT),
      H5Aclose);
  Check(H5Awrite(attr, type, value.c_str()));
  attr.Close();
}
inline void SizeAttribute(hid_t object, const char* name, std::size_t value) {
  Handle space(H5Screate(H5S_SCALAR), H5Sclose);
  Handle attr(
      H5Acreate2(object, name, H5T_STD_U64LE, space, H5P_DEFAULT, H5P_DEFAULT),
      H5Aclose);
  // Buffer type must match H5T_NATIVE_ULLONG.
  // NOLINTNEXTLINE(google-runtime-int)
  unsigned long long integer = value;
  Check(H5Awrite(attr, H5T_NATIVE_ULLONG, &integer));
  attr.Close();
}
inline void DoubleAttribute(hid_t object, const char* name, double value) {
  Handle space(H5Screate(H5S_SCALAR), H5Sclose);
  Handle attr(
      H5Acreate2(object, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT),
      H5Aclose);
  Check(H5Awrite(attr, H5T_NATIVE_DOUBLE, &value));
  attr.Close();
}
inline void IndexVector(
    hid_t group,
    const char* name,
    std::span<const std::size_t> indices) {
  std::vector<NativeIndex> values;
  values.reserve(indices.size());
  for (auto value : indices) {
    if (value >
        static_cast<std::size_t>(std::numeric_limits<NativeIndex>::max())) {
      throw std::overflow_error("orbital identity exceeds int64 range");
    }
    values.push_back(static_cast<NativeIndex>(value));
  }
  const hsize_t size = values.size();
  Handle space(H5Screate_simple(1, &size, nullptr), H5Sclose);
  Handle dataset(
      H5Dcreate2(
          group,
          name,
          H5T_STD_I64LE,
          space,
          H5P_DEFAULT,
          H5P_DEFAULT,
          H5P_DEFAULT),
      H5Dclose);
  if (size) {
    Check(H5Dwrite(
        dataset,
        H5T_NATIVE_LLONG,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        values.data()));
  }
  dataset.Close();
}
template <typename T>
std::vector<T> Read(
    hid_t file,
    const std::string& name,
    std::vector<hsize_t>* shape = nullptr,
    std::size_t* remaining_bytes = nullptr) {
  auto dataset = OpenDataset(file, name);
  Handle type(H5Dget_type(dataset), H5Tclose);
  bool valid;
  if constexpr (std::is_same_v<T, Complex>) {
    valid = H5Tget_class(type) == H5T_COMPOUND && H5Tget_nmembers(type) == 2;
    for (const char* member : {"r", "i"}) {
      const int index = valid ? H5Tget_member_index(type, member) : -1;
      if (index < 0) {
        valid = false;
        break;
      }
      Handle part(H5Tget_member_type(type, index), H5Tclose);
      valid = H5Tget_class(part) == H5T_FLOAT && H5Tget_size(part) == 8;
    }
  } else {
    valid = H5Tget_size(type) == sizeof(T) &&
        H5Tget_class(type) == (std::is_integral_v<T> ? H5T_INTEGER : H5T_FLOAT);
    if constexpr (std::is_integral_v<T>) {
      valid = valid && H5Tget_sign(type) == H5T_SGN_2;
    }
  }
  if (!valid) {
    throw std::invalid_argument("unexpected dataset dtype: " + name);
  }
  auto dims = Shape(dataset);
  std::size_t size = 1;
  for (auto n : dims) {
    size = CheckedProduct({size, n});
  }
  const auto bytes = CheckedProduct({size, sizeof(T)});
  if (remaining_bytes) {
    if (bytes > *remaining_bytes) {
      throw std::runtime_error("input payload exceeds memory budget");
    }
    *remaining_bytes -= bytes;
  }
  std::vector<T> result(size);
  if (size) {
    Check(H5Dread(
        dataset, Type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data()));
  }
  if (shape) {
    *shape = std::move(dims);
  }
  return result;
}
inline void Complete(hid_t file, int value) {
  Handle space(H5Screate(H5S_SCALAR), H5Sclose);
  Handle attribute(
      H5Aexists(file, "complete") > 0 ? H5Aopen(file, "complete", H5P_DEFAULT)
                                      : H5Acreate2(
                                            file,
                                            "complete",
                                            H5T_NATIVE_INT,
                                            space,
                                            H5P_DEFAULT,
                                            H5P_DEFAULT),
      H5Aclose);
  Check(H5Awrite(attribute, H5T_NATIVE_INT, &value));
  attribute.Close();
}
template <typename T>
Handle Matrix(
    hid_t file,
    const std::string& name,
    std::array<std::size_t, 2> shape,
    const std::array<std::size_t, 4>* four = nullptr,
    bool column_chunks = false) {
  std::vector<hsize_t> dims{shape[0], shape[1]};
  if (four) {
    dims.assign(four->begin(), four->end());
  }
  Handle space(
      H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr),
      H5Sclose);
  Handle props(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
  if (shape[0] && shape[1]) {
    auto chunks = dims;
    std::fill(chunks.begin(), chunks.end(), 1);
    if (dims.size() == 2) {
      // Bounded rectangular chunks serve both row and column access. Half
      // writes favor columns; output writes favor rows. A one-element minor
      // axis fragments the opposite pass and the independent audit gathers.
      chunks[0] = std::min<hsize_t>(dims[0], column_chunks ? 128 : 32);
      chunks[1] = std::min<hsize_t>(dims[1], column_chunks ? 32 : 128);
    } else {
      chunks.back() = std::min<hsize_t>(dims.back(), 4096);
    }
    Check(H5Pset_chunk(props, static_cast<int>(chunks.size()), chunks.data()));
    Check(H5Pset_fill_time(props, H5D_FILL_TIME_NEVER));
  }
  Handle access(H5Pcreate(H5P_DATASET_ACCESS), H5Pclose);
  // Bounded application buffers; no additional raw chunk cache per dataset.
  Check(H5Pset_chunk_cache(access, 1, 0, 0));
  Handle links(H5Pcreate(H5P_LINK_CREATE), H5Pclose);
  Check(H5Pset_create_intermediate_group(links, 1));
  return {
      H5Dcreate2(
          file, name.c_str(), FileType<T>(), space, links, props, access),
      H5Dclose};
}
template <typename T>
void Slab(
    hid_t dataset,
    bool write,
    std::array<std::size_t, 2> start,
    std::array<std::size_t, 2> count,
    T* buffer,
    std::size_t tile_elements = std::numeric_limits<std::size_t>::max()) {
  if (!count[0] || !count[1]) {
    return;
  }
  if (!tile_elements) {
    throw std::invalid_argument("I/O tile must hold at least one element");
  }
  if (CheckedProduct({count[0], count[1]}) > tile_elements) {
    for (std::size_t row = 0; row < count[0]; ++row) {
      for (std::size_t col = 0; col < count[1]; col += tile_elements) {
        const auto size = std::min(tile_elements, count[1] - col);
        Slab(
            dataset,
            write,
            {start[0] + row, start[1] + col},
            {1, size},
            buffer + row * count[1] + col);
      }
    }
    return;
  }
  hsize_t offset[2] = {start[0], start[1]}, dims[2] = {count[0], count[1]};
  Handle space(H5Dget_space(dataset), H5Sclose);
  if (H5Sget_simple_extent_ndims(space) == 2) {
    Check(H5Sselect_hyperslab(
        space, H5S_SELECT_SET, offset, nullptr, dims, nullptr));
  } else {
    const auto shape = Shape(dataset);
    if (shape.size() != 4) {
      throw std::invalid_argument(
          "expected pair matrix or rank-four helper dataset");
    }
    Check(H5Sselect_none(space));
    for (std::size_t row = start[0]; row < start[0] + count[0]; ++row) {
      for (std::size_t col = start[1]; col < start[1] + count[1];) {
        const auto length = std::min<std::size_t>(
            shape[3] - col % shape[3], start[1] + count[1] - col);
        hsize_t origin[4] = {
            row / shape[1], row % shape[1], col / shape[3], col % shape[3]};
        hsize_t extent[4] = {1, 1, 1, length};
        Check(H5Sselect_hyperslab(
            space, H5S_SELECT_OR, origin, nullptr, extent, nullptr));
        col += length;
      }
    }
  }
  Handle memory(H5Screate_simple(2, dims, nullptr), H5Sclose);
  if (write) {
    Check(H5Dwrite(dataset, Type<T>(), memory, space, H5P_DEFAULT, buffer));
  } else {
    Check(H5Dread(dataset, Type<T>(), memory, space, H5P_DEFAULT, buffer));
  }
}
// Write a shell-panel column strip directly from its strided half tile.
// This avoids one HDF5 call per MO pair without a second transpose buffer.
template <typename T>
void HalfColumns(
    hid_t dataset,
    std::size_t column,
    std::size_t rows,
    std::size_t columns,
    const T* values,
    std::size_t row_stride,
    std::size_t column_stride,
    std::size_t tile_elements) {
  if (!rows || !columns) {
    return;
  }
  if (!tile_elements) {
    throw std::invalid_argument("zero half-transfer tile");
  }
  const auto tile_cols = std::min(columns, tile_elements);
  const auto tile_rows = std::max<std::size_t>(1, tile_elements / tile_cols);
  for (std::size_t row = 0; row < rows; row += tile_rows) {
    for (std::size_t col = 0; col < columns; col += tile_cols) {
      hsize_t count[2] = {
          std::min(tile_rows, rows - row), std::min(tile_cols, columns - col)};
      hsize_t start[2] = {row, column + col};
      Handle file(H5Dget_space(dataset), H5Sclose);
      Check(H5Sselect_hyperslab(
          file, H5S_SELECT_SET, start, nullptr, count, nullptr));
      hsize_t memory_dims[2] = {count[0], row_stride};
      hsize_t origin[2] = {0, 0}, stride[2] = {1, column_stride};
      Handle memory(H5Screate_simple(2, memory_dims, nullptr), H5Sclose);
      Check(H5Sselect_hyperslab(
          memory, H5S_SELECT_SET, origin, stride, count, nullptr));
      Check(H5Dwrite(
          dataset,
          Type<T>(),
          memory,
          file,
          H5P_DEFAULT,
          values + row * row_stride + col * column_stride));
    }
  }
}
} // namespace ao2mo::h5
