#pragma once

#include "contraction_plan.hpp"

#if defined(WICKQC_USE_TBLIS)
#include "tblis.hpp"
#else
#include "blas.hpp"
#endif

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace wickqc {

namespace detail {

inline std::string Trim(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\n\r");
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = text.find_last_not_of(" \t\n\r");
  return std::string(text.substr(first, last - first + 1));
}

inline std::vector<std::string> Split(std::string_view text, char delimiter) {
  std::vector<std::string> parts;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t end = text.find(delimiter, begin);
    parts.push_back(Trim(text.substr(
        begin,
        end == std::string_view::npos ? text.size() - begin : end - begin)));
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1;
  }
  return parts;
}

inline std::ptrdiff_t ParseSigned(std::string_view text) {
  const std::string value = Trim(text);
  if (value.empty()) {
    throw std::invalid_argument("expected integer, got an empty string");
  }
  std::size_t consumed = 0;
  const auto parsed = std::stoll(value, &consumed);
  if (consumed != value.size()) {
    throw std::invalid_argument("invalid integer: " + value);
  }
  if (parsed < std::numeric_limits<std::ptrdiff_t>::min() ||
      parsed > std::numeric_limits<std::ptrdiff_t>::max()) {
    throw std::out_of_range("integer is outside ptrdiff_t range: " + value);
  }
  return static_cast<std::ptrdiff_t>(parsed);
}

template <typename T>
using RealType = std::conditional_t<
    std::is_same_v<T, std::complex<float>>,
    float,
    std::conditional_t<std::is_same_v<T, std::complex<double>>, double, T>>;

template <typename T>
inline RealType<T> SquaredMagnitude(const T& value) {
  if constexpr (
      std::is_same_v<T, std::complex<float>> ||
      std::is_same_v<T, std::complex<double>>) {
    return std::norm(value);
  } else {
    return value * value;
  }
}

inline std::size_t Product(const std::vector<std::size_t>& values) {
  return std::accumulate(
      values.begin(), values.end(), std::size_t{1}, std::multiplies<>());
}

inline void ValidatePermutation(
    const std::vector<int>& permutation,
    std::size_t rank) {
  if (permutation.size() != rank) {
    throw std::invalid_argument("permutation rank mismatch");
  }
  std::vector<bool> seen(rank, false);
  for (const int axis : permutation) {
    if (axis < 0 || static_cast<std::size_t>(axis) >= rank || seen[axis]) {
      throw std::invalid_argument("invalid permutation");
    }
    seen[axis] = true;
  }
}

} // namespace detail

struct NDArraySlice {
  enum class Kind : std::uint8_t { kAll, kRange, kIndex, kNewAxis };

  Kind kind = Kind::kAll;
  std::optional<std::ptrdiff_t> start;
  std::optional<std::ptrdiff_t> stop;
  std::ptrdiff_t step = 1;

  static NDArraySlice All() {
    return {};
  }

  static NDArraySlice Range(
      std::optional<std::ptrdiff_t> start,
      std::optional<std::ptrdiff_t> stop,
      std::ptrdiff_t step = 1) {
    if (step == 0) {
      throw std::invalid_argument("slice step cannot be zero");
    }
    NDArraySlice slice;
    slice.kind = Kind::kRange;
    slice.start = start;
    slice.stop = stop;
    slice.step = step;
    return slice;
  }

  static NDArraySlice Index(std::ptrdiff_t index) {
    NDArraySlice slice;
    slice.kind = Kind::kIndex;
    slice.start = index;
    slice.stop = index;
    slice.step = 0;
    return slice;
  }

  static NDArraySlice NewAxis() {
    NDArraySlice slice;
    slice.kind = Kind::kNewAxis;
    slice.step = 0;
    return slice;
  }

  [[nodiscard]] bool IsAll() const noexcept {
    return kind == Kind::kAll;
  }
  [[nodiscard]] bool IsRange() const noexcept {
    return kind == Kind::kRange;
  }
  [[nodiscard]] bool IsIndex() const noexcept {
    return kind == Kind::kIndex;
  }
  [[nodiscard]] bool IsNewAxis() const noexcept {
    return kind == Kind::kNewAxis;
  }

  static std::vector<NDArraySlice> Parse(std::string_view expression) {
    std::vector<NDArraySlice> result;
    for (const std::string& raw_token : detail::Split(expression, ',')) {
      const std::string token = detail::Trim(raw_token);
      if (token == "None" || token == "newaxis") {
        result.push_back(NewAxis());
        continue;
      }
      if (token == ":" || token.empty()) {
        result.push_back(All());
        continue;
      }

      const std::vector<std::string> fields = detail::Split(token, ':');
      if (fields.size() == 1) {
        result.push_back(Index(detail::ParseSigned(fields[0])));
        continue;
      }
      if (fields.size() > 3) {
        throw std::invalid_argument("invalid slice: " + token);
      }

      const auto parse_optional =
          [](const std::string& field) -> std::optional<std::ptrdiff_t> {
        if (field.empty()) {
          return std::nullopt;
        }
        return detail::ParseSigned(field);
      };

      const std::optional<std::ptrdiff_t> start = parse_optional(fields[0]);
      const std::optional<std::ptrdiff_t> stop = parse_optional(fields[1]);
      const std::ptrdiff_t step = fields.size() == 3 && !fields[2].empty()
          ? detail::ParseSigned(fields[2])
          : 1;
      result.push_back(Range(start, stop, step));
    }
    return result;
  }
};

inline std::ostream& operator<<(std::ostream& os, const NDArraySlice& slice) {
  if (slice.IsNewAxis()) {
    return os << "None";
  }
  if (slice.IsIndex()) {
    return os << *slice.start;
  }
  if (slice.IsAll()) {
    return os << ':';
  }
  if (slice.start.has_value()) {
    os << *slice.start;
  }
  os << ':';
  if (slice.stop.has_value()) {
    os << *slice.stop;
  }
  if (slice.step != 1) {
    os << ':' << slice.step;
  }
  return os;
}

template <typename T = double>
class NDArray {
 public:
  using ValueType = T;
  using Shape = std::vector<std::size_t>;
  using Strides = std::vector<std::ptrdiff_t>;
  using RealType = detail::RealType<T>;

  NDArray() : NDArray(Shape{}) {}

  explicit NDArray(Shape shape)
      : shape_(std::move(shape)), strides_(ContiguousStrides(shape_)) {
    AllocateForStrides();
  }

  NDArray(Shape shape, Strides strides, T* data)
      : shape_(std::move(shape)), strides_(std::move(strides)), data_(data) {
    ValidateMetadata();
  }

  NDArray(Shape shape, const std::vector<T>& values)
      : NDArray(std::move(shape)) {
    if (values.size() != Size()) {
      throw std::invalid_argument("input data size does not match array shape");
    }
    if (!values.empty()) {
      std::copy(values.begin(), values.end(), data_);
    }
  }

  static NDArray WithStrides(Shape shape, Strides strides) {
    NDArray result;
    result.shape_ = std::move(shape);
    result.strides_ = std::move(strides);
    result.ValidateMetadata();
    result.AllocateForStrides();
    return result;
  }

  static NDArray Zeros(const Shape& shape) {
    return NDArray(shape);
  }

  static NDArray Full(const Shape& shape, const T& value) {
    NDArray result(shape);
    if (result.Size() != 0) {
      std::fill(result.data_, result.data_ + result.Size(), value);
    }
    return result;
  }

  static NDArray Ones(const Shape& shape) {
    return Full(shape, T{1});
  }

  static NDArray Random(const Shape& shape, std::uint64_t seed = 5489U) {
    NDArray result(shape);
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (std::size_t i = 0; i < result.Size(); ++i) {
      if constexpr (std::is_same_v<T, std::complex<double>>) {
        result.data_[i] = T{distribution(generator), distribution(generator)};
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        result.data_[i] =
            T{static_cast<float>(distribution(generator)),
              static_cast<float>(distribution(generator))};
      } else {
        result.data_[i] = static_cast<T>(distribution(generator));
      }
    }
    return result;
  }

  [[nodiscard]] int Rank() const noexcept {
    return static_cast<int>(shape_.size());
  }

  [[nodiscard]] std::size_t Size() const noexcept {
    if (shape_.empty()) {
      return 1;
    }
    return detail::Product(shape_);
  }

  [[nodiscard]] const Shape& shape() const noexcept {
    return shape_;
  }
  [[nodiscard]] const Strides& strides() const noexcept {
    return strides_;
  }
  [[nodiscard]] T* data() noexcept {
    return data_;
  }
  [[nodiscard]] const T* data() const noexcept {
    return data_;
  }

  [[nodiscard]] bool OwnsData() const noexcept {
    return static_cast<bool>(storage_);
  }

  [[nodiscard]] std::size_t MaxStorageSpan() const noexcept {
    if (Size() == 0) {
      return 0;
    }
    const auto [min_offset, max_offset] = StorageBounds();
    return static_cast<std::size_t>(max_offset - min_offset + 1);
  }

  [[nodiscard]] std::vector<std::size_t> DecomposeLinearIndex(
      std::size_t linear_index) const {
    if (linear_index >= Size()) {
      throw std::out_of_range("linear index out of range");
    }
    std::vector<std::size_t> indices(shape_.size(), 0);
    for (int axis = Rank() - 1; axis >= 0; --axis) {
      const std::size_t extent = shape_[axis];
      if (extent == 0) {
        return indices;
      }
      indices[axis] = linear_index % extent;
      linear_index /= extent;
    }
    return indices;
  }

  [[nodiscard]] std::ptrdiff_t LinearOffset(std::size_t linear_index) const {
    if (linear_index >= Size()) {
      throw std::out_of_range("linear index out of range");
    }
    std::ptrdiff_t offset = 0;
    for (int axis = Rank() - 1; axis >= 0; --axis) {
      const std::size_t extent = shape_[axis];
      if (extent == 0) {
        return 0;
      }
      offset +=
          strides_[axis] * static_cast<std::ptrdiff_t>(linear_index % extent);
      linear_index /= extent;
    }
    return offset;
  }

  [[nodiscard]] std::ptrdiff_t Offset(
      const std::vector<std::size_t>& indices) const {
    if (indices.size() != shape_.size()) {
      throw std::invalid_argument("index rank mismatch");
    }
    std::ptrdiff_t offset = 0;
    for (std::size_t axis = 0; axis < shape_.size(); ++axis) {
      if (indices[axis] >= shape_[axis]) {
        throw std::out_of_range("array index out of range");
      }
      offset += strides_[axis] * static_cast<std::ptrdiff_t>(indices[axis]);
    }
    return offset;
  }

  [[nodiscard]] const T& operator[](
      const std::vector<std::size_t>& indices) const {
    return data_[Offset(indices)];
  }

  [[nodiscard]] T& operator[](const std::vector<std::size_t>& indices) {
    return data_[Offset(indices)];
  }

  [[nodiscard]] const T& At(std::initializer_list<std::size_t> indices) const {
    return (*this)[std::vector<std::size_t>(indices)];
  }

  [[nodiscard]] T& At(std::initializer_list<std::size_t> indices) {
    return (*this)[std::vector<std::size_t>(indices)];
  }

  [[nodiscard]] bool IsCOrder() const noexcept {
    std::ptrdiff_t expected = 1;
    for (int axis = Rank() - 1; axis >= 0; --axis) {
      if (shape_[axis] > 1 && strides_[axis] != 0 &&
          strides_[axis] != expected) {
        return false;
      }
      expected *= static_cast<std::ptrdiff_t>(shape_[axis]);
    }
    return true;
  }

  [[nodiscard]] bool IsContiguous() const noexcept {
    std::ptrdiff_t expected = 1;
    for (int axis = Rank() - 1; axis >= 0; --axis) {
      if (shape_[axis] > 1 && strides_[axis] != expected) {
        return false;
      }
      expected *= static_cast<std::ptrdiff_t>(shape_[axis]);
    }
    return true;
  }

  [[nodiscard]] NDArray ReorderC(std::vector<int>& permutation) const {
    permutation.resize(shape_.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::stable_sort(
        permutation.begin(), permutation.end(), [this](int lhs, int rhs) {
          const auto lhs_stride =
              std::abs(static_cast<std::intmax_t>(strides_[lhs]));
          const auto rhs_stride =
              std::abs(static_cast<std::intmax_t>(strides_[rhs]));
          return lhs_stride > rhs_stride;
        });
    return TransposeView(permutation);
  }

  [[nodiscard]] NDArray Slice(
      const std::vector<NDArraySlice>& requested_slices) const {
    std::vector<NDArraySlice> slices = requested_slices;
    std::size_t consumed_axes = 0;
    for (const NDArraySlice& slice : slices) {
      if (!slice.IsNewAxis()) {
        ++consumed_axes;
      }
    }
    if (consumed_axes > shape_.size()) {
      throw std::invalid_argument("too many indices for array");
    }
    while (consumed_axes < shape_.size()) {
      slices.push_back(NDArraySlice::All());
      ++consumed_axes;
    }

    Shape new_shape;
    Strides new_strides;
    std::ptrdiff_t offset = 0;
    std::size_t source_axis = 0;

    for (const NDArraySlice& slice : slices) {
      if (slice.IsNewAxis()) {
        new_shape.push_back(1);
        new_strides.push_back(0);
        continue;
      }

      if (source_axis >= shape_.size()) {
        throw std::invalid_argument("too many indices for array");
      }
      const std::ptrdiff_t extent =
          static_cast<std::ptrdiff_t>(shape_[source_axis]);
      const std::ptrdiff_t stride = strides_[source_axis];

      if (slice.IsIndex()) {
        std::ptrdiff_t index = *slice.start;
        if (index < 0) {
          index += extent;
        }
        if (index < 0 || index >= extent) {
          throw std::out_of_range("slice index out of range");
        }
        offset += index * stride;
        ++source_axis;
        continue;
      }

      const auto normalized = NormalizeSlice(slice, extent);
      if (normalized.length != 0) {
        offset += normalized.start * stride;
      }
      new_shape.push_back(normalized.length);
      new_strides.push_back(stride * normalized.step);
      ++source_axis;
    }

    NDArray result = View(new_shape, new_strides, data_ + offset, storage_);
    return result;
  }

  [[nodiscard]] NDArray Slice(std::string_view expression) const {
    return Slice(NDArraySlice::Parse(expression));
  }

  // Collapse axes that carry the same group id into a diagonal view.
  [[nodiscard]] NDArray DiagView(const std::vector<int>& groups) const {
    if (groups.size() != shape_.size()) {
      throw std::invalid_argument("diagonal group rank mismatch");
    }

    std::unordered_map<int, std::size_t> group_to_axis;
    Shape new_shape;
    Strides new_strides;
    for (std::size_t axis = 0; axis < groups.size(); ++axis) {
      const int group = groups[axis];
      const auto [iterator, inserted] =
          group_to_axis.emplace(group, new_shape.size());
      if (inserted) {
        new_shape.push_back(shape_[axis]);
        new_strides.push_back(strides_[axis]);
      } else {
        const std::size_t target_axis = iterator->second;
        if (new_shape[target_axis] != shape_[axis]) {
          throw std::invalid_argument(
              "repeated einsum index has incompatible dimensions");
        }
        new_strides[target_axis] += strides_[axis];
      }
    }
    return View(new_shape, new_strides, data_, storage_);
  }

  [[nodiscard]] NDArray SumRight(int first_reduced_axis) const {
    if (first_reduced_axis < 0 || first_reduced_axis > Rank()) {
      throw std::out_of_range("sum axis out of range");
    }

    Shape result_shape(shape_.begin(), shape_.begin() + first_reduced_axis);
    NDArray result(result_shape);
    const std::size_t left_size = result.Size();

    Shape reduced_shape(shape_.begin() + first_reduced_axis, shape_.end());
    const std::size_t reduced_size = detail::Product(reduced_shape);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (left_size > 256)
#endif
    for (std::ptrdiff_t left = 0; left < static_cast<std::ptrdiff_t>(left_size);
         ++left) {
      std::vector<std::size_t> left_indices =
          result.DecomposeLinearIndex(static_cast<std::size_t>(left));
      std::ptrdiff_t base_offset = 0;
      for (int axis = 0; axis < first_reduced_axis; ++axis) {
        base_offset +=
            strides_[axis] * static_cast<std::ptrdiff_t>(left_indices[axis]);
      }

      T sum{};
      for (std::size_t reduced = 0; reduced < reduced_size; ++reduced) {
        std::size_t remainder = reduced;
        std::ptrdiff_t offset = base_offset;
        for (int axis = Rank() - 1; axis >= first_reduced_axis; --axis) {
          const std::size_t coordinate = remainder % shape_[axis];
          remainder /= shape_[axis];
          offset += strides_[axis] * static_cast<std::ptrdiff_t>(coordinate);
        }
        sum += data_[offset];
      }
      result.data_[left] = sum;
    }
    return result;
  }

  [[nodiscard]] NDArray BroadcastTo(
      int new_rank,
      const std::vector<int>& original_axes = {}) const {
    if (new_rank < Rank()) {
      throw std::invalid_argument("cannot broadcast to a lower rank");
    }

    Shape new_shape(static_cast<std::size_t>(new_rank), 1);
    Strides new_strides(static_cast<std::size_t>(new_rank), 0);

    if (original_axes.empty()) {
      const int offset = new_rank - Rank();
      for (int axis = 0; axis < Rank(); ++axis) {
        new_shape[offset + axis] = shape_[axis];
        new_strides[offset + axis] = strides_[axis];
      }
    } else {
      if (original_axes.size() != shape_.size()) {
        throw std::invalid_argument("broadcast axis rank mismatch");
      }
      std::vector<bool> seen(static_cast<std::size_t>(new_rank), false);
      for (int axis = 0; axis < Rank(); ++axis) {
        const int target = original_axes[axis];
        if (target < 0 || target >= new_rank || seen[target]) {
          throw std::invalid_argument("invalid broadcast axes");
        }
        seen[target] = true;
        new_shape[target] = shape_[axis];
        new_strides[target] = strides_[axis];
      }
    }

    return View(new_shape, new_strides, data_, storage_);
  }

  [[nodiscard]] T Item() const {
    if (Size() != 1) {
      throw std::logic_error("Item() requires an array with one element");
    }
    return data_[LinearOffset(0)];
  }

  [[nodiscard]] RealType Norm() const {
    RealType sum{};
    for (std::size_t i = 0; i < Size(); ++i) {
      sum += detail::SquaredMagnitude(data_[LinearOffset(i)]);
    }
    using std::sqrt;
    return sqrt(sum);
  }

  [[nodiscard]] NDArray Clone() const {
    NDArray result(shape_);
    Copy(*this, result);
    return result;
  }

  [[nodiscard]] NDArray ToCOrder() const {
    if (IsContiguous()) {
      return *this;
    }
    return Clone();
  }

  [[nodiscard]] NDArray operator*(const T& scalar) const {
    NDArray result(shape_);
    for (std::size_t i = 0; i < Size(); ++i) {
      result.data_[i] = data_[LinearOffset(i)] * scalar;
    }
    return result;
  }

  [[nodiscard]] NDArray operator-() const {
    NDArray result(shape_);
    for (std::size_t i = 0; i < Size(); ++i) {
      result.data_[i] = -data_[LinearOffset(i)];
    }
    return result;
  }

  NDArray& operator+=(const NDArray& other) {
    RequireSameShape(other);
    for (std::size_t i = 0; i < Size(); ++i) {
      data_[LinearOffset(i)] += other.data_[other.LinearOffset(i)];
    }
    return *this;
  }

  NDArray& operator-=(const NDArray& other) {
    RequireSameShape(other);
    for (std::size_t i = 0; i < Size(); ++i) {
      data_[LinearOffset(i)] -= other.data_[other.LinearOffset(i)];
    }
    return *this;
  }

  [[nodiscard]] NDArray operator+(const NDArray& other) const {
    return BinaryElementwise(
        other, [](const T& lhs, const T& rhs) { return lhs + rhs; });
  }

  [[nodiscard]] NDArray operator-(const NDArray& other) const {
    return BinaryElementwise(
        other, [](const T& lhs, const T& rhs) { return lhs - rhs; });
  }

  // No-copy permutation view.
  [[nodiscard]] NDArray TransposeView(
      const std::vector<int>& permutation) const {
    detail::ValidatePermutation(permutation, shape_.size());
    Shape new_shape(shape_.size());
    Strides new_strides(strides_.size());
    for (std::size_t axis = 0; axis < permutation.size(); ++axis) {
      new_shape[axis] = shape_[permutation[axis]];
      new_strides[axis] = strides_[permutation[axis]];
    }
    return View(new_shape, new_strides, data_, storage_);
  }

  static void Copy(
      const NDArray& source,
      NDArray& destination,
      const std::vector<int>& permutation = {},
      T alpha = T{1},
      T beta = T{}) {
    std::vector<int> perm = permutation;
    if (perm.empty()) {
      perm.resize(source.shape_.size());
      std::iota(perm.begin(), perm.end(), 0);
    }
    detail::ValidatePermutation(perm, source.shape_.size());

    Shape expected_shape(perm.size());
    for (std::size_t axis = 0; axis < perm.size(); ++axis) {
      expected_shape[axis] = source.shape_[perm[axis]];
    }
    if (destination.shape_ != expected_shape) {
      throw std::invalid_argument("transpose/copy output shape mismatch");
    }

    for (std::size_t linear = 0; linear < destination.Size(); ++linear) {
      const std::vector<std::size_t> output_index =
          destination.DecomposeLinearIndex(linear);
      std::vector<std::size_t> input_index(source.shape_.size());
      for (std::size_t axis = 0; axis < perm.size(); ++axis) {
        input_index[perm[axis]] = output_index[axis];
      }
      const std::ptrdiff_t source_offset = source.Offset(input_index);
      const std::ptrdiff_t destination_offset =
          destination.Offset(output_index);
      destination.data_[destination_offset] =
          alpha * source.data_[source_offset] +
          beta * destination.data_[destination_offset];
    }
  }

  [[nodiscard]] static NDArray Tensordot(
      const NDArray& lhs,
      const NDArray& rhs,
      const std::vector<int>& lhs_contract_axes,
      const std::vector<int>& rhs_contract_axes,
      const std::vector<int>& lhs_batch_axes = {},
      const std::vector<int>& rhs_batch_axes = {},
      T alpha = T{1}) {
    NDArray result;
    TensordotInto(
        lhs,
        rhs,
        result,
        lhs_contract_axes,
        rhs_contract_axes,
        lhs_batch_axes,
        rhs_batch_axes,
        alpha,
        T{});
    return result;
  }

  static void TensordotInto(
      const NDArray& lhs,
      const NDArray& rhs,
      NDArray& output,
      const std::vector<int>& lhs_contract_axes,
      const std::vector<int>& rhs_contract_axes,
      const std::vector<int>& lhs_batch_axes = {},
      const std::vector<int>& rhs_batch_axes = {},
      T alpha = T{1},
      T beta = T{}) {
    const auto plan = backend::PlanContraction(
        lhs.shape_,
        rhs.shape_,
        lhs_contract_axes,
        rhs_contract_axes,
        lhs_batch_axes,
        rhs_batch_axes);
    if (output.shape_ != plan.output_shape && beta != T{}) {
      throw std::invalid_argument(
          "cannot apply beta to a mismatched tensordot output");
    }
    if (output.shape_ == plan.output_shape && !output.IsContiguous()) {
      throw std::invalid_argument("tensordot output must be C contiguous");
    }

    // Preserve inputs when the caller accumulates into an overlapping view.
    // Ordinary einsum outputs are disjoint, so this requires no data copies.
    std::optional<NDArray> lhs_snapshot;
    std::optional<NDArray> rhs_snapshot;
    if (alpha != T{} && plan.k != 0) {
      if (lhs.Overlaps(output)) {
        lhs_snapshot.emplace(lhs.Clone());
      }
      if (rhs.Overlaps(output)) {
        rhs_snapshot.emplace(rhs.Clone());
      }
    }
    const NDArray& a = lhs_snapshot ? *lhs_snapshot : lhs;
    const NDArray& b = rhs_snapshot ? *rhs_snapshot : rhs;
    if (output.shape_ != plan.output_shape) {
      output = NDArray(plan.output_shape);
    }
    if (output.Size() == 0) {
      return;
    }
    if (plan.k == 0 || alpha == T{}) {
      for (std::size_t i = 0; i < output.Size(); ++i) {
        output.data_[i] = beta == T{} ? T{} : beta * output.data_[i];
      }
      return;
    }

#if defined(WICKQC_USE_TBLIS)
    backend::TblisContract(
        a.data_,
        backend::TBLISMetadata(a.shape_, a.strides_, plan.lhs_indices),
        b.data_,
        backend::TBLISMetadata(b.shape_, b.strides_, plan.rhs_indices),
        output.data_,
        backend::TBLISMetadata(
            output.shape_, output.strides_, plan.output_indices),
        alpha,
        beta);
#else
    const NDArray lhs_matrix = a.TransposeView(plan.lhs_permutation).ToCOrder();
    const NDArray rhs_matrix = b.TransposeView(plan.rhs_permutation).ToCOrder();
    const std::size_t lhs_batch_stride = plan.m * plan.k;
    const std::size_t rhs_batch_stride = plan.n * plan.k;
    const std::size_t output_batch_stride = plan.m * plan.n;
    std::vector<T> temporary;
    if (beta != T{}) {
      temporary.resize(output_batch_stride);
    }
    for (std::size_t batch = 0; batch < plan.batches; ++batch) {
      const T* lhs_ptr = lhs_matrix.data_ + batch * lhs_batch_stride;
      const T* rhs_ptr = rhs_matrix.data_ + batch * rhs_batch_stride;
      T* output_ptr = output.data_ + batch * output_batch_stride;
      if (beta == T{}) {
        blas::Gemm<T>(
            plan.m,
            plan.n,
            plan.k,
            plan.n,
            alpha,
            lhs_ptr,
            rhs_ptr,
            output_ptr);
      } else {
        blas::Gemm<T>(
            plan.m,
            plan.n,
            plan.k,
            plan.n,
            alpha,
            lhs_ptr,
            rhs_ptr,
            temporary.data());
        for (std::size_t i = 0; i < output_batch_stride; ++i) {
          output_ptr[i] = temporary[i] + beta * output_ptr[i];
        }
      }
    }
#endif
  }

  [[nodiscard]] static NDArray Einsum(
      std::string_view expression,
      const std::vector<NDArray>& arrays) {
    const auto parsed = ParseEinsum(expression, arrays);
    return Einsum(parsed.input_labels, parsed.output_labels, arrays);
  }

  // Integer labels connect the equation IR without reparsing text or limiting
  // contractions to a character alphabet. Extents match exactly (no implicit
  // size-one broadcasting of repeated labels).
  [[nodiscard]] static NDArray Einsum(
      const std::vector<std::vector<int>>& input_labels,
      const std::vector<int>& result_labels,
      const std::vector<NDArray>& arrays) {
    if (arrays.empty() || arrays.size() != input_labels.size()) {
      throw std::invalid_argument("einsum operand count does not match labels");
    }
    std::unordered_map<int, std::size_t> extents;
    for (std::size_t operand = 0; operand < arrays.size(); ++operand) {
      if (input_labels[operand].size() != arrays[operand].shape().size()) {
        throw std::invalid_argument("einsum label rank does not match operand");
      }
      for (std::size_t axis = 0; axis < input_labels[operand].size(); ++axis) {
        const int label = input_labels[operand][axis];
        const auto [it, inserted] =
            extents.emplace(label, arrays[operand].shape()[axis]);
        if (!inserted && it->second != arrays[operand].shape()[axis]) {
          throw std::invalid_argument(
              "einsum dimension mismatch for label " + std::to_string(label));
        }
      }
    }
    std::unordered_set<int> seen;
    for (const int label : result_labels) {
      if (!seen.insert(label).second || !extents.contains(label)) {
        throw std::invalid_argument(
            "einsum output label is repeated or absent from inputs");
      }
    }
    std::vector<NDArray> operands = arrays;
    std::vector<std::vector<int>> labels = input_labels;

    // First collapse repeated labels within each operand into diagonal views.
    for (std::size_t operand = 0; operand < operands.size(); ++operand) {
      std::unordered_map<int, int> label_to_group;
      std::vector<int> groups(labels[operand].size());
      std::vector<int> unique_labels;
      for (std::size_t axis = 0; axis < labels[operand].size(); ++axis) {
        const int label = labels[operand][axis];
        auto [iterator, inserted] =
            label_to_group.emplace(label, label_to_group.size());
        groups[axis] = iterator->second;
        if (inserted) {
          unique_labels.push_back(label);
        }
      }
      if (unique_labels.size() != labels[operand].size()) {
        operands[operand] = operands[operand].DiagView(groups);
        labels[operand] = std::move(unique_labels);
      }
    }

    // Sum labels that appear in only one operand and are absent from output.
    std::unordered_set<int> output_labels(
        result_labels.begin(), result_labels.end());
    std::unordered_map<int, int> operand_frequency;
    for (const auto& operand_labels : labels) {
      for (const int label : operand_labels) {
        ++operand_frequency[label];
      }
    }

    for (std::size_t operand = 0; operand < operands.size(); ++operand) {
      std::vector<int> retained_axes;
      std::vector<int> reduced_axes;
      std::vector<int> retained_labels;
      for (std::size_t axis = 0; axis < labels[operand].size(); ++axis) {
        const int label = labels[operand][axis];
        if (operand_frequency[label] == 1 && !output_labels.contains(label)) {
          reduced_axes.push_back(static_cast<int>(axis));
        } else {
          retained_axes.push_back(static_cast<int>(axis));
          retained_labels.push_back(label);
        }
      }
      if (!reduced_axes.empty()) {
        std::vector<int> permutation = retained_axes;
        permutation.insert(
            permutation.end(), reduced_axes.begin(), reduced_axes.end());
        NDArray reordered = operands[operand].TransposeView(permutation);
        operands[operand] = reordered.SumRight(retained_axes.size());
        labels[operand] = std::move(retained_labels);
      }
    }

    NDArray current = operands[0];
    std::vector<int> current_labels = labels[0];

    for (std::size_t operand = 1; operand < operands.size(); ++operand) {
      const std::vector<int>& next_labels = labels[operand];
      std::unordered_map<int, int> current_axis;
      std::unordered_map<int, int> next_axis;
      for (std::size_t axis = 0; axis < current_labels.size(); ++axis) {
        current_axis[current_labels[axis]] = static_cast<int>(axis);
      }
      for (std::size_t axis = 0; axis < next_labels.size(); ++axis) {
        next_axis[next_labels[axis]] = static_cast<int>(axis);
      }

      std::unordered_set<int> future_labels;
      for (std::size_t future = operand + 1; future < labels.size(); ++future) {
        future_labels.insert(labels[future].begin(), labels[future].end());
      }

      std::vector<int> lhs_contract;
      std::vector<int> rhs_contract;
      std::vector<int> lhs_batch;
      std::vector<int> rhs_batch;
      std::vector<int> batch_labels;
      std::unordered_set<int> common_labels;

      for (const int label : current_labels) {
        const auto next = next_axis.find(label);
        if (next == next_axis.end()) {
          continue;
        }
        common_labels.insert(label);
        if (output_labels.contains(label) || future_labels.contains(label)) {
          lhs_batch.push_back(current_axis.at(label));
          rhs_batch.push_back(next->second);
          batch_labels.push_back(label);
        } else {
          lhs_contract.push_back(current_axis.at(label));
          rhs_contract.push_back(next->second);
        }
      }

      std::vector<int> new_labels = batch_labels;
      for (const int label : current_labels) {
        if (!common_labels.contains(label)) {
          new_labels.push_back(label);
        }
      }
      for (const int label : next_labels) {
        if (!common_labels.contains(label)) {
          new_labels.push_back(label);
        }
      }

      current = Tensordot(
          current,
          operands[operand],
          lhs_contract,
          rhs_contract,
          lhs_batch,
          rhs_batch);
      current_labels = std::move(new_labels);
    }

    if (current_labels.size() != result_labels.size()) {
      throw std::runtime_error("einsum output rank mismatch");
    }

    if (current_labels != result_labels) {
      std::unordered_map<int, int> current_axis;
      for (std::size_t axis = 0; axis < current_labels.size(); ++axis) {
        current_axis[current_labels[axis]] = static_cast<int>(axis);
      }
      std::vector<int> permutation(result_labels.size());
      for (std::size_t axis = 0; axis < result_labels.size(); ++axis) {
        const auto iterator = current_axis.find(result_labels[axis]);
        if (iterator == current_axis.end()) {
          throw std::runtime_error("einsum output label disappeared");
        }
        permutation[axis] = iterator->second;
      }
      current = current.TransposeView(permutation);
    }
    return current;
  }

  [[nodiscard]] bool AllClose(
      const NDArray& other,
      RealType relative_tolerance,
      RealType absolute_tolerance) const {
    if (shape_ != other.shape_) {
      return false;
    }
    using std::abs;
    for (std::size_t i = 0; i < Size(); ++i) {
      const T lhs = data_[LinearOffset(i)];
      const T rhs = other.data_[other.LinearOffset(i)];
      const RealType difference = abs(lhs - rhs);
      const RealType scale = std::max(abs(lhs), abs(rhs));
      if (!(difference <= absolute_tolerance + relative_tolerance * scale)) {
        return false;
      }
    }
    return true;
  }

 private:
  struct NormalizedSlice {
    std::ptrdiff_t start = 0;
    std::ptrdiff_t step = 1;
    std::size_t length = 0;
  };

  struct ParsedEinsum {
    std::vector<std::vector<int>> input_labels;
    std::vector<int> output_labels;
  };

  std::shared_ptr<std::vector<T>> storage_;
  Shape shape_;
  Strides strides_;
  T* data_ = nullptr;

  static NDArray View(
      Shape shape,
      Strides strides,
      T* data,
      const std::shared_ptr<std::vector<T>>& owner) {
    NDArray result;
    result.shape_ = std::move(shape);
    result.strides_ = std::move(strides);
    result.data_ = data;
    result.storage_ = owner;
    result.ValidateMetadata();
    return result;
  }

  void ValidateMetadata() const {
    if (shape_.size() != strides_.size()) {
      throw std::invalid_argument("shape/stride rank mismatch");
    }
  }

  static Strides ContiguousStrides(const Shape& shape) {
    Strides strides(shape.size(), 1);
    std::ptrdiff_t current = 1;
    for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis) {
      strides[axis] = current;
      current *= static_cast<std::ptrdiff_t>(shape[axis]);
    }
    return strides;
  }

  [[nodiscard]] std::pair<std::ptrdiff_t, std::ptrdiff_t> StorageBounds()
      const noexcept {
    std::ptrdiff_t min_offset = 0;
    std::ptrdiff_t max_offset = 0;
    for (std::size_t axis = 0; axis < shape_.size(); ++axis) {
      if (shape_[axis] == 0) {
        return {0, -1};
      }
      const std::ptrdiff_t delta =
          strides_[axis] * static_cast<std::ptrdiff_t>(shape_[axis] - 1);
      min_offset += std::min<std::ptrdiff_t>(0, delta);
      max_offset += std::max<std::ptrdiff_t>(0, delta);
    }
    return {min_offset, max_offset};
  }

  [[nodiscard]] bool Overlaps(const NDArray& other) const noexcept {
    if (Size() == 0 || other.Size() == 0) {
      return false;
    }
    const auto [begin, end] = StorageBounds();
    const auto [other_begin, other_end] = other.StorageBounds();
    const std::less<const T*> less;
    return less(data_ + begin, other.data_ + other_end + 1) &&
        less(other.data_ + other_begin, data_ + end + 1);
  }

  void AllocateForStrides() {
    if (Size() == 0) {
      storage_ = std::make_shared<std::vector<T>>();
      data_ = nullptr;
      return;
    }
    const auto [min_offset, max_offset] = StorageBounds();
    const std::size_t storage_size =
        static_cast<std::size_t>(max_offset - min_offset + 1);
    storage_ = std::make_shared<std::vector<T>>(storage_size);
    data_ = storage_->data() - min_offset;
  }

  static NormalizedSlice NormalizeSlice(
      const NDArraySlice& slice,
      std::ptrdiff_t extent) {
    if (slice.IsAll()) {
      return {0, 1, static_cast<std::size_t>(extent)};
    }
    if (!slice.IsRange()) {
      throw std::logic_error("NormalizeSlice called on a non-range slice");
    }

    const std::ptrdiff_t step = slice.step;
    if (step == 0) {
      throw std::invalid_argument("slice step cannot be zero");
    }

    if (step > 0) {
      std::ptrdiff_t start = slice.start.value_or(0);
      std::ptrdiff_t stop = slice.stop.value_or(extent);
      if (slice.start.has_value() && start < 0) {
        start += extent;
      }
      if (slice.stop.has_value() && stop < 0) {
        stop += extent;
      }
      start = std::clamp(start, std::ptrdiff_t{0}, extent);
      stop = std::clamp(stop, std::ptrdiff_t{0}, extent);
      const std::size_t length = start < stop
          ? static_cast<std::size_t>((stop - start - 1) / step + 1)
          : 0;
      return {start, step, length};
    }

    std::ptrdiff_t start = slice.start.value_or(extent - 1);
    std::ptrdiff_t stop = slice.stop.value_or(-1);
    if (slice.start.has_value() && start < 0) {
      start += extent;
    }
    if (slice.stop.has_value() && stop < 0) {
      stop += extent;
    }
    start = std::clamp(start, std::ptrdiff_t{-1}, extent - 1);
    stop = std::clamp(stop, std::ptrdiff_t{-1}, extent - 1);
    const std::ptrdiff_t positive_step = -step;
    const std::size_t length = start > stop
        ? static_cast<std::size_t>((start - stop - 1) / positive_step + 1)
        : 0;
    return {start, step, length};
  }

  void RequireSameShape(const NDArray& other) const {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("array shapes do not match");
    }
  }

  template <typename BinaryOperation>
  [[nodiscard]] NDArray BinaryElementwise(
      const NDArray& other,
      BinaryOperation operation) const {
    if (Rank() != other.Rank()) {
      throw std::invalid_argument("broadcast rank mismatch");
    }

    Shape result_shape = shape_;
    for (std::size_t axis = 0; axis < shape_.size(); ++axis) {
      if (shape_[axis] == other.shape_[axis]) {
        continue;
      }
      if (shape_[axis] == 1 || strides_[axis] == 0) {
        result_shape[axis] = other.shape_[axis];
      } else if (other.shape_[axis] != 1 && other.strides_[axis] != 0) {
        throw std::invalid_argument("incompatible broadcast dimensions");
      }
    }

    NDArray result(result_shape);
    for (std::size_t linear = 0; linear < result.Size(); ++linear) {
      const std::vector<std::size_t> index =
          result.DecomposeLinearIndex(linear);
      std::ptrdiff_t lhs_offset = 0;
      std::ptrdiff_t rhs_offset = 0;
      for (std::size_t axis = 0; axis < index.size(); ++axis) {
        const std::size_t lhs_index =
            shape_[axis] == 1 || strides_[axis] == 0 ? 0 : index[axis];
        const std::size_t rhs_index =
            other.shape_[axis] == 1 || other.strides_[axis] == 0 ? 0
                                                                 : index[axis];
        lhs_offset += strides_[axis] * static_cast<std::ptrdiff_t>(lhs_index);
        rhs_offset +=
            other.strides_[axis] * static_cast<std::ptrdiff_t>(rhs_index);
      }
      result.data_[linear] =
          operation(data_[lhs_offset], other.data_[rhs_offset]);
    }
    return result;
  }

  static ParsedEinsum ParseEinsum(
      std::string_view expression,
      const std::vector<NDArray>& arrays) {
    const std::string script = detail::Trim(expression);
    const std::size_t arrow = script.find("->");
    if (arrow != std::string::npos &&
        script.find("->", arrow + 2) != std::string::npos) {
      throw std::invalid_argument("einsum expression contains multiple arrows");
    }

    const bool explicit_output = arrow != std::string::npos;
    const std::string input_part =
        explicit_output ? script.substr(0, arrow) : script;
    const std::string output_part =
        explicit_output ? script.substr(arrow + 2) : std::string{};
    std::vector<std::string> input_tokens = detail::Split(input_part, ',');
    if (input_tokens.size() != arrays.size()) {
      throw std::invalid_argument("einsum operand count does not match arrays");
    }

    static constexpr int kEllipsisBase = 256;
    std::optional<int> ellipsis_rank;
    const auto explicit_label_count = [](std::string_view token) {
      int count = 0;
      for (std::size_t i = 0; i < token.size();) {
        if (std::isspace(static_cast<unsigned char>(token[i]))) {
          ++i;
        } else if (i + 2 < token.size() && token.substr(i, 3) == "...") {
          i += 3;
        } else {
          ++count;
          ++i;
        }
      }
      return count;
    };

    for (std::size_t operand = 0; operand < input_tokens.size(); ++operand) {
      const std::string compact = detail::Trim(input_tokens[operand]);
      const std::size_t first = compact.find("...");
      const bool has_ellipsis = first != std::string::npos;
      if (has_ellipsis && compact.find("...", first + 3) != std::string::npos) {
        throw std::invalid_argument("multiple ellipses in einsum operand");
      }
      const int explicit_count = explicit_label_count(compact);
      const int rank = arrays[operand].Rank();
      if (has_ellipsis) {
        const int current_ellipsis_rank = rank - explicit_count;
        if (current_ellipsis_rank < 0) {
          throw std::invalid_argument("too many labels for einsum operand");
        }
        if (ellipsis_rank.has_value() &&
            *ellipsis_rank != current_ellipsis_rank) {
          throw std::invalid_argument(
              "ellipsis rank differs between einsum operands");
        }
        ellipsis_rank = current_ellipsis_rank;
      } else if (explicit_count != rank) {
        throw std::invalid_argument("einsum label rank does not match operand");
      }
    }
    const int ell_rank = ellipsis_rank.value_or(0);

    const auto parse_token = [ell_rank](
                                 std::string_view token, bool allow_ellipsis) {
      std::vector<int> labels;
      bool ellipsis_seen = false;
      for (std::size_t i = 0; i < token.size();) {
        const unsigned char ch = static_cast<unsigned char>(token[i]);
        if (std::isspace(ch)) {
          ++i;
          continue;
        }
        if (i + 2 < token.size() && token.substr(i, 3) == "...") {
          if (!allow_ellipsis || ellipsis_seen) {
            throw std::invalid_argument("invalid ellipsis in einsum script");
          }
          for (int axis = 0; axis < ell_rank; ++axis) {
            labels.push_back(kEllipsisBase + axis);
          }
          ellipsis_seen = true;
          i += 3;
          continue;
        }
        if (token[i] == '.' || token[i] == ',' || token[i] == '-' ||
            token[i] == '>') {
          throw std::invalid_argument("illegal character in einsum script");
        }
        labels.push_back(static_cast<int>(ch));
        ++i;
      }
      return labels;
    };

    ParsedEinsum parsed;
    parsed.input_labels.reserve(input_tokens.size());
    std::unordered_map<int, int> input_count;
    for (const std::string& token : input_tokens) {
      std::vector<int> labels = parse_token(token, true);
      for (const int label : labels) {
        ++input_count[label];
      }
      parsed.input_labels.push_back(std::move(labels));
    }

    if (explicit_output) {
      parsed.output_labels = parse_token(output_part, true);
      std::unordered_set<int> seen;
      for (const int label : parsed.output_labels) {
        if (!seen.insert(label).second) {
          throw std::invalid_argument("repeated label in einsum output");
        }
        if (!input_count.contains(label)) {
          throw std::invalid_argument(
              "einsum output label does not occur in an input");
        }
      }
    } else {
      for (int axis = 0; axis < ell_rank; ++axis) {
        parsed.output_labels.push_back(kEllipsisBase + axis);
      }
      std::vector<int> singleton_labels;
      for (const auto& [label, count] : input_count) {
        if (label < kEllipsisBase && count == 1) {
          singleton_labels.push_back(label);
        }
      }
      std::sort(singleton_labels.begin(), singleton_labels.end());
      parsed.output_labels.insert(
          parsed.output_labels.end(),
          singleton_labels.begin(),
          singleton_labels.end());
    }

    return parsed;
  }
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const NDArray<T>& array) {
  os << "NDArray(shape=[";
  for (std::size_t i = 0; i < array.shape().size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << array.shape()[i];
  }
  os << "], strides=[";
  for (std::size_t i = 0; i < array.strides().size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << array.strides()[i];
  }
  os << "])";

  if (array.Size() != 0) {
    os << '\n';
    for (std::size_t i = 0; i < array.Size(); ++i) {
      os << std::setw(16) << std::setprecision(10)
         << array.data()[array.LinearOffset(i)];
      if (!array.shape().empty() && (i + 1) % array.shape().back() == 0) {
        os << '\n';
      }
    }
  }
  return os;
}

} // namespace wickqc
