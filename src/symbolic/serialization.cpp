#include "symbolic/wick.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <istream>
#include <limits>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace wickqc::symbolic {
namespace {

// This codec describes the reference's native binary data layout. It depends
// on the host ABI (size_t, endian order, bool and double representation).
// Production symbolic types and algorithms do not depend on reference types.
class BinaryWriter {
 public:
  explicit BinaryWriter(std::ostream& output) : output_(output) {}

  template <typename T>
  void Scalar(const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    Bytes(reinterpret_cast<const char*>(&value), sizeof(T));
  }

  void String(const std::string& value) {
    Scalar(value.size());
    Bytes(value.data(), value.size());
  }

  template <typename Range>
  void Objects(const Range& values) {
    Scalar(values.size());
    for (const auto& value : values) {
      value.Save(output_);
    }
  }

 private:
  void Bytes(const char* data, std::size_t size) {
    if (size >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
      throw std::length_error("Wick binary field exceeds stream capacity");
    }
    output_.write(data, static_cast<std::streamsize>(size));
    if (!output_) {
      throw std::runtime_error("Unable to write Wick binary data");
    }
  }

  std::ostream& output_;
};

class BinaryReader {
 public:
  explicit BinaryReader(std::istream& input) : input_(input) {}

  template <typename T>
  T Scalar() {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    Bytes(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
  }

  std::string String() {
    auto remaining = Scalar<std::size_t>();
    std::array<char, 4096> buffer{};
    std::string value;
    // Read incrementally so a corrupt length cannot trigger a huge allocation
    // before the stream has supplied its payload.
    while (remaining != 0) {
      const auto size = std::min(remaining, buffer.size());
      Bytes(buffer.data(), size);
      value.append(buffer.data(), size);
      remaining -= size;
    }
    return value;
  }

  template <typename T>
  std::vector<T> Objects() {
    const auto count = Scalar<std::size_t>();
    std::vector<T> values;
    values.reserve(std::min(count, std::size_t{4096}));
    for (std::size_t i = 0; i < count; ++i) {
      values.push_back(T::Load(input_));
    }
    return values;
  }

 private:
  void Bytes(char* data, std::size_t size) {
    input_.read(data, static_cast<std::streamsize>(size));
    if (!input_) {
      throw std::runtime_error("Truncated or unreadable Wick binary data");
    }
  }

  std::istream& input_;
};

constexpr std::array<TensorKind, 5> kTensorWireKinds = {
    TensorKind::kCreation,
    TensorKind::kAnnihilation,
    TensorKind::kSpinFree,
    TensorKind::kDelta,
    TensorKind::kGeneric};

void ValidatePermutation(const SignedPermutation& permutation) {
  auto sorted = permutation.order;
  std::ranges::sort(sorted);
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    if (sorted[i] != i) {
      throw std::invalid_argument("Invalid permutation in Wick binary data");
    }
  }
  if (permutation.sign != 1 && permutation.sign != -1) {
    throw std::invalid_argument("Invalid permutation sign in Wick binary data");
  }
}
} // namespace

void Index::Save(std::ostream& output) const {
  BinaryWriter writer(output);
  writer.String(name);
  if (domain.orbital_spaces > 15 || domain.spins > 3) {
    throw std::invalid_argument(
        "Index domain cannot be encoded in Wick binary data");
  }
  writer.Scalar(
      static_cast<std::uint8_t>(domain.orbital_spaces | (domain.spins << 4)));
}

Index Index::Load(std::istream& input) {
  BinaryReader reader(input);
  Index result;
  result.name = reader.String();
  const auto flags = reader.Scalar<std::uint8_t>();
  if (flags > 63) {
    throw std::invalid_argument("Invalid index domain in Wick binary data");
  }
  result.domain = {
      static_cast<std::uint8_t>(flags & 15U),
      static_cast<std::uint8_t>(flags >> 4)};
  return result;
}

void SignedPermutation::Save(std::ostream& output) const {
  ValidatePermutation(*this);
  BinaryWriter writer(output);
  writer.Scalar(order.size());
  for (auto slot : order) {
    if (slot >
        static_cast<std::size_t>(std::numeric_limits<std::int16_t>::max())) {
      throw std::invalid_argument(
          "Permutation rank exceeds the Wick binary representation");
    }
    writer.Scalar(static_cast<std::int16_t>(slot));
  }
  writer.Scalar(sign < 0);
}

SignedPermutation SignedPermutation::Load(std::istream& input) {
  BinaryReader reader(input);
  const auto count = reader.Scalar<std::size_t>();
  if (count >
      static_cast<std::size_t>(std::numeric_limits<std::int16_t>::max()) + 1) {
    throw std::invalid_argument("Invalid permutation rank in Wick binary data");
  }
  SignedPermutation result;
  for (std::size_t i = 0; i < count; ++i) {
    const auto slot = reader.Scalar<std::int16_t>();
    if (slot < 0) {
      throw std::invalid_argument(
          "Negative permutation slot in Wick binary data");
    }
    result.order.push_back(static_cast<std::size_t>(slot));
  }
  // The native bool representation used by the reference on this ABI is a
  // single byte. Reading into an integer also validates malformed bool bytes.
  static_assert(sizeof(bool) == sizeof(std::uint8_t));
  const auto negative = reader.Scalar<std::uint8_t>();
  if (negative > 1) {
    throw std::invalid_argument(
        "Invalid permutation parity in Wick binary data");
  }
  result.sign = negative == 0 ? 1 : -1;
  ValidatePermutation(result);
  return result;
}

TensorSymmetry TensorSymmetry::FromElements(
    std::vector<SignedPermutation> elements) {
  for (const auto& element : elements) {
    ValidatePermutation(element);
    if (element.order.size() != elements.front().order.size()) {
      throw std::invalid_argument(
          "Tensor symmetry contains inconsistent ranks");
    }
  }
  TensorSymmetry result;
  result.elements_ = std::move(elements);
  return result;
}

void Tensor::Save(std::ostream& output) const {
  BinaryWriter writer(output);
  writer.String(name);
  writer.Objects(indices);
  writer.Objects(symmetry.Elements());
  const auto kind_position = std::ranges::find(kTensorWireKinds, kind);
  if (kind_position == kTensorWireKinds.end()) {
    throw std::invalid_argument("Invalid tensor kind in Wick binary data");
  }
  writer.Scalar(
      static_cast<std::uint8_t>(kind_position - kTensorWireKinds.begin()));
}

Tensor Tensor::Load(std::istream& input) {
  BinaryReader reader(input);
  Tensor result;
  result.name = reader.String();
  result.indices = reader.Objects<Index>();
  auto permutations = reader.Objects<SignedPermutation>();
  for (const auto& permutation : permutations) {
    if (permutation.order.size() != result.indices.size()) {
      throw std::invalid_argument(
          "Tensor and symmetry ranks differ in Wick binary data");
    }
  }
  result.symmetry = TensorSymmetry::FromElements(std::move(permutations));
  const auto kind = reader.Scalar<std::uint8_t>();
  if (kind >= kTensorWireKinds.size()) {
    throw std::invalid_argument("Invalid tensor kind in Wick binary data");
  }
  result.kind = kTensorWireKinds[kind];
  return result;
}

void Term::Save(std::ostream& output) const {
  BinaryWriter writer(output);
  writer.Objects(tensors);
  writer.Objects(std::set<Index>(summed_indices.begin(), summed_indices.end()));
  writer.Scalar(coefficient);
}

Term Term::Load(std::istream& input) {
  BinaryReader reader(input);
  Term result;
  result.tensors = reader.Objects<Tensor>();
  result.summed_indices = reader.Objects<Index>();
  std::ranges::sort(result.summed_indices);
  const auto unique_end = std::ranges::unique(result.summed_indices).begin();
  result.summed_indices.erase(unique_end, result.summed_indices.end());
  result.coefficient = reader.Scalar<double>();
  return result;
}

void Expression::Save(std::ostream& output) const {
  BinaryWriter(output).Objects(terms_);
}

Expression Expression::Load(std::istream& input) {
  return Expression(BinaryReader(input).Objects<Term>());
}

} // namespace wickqc::symbolic
