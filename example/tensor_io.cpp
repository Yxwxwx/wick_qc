#include "tensor_io.hpp"
#include <ios>
#include "backend/ndarray.hpp"
#include "runtime/numeric.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace wickqc::example {
namespace {
std::size_t ByteCount(const NDArray<double>::Shape& shape) {
  std::size_t bytes = sizeof(double);
  for (const auto extent : shape) {
    if (extent != 0 &&
        bytes > std::numeric_limits<std::size_t>::max() / extent) {
      throw std::invalid_argument("Tensor byte count overflows size_t");
    }
    bytes *= extent;
  }
  if (bytes >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::invalid_argument("Tensor is too large for file I/O");
  }
  return bytes;
}
} // namespace

NDArray<double> ReadTensor(
    const std::filesystem::path& path,
    const NDArray<double>::Shape& shape) {
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
  const auto bytes = ByteCount(shape);
  if (std::filesystem::file_size(path) != bytes) {
    throw std::invalid_argument(
        "File size does not match tensor shape: " + path.string());
  }
  NDArray<double> array(shape);
  std::ifstream input(path, std::ios::binary);
  if (bytes != 0 &&
      !input.read(
          reinterpret_cast<char*>(array.data()),
          static_cast<std::streamsize>(bytes))) {
    throw std::runtime_error("Cannot read tensor: " + path.string());
  }
  for (std::size_t i = 0; i < array.Size(); ++i) {
    if (!std::isfinite(array.data()[i])) {
      throw std::invalid_argument("Nonfinite tensor value: " + path.string());
    }
  }
  return array;
}

void WriteTensor(
    const std::filesystem::path& path,
    const NDArray<double>& array) {
  const auto contiguous = array.ToCOrder();
  const auto bytes = ByteCount(contiguous.shape());
  std::ofstream output(path, std::ios::binary);
  if (bytes != 0) {
    output.write(
        reinterpret_cast<const char*>(contiguous.data()),
        static_cast<std::streamsize>(bytes));
  }
  output.close();
  if (!output) {
    throw std::runtime_error("Cannot write tensor: " + path.string());
  }
}

runtime::Dimensions ReadDimensions(const std::filesystem::path& path) {
  std::ptrdiff_t inactive = 0, active = 0, external = 0;
  std::ifstream input(path);
  std::string trailing;
  if (!(input >> inactive >> active >> external) || (input >> trailing) ||
      inactive < 0 || active < 0 || external < 0) {
    throw std::invalid_argument(
        "Expected three nonnegative dimensions (inactive active external): " +
        path.string());
  }
  return {
      {{1, 0}, static_cast<std::size_t>(inactive)},
      {{2, 0}, static_cast<std::size_t>(active)},
      {{8, 0}, static_cast<std::size_t>(external)}};
}
} // namespace wickqc::example
