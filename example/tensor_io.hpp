#pragma once

#include "backend/ndarray.hpp"
#include "runtime/numeric.hpp"

#include <filesystem>

namespace wickqc::example {
// Example interchange format: little-endian IEEE float64, C order, no header.
// Dimensions are read from a separate runtime file, never compiled into a
// kernel.
[[nodiscard]] NDArray<double> ReadTensor(
    const std::filesystem::path& path,
    const NDArray<double>::Shape& shape);
void WriteTensor(
    const std::filesystem::path& path,
    const NDArray<double>& array);
[[nodiscard]] runtime::Dimensions ReadDimensions(
    const std::filesystem::path& path);
} // namespace wickqc::example
