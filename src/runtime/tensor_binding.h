#pragma once

#include "backend/ndarray.hpp"
#include "symbolic/index_domain.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace wickqc::runtime {

using Dimensions = std::map<symbolic::IndexDomain, std::size_t>;
template <typename T = double>
using TensorMap = std::map<std::string, NDArray<T>>;

// Names follow the NumPy emitter (e.g. vIIEE, tEEII, E1); axes retain the
// method's orbital and spin domains. General domains require an explicit size.
struct TensorBinding {
  std::string name;
  std::vector<symbolic::IndexDomain> domains;
  [[nodiscard]] std::vector<std::size_t> Shape(const Dimensions& dimensions) const;
  bool operator==(const TensorBinding&) const = default;
};

} // namespace wickqc::runtime
