#include "runtime/tensor_binding.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace wickqc::runtime {

std::vector<std::size_t> TensorBinding::Shape(
    const Dimensions& dimensions) const {
  std::vector<std::size_t> shape;
  for (const auto domain : domains) {
    const auto found = dimensions.find(domain);
    if (found == dimensions.end()) {
      throw std::invalid_argument(
          "Missing dimension for tensor '" + name + "', orbital mask " +
          std::to_string(domain.orbital_spaces) + ", spin mask " +
          std::to_string(domain.spins));
    }
    shape.push_back(found->second);
  }
  return shape;
}

} // namespace wickqc::runtime
