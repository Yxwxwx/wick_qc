#pragma once

#include "symbolic/wick.h"

#include <string>
#include <utility>
#include <vector>

namespace wickqc::method {

class GHFGenerator {
 public:
  [[nodiscard]] std::vector<std::pair<std::string, symbolic::Expression>> HamiltonianBlocks() const;
  // One coefficient tensor for each block and normal-ordered operator rank.
  // Its axes follow the ordered C/D monomial; remaining indices are reductions.
  [[nodiscard]] std::string GenerateNumpy() const;
};

} // namespace wickqc::method
