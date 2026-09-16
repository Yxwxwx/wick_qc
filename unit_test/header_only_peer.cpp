#include "wick.hpp"

#include <cstddef>
#include <sstream>
#include <string>

namespace wickqc::test {

std::string HeaderOnlyExpression() {
  symbolic::IndexRegistry indices;
  symbolic::SymmetryRegistry symmetries;
  const auto expression =
      symbolic::Expression::Parse("2 A[] + 3 A[]", indices, symmetries)
          .Simplify();
  std::stringstream buffer;
  expression.Save(buffer);
  std::ostringstream output;
  output << symbolic::Expression::Load(buffer);
  return output.str();
}

double HeaderOnlyTrace(std::size_t size) {
  const auto matrix = NDArray<double>::Ones({size, size});
  return NDArray<double>::Einsum("ii->", {matrix}).data()[0];
}

decltype(&symbolic::SignedPermutation::Identity) HeaderOnlyIdentityAddress() {
  return &symbolic::SignedPermutation::Identity;
}

} // namespace wickqc::test
