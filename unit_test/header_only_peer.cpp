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

#if defined(WICKQC_ENABLE_AO2MO)
#include <ao2mo/transform.hpp>

const void* OtherMutex() {
  return &ao2mo::h5::ExecutionMutex();
}

double OtherTransform(const ao2mo::Basis& basis) {
  auto coefficients = std::make_shared<ao2mo::Coefficients<double>>();
  coefficients->nao = coefficients->nmo = 1;
  coefficients->alpha = {1};
  ao2mo::Request<double> request;
  for (auto& index : request.indices) {
    index = {coefficients, {0}};
  }
  ao2mo::Options options;
  options.workspace = ao2mo::Workspace::kIncore;
  return ao2mo::Transform(basis, std::vector{request}, options)
      .blocks.at(0)
      .values.at(0);
}

#endif
