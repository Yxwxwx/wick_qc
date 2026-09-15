#include "method/ccsd.h"

#include "einsum/einsum.h"
#include "equation/equation.h"
#include "symbolic/index_domain.h"
#include "symbolic/wick.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace wickqc::method {

const symbolic::Expression& CCSDGenerator::Hamiltonian() const {
  return h_;
}
using symbolic::Expression;
using symbolic::OrbitalSpace;
using symbolic::TensorSymmetry;

CCSDGenerator::CCSDGenerator(bool antisymmetrized_integrals) {
  indices_.Add(OrbitalSpace::kInactive, "pqrsijklmno");
  indices_.Add(OrbitalSpace::kExternal, "pqrsabcdefg");
  symmetries_.Add(
      "v",
      4,
      antisymmetrized_integrals ? TensorSymmetry::FourAntisymmetric()
                                : TensorSymmetry::QuantumChemistryPhysicists());
  symmetries_.Add("t", 4, TensorSymmetry::FourAntisymmetric());
  const auto one_body = Parse("SUM <pq> h[pq] C[p] D[q]");
  const auto two_body = (antisymmetrized_integrals ? 0.25 : 0.5) *
      Parse("SUM <pqrs> v[pqrs] C[p] C[q] D[s] D[r]");
  h_ = (one_body + two_body).Expand(-1, true).Simplify();
  t_ = (Parse("SUM <ai> t[ai] C[a] D[i]") +
        0.25 * Parse("SUM <abij> t[abij] C[a] C[b] D[j] D[i]"))
           .Expand(-1, true)
           .Simplify();
}

Expression CCSDGenerator::Parse(std::string_view text) const {
  return Expression::Parse(text, indices_, symmetries_);
}

Expression CCSDGenerator::SimilarityTransform(int order, int rank) const {
  if (order < 0 || order > 4) {
    throw std::invalid_argument("CCSD BCH order must be between zero and four");
  }
  Expression result = h_;
  Expression nested = h_;
  for (int level = 0; level < order; ++level) {
    const int remaining_operators =
        rank == 0 ? (order - level - 1) * 2 : (order - level) * rank;
    nested = (1.0 / (level + 1)) *
        Commutator(nested, t_).Expand(remaining_operators).Simplify();
    result = result + nested;
  }
  return result;
}

Expression CCSDGenerator::Energy(int order) const {
  return SimilarityTransform(order, 0).Expand(0).Simplify();
}

Expression CCSDGenerator::Singles(int order) const {
  return (Parse("C[i] D[a]") * SimilarityTransform(order, 2))
      .Expand(0)
      .Simplify();
}

Expression CCSDGenerator::Doubles(int order) const {
  return (Parse("C[i] C[j] D[b] D[a]") * SimilarityTransform(order, 4))
      .Expand(0)
      .Simplify();
}

std::string CCSDGenerator::GenerateNumpy() const {
  auto render = [&](const Expression& expression, std::string_view target) {
    const auto output = symbolic::Tensor::Parse(target, indices_, symmetries_);
    return einsum::RenderNumpy(
        einsum::Program::Lower(
            equation::TensorEquation::FromExpression(expression, output)));
  };
  return render(Energy(), "energy[]") + render(Singles(), "singles[ai]") +
      render(Doubles(), "doubles[abij]");
}

} // namespace wickqc::method
