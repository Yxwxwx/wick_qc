#include "method/uga_ccsd.h"

#include "einsum/einsum.h"

#include <array>
#include <stdexcept>
#include <vector>

namespace wickqc::method {
namespace {
void ValidateOrder(int order) {
  if (order < 0 || order > 4) {
    throw std::invalid_argument(
        "UGA-CCSD BCH order must be between zero and four");
  }
}
} // namespace

using symbolic::Expression;
using symbolic::OrbitalSpace;
using symbolic::TensorSymmetry;

UgaCcsdGenerator::UgaCcsdGenerator(IntegralConvention convention) {
  indices_.Add(OrbitalSpace::kInactive, "pqrsijklmno");
  indices_.Add(OrbitalSpace::kExternal, "pqrsabcdefg");
  const bool chemist = convention == IntegralConvention::kChemist;
  symmetries_.Add(
      "v",
      4,
      chemist ? TensorSymmetry::QuantumChemistryChemists()
              : TensorSymmetry::QuantumChemistryPhysicists());
  symmetries_.Add("t", 4, TensorSymmetry::SpinFree(2));
  const auto fock_definition = std::pair(
      symbolic::Tensor::Parse("h[pq]", indices_, symmetries_),
      Parse(
          chemist ? "f[pq]\n-2 SUM <j> v[pqjj]\n+SUM <j> v[pjjq]"
                  : "f[pq]\n-2 SUM <j> v[pjqj]\n+SUM <j> v[pjjq]"));
  const std::map<std::string, std::pair<symbolic::Tensor, Expression>>
      definitions = {{"h", fock_definition}};
  const auto one_body = Parse("SUM <pq> h[pq] E1[p,q]").Substitute(definitions);
  const auto two_body = Parse(
      chemist ? "0.5 SUM <pqrs> v[pqrs] E2[pr,qs]"
              : "0.5 SUM <pqrs> v[pqrs] E2[pq,rs]");
  const auto reference_energy =
      Parse(
          chemist ? "2 SUM <i> h[ii]\n+2 SUM <ij> v[iijj]\n-SUM <ij> v[ijji]"
                  : "2 SUM <i> h[ii]\n+2 SUM <ij> v[ijij]\n-SUM <ij> v[ijji]")
          .Substitute(definitions);
  h_ = (one_body + two_body - reference_energy).Simplify();
  singles_ = Parse("SUM <ai> t[ai] E1[a,i]");
  doubles_ = Parse("0.5 SUM <abij> t[abij] E1[a,i] E1[b,j]");
  cluster_ = (singles_ + doubles_).Simplify();
}

Expression UgaCcsdGenerator::Parse(std::string_view text) const {
  return Expression::Parse(text, indices_, symmetries_);
}

const Expression& UgaCcsdGenerator::Hamiltonian() const {
  return h_;
}

Expression UgaCcsdGenerator::Energy(int order) const {
  ValidateOrder(order);
  auto nested = h_;
  auto transformed = h_;
  for (int degree = 1; degree <= order; ++degree) {
    nested = (1.0 / degree) * Commutator(nested, cluster_);
    transformed = transformed + nested;
  }
  return transformed.Expand(0).Simplify();
}

Expression UgaCcsdGenerator::Projected(int order, int rank) const {
  ValidateOrder(order);
  auto transformed = h_;
  if (order != 0) {
    transformed = transformed + Commutator(h_, cluster_);
  }
  // The projected BCH polynomial, J. Chem. Phys. 89, 7382 (1988), Eq. 17.
  // Each word gives the ranks of successive cluster operators in a nested
  // commutator; doubles-only contributions start at projection rank two.
  struct Contribution {
    double weight;
    int minimum_rank;
    std::vector<int> word;
  };
  const std::array<Contribution, 6> contributions = {
      {{0.5, 1, {1, 1}},
       {1.0, 1, {2, 1}},
       {0.5, 2, {2, 2}},
       {1.0 / 6.0, 1, {1, 1, 1}},
       {0.5, 2, {2, 1, 1}},
       {1.0 / 24.0, 2, {1, 1, 1, 1}}}};
  for (const auto& contribution : contributions) {
    if (contribution.minimum_rank > rank ||
        static_cast<int>(contribution.word.size()) > order) {
      continue;
    }
    auto nested = h_;
    for (auto cluster_rank : contribution.word) {
      nested = Commutator(nested, cluster_rank == 1 ? singles_ : doubles_);
    }
    transformed = transformed + contribution.weight * nested;
  }
  const auto projector =
      rank == 1 ? Parse("E1[i,a]") : Parse("E1[i,a] E1[j,b]");
  return (projector * transformed).Expand(0).Simplify();
}

Expression UgaCcsdGenerator::Singles(int order) const {
  return Projected(order, 1);
}

Expression UgaCcsdGenerator::Doubles(int order) const {
  return Projected(order, 2);
}

std::string UgaCcsdGenerator::GenerateNumpy() const {
  auto render = [&](const Expression& expression, std::string_view target) {
    return einsum::RenderNumpy(
        einsum::Program::Lower(
            equation::TensorEquation::FromExpression(
                expression,
                symbolic::Tensor::Parse(target, indices_, symmetries_))));
  };
  return render(Energy(), "energy[]") + render(Singles(), "singles[ai]") +
      render(Doubles(), "doubles[abij]");
}

} // namespace wickqc::method
