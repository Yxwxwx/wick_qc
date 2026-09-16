#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "einsum/einsum.hpp"
#include "equation/graph.hpp"
#include "symbolic/index_domain.hpp"
#include "symbolic/wick.hpp"

namespace wickqc::method {

class GHFGenerator {
 public:
  [[nodiscard]] std::vector<std::pair<std::string, symbolic::Expression>>
  HamiltonianBlocks() const;
  // One coefficient tensor for each block and normal-ordered operator rank.
  // Its axes follow the ordered C/D monomial; remaining indices are reductions.
  [[nodiscard]] std::string GenerateNumpy() const;
};

// Spin-orbital CCSD, with a normal-ordered Hamiltonian and factorial BCH
// weights.
class CCSDGenerator {
 public:
  explicit CCSDGenerator(bool antisymmetrized_integrals = true);

  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] symbolic::Expression Energy(int order = 2) const;
  [[nodiscard]] symbolic::Expression Singles(int order = 4) const;
  [[nodiscard]] symbolic::Expression Doubles(int order = 4) const;
  [[nodiscard]] std::string GenerateNumpy() const;
  [[nodiscard]] const symbolic::Expression& Hamiltonian() const;

 private:
  [[nodiscard]] symbolic::Expression SimilarityTransform(int order, int rank)
      const;
  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  symbolic::Expression h_;
  symbolic::Expression t_;
};

inline std::vector<std::pair<std::string, symbolic::Expression>> GHFGenerator::
    HamiltonianBlocks() const {
  std::array<symbolic::IndexRegistry, 4> indices;
  indices[0].Add(
      symbolic::OrbitalSpace::kGeneral, "ijkl", symbolic::Spin::kAlpha);
  indices[1].Add(
      symbolic::OrbitalSpace::kGeneral, "ijkl", symbolic::Spin::kBeta);
  indices[2].Add(
      symbolic::OrbitalSpace::kGeneral, "ij", symbolic::Spin::kAlpha);
  indices[2].Add(symbolic::OrbitalSpace::kGeneral, "kl", symbolic::Spin::kBeta);
  indices[3].Add(symbolic::OrbitalSpace::kGeneral, "ij", symbolic::Spin::kBeta);
  indices[3].Add(
      symbolic::OrbitalSpace::kGeneral, "kl", symbolic::Spin::kAlpha);
  symbolic::SymmetryRegistry symmetries;
  symmetries.Add("v", 4, symbolic::TensorSymmetry::QuantumChemistryChemists());
  auto expand = [&](std::string_view text, std::size_t block) {
    return symbolic::Expression::Parse(text, indices[block], symmetries)
        .Expand()
        .Simplify();
  };
  return {
      {"h1_b", expand("SUM <ij> h[ij] D[i] C[j]", 1)},
      {"h2_aa", expand("0.5 SUM <ijkl> v[ijkl] C[i] C[k] D[l] D[j]", 0)},
      {"h2_bb", expand("0.5 SUM <ijkl> v[ijkl] D[i] D[k] C[l] C[j]", 1)},
      {"h2_ab", expand("0.5 SUM <ijkl> v[ijkl] C[i] D[k] C[l] D[j]", 2)},
      {"h2_ba", expand("0.5 SUM <ijkl> v[ijkl] D[i] C[k] D[l] C[j]", 3)}};
}

inline std::string GHFGenerator::GenerateNumpy() const {
  std::string result;
  for (const auto& [name, expression] : HamiltonianBlocks()) {
    for (auto term : expression.Terms()) {
      symbolic::Tensor output;
      std::erase_if(term.tensors, [&](const symbolic::Tensor& tensor) {
        if (!tensor.IsFermionOperator()) {
          return false;
        }
        output.indices.push_back(tensor.indices.front());
        return true;
      });
      output.name = name + "_rank" + std::to_string(output.indices.size());
      output.symmetry = symbolic::TensorSymmetry::None(output.indices.size());
      std::erase_if(term.summed_indices, [&](const symbolic::Index& index) {
        return std::ranges::find(output.indices, index) != output.indices.end();
      });
      result += einsum::RenderNumpy(
          einsum::Program::Lower(
              equation::TensorEquation::FromExpression(
                  symbolic::Expression(std::move(term)), output)));
    }
  }
  return result;
}

inline const symbolic::Expression& CCSDGenerator::Hamiltonian() const {
  return h_;
}

inline CCSDGenerator::CCSDGenerator(bool antisymmetrized_integrals) {
  indices_.Add(symbolic::OrbitalSpace::kInactive, "pqrsijklmno");
  indices_.Add(symbolic::OrbitalSpace::kExternal, "pqrsabcdefg");
  symmetries_.Add(
      "v",
      4,
      antisymmetrized_integrals
          ? symbolic::TensorSymmetry::FourAntisymmetric()
          : symbolic::TensorSymmetry::QuantumChemistryPhysicists());
  symmetries_.Add("t", 4, symbolic::TensorSymmetry::FourAntisymmetric());
  const auto one_body = Parse("SUM <pq> h[pq] C[p] D[q]");
  const auto two_body = (antisymmetrized_integrals ? 0.25 : 0.5) *
      Parse("SUM <pqrs> v[pqrs] C[p] C[q] D[s] D[r]");
  h_ = (one_body + two_body).Expand(-1, true).Simplify();
  t_ = (Parse("SUM <ai> t[ai] C[a] D[i]") +
        0.25 * Parse("SUM <abij> t[abij] C[a] C[b] D[j] D[i]"))
           .Expand(-1, true)
           .Simplify();
}

inline symbolic::Expression CCSDGenerator::Parse(std::string_view text) const {
  return symbolic::Expression::Parse(text, indices_, symmetries_);
}

inline symbolic::Expression CCSDGenerator::SimilarityTransform(
    int order,
    int rank) const {
  if (order < 0 || order > 4) {
    throw std::invalid_argument("CCSD BCH order must be between zero and four");
  }
  symbolic::Expression result = h_;
  symbolic::Expression nested = h_;
  for (int level = 0; level < order; ++level) {
    const int remaining_operators =
        rank == 0 ? (order - level - 1) * 2 : (order - level) * rank;
    nested = (1.0 / (level + 1)) *
        Commutator(nested, t_).Expand(remaining_operators).Simplify();
    result = result + nested;
  }
  return result;
}

inline symbolic::Expression CCSDGenerator::Energy(int order) const {
  return SimilarityTransform(order, 0).Expand(0).Simplify();
}

inline symbolic::Expression CCSDGenerator::Singles(int order) const {
  return (Parse("C[i] D[a]") * SimilarityTransform(order, 2))
      .Expand(0)
      .Simplify();
}

inline symbolic::Expression CCSDGenerator::Doubles(int order) const {
  return (Parse("C[i] C[j] D[b] D[a]") * SimilarityTransform(order, 4))
      .Expand(0)
      .Simplify();
}

inline std::string CCSDGenerator::GenerateNumpy() const {
  auto render = [&](const symbolic::Expression& expression,
                    std::string_view target) {
    const auto output = symbolic::Tensor::Parse(target, indices_, symmetries_);
    return einsum::RenderNumpy(
        einsum::Program::Lower(
            equation::TensorEquation::FromExpression(expression, output)));
  };
  return render(Energy(), "energy[]") + render(Singles(), "singles[ai]") +
      render(Doubles(), "doubles[abij]");
}

} // namespace wickqc::method
