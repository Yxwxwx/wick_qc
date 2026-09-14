#include "method/ghf.h"

#include "einsum/einsum.h"

#include <algorithm>
#include <array>
#include <set>

namespace wickqc::method {
using symbolic::Expression;
using symbolic::Index;
using symbolic::IndexRegistry;
using symbolic::OrbitalSpace;
using symbolic::Spin;
using symbolic::SymmetryRegistry;
using symbolic::Tensor;
using symbolic::TensorSymmetry;

std::vector<std::pair<std::string, Expression>> GhfGenerator::
    HamiltonianBlocks() const {
  std::array<IndexRegistry, 4> indices;
  indices[0].Add(OrbitalSpace::kGeneral, "ijkl", Spin::kAlpha);
  indices[1].Add(OrbitalSpace::kGeneral, "ijkl", Spin::kBeta);
  indices[2].Add(OrbitalSpace::kGeneral, "ij", Spin::kAlpha);
  indices[2].Add(OrbitalSpace::kGeneral, "kl", Spin::kBeta);
  indices[3].Add(OrbitalSpace::kGeneral, "ij", Spin::kBeta);
  indices[3].Add(OrbitalSpace::kGeneral, "kl", Spin::kAlpha);
  SymmetryRegistry symmetries;
  symmetries.Add("v", 4, TensorSymmetry::QuantumChemistryChemists());
  auto expand = [&](std::string_view text, std::size_t block) {
    return Expression::Parse(text, indices[block], symmetries)
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

std::string GhfGenerator::GenerateNumpy() const {
  std::string result;
  for (const auto& [name, expression] : HamiltonianBlocks()) {
    for (auto term : expression.Terms()) {
      Tensor output;
      std::erase_if(term.tensors, [&](const Tensor& tensor) {
        if (!tensor.IsFermionOperator()) {
          return false;
        }
        output.indices.push_back(tensor.indices.front());
        return true;
      });
      output.name = name + "_rank" + std::to_string(output.indices.size());
      output.symmetry = TensorSymmetry::None(output.indices.size());
      std::erase_if(term.summed_indices, [&](const Index& index) {
        return std::ranges::find(output.indices, index) != output.indices.end();
      });
      result += einsum::RenderNumpy(
          einsum::Program::Lower(
              equation::TensorEquation::FromExpression(
                  Expression(std::move(term)), output)));
    }
  }
  return result;
}

} // namespace wickqc::method
