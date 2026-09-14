#include "method/spatial_mp.h"

#include "einsum/einsum.h"
#include "method/uga_ccsd.h"

#include <stdexcept>

namespace wickqc::method {
namespace {
std::string Axes(int rank) {
  return std::string("abcd").substr(0, rank) +
      std::string("ijkl").substr(0, rank);
}
std::string Excitations(int rank, bool bra) {
  std::string word;
  for (int axis = 0; axis < rank; ++axis) {
    word += " E1[";
    word += bra ? "ijkl"[axis] : "abcd"[axis];
    word += ',';
    word += bra ? "abcd"[axis] : "ijkl"[axis];
    word += ']';
  }
  return word;
}
} // namespace

SpatialMpGenerator::SpatialMpGenerator(int order) : order_(order) {
  if (order < 2 || order > 4) {
    throw std::invalid_argument("Spatial MP order must be two, three, or four");
  }
  indices_.Add(symbolic::OrbitalSpace::kInactive, "pqrsijklmno");
  indices_.Add(symbolic::OrbitalSpace::kExternal, "pqrsabcdefg");
  symmetries_.Add(
      "v", 4, symbolic::TensorSymmetry::QuantumChemistryPhysicists());
  for (int rank = 1; rank <= 4; ++rank) {
    symmetries_.Add(
        "u1",
        2 * static_cast<std::size_t>(rank),
        symbolic::TensorSymmetry::SpinFree(rank));
    symmetries_.Add(
        "u2",
        2 * static_cast<std::size_t>(rank),
        symbolic::TensorSymmetry::SpinFree(rank));
  }
  fock_ = Parse("SUM <p> eps[p] E1[p,p]\n-2 SUM <i> eps[i]");
  const auto canonical_fock = std::pair(
      symbolic::Tensor::Parse("f[pq]", indices_, symmetries_),
      Parse("delta[pq] eps[p]"));
  const auto hamiltonian =
      UgaCcsdGenerator().Hamiltonian().Substitute({{"f", canonical_fock}});
  perturbation_ = (hamiltonian - fock_).Simplify();
}

symbolic::Expression SpatialMpGenerator::Parse(std::string_view text) const {
  return symbolic::Expression::Parse(text, indices_, symmetries_);
}

symbolic::Expression SpatialMpGenerator::Wavefunction(int order, int rank)
    const {
  double factorial = 1.0;
  for (int i = 2; i <= rank; ++i) {
    factorial *= i;
  }
  return (1.0 / factorial) *
      Parse("SUM <" + Axes(rank) + "> u" + std::to_string(order) + "[" +
            Axes(rank) + "]" + Excitations(rank, false));
}

equation::ContractionGraph SpatialMpGenerator::Equations() const {
  equation::ContractionGraph graph;
  auto add = [&](const std::string& output, const symbolic::Expression& value) {
    graph.Add(
        symbolic::Tensor::Parse(output, indices_, symmetries_),
        value.Expand(0).Simplify());
  };
  const auto first = Wavefunction(1, 1) + Wavefunction(1, 2);
  for (int rank = 1; rank <= 2; ++rank) {
    add("residual1_rank" + std::to_string(rank) + "[" + Axes(rank) + "]",
        Parse(Excitations(rank, true)) *
            (fock_ * Wavefunction(1, rank) + perturbation_));
  }
  const auto energy2 = (perturbation_ * first).Expand(0).Simplify();
  graph.Add(
      symbolic::Tensor::Parse("energy2[]", indices_, symmetries_), energy2);
  if (order_ >= 3) {
    add("energy3[]", first.Conjugate() * perturbation_ * first);
  }
  if (order_ >= 4) {
    symbolic::Expression energy4;
    for (int rank = 1; rank <= 4; ++rank) {
      const auto second = Wavefunction(2, rank);
      add("residual2_rank" + std::to_string(rank) + "[" + Axes(rank) + "]",
          Parse(Excitations(rank, true)) *
              (fock_ * second + perturbation_ * first));
      // F_N preserves excitation rank. The 2n+1 formula used by block2's
      // MP driver is E4 = -<2|F_N|2> - E2 <1|1>, with E1 = 0 for RHF.
      energy4 = energy4 - second.Conjugate() * fock_ * second;
    }
    const auto norm1 = (first.Conjugate() * first).Expand(0).Simplify();
    add("energy4[]", energy4 - energy2 * norm1);
  }
  return graph;
}

std::string SpatialMpGenerator::GenerateNumpy(bool optimize) const {
  const auto graph = Equations();
  if (optimize) {
    return einsum::RenderNumpy(graph.Simplify());
  }
  std::string result;
  for (const auto& node : graph.Nodes()) {
    result += einsum::RenderNumpy(
        einsum::Program::Lower(
            equation::TensorEquation::FromExpression(
                node.expression, node.output)));
  }
  return result;
}
} // namespace wickqc::method
