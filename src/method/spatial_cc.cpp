#include "method/spatial_cc.h"

#include "einsum/einsum.h"
#include "equation/equation.h"
#include "method/integral_convention.h"
#include "method/uga_ccsd.h"
#include "symbolic/index_domain.h"
#include "symbolic/wick.h"

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace wickqc::method {
namespace {
constexpr std::string_view kVirtual = "abcd";
constexpr std::string_view kOccupied = "ijkl";

std::string ExcitationWord(int rank, bool conjugate) {
  std::string word;
  for (int axis = 0; axis < rank; ++axis) {
    word += " E1[";
    word += conjugate ? kOccupied[axis] : kVirtual[axis];
    word += ',';
    word += conjugate ? kVirtual[axis] : kOccupied[axis];
    word += ']';
  }
  return word;
}

std::string Axes(int rank) {
  return std::string(kVirtual.substr(0, rank)) +
      std::string(kOccupied.substr(0, rank));
}
} // namespace

SpatialCcGenerator::SpatialCcGenerator(
    int excitation_rank,
    IntegralConvention convention) {
  if (excitation_rank < 1 || excitation_rank > 4) {
    throw std::invalid_argument(
        "Spatial CC excitation rank must be between one and four");
  }
  indices_.Add(symbolic::OrbitalSpace::kInactive, "pqrsijklmno");
  indices_.Add(symbolic::OrbitalSpace::kExternal, "pqrsabcdefg");
  symmetries_.Add(
      "v",
      4,
      convention == IntegralConvention::kChemist
          ? symbolic::TensorSymmetry::QuantumChemistryChemists()
          : symbolic::TensorSymmetry::QuantumChemistryPhysicists());
  double factorial = 1.0;
  for (int rank = 1; rank <= excitation_rank; ++rank) {
    factorial *= rank;
    symmetries_.Add(
        "t",
        2 * static_cast<std::size_t>(rank),
        symbolic::TensorSymmetry::SpinFree(rank));
    cluster_ranks_.push_back(
        (1.0 / factorial) *
        Parse(
            "SUM <" + Axes(rank) + "> t[" + Axes(rank) + "]" +
            ExcitationWord(rank, false)));
  }
  hamiltonian_ = UgaCcsdGenerator(convention).Hamiltonian();
}

symbolic::Expression SpatialCcGenerator::Parse(std::string_view text) const {
  return symbolic::Expression::Parse(text, indices_, symmetries_);
}

const symbolic::Expression& SpatialCcGenerator::Hamiltonian() const {
  return hamiltonian_;
}

symbolic::Expression SpatialCcGenerator::Projected(int rank, int bch_order)
    const {
  if (rank < 0 || rank > static_cast<int>(cluster_ranks_.size()) ||
      bch_order < 0 || bch_order > 4) {
    throw std::invalid_argument("Invalid spatial CC projection or BCH order");
  }
  const auto projector =
      rank == 0 ? Parse("1") : Parse(ExcitationWord(rank, true));
  const bool batch_projection = cluster_ranks_.size() == 4;
  auto project_batch = [&](const symbolic::Expression& expression) {
    return (projector * expression).Expand(0).Simplify();
  };
  auto result = batch_projection ? project_batch(hamiltonian_) : hamiltonian_;
  // Pure excitation operators commute. Enumerate their rank multisets in the
  // same BCH ordering as the supplied UGA equations (higher ranks innermost).
  // A two-body Hamiltonian can lower excitation rank by at most two.
  std::vector<int> word;
  for (int degree = 1; degree <= bch_order; ++degree) {
    std::function<void(int, int)> enumerate = [&](int minimum, int total_rank) {
      if (static_cast<int>(word.size()) == degree) {
        auto nested = hamiltonian_;
        double coefficient = 1.0;
        int previous = 0, repetitions = 0;
        for (auto it = word.rbegin(); it != word.rend(); ++it) {
          repetitions = *it == previous ? repetitions + 1 : 1;
          previous = *it;
          coefficient /= repetitions;
          nested = Commutator(nested, cluster_ranks_[*it - 1]);
          if (batch_projection) {
            // Keep the full tagged operator word between commutators. The
            // compact reference-vacuum representation would drop operators
            // still needed by a later contraction.
            nested = nested.Expand(-1, false, false).Simplify();
          }
        }
        const auto contribution = coefficient * nested;
        result = result +
            (batch_projection ? project_batch(contribution) : contribution);
        return;
      }
      for (int next = minimum; next <= static_cast<int>(cluster_ranks_.size());
           ++next) {
        if (total_rank + next > rank + 2) {
          break;
        }
        word.push_back(next);
        enumerate(next, total_rank + next);
        word.pop_back();
      }
    };
    enumerate(1, 0);
  }
  return batch_projection ? result.Simplify()
                          : (projector * result).Expand(0).Simplify();
}

equation::ContractionGraph SpatialCcGenerator::Equations() const {
  equation::ContractionGraph graph;
  for (int rank = 0; rank <= static_cast<int>(cluster_ranks_.size()); ++rank) {
    const auto name = rank == 0 ? "energy" : "residual" + std::to_string(rank);
    graph.Add(
        symbolic::Tensor::Parse(
            name + "[" + Axes(rank) + "]", indices_, symmetries_),
        Projected(rank));
  }
  return graph;
}

std::string SpatialCcGenerator::GenerateNumpy(bool optimize) const {
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
