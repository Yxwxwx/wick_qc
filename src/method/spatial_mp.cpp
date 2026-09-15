#include "method/spatial_mp.h"

#include "einsum/einsum.h"
#include "equation/equation.h"
#include "method/integral_convention.h"
#include "method/uga_ccsd.h"
#include "symbolic/index_domain.h"
#include "symbolic/wick.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace wickqc::method {
namespace {
std::string Axis(int axis, bool occupied) {
  if (axis < 4) {
    return std::string(1, occupied ? "ijkl"[axis] : "abcd"[axis]);
  }
  return std::string(occupied ? "i" : "a") + std::to_string(axis);
}
std::string Axes(int rank) {
  if (rank <= 4) {
    return std::string("abcd").substr(0, rank) +
        std::string("ijkl").substr(0, rank);
  }
  std::string result;
  for (bool occupied : {false, true}) {
    for (int axis = 0; axis < rank; ++axis) {
      result += Axis(axis, occupied) + " ";
    }
  }
  return result;
}
std::string Excitations(int rank, bool bra) {
  std::string word;
  for (int axis = 0; axis < rank; ++axis) {
    // Spaces make multi-character index names unambiguous to the parser.
    word += " E1[" + Axis(axis, bra) + (axis < 4 ? "," : ", ") +
        Axis(axis, !bra) + "]";
  }
  return word;
}
} // namespace

SpatialMpGenerator::SpatialMpGenerator(
    int order,
    IntegralConvention convention,
    int maximum_excitation_rank)
    : order_(order), maximum_excitation_rank_(maximum_excitation_rank) {
  if (order < 2 || maximum_excitation_rank < 0) {
    throw std::invalid_argument(
        "Spatial MP order must be at least two, and the excitation bound nonnegative");
  }
  indices_.Add(symbolic::OrbitalSpace::kInactive, "pqrsijklmno");
  indices_.Add(symbolic::OrbitalSpace::kExternal, "pqrsabcdefg");
  symmetries_.Add(
      "v",
      4,
      convention == IntegralConvention::kChemist
          ? symbolic::TensorSymmetry::QuantumChemistryChemists()
          : symbolic::TensorSymmetry::QuantumChemistryPhysicists());
  for (int axis = 4; axis < WavefunctionRank(order_ / 2); ++axis) {
    indices_.Add(symbolic::OrbitalSpace::kInactive, Axis(axis, true) + " ");
    indices_.Add(symbolic::OrbitalSpace::kExternal, Axis(axis, false) + " ");
  }
  for (int n = 1; n <= order_ / 2; ++n) {
    for (int rank = 1; rank <= WavefunctionRank(n); ++rank) {
      symmetries_.Add(
          "u" + std::to_string(n),
          2 * static_cast<std::size_t>(rank),
          symbolic::TensorSymmetry::SpinFree(rank));
    }
  }
  fock_ = Parse("SUM <p> eps[p] E1[p,p]\n-2 SUM <i> eps[i]");
  const auto canonical_fock = std::pair(
      symbolic::Tensor::Parse("f[pq]", indices_, symmetries_),
      Parse("delta[pq] eps[p]"));
  const auto hamiltonian = UgaCcsdGenerator(convention)
                               .Hamiltonian()
                               .Substitute({{"f", canonical_fock}});
  perturbation_ = (hamiltonian - fock_).Simplify();
}

symbolic::Expression SpatialMpGenerator::Parse(std::string_view text) const {
  return symbolic::Expression::Parse(text, indices_, symmetries_);
}

int SpatialMpGenerator::WavefunctionRank(int order) const {
  return maximum_excitation_rank_ == 0
      ? 2 * order
      : std::min(2 * order, maximum_excitation_rank_);
}

symbolic::Expression SpatialMpGenerator::Wavefunction(int order, int rank)
    const {
  if (rank > WavefunctionRank(order)) {
    return {};
  }
  double factorial = 1.0;
  for (int i = 2; i <= rank; ++i) {
    factorial *= i;
  }
  if (!std::isfinite(factorial)) {
    throw std::invalid_argument("MP excitation factorial exceeds double range");
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
  for (int rank = 1; rank <= WavefunctionRank(1); ++rank) {
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
    for (int rank = 1; rank <= WavefunctionRank(2); ++rank) {
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
  // Continue the same intermediate-normalized Rayleigh-Schroedinger hierarchy.
  // Q is enforced by the excited projectors, and <0|u_n>=0, E1=0 for RHF.
  // Lower energies are graph values, not independent caller-supplied scalars.
  const auto energy = [&](int n) {
    return Parse("energy" + std::to_string(n) + "[]");
  };
  const auto overlap = [&](int a, int b) {
    symbolic::Expression result;
    for (int rank = 1;
         rank <= std::min(WavefunctionRank(a), WavefunctionRank(b));
         ++rank) {
      result = result +
          (Wavefunction(a, rank).Conjugate() * Wavefunction(b, rank))
              .Expand(0)
              .Simplify();
    }
    return result.Simplify();
  };
  for (int n = 5; n <= order_; ++n) {
    const int m = n / 2;
    if (n % 2 == 0) {
      for (int rank = 1; rank <= WavefunctionRank(m); ++rank) {
        const auto projector = Parse(Excitations(rank, true));
        auto residual =
            (projector * fock_ * Wavefunction(m, rank)).Expand(0).Simplify();
        for (int previous = std::max(1, rank - 2);
             previous <= std::min(WavefunctionRank(m - 1), rank + 2);
             ++previous) {
          residual = residual +
              (projector * perturbation_ * Wavefunction(m - 1, previous))
                  .Expand(0)
                  .Simplify();
        }
        for (int k = 2; k < m; ++k) {
          if (rank <= WavefunctionRank(m - k)) {
            residual = residual -
                energy(k) *
                    (projector * Wavefunction(m - k, rank))
                        .Expand(0)
                        .Simplify();
          }
        }
        add("residual" + std::to_string(m) + "_rank" + std::to_string(rank) +
                "[" + Axes(rank) + "]",
            residual);
      }
    }
    symbolic::Expression correction;
    if (n % 2 == 0) {
      for (int rank = 1; rank <= WavefunctionRank(m); ++rank) {
        const auto psi = Wavefunction(m, rank);
        correction =
            correction - (psi.Conjugate() * fock_ * psi).Expand(0).Simplify();
      }
    } else {
      for (int a = 1; a <= WavefunctionRank(m); ++a) {
        for (int b = std::max(1, a - 2);
             b <= std::min(WavefunctionRank(m), a + 2);
             ++b) {
          correction = correction +
              (Wavefunction(m, a).Conjugate() * perturbation_ *
               Wavefunction(m, b))
                  .Expand(0)
                  .Simplify();
        }
      }
    }
    // Wigner's 2m+1 rule. Even orders eliminate V|u_(m-1)> with the
    // order-m residual. The remaining normalization sums reproduce block2's
    // E4=-<2|F_N|2>-E2<1|1> and E5=<2|V|2>-2E2<2|1>-E3<1|1>.
    const int bound = n % 2 == 0 ? m - 1 : m;
    for (int a = 1; a <= bound; ++a) {
      for (int b = 1; b <= bound; ++b) {
        if (n - a - b >= 2) {
          correction = correction - energy(n - a - b) * overlap(a, b);
        }
      }
    }
    add("energy" + std::to_string(n) + "[]", correction);
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
