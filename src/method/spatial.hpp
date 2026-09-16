#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "einsum/einsum.hpp"
#include "equation/graph.hpp"
#include "method/specification.hpp"
#include "symbolic/index_domain.hpp"
#include "symbolic/wick.hpp"

namespace wickqc::method {

// Canonical closed-shell MPn with intermediate-normalized wavefunctions. The
// spin-free projected linear systems retain their overlap metric; these
// residuals are not denominator-divided updates.
class SpatialMPGenerator {
 public:
  explicit SpatialMPGenerator(
      int order = 4,
      IntegralConvention convention = IntegralConvention::kPhysicist,
      int maximum_excitation_rank = 0);
  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] equation::ContractionGraph Equations() const;
  [[nodiscard]] std::string GenerateNumpy(bool optimize = false) const;

 private:
  [[nodiscard]] symbolic::Expression Wavefunction(int order, int rank) const;
  [[nodiscard]] int WavefunctionRank(int order) const;
  int order_;
  // Zero retains every rank (through 2 * wavefunction order). A physical
  // electron/hole rank bound can be supplied for runtime generation.
  int maximum_excitation_rank_;
  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  symbolic::Expression fock_;
  symbolic::Expression perturbation_;
};

// Closed-shell spin-free CC with pair-symmetric spatial amplitudes T_1...T_n.
// Residuals use the covariant E1...E1 projectors of the supplied UGA fixture.
class SpatialCCGenerator {
 public:
  explicit SpatialCCGenerator(
      int excitation_rank = 2,
      IntegralConvention convention = IntegralConvention::kPhysicist);
  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] const symbolic::Expression& Hamiltonian() const;
  [[nodiscard]] symbolic::Expression Projected(int rank, int bch_order = 4)
      const;
  [[nodiscard]] equation::ContractionGraph Equations() const;
  [[nodiscard]] std::string GenerateNumpy(bool optimize = false) const;

 private:
  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  symbolic::Expression hamiltonian_;
  std::vector<symbolic::Expression> cluster_ranks_;
};

// Closed-shell unitary-group CCSD with pair-symmetric spatial amplitudes.
class UGACCSDGenerator {
 public:
  explicit UGACCSDGenerator(
      IntegralConvention convention = IntegralConvention::kPhysicist);

  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] const symbolic::Expression& Hamiltonian() const;
  [[nodiscard]] symbolic::Expression Energy(int order = 2) const;
  [[nodiscard]] symbolic::Expression Singles(int order = 4) const;
  [[nodiscard]] symbolic::Expression Doubles(int order = 4) const;
  [[nodiscard]] std::string GenerateNumpy() const;

 private:
  [[nodiscard]] symbolic::Expression Projected(int order, int rank) const;
  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  symbolic::Expression h_;
  symbolic::Expression singles_;
  symbolic::Expression doubles_;
  symbolic::Expression cluster_;
};

namespace spatial_mp_detail {
inline std::string Axis(int axis, bool occupied) {
  if (axis < 4) {
    return std::string(1, occupied ? "ijkl"[axis] : "abcd"[axis]);
  }
  return std::string(occupied ? "i" : "a") + std::to_string(axis);
}
inline std::string Axes(int rank) {
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
inline std::string Excitations(int rank, bool bra) {
  std::string word;
  for (int axis = 0; axis < rank; ++axis) {
    // Spaces make multi-character index names unambiguous to the parser.
    word += " E1[" + Axis(axis, bra) + (axis < 4 ? "," : ", ") +
        Axis(axis, !bra) + "]";
  }
  return word;
}
} // namespace spatial_mp_detail

inline SpatialMPGenerator::SpatialMPGenerator(
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
    indices_.Add(
        symbolic::OrbitalSpace::kInactive,
        spatial_mp_detail::Axis(axis, true) + " ");
    indices_.Add(
        symbolic::OrbitalSpace::kExternal,
        spatial_mp_detail::Axis(axis, false) + " ");
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
  const auto hamiltonian = UGACCSDGenerator(convention)
                               .Hamiltonian()
                               .Substitute({{"f", canonical_fock}});
  perturbation_ = (hamiltonian - fock_).Simplify();
}

inline symbolic::Expression SpatialMPGenerator::Parse(
    std::string_view text) const {
  return symbolic::Expression::Parse(text, indices_, symmetries_);
}

inline int SpatialMPGenerator::WavefunctionRank(int order) const {
  return maximum_excitation_rank_ == 0
      ? 2 * order
      : std::min(2 * order, maximum_excitation_rank_);
}

inline symbolic::Expression SpatialMPGenerator::Wavefunction(
    int order,
    int rank) const {
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
      Parse("SUM <" + spatial_mp_detail::Axes(rank) + "> u" +
            std::to_string(order) + "[" + spatial_mp_detail::Axes(rank) + "]" +
            spatial_mp_detail::Excitations(rank, false));
}

inline equation::ContractionGraph SpatialMPGenerator::Equations() const {
  equation::ContractionGraph graph;
  auto add = [&](const std::string& output, const symbolic::Expression& value) {
    graph.Add(
        symbolic::Tensor::Parse(output, indices_, symmetries_),
        value.Expand(0).Simplify());
  };
  const auto first = Wavefunction(1, 1) + Wavefunction(1, 2);
  for (int rank = 1; rank <= WavefunctionRank(1); ++rank) {
    add("residual1_rank" + std::to_string(rank) + "[" +
            spatial_mp_detail::Axes(rank) + "]",
        Parse(spatial_mp_detail::Excitations(rank, true)) *
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
      add("residual2_rank" + std::to_string(rank) + "[" +
              spatial_mp_detail::Axes(rank) + "]",
          Parse(spatial_mp_detail::Excitations(rank, true)) *
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
        const auto projector =
            Parse(spatial_mp_detail::Excitations(rank, true));
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
                "[" + spatial_mp_detail::Axes(rank) + "]",
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

inline std::string SpatialMPGenerator::GenerateNumpy(bool optimize) const {
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
namespace spatial_cc_detail {
inline constexpr std::string_view kVirtual = "abcd";
inline constexpr std::string_view kOccupied = "ijkl";

inline std::string ExcitationWord(int rank, bool conjugate) {
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

inline std::string Axes(int rank) {
  return std::string(kVirtual.substr(0, rank)) +
      std::string(kOccupied.substr(0, rank));
}
} // namespace spatial_cc_detail

inline SpatialCCGenerator::SpatialCCGenerator(
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
            "SUM <" + spatial_cc_detail::Axes(rank) + "> t[" +
            spatial_cc_detail::Axes(rank) + "]" +
            spatial_cc_detail::ExcitationWord(rank, false)));
  }
  hamiltonian_ = UGACCSDGenerator(convention).Hamiltonian();
}

inline symbolic::Expression SpatialCCGenerator::Parse(
    std::string_view text) const {
  return symbolic::Expression::Parse(text, indices_, symmetries_);
}

inline const symbolic::Expression& SpatialCCGenerator::Hamiltonian() const {
  return hamiltonian_;
}

inline symbolic::Expression SpatialCCGenerator::Projected(
    int rank,
    int bch_order) const {
  if (rank < 0 || rank > static_cast<int>(cluster_ranks_.size()) ||
      bch_order < 0 || bch_order > 4) {
    throw std::invalid_argument("Invalid spatial CC projection or BCH order");
  }
  const auto projector = rank == 0
      ? Parse("1")
      : Parse(spatial_cc_detail::ExcitationWord(rank, true));
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

inline equation::ContractionGraph SpatialCCGenerator::Equations() const {
  equation::ContractionGraph graph;
  for (int rank = 0; rank <= static_cast<int>(cluster_ranks_.size()); ++rank) {
    const auto name = rank == 0 ? "energy" : "residual" + std::to_string(rank);
    graph.Add(
        symbolic::Tensor::Parse(
            name + "[" + spatial_cc_detail::Axes(rank) + "]",
            indices_,
            symmetries_),
        Projected(rank));
  }
  return graph;
}

inline std::string SpatialCCGenerator::GenerateNumpy(bool optimize) const {
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
namespace uga_ccsd_detail {
inline void ValidateOrder(int order) {
  if (order < 0 || order > 4) {
    throw std::invalid_argument(
        "UGA-CCSD BCH order must be between zero and four");
  }
}
} // namespace uga_ccsd_detail

inline UGACCSDGenerator::UGACCSDGenerator(IntegralConvention convention) {
  indices_.Add(symbolic::OrbitalSpace::kInactive, "pqrsijklmno");
  indices_.Add(symbolic::OrbitalSpace::kExternal, "pqrsabcdefg");
  const bool chemist = convention == IntegralConvention::kChemist;
  symmetries_.Add(
      "v",
      4,
      chemist ? symbolic::TensorSymmetry::QuantumChemistryChemists()
              : symbolic::TensorSymmetry::QuantumChemistryPhysicists());
  symmetries_.Add("t", 4, symbolic::TensorSymmetry::SpinFree(2));
  const auto fock_definition = std::pair(
      symbolic::Tensor::Parse("h[pq]", indices_, symmetries_),
      Parse(
          chemist ? "f[pq]\n-2 SUM <j> v[pqjj]\n+SUM <j> v[pjjq]"
                  : "f[pq]\n-2 SUM <j> v[pjqj]\n+SUM <j> v[pjjq]"));
  const std::map<std::string, std::pair<symbolic::Tensor, symbolic::Expression>>
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

inline symbolic::Expression UGACCSDGenerator::Parse(
    std::string_view text) const {
  return symbolic::Expression::Parse(text, indices_, symmetries_);
}

inline const symbolic::Expression& UGACCSDGenerator::Hamiltonian() const {
  return h_;
}

inline symbolic::Expression UGACCSDGenerator::Energy(int order) const {
  uga_ccsd_detail::ValidateOrder(order);
  auto nested = h_;
  auto transformed = h_;
  for (int degree = 1; degree <= order; ++degree) {
    nested = (1.0 / degree) * Commutator(nested, cluster_);
    transformed = transformed + nested;
  }
  return transformed.Expand(0).Simplify();
}

inline symbolic::Expression UGACCSDGenerator::Projected(int order, int rank)
    const {
  uga_ccsd_detail::ValidateOrder(order);
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

inline symbolic::Expression UGACCSDGenerator::Singles(int order) const {
  return Projected(order, 1);
}

inline symbolic::Expression UGACCSDGenerator::Doubles(int order) const {
  return Projected(order, 2);
}

inline std::string UGACCSDGenerator::GenerateNumpy() const {
  auto render = [&](const symbolic::Expression& expression,
                    std::string_view target) {
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
