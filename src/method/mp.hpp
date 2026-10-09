#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backend/lapack.hpp"
#include "method/rhf.hpp"
#include "runtime/spatial.hpp"

namespace wickqc::method {

struct MPResult {
  std::vector<double> order_energies;
  double correlation_energy = 0;
  double total_energy = 0;
};

namespace mp_detail {

using Pair = std::pair<std::size_t, std::size_t>; // virtual, occupied
using Pairs = std::vector<Pair>;

inline std::string AmplitudeName(int order, int rank) {
  return "u" + std::to_string(order) +
      std::string(static_cast<std::size_t>(rank), 'E') +
      std::string(static_cast<std::size_t>(rank), 'I');
}

inline std::string ResidualName(int order, int rank) {
  return "residual" + std::to_string(order) + "_rank" + std::to_string(rank);
}

inline std::vector<std::size_t> Indices(const Pairs& pairs) {
  std::vector<std::size_t> result(2 * pairs.size());
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    result[k] = pairs[k].first;
    result[k + pairs.size()] = pairs[k].second;
  }
  return result;
}

template <typename Function>
void ForMultisets(std::size_t extent, int rank, Function&& function) {
  std::vector<std::size_t> indices;
  const auto visit = [&](const auto& self, std::size_t first) -> void {
    if (indices.size() == static_cast<std::size_t>(rank)) {
      function(indices);
      return;
    }
    for (std::size_t index = first; index < extent; ++index) {
      indices.push_back(index);
      self(self, index);
      indices.pop_back();
    }
  };
  visit(visit, 0);
}

// E[a,i] = sum_spin c+(a,spin)c(i,spin). Its action on the closed-shell
// determinant gives the exact spin-free overlap metric at any excitation rank.
inline std::map<std::string, double> ExcitedDeterminants(
    const Pairs& pairs,
    std::size_t occupied,
    std::size_t virtuals) {
  std::string reference(2 * (occupied + virtuals), '0');
  std::fill_n(reference.begin(), 2 * occupied, '1');
  std::map<std::string, double> result;
  const auto apply = [&](const auto& self,
                         std::size_t remaining,
                         const std::string& state,
                         double coefficient) -> void {
    if (remaining == 0) {
      result[state] += coefficient;
      return;
    }
    const auto [external, inactive] = pairs[remaining - 1];
    for (std::size_t spin = 0; spin < 2; ++spin) {
      const auto hole = 2 * inactive + spin;
      const auto particle = 2 * (occupied + external) + spin;
      if (state[hole] == '0' || state[particle] == '1') {
        continue;
      }
      std::string next = state;
      const auto before_hole =
          std::count(next.begin(), next.begin() + hole, '1');
      next[hole] = '0';
      const auto before_particle =
          std::count(next.begin(), next.begin() + particle, '1');
      next[particle] = '1';
      const double sign = (before_hole + before_particle) % 2 ? -1.0 : 1.0;
      self(self, remaining - 1, next, coefficient * sign);
    }
  };
  apply(apply, pairs.size(), reference, 1.0);
  return result;
}

inline double Overlap(
    const std::map<std::string, double>& left,
    const std::map<std::string, double>& right) {
  double value = 0;
  for (const auto& [determinant, coefficient] : left) {
    if (const auto found = right.find(determinant); found != right.end()) {
      value += coefficient * found->second;
    }
  }
  return value;
}

inline double OrbitWeight(const Pairs& pairs) {
  double weight = 1;
  for (std::size_t first = 0; first < pairs.size();) {
    std::size_t end = first + 1;
    while (end < pairs.size() && pairs[end] == pairs[first]) {
      weight /= static_cast<double>(end - first + 1);
      ++end;
    }
    first = end;
  }
  return weight;
}

inline std::vector<Pairs> Pairings(
    const std::vector<std::size_t>& external,
    const std::vector<std::size_t>& inactive) {
  auto permutation = external;
  std::set<Pairs> unique;
  do {
    Pairs pairs;
    for (std::size_t k = 0; k < permutation.size(); ++k) {
      pairs.emplace_back(permutation[k], inactive[k]);
    }
    std::ranges::sort(pairs);
    unique.insert(std::move(pairs));
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  return {unique.begin(), unique.end()};
}

inline double ExcitationGap(const RHFData& data, const Pairs& pairs) {
  double gap = 0;
  for (const auto [external, inactive] : pairs) {
    gap += data.orbital_energies.At({data.occupied + external}) -
        data.orbital_energies.At({inactive});
  }
  if (!std::isfinite(gap) || gap <= 0) {
    throw std::invalid_argument("MP excitation has a nonpositive orbital gap");
  }
  return gap;
}

inline void SolveRank(
    const RHFData& data,
    int rank,
    const NDArray<double>& source,
    NDArray<double>& amplitudes) {
  const auto virtuals = data.orbital_energies.Size() - data.occupied;
  ForMultisets(virtuals, rank, [&](const auto& external) {
    ForMultisets(data.occupied, rank, [&](const auto& inactive) {
      const auto pairings = Pairings(external, inactive);
      const auto size = pairings.size();
      std::vector<std::map<std::string, double>> states;
      states.reserve(size);
      for (const auto& pairs : pairings) {
        states.push_back(ExcitedDeterminants(pairs, data.occupied, virtuals));
      }
      const double gap = ExcitationGap(data, pairings.front());
      std::vector<double> metric(size * size), rhs(size);
      for (std::size_t row = 0; row < size; ++row) {
        rhs[row] = -source[Indices(pairings[row])];
        for (std::size_t column = 0; column < size; ++column) {
          metric[row * size + column] = gap * OrbitWeight(pairings[column]) *
              Overlap(states[row], states[column]);
        }
      }
      const auto solution = lapack::LeastSquares(metric, size, size, rhs);
      for (std::size_t row = 0; row < size; ++row) {
        double reconstructed = 0;
        for (std::size_t column = 0; column < size; ++column) {
          reconstructed +=
              metric[row * size + column] * solution.solution[column];
        }
        if (std::abs(reconstructed - rhs[row]) >
            1e-8 * std::max(1.0, std::abs(rhs[row]))) {
          throw std::runtime_error(
              "MP spin-free projected residual lies outside the excitation metric range");
        }
      }
      for (std::size_t column = 0; column < size; ++column) {
        auto pairs = pairings[column];
        do {
          amplitudes[Indices(pairs)] = solution.solution[column];
        } while (std::next_permutation(pairs.begin(), pairs.end()));
      }
    });
  });
}

} // namespace mp_detail

// Compiles each required order once. Inputs() lets AO2MO transform exactly the
// blocks needed by every stage before the numerical solve starts.
class SpatialMPPlan {
 public:
  SpatialMPPlan(
      int order,
      std::size_t occupied,
      std::size_t virtuals,
      IntegralConvention convention = IntegralConvention::kChemist,
      runtime::SpatialEvaluator::KernelLookup lookup = nullptr)
      : order_(order),
        convention_(convention),
        occupied_(occupied),
        virtuals_(virtuals) {
    if (order < 2 || order == std::numeric_limits<int>::max()) {
      throw std::invalid_argument("MP perturbation order must be at least two");
    }
    if (occupied == 0 || virtuals == 0 ||
        occupied >
            static_cast<std::size_t>(std::numeric_limits<int>::max() / 2) ||
        virtuals >
            static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) {
      throw std::invalid_argument(
          "MP occupied/virtual orbital counts are invalid");
    }
    physical_rank_ = static_cast<int>(2 * std::min(occupied, virtuals));
    for (int wavefunction_order = 1; wavefunction_order <= order / 2;
         ++wavefunction_order) {
      stages_.push_back(MakeEvaluator(2 * wavefunction_order, lookup));
      AddInputs(stages_.back().Inputs());
    }
    if (order % 2 == 0) {
      final_ = stages_.size() - 1;
    } else {
      stages_.push_back(MakeEvaluator(order, lookup));
      final_ = stages_.size() - 1;
      AddInputs(stages_.back().Inputs());
    }
  }

  [[nodiscard]] const std::vector<runtime::TensorBinding>& Inputs() const {
    return inputs_;
  }

  [[nodiscard]] MPResult Solve(const RHFData& data, double reference_energy)
      const {
    rhf_detail::Validate(data, reference_energy);
    if (data.occupied != occupied_ ||
        data.orbital_energies.Size() != occupied_ + virtuals_) {
      throw std::invalid_argument("MP data orbital counts differ from plan");
    }
    std::map<std::string, NDArray<double>> amplitudes;
    for (std::size_t stage_index = 0;
         stage_index < stages_.size() - (order_ % 2);
         ++stage_index) {
      const auto& stage = stages_[stage_index];
      const int wavefunction_order = static_cast<int>(stage_index) + 1;
      for (const auto& binding : stage.Inputs()) {
        if (binding.name.starts_with("u") &&
            !amplitudes.contains(binding.name)) {
          amplitudes.emplace(
              binding.name, NDArray<double>(binding.Shape(data.Dimensions())));
        }
      }
      const auto source = stage.Evaluate(
          data.Bind(stage.Inputs(), amplitudes, convention_),
          data.Dimensions());
      for (int rank = 1; rank <= 2 * wavefunction_order; ++rank) {
        const auto name = mp_detail::AmplitudeName(wavefunction_order, rank);
        const auto residual =
            source.find(mp_detail::ResidualName(wavefunction_order, rank));
        if (residual != source.end() && rank <= physical_rank_) {
          mp_detail::SolveRank(
              data, rank, residual->second, amplitudes.at(name));
        }
      }
    }
    const auto& final = stages_[final_];
    const auto values = final.Evaluate(
        data.Bind(final.Inputs(), amplitudes, convention_), data.Dimensions());
    MPResult result;
    for (int n = 2; n <= order_; ++n) {
      const double energy = values.at("energy" + std::to_string(n)).Item();
      if (!std::isfinite(energy)) {
        throw std::runtime_error(
            "Non-finite MP" + std::to_string(n) + " energy");
      }
      result.order_energies.push_back(energy);
      result.correlation_energy += energy;
    }
    result.total_energy = reference_energy + result.correlation_energy;
    return result;
  }

 private:
  [[nodiscard]] runtime::SpatialEvaluator MakeEvaluator(
      int target,
      runtime::SpatialEvaluator::KernelLookup lookup) const {
    const int bound = target <= 4 ? 0 : physical_rank_;
    return runtime::SpatialEvaluator(
        {SpatialFamily::kMP, target, convention_, bound},
        target <= 4 ? runtime::GenerationPolicy::kPreferPrecompiled
                    : runtime::GenerationPolicy::kRuntimeOnly,
        lookup);
  }

  void AddInputs(std::span<const runtime::TensorBinding> bindings) {
    for (const auto& binding : bindings) {
      if (std::ranges::none_of(inputs_, [&](const auto& existing) {
            return existing.name == binding.name;
          })) {
        inputs_.push_back(binding);
      }
    }
  }

  int order_;
  int physical_rank_ = 0;
  IntegralConvention convention_;
  std::size_t occupied_ = 0, virtuals_ = 0;
  std::size_t final_ = 0;
  std::vector<runtime::SpatialEvaluator> stages_;
  std::vector<runtime::TensorBinding> inputs_;
};

inline MPResult SolveSpatialMP(
    const RHFData& data,
    int order,
    double reference_energy,
    IntegralConvention convention = IntegralConvention::kChemist,
    runtime::SpatialEvaluator::KernelLookup lookup = nullptr) {
  (void)data.Dimensions();
  const auto plan = SpatialMPPlan(
      order,
      data.occupied,
      data.orbital_energies.Size() - data.occupied,
      convention,
      lookup);
  return plan.Solve(data, reference_energy);
}

} // namespace wickqc::method
