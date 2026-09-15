#include "symbolic/wick.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cctype>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <ios>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include "symbolic/index_domain.h"

namespace wickqc::symbolic {
namespace {

constexpr std::uint8_t ToMask(OrbitalSpace space) {
  return static_cast<std::uint8_t>(space);
}

constexpr std::uint8_t ToMask(Spin spin) {
  return static_cast<std::uint8_t>(spin);
}

std::string Trim(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r");
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = text.find_last_not_of(" \t\r");
  return std::string(text.substr(first, last - first + 1));
}

std::vector<std::string> SplitIndexNames(std::string_view text) {
  std::vector<std::string> names;
  const bool has_separator =
      text.find_first_of(" \t") != std::string_view::npos;
  if (!has_separator) {
    for (char character : text) {
      if (character != ',') {
        names.emplace_back(1, character);
      }
    }
    return names;
  }

  std::string name;
  for (char character : text) {
    if (character == ',' || character == ' ' || character == '\t') {
      if (!name.empty()) {
        names.push_back(std::move(name));
        name.clear();
      }
    } else {
      name.push_back(character);
    }
  }
  if (!name.empty()) {
    names.push_back(std::move(name));
  }
  return names;
}

SignedPermutation Compose(
    const SignedPermutation& lhs,
    const SignedPermutation& rhs) {
  assert(lhs.order.size() == rhs.order.size());
  SignedPermutation result;
  result.order.resize(lhs.order.size());
  result.sign = lhs.sign * rhs.sign;
  for (std::size_t index = 0; index < result.order.size(); ++index) {
    result.order[index] = lhs.order[rhs.order[index]];
  }
  return result;
}

std::string DomainKey(const IndexDomain& domain) {
  return std::to_string(domain.orbital_spaces) + ":" +
      std::to_string(domain.spins);
}

std::string IndexKey(const Index& index) {
  return DomainKey(index.domain) + ":" + index.name;
}

std::string TensorKey(const Tensor& tensor) {
  std::ostringstream output;
  output << static_cast<int>(tensor.kind) << ':' << tensor.name << '[';
  for (const auto& index : tensor.indices) {
    output << IndexKey(index) << ';';
  }
  output << ']';
  return output.str();
}

std::string TermKey(const Term& term) {
  std::ostringstream output;
  for (const auto& index : term.summed_indices) {
    output << "S{" << IndexKey(index) << '}';
  }
  for (const auto& tensor : term.tensors) {
    output << "T{" << TensorKey(tensor) << '}';
  }
  return output.str();
}

std::vector<Index> AllIndices(const Term& term) {
  std::vector<Index> result = term.summed_indices;
  for (const auto& tensor : term.tensors) {
    result.insert(result.end(), tensor.indices.begin(), tensor.indices.end());
  }
  return result;
}

bool IsSummed(const Term& term, const Index& index) {
  return std::ranges::find(term.summed_indices, index) !=
      term.summed_indices.end();
}

void EraseSummed(Term& term, const Index& index) {
  std::erase(term.summed_indices, index);
}

void AddDeltaTensor(Term& term, const Index& lhs, const Index& rhs) {
  Tensor delta;
  delta.name = "delta";
  delta.indices = {lhs, rhs};
  delta.kind = TensorKind::kDelta;
  delta.symmetry = TensorSymmetry::TwoSymmetric();
  term.tensors.push_back(std::move(delta));
}

void AppendDeltas(
    Term& term,
    const std::vector<std::pair<Index, Index>>& deltas) {
  // Expansion records the matching. Index elimination belongs to Simplify,
  // after all contraction deltas and uncontracted operators are present.
  for (const auto& [lhs, rhs] : deltas) {
    AddDeltaTensor(term, lhs, rhs);
  }
}

bool TensorRepresentativeLess(const Tensor& lhs, const Tensor& rhs);

bool IsExternal(const IndexDomain& domain) {
  return (domain.orbital_spaces & ToMask(OrbitalSpace::kExternal)) != 0;
}

bool ContractionDomainsMatch(const IndexDomain& lhs, const IndexDomain& rhs) {
  // An unspecified type is not a wildcard in a Wick contraction. General
  // indices contract with general indices; typed indices need a common space
  // and exactly the same explicit spin flags.
  return lhs.spins == rhs.spins &&
      ((lhs.orbital_spaces == 0 && rhs.orbital_spaces == 0) ||
       (lhs.orbital_spaces & rhs.orbital_spaces) != 0);
}

struct WickExpansion {
  int sign = 1;
  std::vector<Tensor> uncontracted;
  std::vector<std::pair<Index, Index>> deltas;
};

struct SpinOperator {
  TensorKind kind = TensorKind::kGeneric;
  Index index;
  std::size_t spin_label = 0;
  std::size_t source_position = 0;
};

struct SpinWickExpansion {
  int sign = 1;
  std::vector<SpinOperator> uncontracted;
  std::vector<std::pair<Index, Index>> orbital_deltas;
  std::vector<std::pair<std::size_t, std::size_t>> spin_deltas;
  std::vector<std::pair<std::size_t, std::size_t>> contractions;
};

class DisjointSet {
 public:
  explicit DisjointSet(std::size_t size) : parent_(size) {
    for (std::size_t index = 0; index < size; ++index) {
      parent_[index] = index;
    }
  }

  std::size_t Find(std::size_t index) {
    if (parent_[index] != index) {
      parent_[index] = Find(parent_[index]);
    }
    return parent_[index];
  }

  void Unite(std::size_t lhs, std::size_t rhs) {
    lhs = Find(lhs);
    rhs = Find(rhs);
    if (lhs != rhs) {
      parent_[std::max(lhs, rhs)] = std::min(lhs, rhs);
    }
  }

 private:
  std::vector<std::size_t> parent_;
};

Tensor TaggedOperator(const SpinOperator& op) {
  return {
      (op.kind == TensorKind::kCreation ? "C" : "D") +
          std::to_string(op.spin_label),
      {op.index},
      op.kind,
      TensorSymmetry::None(1)};
}

bool CanContract(
    const SpinOperator& lhs,
    const SpinOperator& rhs,
    bool spin_free) {
  const auto& a = lhs.index.domain;
  const auto& b = rhs.index.domain;
  const bool compatible = spin_free
      ? ((a.orbital_spaces & b.orbital_spaces) | (a.spins & b.spins)) != 0
      : ContractionDomainsMatch(a, b);
  return compatible && lhs.kind != rhs.kind &&
      TensorRepresentativeLess(TaggedOperator(rhs), TaggedOperator(lhs));
}

// Sorting the whole operator word first fixes the order for every residual
// subsequence. In explicit spin-free output, nonactive lines precede active
// lines and the two halves have opposite spin-label order.
std::vector<std::size_t> SpinOperatorOrder(
    const std::vector<SpinOperator>& operators,
    bool compact) {
  std::vector<std::size_t> order(operators.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  auto by_tensor = [&](auto a, auto b) {
    return TensorRepresentativeLess(
        TaggedOperator(operators[a]), TaggedOperator(operators[b]));
  };
  if (compact) {
    std::ranges::stable_sort(order, [&](auto a, auto b) {
      return operators[a].kind < operators[b].kind;
    });
    return order;
  }
  std::ranges::stable_sort(order, by_tensor);
  auto active = [&](auto position) {
    return (operators[position].index.domain.orbital_spaces &
            (ToMask(OrbitalSpace::kActive) | ToMask(OrbitalSpace::kSingle))) !=
        0;
  };
  auto boundary = std::stable_partition(
      order.begin(), order.end(), [&](auto i) { return !active(i); });
  const auto boundary_offset = std::min<std::ptrdiff_t>(
      static_cast<std::ptrdiff_t>(order.size() / 2), boundary - order.begin());
  const auto half = order.begin() + boundary_offset;
  auto by_spin = [&](auto a, auto b, bool ascending) {
    auto ta = TaggedOperator(operators[a]);
    auto tb = TaggedOperator(operators[b]);
    // The Fermi class comparison is extracted below, independent of names.
    const auto sa = ta.indices.front().domain.orbital_spaces;
    const auto sb = tb.indices.front().domain.orbital_spaces;
    auto occupied = std::min(sa, sb);
    if (occupied == 0 || occupied == ToMask(OrbitalSpace::kExternal) ||
        (occupied == ToMask(OrbitalSpace::kActive) && sa == sb)) {
      occupied = ToMask(OrbitalSpace::kInactive);
    }
    auto fermi = [&](const SpinOperator& op) {
      const int annihilation = op.kind == TensorKind::kAnnihilation;
      const int hole = (op.index.domain.orbital_spaces & occupied) != 0;
      return annihilation | ((annihilation ^ hole) << 1);
    };
    if (fermi(operators[a]) != fermi(operators[b])) {
      return fermi(operators[a]) < fermi(operators[b]);
    }
    if (operators[a].spin_label != operators[b].spin_label) {
      return ascending ? operators[a].spin_label < operators[b].spin_label
                       : operators[a].spin_label > operators[b].spin_label;
    }
    return by_tensor(a, b);
  };
  std::stable_sort(
      order.begin(), half, [&](auto a, auto b) { return by_spin(a, b, true); });
  std::stable_sort(
      half, boundary, [&](auto a, auto b) { return by_spin(a, b, false); });
  std::stable_sort(boundary, order.end(), [&](auto a, auto b) {
    return by_spin(a, b, false);
  });
  return order;
}

using SpinContractionPair = std::pair<std::size_t, std::size_t>;

struct InactiveEndpoints {
  std::set<std::size_t> left;
  std::set<std::size_t> right;

  bool UnmatchedOnBothSides(
      const std::vector<bool>& contracted,
      std::size_t before_left,
      std::size_t before_right) const {
    const auto unmatched = [&](const auto& endpoints, std::size_t before) {
      const auto completed = std::ranges::count_if(
          endpoints,
          [&](std::size_t position) { return contracted[position]; });
      const auto suffix_capacity = std::ranges::count_if(
          endpoints, [&](std::size_t position) { return position >= before; });
      return completed + suffix_capacity < endpoints.size();
    };
    return unmatched(left, before_left) && unmatched(right, before_right);
  }
};

int SpinContractionSign(const std::vector<SpinContractionPair>& contractions) {
  int parity = 0;
  for (std::size_t level = 0; level < contractions.size(); ++level) {
    const auto [creation, annihilation] = contractions[level];
    parity ^= static_cast<int>(((creation ^ annihilation) & 1U) ^ 1U);
    for (std::size_t previous = 0; previous < level; ++previous) {
      const auto [lhs, rhs] = contractions[previous];
      parity ^= static_cast<int>(
          (lhs < creation && rhs > creation && rhs < annihilation) ||
          (lhs > creation && lhs < annihilation && rhs > annihilation));
    }
  }
  return parity == 0 ? 1 : -1;
}

void TraverseSpinContractions(
    const std::vector<SpinOperator>& operators,
    const std::vector<SpinContractionPair>& candidates,
    std::size_t candidate_begin,
    std::vector<SpinContractionPair>& contractions,
    std::vector<bool>& contracted,
    int sign,
    bool skip_contractions,
    const InactiveEndpoints* inactive,
    const std::vector<std::size_t>& order,
    int max_uncontracted,
    std::vector<SpinWickExpansion>& expansions) {
  // The ordered inactive-endpoint rule uses separate left and right cutoffs.
  // Checking only the final residual operators retains additional branches.
  if (inactive && !contractions.empty() &&
      inactive->UnmatchedOnBothSides(
          contracted,
          contractions.back().first + 1,
          contractions.back().second + 1)) {
    return;
  }
  const bool representable = !inactive ||
      !inactive->UnmatchedOnBothSides(
          contracted, operators.size(), operators.size());
  // As in the reference traversal, continue searching deeper matchings but
  // do not materialize expansions excluded by the requested residual rank.
  const auto remaining = operators.size() - contractions.size() * 2;
  if (representable &&
      (max_uncontracted < 0 ||
       remaining <= static_cast<std::size_t>(max_uncontracted))) {
    std::vector<SpinOperator> uncontracted;
    uncontracted.reserve(operators.size() - contractions.size() * 2);
    int order_sign = 1;
    for (auto position : order) {
      if (!contracted[position]) {
        for (const auto& previous : uncontracted) {
          if (previous.source_position > position) {
            order_sign = -order_sign;
          }
        }
        uncontracted.push_back(operators[position]);
      }
    }

    std::vector<std::pair<Index, Index>> orbital_deltas;
    std::vector<std::pair<std::size_t, std::size_t>> spin_deltas;
    orbital_deltas.reserve(contractions.size());
    spin_deltas.reserve(contractions.size());
    for (const auto [lhs, rhs] : contractions) {
      orbital_deltas.emplace_back(operators[lhs].index, operators[rhs].index);
      spin_deltas.emplace_back(
          operators[lhs].spin_label, operators[rhs].spin_label);
    }
    expansions.push_back(
        {sign * SpinContractionSign(contractions) * order_sign,
         std::move(uncontracted),
         std::move(orbital_deltas),
         std::move(spin_deltas),
         contractions});
  }

  if (skip_contractions) {
    return;
  }
  for (std::size_t candidate = candidates.size();
       candidate-- > candidate_begin;) {
    const auto [lhs, rhs] = candidates[candidate];
    if (contracted[lhs] || contracted[rhs]) {
      continue;
    }
    contracted[lhs] = true;
    contracted[rhs] = true;
    contractions.push_back(candidates[candidate]);

    std::size_t next_begin = candidate_begin;
    while (next_begin < candidates.size() &&
           candidates[next_begin].first <= lhs) {
      ++next_begin;
    }
    TraverseSpinContractions(
        operators,
        candidates,
        next_begin,
        contractions,
        contracted,
        sign,
        false,
        inactive,
        order,
        max_uncontracted,
        expansions);

    contractions.pop_back();
    contracted[lhs] = false;
    contracted[rhs] = false;
  }
}

void EnumerateSpinWick(
    const std::vector<SpinOperator>& operators,
    int sign,
    bool skip_contractions,
    std::vector<SpinWickExpansion>& expansions,
    bool spin_free = false,
    bool compact = true,
    std::vector<bool>* single_endpoints = nullptr,
    const std::vector<std::size_t>* explicit_order = nullptr,
    int max_uncontracted = -1) {
  std::vector<SpinContractionPair> candidates;
  InactiveEndpoints inactive;
  for (std::size_t lhs = 0; lhs < operators.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < operators.size(); ++rhs) {
      if (CanContract(operators[lhs], operators[rhs], spin_free)) {
        if (single_endpoints &&
            (operators[lhs].index.domain.orbital_spaces &
             operators[rhs].index.domain.orbital_spaces &
             ToMask(OrbitalSpace::kSingle)) != 0) {
          (*single_endpoints)[lhs] = (*single_endpoints)[rhs] = true;
        }
        candidates.emplace_back(lhs, rhs);
        if ((operators[lhs].index.domain.orbital_spaces &
             operators[rhs].index.domain.orbital_spaces &
             ToMask(OrbitalSpace::kInactive)) != 0) {
          inactive.left.insert(lhs);
          inactive.right.insert(rhs);
        }
      }
    }
  }

  std::vector<SpinContractionPair> contractions;
  std::vector<bool> contracted(operators.size(), false);
  TraverseSpinContractions(
      operators,
      candidates,
      0,
      contractions,
      contracted,
      sign,
      skip_contractions,
      spin_free && compact ? &inactive : nullptr,
      explicit_order ? *explicit_order : SpinOperatorOrder(operators, compact),
      max_uncontracted,
      expansions);
}

void EnumerateWick(
    const std::vector<Tensor>& operators,
    bool skip_contractions,
    std::vector<WickExpansion>& expansions) {
  // Use the same ordered contraction matchings for both operator notations.
  // The label retains the source position; spin summation is not used here.
  std::vector<SpinOperator> positions;
  positions.reserve(operators.size());
  for (std::size_t i = 0; i < operators.size(); ++i) {
    positions.push_back(
        {operators[i].kind, operators[i].indices.front(), i, i});
  }
  std::vector<std::size_t> order(operators.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::ranges::stable_sort(order, [&](auto a, auto b) {
    return TensorRepresentativeLess(operators[a], operators[b]);
  });
  std::vector<SpinWickExpansion> matchings;
  EnumerateSpinWick(
      positions, 1, skip_contractions, matchings, false, true, nullptr, &order);
  for (auto& matching : matchings) {
    std::vector<Tensor> remaining;
    remaining.reserve(matching.uncontracted.size());
    for (const auto& position : matching.uncontracted) {
      remaining.push_back(operators[position.spin_label]);
    }
    expansions.push_back(
        {matching.sign,
         std::move(remaining),
         std::move(matching.orbital_deltas)});
  }
}

struct SpinFreeRemainder {
  int sign = 1;
  std::size_t closed_spin_loops = 0;
  std::optional<Tensor> tensor;
};

std::optional<SpinFreeRemainder> AnalyzeSpinFreeRemainder(
    const SpinWickExpansion& expansion,
    std::size_t spin_count) {
  DisjointSet spins(spin_count);
  for (const auto& [lhs, rhs] : expansion.spin_deltas) {
    spins.Unite(lhs, rhs);
  }

  std::set<std::tuple<TensorKind, std::string, std::size_t>> seen_operators;
  std::map<std::size_t, std::vector<std::size_t>> creators_by_spin;
  std::map<std::size_t, std::vector<std::size_t>> annihilators_by_spin;
  bool annihilation_seen = false;
  std::vector<std::size_t> creator_positions;
  std::vector<std::size_t> annihilator_positions;
  for (std::size_t position = 0; position < expansion.uncontracted.size();
       ++position) {
    const auto& spin_operator = expansion.uncontracted[position];
    const std::size_t root = spins.Find(spin_operator.spin_label);
    if (!seen_operators
             .insert({spin_operator.kind, IndexKey(spin_operator.index), root})
             .second) {
      return std::nullopt;
    }
    if (spin_operator.kind == TensorKind::kCreation) {
      if (annihilation_seen) {
        return std::nullopt;
      }
      creators_by_spin[root].push_back(position);
      creator_positions.push_back(position);
    } else {
      annihilation_seen = true;
      annihilators_by_spin[root].push_back(position);
      annihilator_positions.push_back(position);
    }
  }

  std::set<std::size_t> all_roots;
  for (std::size_t label = 0; label < spin_count; ++label) {
    all_roots.insert(spins.Find(label));
  }
  std::size_t closed_spin_loops = 0;
  for (const auto root : all_roots) {
    const std::size_t creator_count = creators_by_spin[root].size();
    const std::size_t annihilator_count = annihilators_by_spin[root].size();
    if (creator_count == 0 && annihilator_count == 0) {
      ++closed_spin_loops;
    } else if (creator_count != 1 || annihilator_count != 1) {
      return std::nullopt;
    }
  }

  if (creator_positions.size() != annihilator_positions.size()) {
    return std::nullopt;
  }
  if (creator_positions.empty()) {
    return SpinFreeRemainder{1, closed_spin_loops, std::nullopt};
  }

  std::vector<std::size_t> creator_roots;
  creator_roots.reserve(creator_positions.size());
  for (const auto position : creator_positions) {
    creator_roots.push_back(
        spins.Find(expansion.uncontracted[position].spin_label));
  }
  std::vector<std::size_t> annihilator_roots;
  annihilator_roots.reserve(annihilator_positions.size());
  std::map<std::size_t, Index> annihilator_index_by_root;
  for (const auto position : annihilator_positions) {
    const std::size_t root =
        spins.Find(expansion.uncontracted[position].spin_label);
    annihilator_roots.push_back(root);
    annihilator_index_by_root[root] = expansion.uncontracted[position].index;
  }

  std::vector<std::size_t> desired_annihilator_roots(
      creator_roots.rbegin(), creator_roots.rend());
  std::map<std::size_t, std::size_t> current_position;
  for (std::size_t position = 0; position < annihilator_roots.size();
       ++position) {
    current_position[annihilator_roots[position]] = position;
  }
  std::size_t inversions = 0;
  for (std::size_t i = 0; i < desired_annihilator_roots.size(); ++i) {
    for (std::size_t j = i + 1; j < desired_annihilator_roots.size(); ++j) {
      inversions += current_position.at(desired_annihilator_roots[i]) >
          current_position.at(desired_annihilator_roots[j]);
    }
  }

  Tensor tensor;
  tensor.name = "E" + std::to_string(creator_positions.size());
  tensor.kind = TensorKind::kSpinFree;
  tensor.symmetry = TensorSymmetry::SpinFree(creator_positions.size());
  for (const auto position : creator_positions) {
    tensor.indices.push_back(expansion.uncontracted[position].index);
  }
  for (const auto root : creator_roots) {
    tensor.indices.push_back(annihilator_index_by_root.at(root));
  }
  return SpinFreeRemainder{
      inversions % 2 == 0 ? 1 : -1, closed_spin_loops, std::move(tensor)};
}

struct SpinWord {
  Term scalar;
  std::vector<SpinOperator> operators;
  std::size_t spin_count = 0;
  int sign = 1;
};

SpinWord ResolveSpinWord(const Term& source) {
  SpinWord word;
  word.scalar = source;
  word.scalar.tensors.clear();
  std::map<int, std::vector<std::size_t>> tagged_positions;
  for (const auto& tensor : source.tensors) {
    if (tensor.IsFermionOperator()) {
      tagged_positions[tensor.SpinTag()].push_back(word.operators.size());
      word.operators.push_back({tensor.kind, tensor.indices.front(), 0});
    } else if (tensor.kind == TensorKind::kSpinFree) {
      if (tensor.indices.size() % 2 != 0) {
        throw std::invalid_argument("Spin-free operator must have even rank");
      }
      const auto rank = tensor.indices.size() / 2;
      // E/R notation stores creators followed by their paired annihilators;
      // the physical word reverses the annihilator half.
      for (std::size_t i = 0; i < 2 * rank; ++i) {
        word.operators.push_back(
            {i < rank ? TensorKind::kCreation : TensorKind::kAnnihilation,
             tensor.indices[i],
             word.spin_count + i % rank});
      }
      if ((rank * (rank - 1) / 2) % 2 != 0) {
        word.sign = -word.sign;
      }
      word.spin_count += rank;
    } else {
      word.scalar.tensors.push_back(tensor);
    }
  }
  for (const auto& [tag, positions] : tagged_positions) {
    if (positions.size() != 2) {
      throw std::invalid_argument(
          "Summed spin label " + std::to_string(tag) +
          " must occur on exactly two operators");
    }
    for (auto position : positions) {
      word.operators[position].spin_label = word.spin_count;
    }
    ++word.spin_count;
  }
  // Spin sums have no externally significant numbers. Number the lines by
  // their creation/annihilation indices, with stable ties in word order.
  std::vector<std::vector<std::size_t>> lines(word.spin_count);
  for (std::size_t i = 0; i < word.operators.size(); ++i) {
    word.operators[i].source_position = i;
    lines[word.operators[i].spin_label].push_back(i);
  }
  std::ranges::sort(lines, [](const auto& a, const auto& b) {
    return a.front() < b.front();
  });
  std::ranges::stable_sort(lines, [&](const auto& a, const auto& b) {
    auto key = [&](const auto& line) {
      const auto& first = word.operators[line[0]];
      const auto& second = word.operators[line[1]];
      return first.kind == TensorKind::kCreation
          ? std::pair(first.index, second.index)
          : std::pair(second.index, first.index);
    };
    return key(a) < key(b);
  });
  for (std::size_t label = 0; label < lines.size(); ++label) {
    for (auto position : lines[label]) {
      word.operators[position].spin_label = label;
    }
  }
  return word;
}

std::vector<Term> ExpandSpinFreeTerm(
    const Term& source,
    int max_uncontracted,
    bool skip_contractions,
    bool compact) {
  const auto word = ResolveSpinWord(source);
  const auto& operators = word.operators;
  std::vector<SpinWickExpansion> expansions;
  std::vector<bool> single_endpoints(operators.size());
  EnumerateSpinWick(
      operators,
      1,
      skip_contractions,
      expansions,
      true,
      compact,
      &single_endpoints,
      nullptr,
      max_uncontracted);
  std::vector<Term> result;
  for (const auto& expansion : expansions) {
    DisjointSet spins(word.spin_count);
    std::vector<bool> single_line(word.spin_count);
    for (std::size_t i = 0; i < operators.size(); ++i) {
      single_line[operators[i].spin_label] =
          single_line[operators[i].spin_label] || single_endpoints[i];
    }
    double loop_weight = 1.0;
    std::vector<std::size_t> single_closures;
    for (std::size_t i = 0; i < expansion.contractions.size(); ++i) {
      const auto [a, b] = expansion.contractions[i];
      const auto left = spins.Find(operators[a].spin_label);
      const auto right = spins.Find(operators[b].spin_label);
      const bool single = single_line[left] || single_line[right];
      if (left == right) {
        if (single) {
          single_closures.push_back(i);
        } else {
          loop_weight *= 2.0;
        }
      }
      spins.Unite(left, right);
      single_line[spins.Find(left)] = single;
    }

    Term term = word.scalar;
    term.coefficient *= word.sign * expansion.sign * loop_weight;
    AppendDeltas(term, expansion.orbital_deltas);
    std::vector<Tensor> residual;
    std::map<std::size_t, std::size_t> residual_tags;
    for (auto op : expansion.uncontracted) {
      if (!compact) {
        const auto root = spins.Find(op.spin_label);
        auto [it, inserted] =
            residual_tags.try_emplace(root, residual_tags.size());
        op.spin_label = it->second;
      }
      residual.push_back(TaggedOperator(op));
    }
    if (compact) {
      const auto remainder =
          AnalyzeSpinFreeRemainder(expansion, word.spin_count);
      if (!remainder) {
        continue;
      }
      term.coefficient *= remainder->sign;
      if (remainder->tensor) {
        term.tensors.push_back(*remainder->tensor);
      }
    } else {
      term.tensors.insert(term.tensors.end(), residual.begin(), residual.end());
    }

    // A closed single-occupancy line contributes a delta and the symmetric
    // half-weighted pair. Preserve individual branches, since Expand is a
    // matching enumeration and Simplify owns term aggregation.
    std::size_t orientation_count = 1;
    if (single_closures.size() >= std::numeric_limits<std::size_t>::digits) {
      throw std::overflow_error("Too many single-occupancy loops to expand");
    }
    const auto choices = std::size_t{1} << single_closures.size();
    for (std::size_t subset = 1; subset < choices; ++subset) {
      // The reference enumerator retains repeated orientations across subsets.
      // Their total weight is unchanged; retain that raw expansion convention.
      const auto multiplicity = std::popcount(subset);
      if (orientation_count >
          (std::numeric_limits<std::size_t>::max() >> multiplicity)) {
        throw std::overflow_error(
            "Single-occupancy expansion exceeds addressable size");
      }
      orientation_count <<= multiplicity;
      std::set<std::size_t> replaced;
      for (std::size_t bit = 0; bit < single_closures.size(); ++bit) {
        if ((subset & (std::size_t{1} << bit)) != 0) {
          replaced.insert(single_closures[bit]);
        }
      }
      for (std::size_t orientation = 0; orientation < orientation_count;
           ++orientation) {
        Term correction = word.scalar;
        correction.coefficient =
            term.coefficient / static_cast<double>(orientation_count);
        for (std::size_t i = 0; i < expansion.orbital_deltas.size(); ++i) {
          if (!replaced.contains(i)) {
            const auto& [a, b] = expansion.orbital_deltas[i];
            AddDeltaTensor(correction, a, b);
          }
        }
        auto directions = orientation;
        for (std::size_t bit = 0; bit < single_closures.size(); ++bit) {
          const auto closure = single_closures[bit];
          if (!replaced.contains(closure)) {
            continue;
          }
          const auto [a, b] = expansion.contractions[closure];
          auto left = operators[a];
          auto right = operators[b];
          if (!compact) {
            left.spin_label = right.spin_label = residual_tags.size() + bit;
          }
          if ((directions & 1U) == 0) {
            std::swap(left, right);
          }
          correction.tensors.push_back(TaggedOperator(left));
          correction.tensors.push_back(TaggedOperator(right));
          directions >>= 1;
        }
        correction.tensors.insert(
            correction.tensors.end(), residual.begin(), residual.end());
        result.push_back(std::move(correction));
      }
    }
    result.push_back(std::move(term));
  }
  return result;
}

std::vector<IndexDomain> ConcreteDomains(IndexDomain domain) {
  std::vector<std::uint8_t> orbital_spaces;
  if (domain.orbital_spaces == 0) {
    orbital_spaces.push_back(0);
  } else {
    for (const auto space :
         {OrbitalSpace::kInactive,
          OrbitalSpace::kActive,
          OrbitalSpace::kSingle,
          OrbitalSpace::kExternal}) {
      const auto mask = ToMask(space);
      if ((domain.orbital_spaces & mask) != 0) {
        orbital_spaces.push_back(mask);
      }
    }
  }

  std::vector<std::uint8_t> spins;
  if (domain.spins == 0) {
    spins.push_back(0);
  } else {
    for (const auto spin : {Spin::kAlpha, Spin::kBeta}) {
      const auto mask = ToMask(spin);
      if ((domain.spins & mask) != 0) {
        spins.push_back(mask);
      }
    }
  }

  std::vector<IndexDomain> result;
  for (const auto orbital_space : orbital_spaces) {
    for (const auto spin : spins) {
      result.push_back({orbital_space, spin});
    }
  }
  return result;
}

void EnumerateDomainAssignments(
    const Term& term,
    const std::vector<Index>& names,
    const std::map<Index, std::vector<IndexDomain>>& options,
    std::size_t position,
    std::map<Index, IndexDomain>& assignment,
    std::vector<Term>& result) {
  if (position == 0) {
    Term concrete = term;
    for (auto& tensor : concrete.tensors) {
      for (auto& index : tensor.indices) {
        index.domain = assignment.at(index);
      }
    }
    for (auto& index : concrete.summed_indices) {
      index.domain = assignment.at(index);
    }
    std::ranges::sort(concrete.summed_indices);
    concrete.summed_indices.erase(
        std::unique(
            concrete.summed_indices.begin(), concrete.summed_indices.end()),
        concrete.summed_indices.end());
    result.push_back(std::move(concrete));
    return;
  }

  const Index& name = names[position - 1];
  for (const auto domain : options.at(name)) {
    assignment[name] = domain;
    EnumerateDomainAssignments(
        term, names, options, position - 1, assignment, result);
  }
}

std::vector<Term> SplitTermDomains(const Term& term) {
  if (term.coefficient == 0.0 ||
      std::ranges::any_of(term.tensors, [](const auto& tensor) {
        return tensor.symmetry.Elements().empty();
      })) {
    return {};
  }
  std::map<Index, std::vector<IndexDomain>> options;
  for (const auto& index : AllIndices(term)) {
    auto& alternatives = options[index];
    alternatives = IsSummed(term, index)
        ? ConcreteDomains({index.domain.orbital_spaces, 0})
        : std::vector<IndexDomain>{index.domain};
    for (auto& domain : alternatives) {
      domain.spins = index.domain.spins;
    }
  }
  std::vector<Index> names;
  names.reserve(options.size());
  for (const auto& [name, domains] : options) {
    static_cast<void>(domains);
    names.push_back(name);
  }
  std::map<Index, IndexDomain> assignment;
  std::vector<Term> result;
  EnumerateDomainAssignments(
      term, names, options, names.size(), assignment, result);
  return result;
}

void CanonicalizeTensor(Tensor& tensor, double& coefficient) {
  std::vector<Index> best = tensor.indices;
  int best_sign = 1;
  for (const auto& permutation : tensor.symmetry.Elements()) {
    if (permutation.order.size() != tensor.indices.size()) {
      continue;
    }
    std::vector<Index> candidate(tensor.indices.size());
    for (std::size_t index = 0; index < candidate.size(); ++index) {
      candidate[index] = tensor.indices[permutation.order[index]];
    }
    if (candidate < best) {
      best = std::move(candidate);
      best_sign = permutation.sign;
    }
  }
  tensor.indices = std::move(best);
  coefficient *= best_sign;
}

int TensorStorageKind(TensorKind kind) {
  switch (kind) {
    case TensorKind::kCreation:
      return 0;
    case TensorKind::kAnnihilation:
      return 1;
    case TensorKind::kSpinFree:
      return 2;
    case TensorKind::kDelta:
      return 3;
    case TensorKind::kGeneric:
      return 4;
  }
  return 4;
}

int FermionSortOrder(const Tensor& tensor, std::uint8_t occupied_space) {
  const int annihilation = tensor.kind == TensorKind::kAnnihilation;
  const int occupied = !tensor.indices.empty() &&
      (tensor.indices.front().domain.orbital_spaces & occupied_space) != 0;
  return annihilation | ((annihilation ^ occupied) << 1);
}

bool TensorRepresentativeLess(const Tensor& lhs, const Tensor& rhs) {
  const int comparison = lhs.CompareFermiClass(rhs);
  if (comparison != 0) {
    return comparison < 0;
  }
  if (lhs.name != rhs.name) {
    return lhs.name < rhs.name;
  }
  if (lhs.kind != rhs.kind) {
    return TensorStorageKind(lhs.kind) < TensorStorageKind(rhs.kind);
  }
  return lhs.indices < rhs.indices;
}

bool TermRepresentativeLess(const Term& lhs, const Term& rhs) {
  if (lhs.tensors.size() != rhs.tensors.size()) {
    return lhs.tensors.size() < rhs.tensors.size();
  }
  if (lhs.summed_indices.size() != rhs.summed_indices.size()) {
    return lhs.summed_indices.size() < rhs.summed_indices.size();
  }
  if (std::lexicographical_compare(
          lhs.tensors.begin(),
          lhs.tensors.end(),
          rhs.tensors.begin(),
          rhs.tensors.end(),
          TensorRepresentativeLess)) {
    return true;
  }
  if (std::lexicographical_compare(
          rhs.tensors.begin(),
          rhs.tensors.end(),
          lhs.tensors.begin(),
          lhs.tensors.end(),
          TensorRepresentativeLess)) {
    return false;
  }
  auto lhs_summed = lhs.summed_indices;
  auto rhs_summed = rhs.summed_indices;
  std::ranges::sort(lhs_summed);
  std::ranges::sort(rhs_summed);
  if (lhs_summed != rhs_summed) {
    return lhs_summed < rhs_summed;
  }
  return lhs.coefficient < rhs.coefficient;
}

void RemoveDuplicateDeltas(Term& term) {
  std::set<std::string> deltas;
  std::erase_if(term.tensors, [&](const Tensor& tensor) {
    if (tensor.kind != TensorKind::kDelta) {
      return false;
    }
    Tensor canonical = tensor;
    double ignored_coefficient = 1.0;
    CanonicalizeTensor(canonical, ignored_coefficient);
    return !deltas.insert(TensorKey(canonical)).second;
  });
}

void ReduceDeltas(Term& term) {
  std::vector<bool> erased(term.tensors.size(), false);
  for (std::size_t position = 0; position < term.tensors.size(); ++position) {
    const auto& tensor = term.tensors[position];
    if (tensor.kind != TensorKind::kDelta) {
      continue;
    }
    const Index lhs = tensor.indices[0];
    const Index rhs = tensor.indices[1];
    if (!ContractionDomainsMatch(lhs.domain, rhs.domain)) {
      term.coefficient = 0;
      return;
    }
    const bool lhs_summed = IsSummed(term, lhs);
    const bool rhs_summed = IsSummed(term, rhs);
    if (lhs == rhs) {
      // delta_ii is one inside an existing reduction. It carries a dimension
      // only when removing it would leave the bound index without an operand.
      bool consumed_elsewhere = false;
      for (std::size_t other = 0; other < term.tensors.size(); ++other) {
        if (other != position && !erased[other]) {
          consumed_elsewhere |=
              std::ranges::find(term.tensors[other].indices, lhs) !=
              term.tensors[other].indices.end();
        }
      }
      erased[position] = !lhs_summed || consumed_elsewhere;
      continue;
    }
    Index retained = lhs_summed ? rhs : rhs_summed ? lhs : std::min(lhs, rhs);
    if (lhs_summed || rhs_summed) {
      bool consumed_elsewhere = false;
      for (std::size_t other = 0; other < term.tensors.size(); ++other) {
        if (other == position || erased[other]) {
          continue;
        }
        for (const auto& index : term.tensors[other].indices) {
          consumed_elsewhere |= index == lhs || index == rhs;
        }
      }
      EraseSummed(term, lhs_summed ? lhs : rhs);
      retained.domain.orbital_spaces =
          lhs.domain.orbital_spaces & rhs.domain.orbital_spaces;
      retained.domain.spins = lhs.domain.spins & rhs.domain.spins;
      erased[position] = true;
      if (lhs_summed && rhs_summed && !consumed_elsewhere) {
        term.tensors[position].indices = {retained, retained};
        erased[position] = false;
      }
    }
    for (std::size_t other = 0; other < term.tensors.size(); ++other) {
      if (other == position || erased[other]) {
        continue;
      }
      for (auto& index : term.tensors[other].indices) {
        if (index == lhs || index == rhs) {
          index = retained;
        }
      }
    }
    for (auto& index : term.summed_indices) {
      if (index == lhs || index == rhs) {
        index = retained;
      }
    }
  }
  std::vector<Tensor> tensors;
  for (std::size_t i = 0; i < term.tensors.size(); ++i) {
    if (!erased[i]) {
      tensors.push_back(std::move(term.tensors[i]));
    }
  }
  term.tensors = std::move(tensors);
  RemoveDuplicateDeltas(term);
}

struct PrefixLabeling {
  // Keep every consistent dummy assignment for the selected tensor prefix.
  // Later tensors refine these assignments without changing earlier choices.
  std::map<Index, std::size_t> names;
  int sign = 1;
  auto operator<=>(const PrefixLabeling&) const = default;
};

struct TensorChoice {
  Tensor tensor;
  std::set<PrefixLabeling> labelings;
};

std::string AbstractName(std::size_t ordinal) {
  // Labels are private keys; their ordering precedes ordinary orbital names.
  std::ostringstream name;
  name << '\x01' << std::setw(10) << std::setfill('0') << ordinal;
  return name.str();
}

TensorChoice ExtendCanonicalPrefix(
    const Tensor& source,
    const std::set<Index>& summed,
    const std::set<PrefixLabeling>& prefixes) {
  TensorChoice choice;
  bool initialized = false;
  for (const auto& permutation : source.symmetry.Elements()) {
    for (const auto& prefix : prefixes) {
      PrefixLabeling next = prefix;
      next.sign *= permutation.sign;
      Tensor tensor = source;
      for (std::size_t slot = 0; slot < tensor.indices.size(); ++slot) {
        auto index = source.indices[permutation.order[slot]];
        if (summed.contains(index)) {
          const auto [position, inserted] =
              next.names.try_emplace(index, next.names.size());
          index.name = AbstractName(position->second);
        }
        tensor.indices[slot] = std::move(index);
      }
      if (!initialized || tensor.indices < choice.tensor.indices) {
        choice.tensor = std::move(tensor);
        choice.labelings.clear();
        choice.labelings.insert(std::move(next));
        initialized = true;
      } else if (tensor.indices == choice.tensor.indices) {
        choice.labelings.insert(std::move(next));
      }
    }
  }
  if (!initialized) {
    throw std::invalid_argument("Tensor has no valid permutation");
  }
  return choice;
}

std::vector<Tensor> CanonicalizeTaggedWord(
    std::vector<Tensor> operators,
    const std::set<Index>& summed,
    PrefixLabeling& labels) {
  auto same_group = [](const Tensor& a, const Tensor& b) {
    return a.kind == b.kind &&
        a.indices.front().domain == b.indices.front().domain;
  };
  // Remaining dummy indices are ranked by their occurrences in successive
  // operator groups. Indices already fixed by tensor factors keep their labels.
  std::map<Index, std::vector<std::pair<std::size_t, int>>> occurrences;
  std::vector<Index> unresolved;
  std::size_t group = 0;
  for (std::size_t i = 0; i < operators.size(); ++i) {
    if (i != 0 && !same_group(operators[i - 1], operators[i])) {
      ++group;
    }
    const auto& index = operators[i].indices.front();
    if (!summed.contains(index) || labels.names.contains(index)) {
      continue;
    }
    auto [it, inserted] = occurrences.try_emplace(index);
    if (inserted) {
      unresolved.push_back(index);
    }
    if (it->second.empty() || it->second.back().first != group) {
      it->second.emplace_back(group, -1);
    } else {
      --it->second.back().second;
    }
  }
  std::ranges::stable_sort(unresolved, [&](const auto& a, const auto& b) {
    const auto& left = occurrences.at(a);
    const auto& right = occurrences.at(b);
    const auto common = std::min(left.size(), right.size());
    for (std::size_t i = 0; i < common; ++i) {
      if (left[i] != right[i]) {
        return left[i] < right[i];
      }
    }
    return left.size() > right.size();
  });
  for (const auto& index : unresolved) {
    labels.names.emplace(index, labels.names.size());
  }
  for (auto& op : operators) {
    auto& index = op.indices.front();
    if (labels.names.contains(index)) {
      index.name = AbstractName(labels.names.at(index));
    }
  }
  std::vector<Tensor> result;
  std::map<int, int> spin_names;
  for (std::size_t begin = 0; begin < operators.size();) {
    std::size_t end = begin + 1;
    while (end < operators.size() &&
           same_group(operators[begin], operators[end])) {
      ++end;
    }
    std::vector<std::size_t> order;
    for (auto i = begin; i < end; ++i) {
      order.push_back(i);
    }
    std::ranges::stable_sort(order, [&](auto a, auto b) {
      return operators[a].indices < operators[b].indices;
    });
    for (std::size_t i = 0; i < order.size(); ++i) {
      for (std::size_t j = i + 1; j < order.size(); ++j) {
        if (order[i] > order[j]) {
          labels.sign = -labels.sign;
        }
      }
      auto tensor = operators[order[i]];
      auto [it, inserted] = spin_names.try_emplace(
          tensor.SpinTag(), static_cast<int>(spin_names.size()));
      tensor.SetSpinTag(it->second);
      result.push_back(std::move(tensor));
    }
    begin = end;
  }
  return result;
}

Term CanonicalizeTerm(Term term) {
  std::set<Index> summed(
      term.summed_indices.begin(), term.summed_indices.end());
  std::vector<Tensor> factors;
  std::vector<Tensor> operators;
  for (auto tensor : term.tensors) {
    CanonicalizeTensor(tensor, term.coefficient);
    if (tensor.kind == TensorKind::kGeneric ||
        tensor.kind == TensorKind::kDelta) {
      factors.push_back(std::move(tensor));
    } else {
      operators.push_back(std::move(tensor));
    }
  }
  auto family = [](const Tensor& tensor) {
    return std::pair(tensor.name, tensor.indices.size());
  };
  std::sort(
      factors.begin(), factors.end(), [&](const Tensor& a, const Tensor& b) {
        return family(a) < family(b);
      });
  term.tensors.clear();
  std::set<PrefixLabeling> prefixes{PrefixLabeling{}};
  for (std::size_t first = 0; first < factors.size(); ++first) {
    std::size_t selected = first;
    auto best = ExtendCanonicalPrefix(factors[first], summed, prefixes);
    for (std::size_t next = first + 1; next < factors.size() &&
         family(factors[next]) == family(factors[first]);
         ++next) {
      auto candidate = ExtendCanonicalPrefix(factors[next], summed, prefixes);
      if (candidate.tensor.indices < best.tensor.indices) {
        best = std::move(candidate);
        selected = next;
      }
    }
    std::swap(factors[first], factors[selected]);
    term.tensors.push_back(std::move(best.tensor));
    prefixes = std::move(best.labelings);
  }
  if (!operators.empty() && std::ranges::all_of(operators, [](const auto& op) {
        return op.SpinTag() >= 0;
      })) {
    auto labeling = *prefixes.begin();
    auto canonical =
        CanonicalizeTaggedWord(std::move(operators), summed, labeling);
    term.tensors.insert(term.tensors.end(), canonical.begin(), canonical.end());
    prefixes = {std::move(labeling)};
  } else {
    for (const auto& op : operators) {
      auto next = ExtendCanonicalPrefix(op, summed, prefixes);
      term.tensors.push_back(std::move(next.tensor));
      prefixes = std::move(next.labelings);
    }
  }
  const auto& selected = *prefixes.begin();
  term.coefficient *= selected.sign;
  for (auto& index : term.summed_indices) {
    if (selected.names.contains(index)) {
      index.name = AbstractName(selected.names.at(index));
    }
  }
  std::ranges::sort(term.summed_indices);
  return term;
}

std::set<std::string> UsedIndexNames(const Term& term) {
  std::set<std::string> result;
  for (const auto& index : AllIndices(term)) {
    result.insert(index.name);
  }
  return result;
}

std::set<std::string> SummedIndexNames(const Term& term) {
  std::set<std::string> result;
  for (const auto& index : term.summed_indices) {
    result.insert(index.name);
  }
  return result;
}

std::string FreshRelatedName(
    std::string_view source,
    std::set<std::string>& occupied) {
  if (source.empty()) {
    throw std::invalid_argument("Cannot rename an empty index name");
  }
  for (int offset = 1; offset < 100; ++offset) {
    std::string candidate(source);
    int first = static_cast<unsigned char>(candidate.front()) + offset;
    if (first >= 123) {
      first -= 58;
    }
    candidate.front() = static_cast<char>(first);
    if (occupied.insert(candidate).second) {
      return candidate;
    }
  }
  throw std::invalid_argument("Unable to find a fresh product index name");
}

void RenameSummedIndex(Term& term, std::string_view name, std::string_view to) {
  for (auto& tensor : term.tensors) {
    for (auto& index : tensor.indices) {
      if (index.name == name) {
        index.name = to;
      }
    }
  }
  for (auto& index : term.summed_indices) {
    if (index.name == name) {
      index.name = to;
    }
  }
}

std::pair<Term, Term> AlphaRenameProduct(Term lhs, Term rhs) {
  const auto lhs_used = UsedIndexNames(lhs);
  const auto rhs_used = UsedIndexNames(rhs);
  const auto lhs_summed = SummedIndexNames(lhs);
  const auto rhs_summed = SummedIndexNames(rhs);

  std::set<std::string> shared_summed;
  std::ranges::set_intersection(
      lhs_summed,
      rhs_summed,
      std::inserter(shared_summed, shared_summed.end()));

  std::set<std::string> rename_lhs;
  for (const auto& name : lhs_summed) {
    if (rhs_used.contains(name) && !shared_summed.contains(name)) {
      rename_lhs.insert(name);
    }
  }
  std::set<std::string> rename_rhs;
  for (const auto& name : rhs_summed) {
    if (lhs_used.contains(name)) {
      rename_rhs.insert(name);
    }
  }

  std::set<std::string> occupied = lhs_used;
  occupied.insert(rhs_used.begin(), rhs_used.end());
  std::map<std::string, std::string> replacements;
  for (const auto& name : occupied) {
    if (rename_lhs.contains(name) || rename_rhs.contains(name)) {
      replacements.emplace(name, FreshRelatedName(name, occupied));
    }
  }
  for (const auto& name : rename_lhs) {
    RenameSummedIndex(lhs, name, replacements.at(name));
  }
  for (const auto& name : rename_rhs) {
    RenameSummedIndex(rhs, name, replacements.at(name));
  }
  return {std::move(lhs), std::move(rhs)};
}

void FreshenProductSpinTags(Term& lhs, Term& rhs) {
  auto counts = [](const Term& term) {
    std::map<int, std::size_t> result;
    for (const auto& tensor : term.tensors) {
      if (const auto tag = tensor.SpinTag(); tag >= 0) {
        ++result[tag];
      }
    }
    return result;
  };
  auto left = counts(lhs);
  auto right = counts(rhs);
  auto rename_collisions = [&](Term& term, auto& own, const auto& other) {
    const auto original = own;
    for (const auto& [tag, count] : original) {
      if (count <= 1 || !other.contains(tag)) {
        continue;
      }
      int fresh = 0;
      while (left.contains(fresh) || right.contains(fresh)) {
        ++fresh;
      }
      own.erase(tag);
      own[fresh] = count;
      for (auto& tensor : term.tensors) {
        if (tensor.SpinTag() == tag) {
          tensor.SetSpinTag(fresh);
        }
      }
    }
  };
  rename_collisions(rhs, right, left);
  rename_collisions(lhs, left, right);
}

struct ParsedLine {
  double coefficient = 1.0;
  std::vector<std::string> summed_names;
  std::vector<std::uint8_t> summed_spaces;
  std::vector<std::pair<std::string, std::vector<std::string>>> tensors;
};

std::string NormalizeLineBreaks(std::string_view input) {
  std::string result;
  result.reserve(input.size());
  int nesting = 0;
  for (std::size_t index = 0; index < input.size(); ++index) {
    if (input[index] == '\\' && index + 1 < input.size() &&
        input[index + 1] == 'n') {
      result.push_back('\n');
      ++index;
    } else {
      const char c = input[index];
      const bool exponent_sign = index > 1 &&
          (input[index - 1] == 'e' || input[index - 1] == 'E') &&
          (std::isdigit(static_cast<unsigned char>(input[index - 2])) ||
           input[index - 2] == '.');
      bool separate_coefficient = c == '(';
      if (nesting == 0 && separate_coefficient) {
        const auto newline = result.find_last_of('\n');
        const auto prefix = Trim(
            std::string_view(result).substr(
                newline == std::string::npos ? 0 : newline + 1));
        // Keep an outer sign attached to its parenthesized coefficient.
        separate_coefficient = prefix != "+" && prefix != "-";
      }
      if (nesting == 0 &&
          (((c == '+' || c == '-') && !exponent_sign) ||
           separate_coefficient)) {
        result.push_back('\n');
      }
      result.push_back(c);
      nesting += c == '(' || c == '[' || c == '{' || c == '<';
      nesting -= c == ')' || c == ']' || c == '}' || c == '>';
    }
  }
  return result;
}

std::string ReadDelimited(
    std::string_view line,
    std::size_t& position,
    char open,
    char close) {
  if (position >= line.size() || line[position] != open) {
    throw std::invalid_argument("Expected delimited index list");
  }
  const std::size_t end = line.find(close, position + 1);
  if (end == std::string_view::npos) {
    throw std::invalid_argument("Unterminated index list");
  }
  std::string content(line.substr(position + 1, end - position - 1));
  position = end + 1;
  return content;
}

void SkipSpace(std::string_view line, std::size_t& position) {
  while (position < line.size() &&
         (line[position] == ' ' || line[position] == '\t' ||
          line[position] == '\r')) {
    ++position;
  }
}

ParsedLine ParseLine(std::string_view input) {
  const std::string owned = Trim(input);
  const std::string_view line = owned;
  ParsedLine result;
  if (line.empty()) {
    result.coefficient = 0.0;
    return result;
  }

  std::size_t position = 0;
  SkipSpace(line, position);
  int sign = 1;
  if (position < line.size() &&
      (line[position] == '+' || line[position] == '-')) {
    sign = line[position] == '-' ? -1 : 1;
    ++position;
    SkipSpace(line, position);
  }

  const bool parenthesized = position < line.size() && line[position] == '(';
  if (parenthesized) {
    ++position;
    SkipSpace(line, position);
  }
  const char* number_begin = line.data() + position;
  char* number_end = nullptr;
  const double parsed_number = std::strtod(number_begin, &number_end);
  if (number_end != number_begin) {
    result.coefficient = sign * parsed_number;
    position += static_cast<std::size_t>(number_end - number_begin);
    SkipSpace(line, position);
    if (position < line.size() && line[position] == '*') {
      ++position;
    }
  } else {
    result.coefficient = sign;
  }
  if (parenthesized) {
    SkipSpace(line, position);
    if (number_end == number_begin || position == line.size() ||
        line[position] != ')') {
      throw std::invalid_argument(
          "Expected a numeric coefficient in parentheses");
    }
    ++position;
  }

  SkipSpace(line, position);
  if (line.substr(position).starts_with("SUM")) {
    position += 3;
    SkipSpace(line, position);
    const auto content = ReadDelimited(line, position, '<', '>');
    const auto separator = content.find('|');
    result.summed_names = SplitIndexNames(content.substr(0, separator));
    if (separator != std::string::npos) {
      for (char code : content.substr(separator + 1)) {
        const auto where = std::string_view("IASE").find(code);
        if (where != std::string_view::npos) {
          result.summed_spaces.push_back(
              static_cast<std::uint8_t>(1U << where));
        }
      }
    }
  } else if (line.substr(position).starts_with("\\sum_")) {
    position += 5;
    result.summed_names =
        SplitIndexNames(ReadDelimited(line, position, '{', '}'));
  }

  while (position < line.size()) {
    SkipSpace(line, position);
    if (position == line.size()) {
      break;
    }
    const std::size_t name_begin = position;
    while (position < line.size() && line[position] != '[' &&
           (line[position] != '_' || position + 1 >= line.size() ||
            (line[position + 1] != '{' &&
             line.find('[', position) != std::string_view::npos))) {
      if (line[position] == ' ' || line[position] == '\t') {
        break;
      }
      ++position;
    }
    const std::string name(line.substr(name_begin, position - name_begin));
    if (name.empty()) {
      throw std::invalid_argument(
          "Expected tensor name in expression: " + owned);
    }
    SkipSpace(line, position);
    std::string index_text;
    if (position < line.size() && line[position] == '[') {
      index_text = ReadDelimited(line, position, '[', ']');
    } else if (position < line.size() && line[position] == '_') {
      ++position;
      if (position < line.size() && line[position] == '{') {
        index_text = ReadDelimited(line, position, '{', '}');
      } else {
        const auto begin = position;
        while (position < line.size() &&
               !std::isspace(static_cast<unsigned char>(line[position]))) {
          ++position;
        }
        index_text = line.substr(begin, position - begin);
      }
    }
    result.tensors.emplace_back(name, SplitIndexNames(index_text));
  }
  return result;
}

TensorKind ClassifyTensor(std::string_view name, std::size_t rank) {
  const bool tagged = name.size() > 1 && name[1] >= '0' && name[1] <= '9';
  if (rank == 1 &&
      (name == "C" || name == "Ca" || name == "Cb" ||
       (name.front() == 'C' && tagged))) {
    return TensorKind::kCreation;
  }
  if (rank == 1 &&
      (name == "D" || name == "Da" || name == "Db" ||
       (name.front() == 'D' && tagged))) {
    return TensorKind::kAnnihilation;
  }
  if (name == "delta" && rank == 2) {
    return TensorKind::kDelta;
  }
  if (name.size() == 2 && (name.front() == 'E' || name.front() == 'R') &&
      name.back() >= '0' && name.back() <= '9' &&
      rank == 2 * static_cast<std::size_t>(name.back() - '0')) {
    return TensorKind::kSpinFree;
  }
  if (name.size() == rank + 1 && name.front() == 'E' &&
      std::ranges::all_of(
          name.substr(1), [](char c) { return c == 'C' || c == 'D'; })) {
    return TensorKind::kSpinFree;
  }
  return TensorKind::kGeneric;
}

std::string DisplayIndex(const Index& index) {
  if (!index.name.empty() && index.name.front() == '@') {
    const auto separator = index.name.find_last_of(':');
    return "d" + index.name.substr(separator + 1);
  }
  return index.name;
}

} // namespace

std::strong_ordering Index::operator<=>(const Index& other) const {
  if (const auto domain_order = domain <=> other.domain; domain_order != 0) {
    return domain_order;
  }
  return name <=> other.name;
}

bool Index::HasTypes() const noexcept {
  return domain != IndexDomain{};
}

bool Index::IsShort() const noexcept {
  return name.size() == 1;
}

Index Index::Untyped() const {
  return {name, {}};
}

std::size_t Index::Hash() const noexcept {
  return std::hash<std::string>{}(name);
}

std::strong_ordering SignedPermutation::operator<=>(
    const SignedPermutation& other) const {
  if (sign != other.sign) {
    return other.sign <=> sign;
  }
  return order <=> other.order;
}

SignedPermutation SignedPermutation::Identity(std::size_t rank) {
  SignedPermutation result;
  result.order.resize(rank);
  std::iota(result.order.begin(), result.order.end(), 0);
  return result;
}

SignedPermutation SignedPermutation::Compose(
    const SignedPermutation& other) const {
  const auto identity = Identity(order.size());
  for (const auto* permutation : {this, &other}) {
    auto sorted = permutation->order;
    std::ranges::sort(sorted);
    if (sorted != identity.order ||
        (permutation->sign != 1 && permutation->sign != -1)) {
      throw std::invalid_argument(
          "Cannot compose invalid or differently ranked permutations");
    }
  }
  return wickqc::symbolic::Compose(*this, other);
}

std::size_t SignedPermutation::Hash() const noexcept {
  // An opaque ordered-value hash; its numeric value is not an interchange ID.
  std::size_t value = sign < 0;
  value = 31 * value + order.size();
  for (const auto axis : order) {
    value = 31 * value + axis;
  }
  return value;
}

void IndexRegistry::Add(OrbitalSpace space, std::string_view names, Spin spin) {
  for (const auto& name : SplitIndexNames(names)) {
    auto& domain = domains_[name];
    domain.orbital_spaces |= ToMask(space);
    domain.spins |= ToMask(spin);
  }
}

Index IndexRegistry::Resolve(std::string_view name) const {
  const auto iterator = domains_.find(std::string(name));
  if (iterator == domains_.end()) {
    return {std::string(name), {}};
  }
  return {iterator->first, iterator->second};
}

std::vector<Index> IndexRegistry::Parse(std::string_view names) const {
  std::vector<Index> result;
  for (const auto& name : SplitIndexNames(names)) {
    result.push_back(Resolve(name));
  }
  return result;
}

std::set<Index> IndexRegistry::ParseSet(std::string_view names) const {
  const auto parsed = Parse(names);
  return {parsed.begin(), parsed.end()};
}

std::vector<IndexDomain> IndexRegistry::ConcreteDomains(
    std::string_view name) const {
  return wickqc::symbolic::ConcreteDomains(Resolve(name).domain);
}

TensorSymmetry::TensorSymmetry(
    std::size_t rank,
    const std::vector<SignedPermutation>& generators) {
  SignedPermutation identity;
  identity.order.resize(rank);
  for (std::size_t index = 0; index < rank; ++index) {
    identity.order[index] = index;
  }
  elements_.push_back(identity);
  for (const auto& generator : generators) {
    auto order = generator.order;
    std::ranges::sort(order);
    if (order != identity.order ||
        (generator.sign != 1 && generator.sign != -1)) {
      throw std::invalid_argument("Invalid tensor-symmetry permutation");
    }
  }
  std::set<SignedPermutation> visited{identity};
  for (std::size_t position = 0; position < elements_.size(); ++position) {
    for (const auto& generator : generators) {
      if (generator.order.size() != rank ||
          (generator.sign != 1 && generator.sign != -1)) {
        throw std::invalid_argument("Invalid tensor-symmetry generator");
      }
      auto candidate = Compose(elements_[position], generator);
      if (visited.insert(candidate).second) {
        elements_.push_back(std::move(candidate));
      }
    }
  }
}

TensorSymmetry TensorSymmetry::None(std::size_t rank) {
  return TensorSymmetry(rank, {});
}

const std::vector<SignedPermutation>& TensorSymmetry::Elements() const {
  return elements_;
}

TensorSymmetry TensorSymmetry::TwoSymmetric() {
  return TensorSymmetry(2, {{{1, 0}, 1}});
}

TensorSymmetry TensorSymmetry::TwoAntisymmetric() {
  auto swap = SignedPermutation::Identity(2);
  std::ranges::reverse(swap.order);
  swap.sign = -1;
  return TensorSymmetry(2, {swap});
}

TensorSymmetry TensorSymmetry::CanonicalTransformation() {
  auto transpose = SignedPermutation::Identity(4);
  std::rotate(
      transpose.order.begin(),
      transpose.order.begin() + 2,
      transpose.order.end());
  transpose.sign = -1;
  auto exchange = SignedPermutation::Identity(4);
  std::swap(exchange.order[0], exchange.order[1]);
  std::swap(exchange.order[2], exchange.order[3]);
  return TensorSymmetry(4, {transpose, exchange});
}

TensorSymmetry TensorSymmetry::PairAntisymmetric(std::size_t order) {
  if (order > std::numeric_limits<std::size_t>::max() / 2) {
    throw std::length_error("Pair-antisymmetric tensor rank overflows size_t");
  }
  std::vector<SignedPermutation> generators;
  for (const std::size_t start : {std::size_t{0}, order}) {
    for (std::size_t axis = 1; axis < order; ++axis) {
      auto exchange = SignedPermutation::Identity(2 * order);
      std::swap(exchange.order[start], exchange.order[start + axis]);
      exchange.sign = -1;
      generators.push_back(std::move(exchange));
    }
  }
  return TensorSymmetry(2 * order, generators);
}

TensorSymmetry TensorSymmetry::All(std::size_t rank) {
  auto permutation = SignedPermutation::Identity(rank);
  std::vector<SignedPermutation> elements;
  do {
    elements.push_back(permutation);
  } while (std::next_permutation(
      permutation.order.begin(), permutation.order.end()));
  return FromElements(std::move(elements));
}

TensorSymmetry TensorSymmetry::FourAntisymmetric() {
  return TensorSymmetry(
      4, {{{2, 3, 0, 1}, 1}, {{1, 0, 2, 3}, -1}, {{0, 1, 3, 2}, -1}});
}

TensorSymmetry TensorSymmetry::QuantumChemistryChemists() {
  return TensorSymmetry(
      4, {{{2, 3, 0, 1}, 1}, {{1, 0, 2, 3}, 1}, {{0, 1, 3, 2}, 1}});
}

TensorSymmetry TensorSymmetry::QuantumChemistryPhysicists() {
  return TensorSymmetry(
      4, {{{0, 3, 2, 1}, 1}, {{2, 1, 0, 3}, 1}, {{1, 0, 3, 2}, 1}});
}

TensorSymmetry TensorSymmetry::SpinFree(std::size_t order, bool hermitian) {
  std::vector<SignedPermutation> generators;
  for (std::size_t other = 1; other < order; ++other) {
    SignedPermutation permutation;
    permutation.order.resize(order * 2);
    for (std::size_t index = 0; index < order * 2; ++index) {
      permutation.order[index] = index;
    }
    std::swap(permutation.order[0], permutation.order[other]);
    std::swap(permutation.order[order], permutation.order[order + other]);
    generators.push_back(std::move(permutation));
  }
  if (hermitian && order != 0) {
    SignedPermutation transpose;
    transpose.order.resize(order * 2);
    for (std::size_t index = 0; index < order; ++index) {
      transpose.order[index] = order + index;
      transpose.order[order + index] = index;
    }
    generators.push_back(std::move(transpose));
  }
  return TensorSymmetry(order * 2, generators);
}

void SymmetryRegistry::Add(
    std::string name,
    std::size_t rank,
    TensorSymmetry symmetry) {
  symmetries_[{std::move(name), rank}] = std::move(symmetry);
}

TensorSymmetry SymmetryRegistry::Lookup(std::string_view name, std::size_t rank)
    const {
  const auto iterator = symmetries_.find({std::string(name), rank});
  if (iterator != symmetries_.end()) {
    return iterator->second;
  }
  return TensorSymmetry::None(rank);
}

bool Tensor::IsFermionOperator() const noexcept {
  return kind == TensorKind::kCreation || kind == TensorKind::kAnnihilation;
}

bool Tensor::operator==(const Tensor& other) const {
  return name == other.name && kind == other.kind && indices == other.indices;
}

int Tensor::CompareFermiClass(const Tensor& rhs) const {
  constexpr auto inactive = static_cast<std::uint8_t>(OrbitalSpace::kInactive);
  constexpr auto active = static_cast<std::uint8_t>(OrbitalSpace::kActive);
  constexpr auto external = static_cast<std::uint8_t>(OrbitalSpace::kExternal);
  const auto lhs_space =
      indices.empty() ? std::uint8_t{0} : indices.front().domain.orbital_spaces;
  const auto rhs_space = rhs.indices.empty()
      ? std::uint8_t{0}
      : rhs.indices.front().domain.orbital_spaces;
  auto occupied_space = std::min(lhs_space, rhs_space);
  if (occupied_space == 0 || occupied_space == external ||
      (occupied_space == active && std::max(lhs_space, rhs_space) == active)) {
    occupied_space = inactive;
  }
  const int lhs_order = FermionSortOrder(*this, occupied_space);
  const int rhs_order = FermionSortOrder(rhs, occupied_space);
  return (lhs_order > rhs_order) - (lhs_order < rhs_order);
}

std::vector<std::map<std::string, std::string>> Tensor::IndexPermutations()
    const {
  std::map<IndexDomain, std::vector<Index>> groups;
  std::map<std::string, std::string> identity;
  for (const auto& index : indices) {
    groups[index.domain].push_back(index);
    identity[index.name] = index.name;
  }
  std::vector<std::map<std::string, std::string>> result{identity};
  for (const auto& [domain, indices] : groups) {
    std::vector<std::size_t> permutation(indices.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::vector<std::map<std::string, std::string>> extended;
    do {
      for (auto mapping : result) {
        for (std::size_t i = 0; i < indices.size(); ++i) {
          mapping[indices[i].name] = indices[permutation[i]].name;
        }
        extended.push_back(std::move(mapping));
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    result = std::move(extended);
  }
  return result;
}

std::map<std::string, std::string> Tensor::IndexMapTo(
    const Tensor& other) const {
  if (indices.size() != other.indices.size()) {
    return {};
  }
  std::map<IndexDomain, std::array<std::vector<std::string>, 2>> groups;
  for (const auto& index : indices) {
    groups[index.domain][0].push_back(index.name);
  }
  for (const auto& index : other.indices) {
    groups[index.domain][1].push_back(index.name);
  }
  std::map<std::string, std::string> names;
  for (const auto& [domain, pair] : groups) {
    if (pair[0].size() != pair[1].size()) {
      return {};
    }
    for (std::size_t i = 0; i < pair[0].size(); ++i) {
      names[pair[0][i]] = pair[1][i];
    }
  }
  return names;
}
bool Tensor::operator<(const Tensor& other) const {
  return TensorRepresentativeLess(*this, other);
}

Tensor Tensor::Canonicalize(double& coefficient) const {
  Tensor result = *this;
  CanonicalizeTensor(result, coefficient);
  return result;
}

Tensor Tensor::RestrictSymmetry() const {
  Tensor result = *this;
  std::vector<SignedPermutation> compatible;
  for (const auto& permutation : symmetry.Elements()) {
    const auto permuted = Permute(permutation);
    bool valid = true;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      valid = valid &&
          ContractionDomainsMatch(
                  indices[i].domain, permuted.indices[i].domain);
    }
    if (valid &&
        std::ranges::find(compatible, permutation) == compatible.end()) {
      compatible.push_back(permutation);
    }
  }
  result.symmetry = TensorSymmetry::FromElements(std::move(compatible));
  return result;
}

std::string Tensor::ToString(const SignedPermutation& permutation) const {
  const auto permuted = Permute(permutation);
  const std::string separator =
      std::ranges::all_of(indices, &Index::IsShort) ? "" : " ";
  std::ostringstream output;
  output << (permutation.sign < 0 ? "-" : "") << name << '[' << separator;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (kind == TensorKind::kSpinFree && i * 2 == indices.size()) {
      output << ',' << separator;
    }
    output << permuted.indices[i].name << separator;
  }
  output << ']';
  return output.str();
}

std::string Tensor::PermutationRules() const {
  std::string result;
  for (const auto& permutation : symmetry.Elements()) {
    if (!result.empty()) {
      result += " == ";
    }
    result += ToString(permutation);
  }
  return result;
}

Tensor Tensor::Permute(const SignedPermutation& permutation) const {
  auto sorted = permutation.order;
  std::ranges::sort(sorted);
  if (sorted.size() != indices.size()) {
    throw std::invalid_argument("Tensor and permutation ranks differ");
  }
  Tensor result = *this;
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    if (sorted[i] != i) {
      throw std::invalid_argument("Invalid tensor permutation");
    }
    result.indices[i] = indices[permutation.order[i]];
  }
  return result;
}

bool Term::operator==(const Term& other) const {
  return coefficient == other.coefficient && SameForm(other);
}

bool Term::operator<(const Term& other) const {
  auto left = *this;
  auto right = other;
  for (auto* term : {&left, &right}) {
    std::ranges::sort(term->summed_indices);
    term->summed_indices.erase(
        std::unique(term->summed_indices.begin(), term->summed_indices.end()),
        term->summed_indices.end());
  }
  return TermRepresentativeLess(left, right);
}

std::set<Index> Term::UsedIndices() const {
  std::set<Index> result;
  for (const auto& tensor : tensors) {
    result.insert(tensor.indices.begin(), tensor.indices.end());
  }
  return result;
}

std::set<std::string> Term::UsedIndexNames() const {
  std::set<std::string> result;
  for (const auto& index : UsedIndices()) {
    result.insert(index.name);
  }
  return result;
}

std::set<std::string> Term::SummedIndexNames() const {
  return wickqc::symbolic::SummedIndexNames(*this);
}

std::map<int, int> Term::SpinTagCounts() const {
  std::map<int, int> result;
  for (const auto& tensor : tensors) {
    if (const auto tag = tensor.SpinTag(); tag >= 0) {
      ++result[tag];
    }
  }
  return result;
}

bool Term::HasOperatorsIn(OrbitalSpace space) const {
  return std::ranges::any_of(tensors, [&](const auto& tensor) {
    return (tensor.kind == TensorKind::kSpinFree ||
            tensor.IsFermionOperator()) &&
        std::ranges::any_of(tensor.indices, [&](const auto& index) {
             return (index.domain.orbital_spaces & ToMask(space)) != 0;
           });
  });
}

bool Term::SameForm(const Term& other) const {
  if (tensors != other.tensors) {
    return false;
  }
  auto left = summed_indices;
  auto right = other.summed_indices;
  std::ranges::sort(left);
  std::ranges::sort(right);
  left.erase(std::unique(left.begin(), left.end()), left.end());
  right.erase(std::unique(right.begin(), right.end()), right.end());
  return left == right;
}

Term Term::Canonicalize() const {
  return CanonicalizeTerm(*this);
}

Term Term::SortFactors() const {
  Term result = *this;
  for (auto& tensor : result.tensors) {
    CanonicalizeTensor(tensor, result.coefficient);
  }
  const auto end = std::stable_partition(
      result.tensors.begin(), result.tensors.end(), [](const auto& tensor) {
        return tensor.kind == TensorKind::kGeneric ||
            tensor.kind == TensorKind::kDelta;
      });
  std::sort(result.tensors.begin(), end, TensorRepresentativeLess);
  return result;
}

int Tensor::SpinTag() const {
  if (!IsFermionOperator() || name.size() < 2 || name[1] < '0' ||
      name[1] > '9') {
    return -1;
  }
  return std::stoi(name.substr(1));
}

void Tensor::SetSpinTag(int tag) {
  if (IsFermionOperator() && (name.size() == 1 || SpinTag() >= 0)) {
    name = name.substr(0, 1) + (tag < 0 ? std::string{} : std::to_string(tag));
  }
}

Tensor Tensor::Parse(
    std::string_view text,
    const IndexRegistry& indices,
    const SymmetryRegistry& symmetries) {
  const auto expression = Expression::Parse(text, indices, symmetries);
  if (expression.Terms().size() != 1 ||
      expression.Terms().front().coefficient != 1.0 ||
      !expression.Terms().front().summed_indices.empty() ||
      expression.Terms().front().tensors.size() != 1) {
    throw std::invalid_argument("Expected exactly one unscaled tensor");
  }
  return expression.Terms().front().tensors.front();
}

Expression::Expression(Term term) : terms_{std::move(term)} {}

Expression::Expression(std::vector<Term> terms) : terms_(std::move(terms)) {}

const std::vector<Term>& Expression::Terms() const noexcept {
  return terms_;
}

bool Expression::operator==(const Expression& other) const {
  return terms_ == other.terms_;
}

bool Expression::operator<(const Expression& other) const {
  return std::lexicographical_compare(
      terms_.begin(), terms_.end(), other.terms_.begin(), other.terms_.end());
}

bool Expression::Empty() const noexcept {
  return terms_.empty();
}

Expression Expression::Parse(
    std::string_view text,
    const IndexRegistry& indices,
    const SymmetryRegistry& symmetries) {
  std::vector<Term> terms;
  std::istringstream input(NormalizeLineBreaks(text));
  std::string line;
  while (std::getline(input, line)) {
    const auto parsed = ParseLine(line);
    if (Trim(line).empty()) {
      continue;
    }
    Term term;
    term.coefficient = parsed.coefficient;
    std::map<std::string, IndexDomain> overrides;
    for (std::size_t i = 0; i < parsed.summed_names.size(); ++i) {
      auto index = indices.Resolve(parsed.summed_names[i]);
      if (parsed.summed_spaces.size() == parsed.summed_names.size()) {
        index.domain = {parsed.summed_spaces[i], 0};
        overrides[index.name] = index.domain;
      }
      term.summed_indices.push_back(std::move(index));
    }
    std::ranges::sort(term.summed_indices);
    term.summed_indices.erase(
        std::unique(term.summed_indices.begin(), term.summed_indices.end()),
        term.summed_indices.end());
    for (const auto& [name, index_names] : parsed.tensors) {
      Tensor tensor;
      tensor.name = name;
      for (const auto& index_name : index_names) {
        auto index = indices.Resolve(index_name);
        if (overrides.contains(index.name)) {
          index.domain = overrides.at(index.name);
        }
        tensor.indices.push_back(std::move(index));
      }
      tensor.kind = ClassifyTensor(name, tensor.indices.size());
      if (tensor.kind == TensorKind::kDelta) {
        tensor.symmetry = TensorSymmetry::TwoSymmetric();
      } else if (tensor.kind == TensorKind::kSpinFree) {
        tensor.symmetry = name.size() == 2
            ? TensorSymmetry::SpinFree(
                  tensor.indices.size() / 2, name.front() == 'R')
            : TensorSymmetry::None(tensor.indices.size());
      } else {
        tensor.symmetry = symmetries.Lookup(name, tensor.indices.size());
      }
      term.tensors.push_back(std::move(tensor));
    }
    terms.push_back(std::move(term));
  }
  return Expression(std::move(terms));
}

std::pair<Tensor, Expression> Expression::ParseDefinition(
    std::string_view text,
    const IndexRegistry& indices,
    const SymmetryRegistry& symmetries) {
  const auto separator = text.find('=');
  if (separator == std::string_view::npos ||
      text.find('=', separator + 1) != std::string_view::npos) {
    throw std::invalid_argument("A tensor definition requires exactly one '='");
  }
  return {
      Tensor::Parse(text.substr(0, separator), indices, symmetries),
      Parse(text.substr(separator + 1), indices, symmetries)};
}

Expression Expression::SplitIndexDomains() const {
  std::vector<Term> result;
  for (const auto& source : terms_) {
    auto split = SplitTermDomains(source);
    result.insert(
        result.end(),
        std::make_move_iterator(split.begin()),
        std::make_move_iterator(split.end()));
  }
  return Expression(std::move(result));
}

Expression Expression::Expand(
    int max_uncontracted,
    bool skip_contractions,
    bool compact_spin_free) const {
  return SplitIndexDomains().NormalOrder(
      max_uncontracted, skip_contractions, compact_spin_free);
}

Expression Expression::NormalOrder(
    int max_uncontracted,
    bool skip_contractions,
    bool compact_spin_free) const {
  std::vector<Term> result;
  for (const auto& concrete : terms_) {
    if (std::ranges::any_of(concrete.tensors, [](const Tensor& tensor) {
          return tensor.kind == TensorKind::kSpinFree || tensor.SpinTag() >= 0;
        })) {
      auto expanded = ExpandSpinFreeTerm(
          concrete, max_uncontracted, skip_contractions, compact_spin_free);
      result.insert(
          result.end(),
          std::make_move_iterator(expanded.begin()),
          std::make_move_iterator(expanded.end()));
      continue;
    }
    std::vector<Tensor> operators;
    Term base = concrete;
    std::erase_if(base.tensors, [&](const Tensor& tensor) {
      if (tensor.IsFermionOperator()) {
        operators.push_back(tensor);
        return true;
      }
      return false;
    });
    if (operators.empty()) {
      result.push_back(std::move(base));
      continue;
    }

    std::vector<WickExpansion> expansions;
    EnumerateWick(operators, skip_contractions, expansions);
    for (auto& expansion : expansions) {
      if (max_uncontracted >= 0 &&
          static_cast<int>(expansion.uncontracted.size()) > max_uncontracted) {
        continue;
      }
      Term term = base;
      term.coefficient *= expansion.sign;
      AppendDeltas(term, expansion.deltas);
      term.tensors.insert(
          term.tensors.end(),
          expansion.uncontracted.begin(),
          expansion.uncontracted.end());
      result.push_back(std::move(term));
    }
  }
  return Expression(std::move(result));
}

Expression Expression::SortFactors() const {
  auto result = terms_;
  for (auto& term : result) {
    term = term.SortFactors();
  }
  return Expression(std::move(result));
}

Expression Expression::SimplifyDeltas() const {
  auto result = terms_;
  for (auto& term : result) {
    ReduceDeltas(term);
  }
  return Expression(std::move(result));
}

Expression Expression::RemoveZeros(double tolerance) const {
  if (tolerance < 0 || !std::isfinite(tolerance)) {
    throw std::invalid_argument(
        "A simplification tolerance must be finite and nonnegative");
  }
  auto result = terms_;
  std::erase_if(result, [&](const auto& term) {
    return std::abs(term.coefficient) <= tolerance;
  });
  return Expression(std::move(result));
}

Expression Expression::Simplify(double tolerance) const {
  return SimplifyDeltas().RemoveZeros(tolerance).MergeTerms(tolerance);
}

Expression Expression::MergeTerms(double tolerance) const {
  if (tolerance < 0 || !std::isfinite(tolerance)) {
    throw std::invalid_argument(
        "A simplification tolerance must be finite and nonnegative");
  }
  struct Aggregate {
    Term representative;
    double canonical_coefficient = 0.0;
    double representative_sign = 1.0;
  };

  std::map<std::string, std::size_t> unique_terms;
  std::vector<Aggregate> aggregates;
  for (auto term : terms_) {
    auto unit = term;
    unit.coefficient = 1.0;
    auto canonical = CanonicalizeTerm(std::move(unit));
    if (canonical.coefficient == 0.0) {
      continue;
    }
    const std::string key = TermKey(canonical);
    const double representative_sign = canonical.coefficient;
    canonical.coefficient *= term.coefficient;
    auto [iterator, inserted] =
        unique_terms.try_emplace(key, aggregates.size());
    if (inserted) {
      aggregates.push_back(
          {std::move(term), canonical.coefficient, representative_sign});
    } else {
      aggregates[iterator->second].canonical_coefficient +=
          canonical.coefficient;
    }
  }

  std::vector<Term> simplified;
  simplified.reserve(unique_terms.size());
  for (auto& aggregate : aggregates) {
    if (std::abs(aggregate.canonical_coefficient) > tolerance) {
      aggregate.representative.coefficient =
          aggregate.canonical_coefficient / aggregate.representative_sign;
      simplified.push_back(std::move(aggregate.representative));
    }
  }
  std::ranges::sort(simplified, TermRepresentativeLess);
  return Expression(std::move(simplified));
}

Expression Expression::Conjugate() const {
  auto terms = terms_;
  for (auto& term : terms) {
    std::vector<Tensor> adjoint_operators;
    for (const auto& tensor : term.tensors) {
      if (tensor.kind != TensorKind::kSpinFree && !tensor.IsFermionOperator()) {
        continue;
      }
      Tensor adjoint = tensor;
      if (adjoint.kind == TensorKind::kSpinFree) {
        const std::size_t order = adjoint.indices.size() / 2;
        for (std::size_t index = 0; index < order; ++index) {
          std::swap(adjoint.indices[index], adjoint.indices[order + index]);
        }
      } else if (adjoint.kind == TensorKind::kCreation) {
        adjoint.kind = TensorKind::kAnnihilation;
        adjoint.name.front() = 'D';
      } else {
        adjoint.kind = TensorKind::kCreation;
        adjoint.name.front() = 'C';
      }
      adjoint_operators.push_back(std::move(adjoint));
    }
    std::ranges::reverse(adjoint_operators);
    std::size_t position = 0;
    for (auto& tensor : term.tensors) {
      if (tensor.kind == TensorKind::kSpinFree || tensor.IsFermionOperator()) {
        tensor = std::move(adjoint_operators[position++]);
      }
    }
  }
  return Expression(std::move(terms));
}

Expression Expression::RenameIndices(
    const std::map<std::string, std::string>& names) const {
  for (const auto& [from, to] : names) {
    if (from.empty() || to.empty()) {
      throw std::invalid_argument("Index names must not be empty");
    }
  }
  auto result = terms_;
  auto rename = [&](Index& index) {
    if (const auto found = names.find(index.name); found != names.end()) {
      index.name = found->second;
    }
  };
  for (auto& term : result) {
    for (auto& tensor : term.tensors) {
      for (auto& index : tensor.indices) {
        rename(index);
      }
    }
    for (auto& index : term.summed_indices) {
      rename(index);
    }
    std::ranges::sort(term.summed_indices);
    term.summed_indices.erase(
        std::unique(term.summed_indices.begin(), term.summed_indices.end()),
        term.summed_indices.end());
  }
  return Expression(std::move(result));
}

Expression Expression::Substitute(
    const std::map<std::string, std::pair<Tensor, Expression>>& definitions)
    const {
  std::vector<Term> result;
  for (const auto& source : terms_) {
    const auto original_names = UsedIndexNames(source);
    std::vector<Term> partials{
        Term{source.coefficient, {}, source.summed_indices}};
    for (const auto& tensor : source.tensors) {
      const auto definition = definitions.find(tensor.name);
      if (definition == definitions.end() ||
          definition->second.first.indices.size() != tensor.indices.size()) {
        for (auto& partial : partials) {
          partial.tensors.push_back(tensor);
        }
        continue;
      }
      const auto& [formal, replacement] = definition->second;
      std::vector<Term> expanded;
      for (const auto& partial : partials) {
        for (const auto& value : replacement.Terms()) {
          auto occupied = UsedIndexNames(partial);
          occupied.insert(original_names.begin(), original_names.end());
          std::map<Index, Index> bindings;
          for (std::size_t slot = 0; slot < tensor.indices.size(); ++slot) {
            const auto& actual = tensor.indices[slot];
            if (!formal.indices[slot].domain.IsCompatibleWith(actual.domain)) {
              throw std::invalid_argument(
                  "Incompatible index space substituting '" + tensor.name +
                  "'");
            }
            const auto [found, inserted] =
                bindings.emplace(formal.indices[slot], actual);
            if (!inserted && found->second != actual) {
              throw std::invalid_argument(
                  "Repeated formal index has inconsistent arguments");
            }
          }
          Term next = partial;
          next.coefficient *= value.coefficient;
          for (const auto& dummy : value.summed_indices) {
            if (bindings.contains(dummy)) {
              throw std::invalid_argument(
                  "Definition output index is also summed");
            }
            Index fresh = dummy;
            if (occupied.contains(fresh.name)) {
              fresh.name = FreshRelatedName(dummy.name, occupied);
            } else {
              occupied.insert(fresh.name);
            }
            bindings.emplace(dummy, fresh);
            next.summed_indices.push_back(fresh);
          }
          for (auto factor : value.tensors) {
            for (auto& index : factor.indices) {
              const auto found = bindings.find(index);
              if (found == bindings.end()) {
                throw std::invalid_argument(
                    "Unbound definition index '" + index.name + "'");
              }
              index = found->second;
            }
            next.tensors.push_back(std::move(factor));
          }
          expanded.push_back(std::move(next));
        }
      }
      partials = std::move(expanded);
    }
    result.insert(
        result.end(),
        std::make_move_iterator(partials.begin()),
        std::make_move_iterator(partials.end()));
  }
  return Expression(std::move(result));
}

Expression Expression::RemoveExternal() const {
  auto terms = terms_;
  std::erase_if(terms, [](const Term& term) {
    return term.HasOperatorsIn(OrbitalSpace::kExternal);
  });
  return Expression(std::move(terms));
}

Expression Expression::RemoveInactive() const {
  auto terms = terms_;
  std::erase_if(terms, [](const Term& term) {
    return term.HasOperatorsIn(OrbitalSpace::kInactive);
  });
  return Expression(std::move(terms));
}

Expression Expression::AddSpinFreeTransposeSymmetry() const {
  auto terms = terms_;
  for (auto& term : terms) {
    if (std::ranges::count_if(term.tensors, [](const Tensor& tensor) {
          return tensor.kind == TensorKind::kSpinFree;
        }) != 1) {
      continue;
    }
    for (auto& tensor : term.tensors) {
      if (tensor.kind == TensorKind::kSpinFree) {
        tensor.symmetry =
            TensorSymmetry::SpinFree(tensor.indices.size() / 2, true);
      }
    }
  }
  return Expression(std::move(terms));
}

Expression operator+(const Expression& lhs, const Expression& rhs) {
  std::vector<Term> terms = lhs.terms_;
  terms.insert(terms.end(), rhs.terms_.begin(), rhs.terms_.end());
  return Expression(std::move(terms));
}

Expression operator-(const Expression& lhs, const Expression& rhs) {
  return lhs + (-1.0 * rhs);
}

Expression operator*(const Expression& lhs, const Expression& rhs) {
  std::vector<Term> products;
  products.reserve(lhs.terms_.size() * rhs.terms_.size());
  for (const auto& original_lhs_term : lhs.terms_) {
    for (const auto& original_rhs_term : rhs.terms_) {
      auto [lhs_term, rhs_term] =
          AlphaRenameProduct(original_lhs_term, original_rhs_term);
      FreshenProductSpinTags(lhs_term, rhs_term);
      Term product = lhs_term;
      product.coefficient *= rhs_term.coefficient;
      product.tensors.insert(
          product.tensors.end(),
          rhs_term.tensors.begin(),
          rhs_term.tensors.end());
      product.summed_indices.insert(
          product.summed_indices.end(),
          rhs_term.summed_indices.begin(),
          rhs_term.summed_indices.end());
      products.push_back(std::move(product));
    }
  }
  return Expression(std::move(products));
}

Expression operator*(double scalar, const Expression& expression) {
  auto terms = expression.terms_;
  for (auto& term : terms) {
    term.coefficient *= scalar;
  }
  return Expression(std::move(terms));
}

Expression operator*(const Expression& expression, double scalar) {
  return scalar * expression;
}

Expression FullySummedProduct(const Expression& lhs, const Expression& rhs) {
  auto terms = (lhs * rhs).Terms();
  for (auto& term : terms) {
    std::set<Index> bound(
        term.summed_indices.begin(), term.summed_indices.end());
    for (const auto& tensor : term.tensors) {
      bound.insert(tensor.indices.begin(), tensor.indices.end());
    }
    term.summed_indices.assign(bound.begin(), bound.end());
  }
  return Expression(std::move(terms));
}

Expression Commutator(const Expression& lhs, const Expression& rhs) {
  return lhs * rhs - rhs * lhs;
}

std::ostream& operator<<(std::ostream& output, const Expression& expression) {
  output << "Expression{" << expression.terms_.size() << " terms";
  for (const auto& term : expression.terms_) {
    output << "\n  " << std::showpos << std::setprecision(12)
           << term.coefficient << std::noshowpos;
    if (!term.summed_indices.empty()) {
      output << " SUM<";
      for (const auto& index : term.summed_indices) {
        output << DisplayIndex(index);
      }
      output << '>';
    }
    for (const auto& tensor : term.tensors) {
      output << ' ' << tensor.name << '[';
      for (std::size_t index = 0; index < tensor.indices.size(); ++index) {
        if (index != 0) {
          output << ',';
        }
        output << DisplayIndex(tensor.indices[index]);
      }
      output << ']';
    }
  }
  output << "\n}";
  return output;
}

} // namespace wickqc::symbolic
