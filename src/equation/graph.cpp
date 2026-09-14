#include "equation/graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace wickqc::equation {
namespace {
using symbolic::Expression;
using symbolic::Index;
using symbolic::IndexDomain;
using symbolic::SignedPermutation;
using symbolic::Tensor;
using symbolic::TensorKind;
using symbolic::TensorSymmetry;
using symbolic::Term;
using IndexSet = std::set<Index>;
using NameMap = std::map<std::string, std::string>;

IndexSet Axes(const Tensor& tensor) {
  return {tensor.indices.begin(), tensor.indices.end()};
}

double Size(const IndexSet& indices, const GraphOptions& options) {
  double size = 1.0;
  for (const auto& index : indices) {
    const auto found = options.index_scales.find(index.domain);
    size *= found == options.index_scales.end()
        ? options.index_scales.at(IndexDomain{})
        : found->second;
  }
  return size;
}

std::vector<IndexSet> FutureAxes(
    const std::vector<Tensor>& operands,
    const Tensor& output) {
  std::vector<IndexSet> future(operands.size(), Axes(output));
  for (std::size_t i = operands.size(); i > 1; --i) {
    future[i - 2] = future[i - 1];
    future[i - 2].insert(
        operands[i - 1].indices.begin(), operands[i - 1].indices.end());
  }
  return future;
}

IndexSet EliminateUnused(IndexSet& indices, const IndexSet& future) {
  IndexSet eliminated;
  for (auto it = indices.begin(); it != indices.end();) {
    if (future.contains(*it)) {
      ++it;
    } else {
      eliminated.insert(*it);
      it = indices.erase(it);
    }
  }
  return eliminated;
}

bool SameDomain(const IndexDomain& a, const IndexDomain& b) {
  return a.spins == b.spins &&
      ((a.orbital_spaces == 0 && b.orbital_spaces == 0) ||
       (a.orbital_spaces & b.orbital_spaces) != 0);
}

// Restrict operand symmetries to the surviving axes of a binary contraction.
// Contracted axes must map among themselves with the same renaming in both
// operands. Repeated surviving indices are identified only after restriction.
Tensor BinaryIntermediate(
    const Tensor& left,
    const Tensor& right,
    const IndexSet& eliminated,
    const std::string& name) {
  std::vector<Index> occurrences;
  std::map<Index, int> reductions;
  std::vector<IndexDomain> reduction_domains;
  for (const auto& index : eliminated) {
    reductions[index] = -1 - static_cast<int>(reductions.size());
    reduction_domains.push_back(index.domain);
  }
  auto label_slots = [&](const Tensor& tensor) {
    std::vector<int> slots;
    for (const auto& index : tensor.indices) {
      if (reductions.contains(index)) {
        slots.push_back(reductions.at(index));
      } else {
        slots.push_back(static_cast<int>(occurrences.size()));
        occurrences.push_back(index);
      }
    }
    return slots;
  };
  const auto left_slots = label_slots(left);
  const auto right_slots = label_slots(right);
  struct Restriction {
    std::map<int, int> reductions;
    std::vector<std::size_t> free;
  };
  auto restrict =
      [&](const SignedPermutation& permutation,
          const std::vector<int>& slots,
          const std::vector<int>& targets) -> std::optional<Restriction> {
    Restriction result;
    for (std::size_t i = 0; i < slots.size(); ++i) {
      const int from = slots[i];
      const int to = targets[permutation.order[i]];
      if ((from < 0) != (to < 0)) {
        return std::nullopt;
      }
      if (from < 0) {
        // Dummy renaming cannot exchange different orbital or spin domains.
        if (!SameDomain(
                reduction_domains[-1 - from], reduction_domains[-1 - to])) {
          return std::nullopt;
        }
        const auto [it, inserted] = result.reductions.emplace(from, to);
        if (!inserted && it->second != to) {
          return std::nullopt;
        }
      } else {
        result.free.push_back(static_cast<std::size_t>(to));
      }
    }
    return result;
  };
  const bool equal_operands = left.name == right.name &&
      left.kind == right.kind && left.indices.size() == right.indices.size();
  std::vector<SignedPermutation> valid_exchanges;
  if (equal_operands) {
    for (const auto& a : left.symmetry.Elements()) {
      for (const auto& b : right.symmetry.Elements()) {
        const auto ra = restrict(a, left_slots, right_slots);
        const auto rb = restrict(b, right_slots, left_slots);
        if (ra && rb && ra->reductions == rb->reductions) {
          auto order = ra->free;
          order.insert(order.end(), rb->free.begin(), rb->free.end());
          valid_exchanges.push_back({std::move(order), a.sign * b.sign});
        }
      }
    }
  }
  std::vector<SignedPermutation> lifted;
  for (const auto& a : left.symmetry.Elements()) {
    const auto ra = restrict(a, left_slots, left_slots);
    if (!ra) {
      continue;
    }
    for (const auto& b : right.symmetry.Elements()) {
      const auto rb = restrict(b, right_slots, right_slots);
      if (!rb || ra->reductions != rb->reductions) {
        continue;
      }
      auto order = ra->free;
      order.insert(order.end(), rb->free.begin(), rb->free.end());
      lifted.push_back({std::move(order), a.sign * b.sign});
      bool exchange = equal_operands && ra->free.size() == rb->free.size();
      for (std::size_t axis = 0; exchange && axis < ra->free.size(); ++axis) {
        exchange = ra->free[axis] + ra->free.size() == rb->free[axis];
      }
      if (exchange) {
        auto swapped = rb->free;
        swapped.insert(swapped.end(), ra->free.begin(), ra->free.end());
        const SignedPermutation candidate{std::move(swapped), a.sign * b.sign};
        // A surviving-block swap must lift to a consistent map of all slots,
        // including reductions and their domains.
        if (std::ranges::find(valid_exchanges, candidate) !=
            valid_exchanges.end()) {
          lifted.push_back(candidate);
        }
      }
    }
  }
  std::vector<Index> output;
  std::map<Index, std::size_t> unique;
  for (const auto& index : occurrences) {
    const auto [it, inserted] = unique.emplace(index, unique.size());
    if (inserted) {
      output.push_back(index);
    }
  }
  std::vector<SignedPermutation> restricted;
  for (const auto& permutation : lifted) {
    SignedPermutation result{
        std::vector<std::size_t>(output.size(), output.size()),
        permutation.sign};
    bool compatible = true;
    for (std::size_t i = 0; i < occurrences.size() && compatible; ++i) {
      const auto from = unique.at(occurrences[i]);
      const auto to = unique.at(occurrences[permutation.order[i]]);
      if (result.order[from] != output.size() && result.order[from] != to) {
        compatible = false;
      }
      result.order[from] = to;
      compatible =
          compatible && SameDomain(output[from].domain, output[to].domain);
    }
    if (compatible) {
      restricted.push_back(std::move(result));
    }
  }
  return {
      name,
      output,
      TensorKind::kGeneric,
      TensorSymmetry(output.size(), restricted)};
}

Term Rename(const Term& term, const NameMap& names) {
  return Expression(term).RenameIndices(names).Terms().front();
}

NameMap ComposeNames(NameMap first, const NameMap& second) {
  for (auto& [from, to] : first) {
    if (second.contains(to)) {
      to = second.at(to);
    }
  }
  for (const auto& [from, to] : second) {
    first.try_emplace(from, to);
  }
  return first;
}

std::vector<NameMap> OutputPermutations(const Tensor& output) {
  return output.IndexPermutations();
}

Term UnitCanonical(Term term) {
  term.coefficient = 1.0;
  return term.Canonicalize();
}

NameMap TypeMatchedNames(const Tensor& a, const Tensor& b) {
  return a.IndexMapTo(b);
}
} // namespace

GraphOptions::GraphOptions() {
  for (std::uint8_t spin = 0; spin < 3; ++spin) {
    index_scales[{1, spin}] = 4.0;
    index_scales[{2, spin}] = 8.0;
    index_scales[{8, spin}] = 12.0;
  }
  index_scales[{}] = 4.0;
}

ContractionGraph::ContractionGraph(
    std::vector<EquationNode> nodes,
    GraphOptions options)
    : nodes_(std::move(nodes)), options_(std::move(options)) {}

void ContractionGraph::Add(Tensor output, Expression expression) {
  nodes_.push_back({std::move(output), std::move(expression)});
}

const std::vector<EquationNode>& ContractionGraph::Nodes() const {
  return nodes_;
}

const GraphOptions& ContractionGraph::Options() const {
  return options_;
}

bool ContractionGraph::IsIntermediate(const std::string& name) const {
  return name.starts_with(options_.intermediate_prefix);
}

int ContractionGraph::LastIntermediateId() const {
  int last = 0;
  auto inspect = [&](const Tensor& tensor) {
    if (!IsIntermediate(tensor.name)) {
      return;
    }
    const auto digits = tensor.name.substr(options_.intermediate_prefix.size());
    if (!digits.empty() && std::ranges::all_of(digits, [](char c) {
          return c >= '0' && c <= '9';
        })) {
      last = std::max(last, std::stoi(digits));
    }
  };
  for (const auto& node : nodes_) {
    inspect(node.output);
    for (const auto& term : node.expression.Terms()) {
      for (const auto& tensor : term.tensors) {
        inspect(tensor);
      }
    }
  }
  return last;
}

ContractionGraph ContractionGraph::OrderContractions() const {
  auto result = *this;
  for (auto& node : result.nodes_) {
    auto terms = node.expression.Terms();
    for (auto& term : terms) {
      if (term.tensors.size() < 2) {
        continue;
      }
      std::vector<std::size_t> order(term.tensors.size());
      std::iota(order.begin(), order.end(), 0);
      std::vector<Tensor> best;
      double best_cost = 0.0;
      do {
        if (order[0] > order[1]) {
          continue;
        }
        std::vector<Tensor> operands;
        operands.reserve(order.size());
        for (auto i : order) {
          operands.push_back(term.tensors[i]);
        }
        const auto future = FutureAxes(operands, node.output);
        auto live = Axes(operands.front());
        double cost = 0.0;
        for (std::size_t i = 1; i < operands.size(); ++i) {
          live.insert(operands[i].indices.begin(), operands[i].indices.end());
          cost += Size(live, options_);
          EliminateUnused(live, future[i]);
        }
        if (best.empty() || cost < best_cost) {
          best = std::move(operands);
          best_cost = cost;
        }
      } while (std::next_permutation(order.begin(), order.end()));
      term.tensors = std::move(best);
    }
    node.expression = Expression(std::move(terms));
  }
  return result;
}

ContractionGraph ContractionGraph::SplitBinary() const {
  auto result = *this;
  int next = LastIntermediateId();
  std::vector<EquationNode> intermediates;
  for (auto& node : result.nodes_) {
    auto terms = node.expression.Terms();
    for (auto& term : terms) {
      if (term.tensors.size() <= 2) {
        continue;
      }
      const auto future = FutureAxes(term.tensors, node.output);
      auto partial = term.tensors.front();
      auto live = Axes(partial);
      for (std::size_t i = 1; i < term.tensors.size(); ++i) {
        const auto& operand = term.tensors[i];
        live.insert(operand.indices.begin(), operand.indices.end());
        const auto reduced = EliminateUnused(live, future[i]);
        Term binary{1.0, {partial, operand}, {reduced.begin(), reduced.end()}};
        if (i + 1 == term.tensors.size()) {
          binary.coefficient = term.coefficient;
          term = std::move(binary);
          break;
        }
        auto output = BinaryIntermediate(
            partial,
            operand,
            reduced,
            options_.intermediate_prefix + std::to_string(++next));
        intermediates.push_back({output, Expression(std::move(binary))});
        partial = std::move(output);
      }
    }
    node.expression = Expression(std::move(terms));
  }
  result.nodes_.insert(
      result.nodes_.end(), intermediates.begin(), intermediates.end());
  return result;
}

ContractionGraph ContractionGraph::FactorPermutations() const {
  ContractionGraph result({}, options_);
  int next = LastIntermediateId();
  struct Orbit {
    Term representative;
    std::vector<Term> images;
    std::vector<std::pair<int, double>> weights;
  };
  for (const auto& node : nodes_) {
    const auto permutations = OutputPermutations(node.output);
    std::vector<Orbit> orbits;
    for (const auto& term : node.expression.Terms()) {
      if (term.coefficient == 0.0) {
        continue;
      }
      const auto canonical = UnitCanonical(term);
      if (canonical.coefficient == 0.0) {
        continue;
      }
      bool found = false;
      for (auto& orbit : orbits) {
        for (std::size_t i = 0; i < orbit.images.size(); ++i) {
          if (canonical.SameForm(orbit.images[i])) {
            orbit.weights.emplace_back(
                static_cast<int>(i),
                term.coefficient * canonical.coefficient *
                    orbit.images[i].coefficient);
            found = true;
            break;
          }
        }
        if (found) {
          break;
        }
      }
      if (!found) {
        Orbit orbit{term, {}, {{-1, term.coefficient}}};
        orbit.representative.coefficient = 1.0;
        for (const auto& permutation : permutations) {
          orbit.images.push_back(
              UnitCanonical(Rename(orbit.representative, permutation)));
        }
        orbits.push_back(std::move(orbit));
      }
    }
    for (auto& orbit : orbits) {
      const double scale = orbit.weights.front().second;
      if (scale == 0.0) {
        throw std::invalid_argument(
            "Remove zero terms before factoring graph permutations");
      }
      orbit.representative.coefficient *= scale;
      for (auto& [permutation, weight] : orbit.weights) {
        weight /= scale;
      }
      std::ranges::sort(orbit.weights, {}, &std::pair<int, double>::first);
    }
    auto pattern_less = [](const Orbit& a, const Orbit& b) {
      if (a.weights.size() != b.weights.size()) {
        return a.weights.size() < b.weights.size();
      }
      for (std::size_t i = 0; i < a.weights.size(); ++i) {
        if (a.weights[i].first != b.weights[i].first) {
          return a.weights[i].first < b.weights[i].first;
        }
      }
      for (std::size_t i = 0; i < a.weights.size(); ++i) {
        if (std::abs(a.weights[i].second - b.weights[i].second) > 1.0e-12) {
          return a.weights[i].second < b.weights[i].second;
        }
      }
      return false;
    };
    std::vector<std::size_t> order(orbits.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](auto a, auto b) {
      return pattern_less(orbits[a], orbits[b]);
    });
    std::vector<Term> assembly;
    for (std::size_t position = 0; position < order.size(); ++position) {
      const auto& orbit = orbits[order[position]];
      if (position == 0 || pattern_less(orbits[order[position - 1]], orbit)) {
        auto output = node.output;
        output.name = options_.intermediate_prefix + std::to_string(++next);
        assembly.push_back({1.0, {output}, {}});
        std::vector<IndexTransform> transforms;
        for (const auto& outer : node.transforms) {
          for (const auto& [index, weight] : orbit.weights) {
            transforms.push_back(
                {weight * outer.coefficient,
                 index < 0 ? outer.names
                           : ComposeNames(permutations[index], outer.names)});
          }
        }
        result.nodes_.push_back(
            {std::move(output),
             Expression(orbit.representative),
             std::move(transforms)});
      } else {
        result.nodes_.back().expression =
            result.nodes_.back().expression + Expression(orbit.representative);
      }
    }
    if (assembly.size() == 1) {
      result.nodes_.back().output = node.output;
    } else {
      result.Add(node.output, Expression(std::move(assembly)));
    }
  }
  return result;
}

ContractionGraph ContractionGraph::FactorCommonOperands() const {
  ContractionGraph result({}, options_);
  int next = LastIntermediateId();
  for (const auto& node : nodes_) {
    if (node.expression.Terms().size() == 1) {
      result.nodes_.push_back(node);
      continue;
    }
    std::vector<Term> untouched, groups;
    std::vector<EquationNode> generated;
    for (const auto& term : node.expression.Terms()) {
      bool eligible = term.tensors.size() == 2;
      if (eligible) {
        const auto a = Axes(term.tensors[0]);
        const auto b = Axes(term.tensors[1]);
        const IndexSet reductions(
            term.summed_indices.begin(), term.summed_indices.end());
        eligible = a.size() == term.tensors[0].indices.size() &&
            b.size() == term.tensors[1].indices.size();
        for (const auto& index : a) {
          eligible =
              eligible && (!b.contains(index) || reductions.contains(index));
        }
      }
      if (!eligible) {
        untouched.push_back(term);
        continue;
      }
      bool matched = false;
      for (auto& group : groups) {
        if (group.summed_indices.size() != term.summed_indices.size()) {
          continue;
        }
        for (std::size_t own = 0; own < 2 && !matched; ++own) {
          for (std::size_t common = 0; common < 2 && !matched; ++common) {
            const auto& candidate = term.tensors[own];
            const auto& pattern = group.tensors[common];
            if (candidate.name != pattern.name ||
                candidate.indices.size() != pattern.indices.size() ||
                term.tensors[1 - own].indices.size() !=
                    group.tensors[1 - common].indices.size()) {
              continue;
            }
            for (const auto& symmetry : candidate.symmetry.Elements()) {
              Term permuted = term;
              permuted.tensors[own] = candidate.Permute(symmetry);
              auto names = TypeMatchedNames(permuted.tensors[own], pattern);
              if (names.empty()) {
                continue;
              }
              auto bound_name = [](const auto& indices,
                                   const std::string& name) {
                return std::ranges::any_of(
                    indices, [&](const auto& i) { return i.name == name; });
              };
              std::erase_if(names, [&](const auto& pair) {
                return !bound_name(term.summed_indices, pair.first) ||
                    !bound_name(group.summed_indices, pair.second);
              });
              permuted = Rename(permuted, names);
              if (!(permuted.tensors[own] == pattern)) {
                continue;
              }
              const auto& a = group.tensors[1 - common];
              const auto& b = permuted.tensors[1 - own];
              // A shared factor alone does not align the remaining free
              // axes. Their sum must have one well-defined tensor shape.
              if (Axes(a) != Axes(b)) {
                continue;
              }
              std::vector<SignedPermutation> intersection;
              for (const auto& permutation : a.symmetry.Elements()) {
                if (std::ranges::find(b.symmetry.Elements(), permutation) !=
                    b.symmetry.Elements().end()) {
                  intersection.push_back(permutation);
                }
              }
              Tensor combined{
                  options_.intermediate_prefix + std::to_string(++next),
                  a.indices,
                  TensorKind::kGeneric,
                  TensorSymmetry(a.indices.size(), intersection)};
              Expression sum;
              auto absorb = [&](const Tensor& operand, double scale) {
                for (auto& previous : generated) {
                  if (previous.output == operand) {
                    sum = sum + scale * previous.expression;
                    previous.expression = Expression{};
                    return;
                  }
                }
                sum = sum + Expression(Term{scale, {operand}, {}});
              };
              absorb(a, group.coefficient);
              absorb(b, symmetry.sign * permuted.coefficient);
              generated.push_back({combined, std::move(sum)});
              group.coefficient = 1.0;
              group.tensors[1 - common] = std::move(combined);
              matched = true;
              break;
            }
          }
        }
        if (matched) {
          break;
        }
      }
      if (!matched) {
        groups.push_back(term);
      }
    }
    for (auto& intermediate : generated) {
      if (!intermediate.expression.Empty()) {
        result.nodes_.push_back(std::move(intermediate));
      }
    }
    untouched.insert(untouched.end(), groups.begin(), groups.end());
    result.nodes_.push_back(
        {node.output, Expression(std::move(untouched)), node.transforms});
  }
  return result;
}

ContractionGraph ContractionGraph::MergeIntermediates() const {
  struct Alias {
    std::string name;
    std::vector<std::size_t> axes;
    double scale;
  };
  using Signature = std::pair<std::string, std::size_t>;
  std::vector<std::vector<Term>> canonical(nodes_.size());
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].transforms == std::vector<IndexTransform>{{}}) {
      for (const auto& term : nodes_[i].expression.Terms()) {
        canonical[i].push_back(UnitCanonical(term));
      }
    }
  }
  std::vector<std::size_t> kept, deferred;
  std::map<Signature, Alias> aliases;
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].transforms != std::vector<IndexTransform>{{}}) {
      deferred.push_back(i);
      continue;
    }
    bool found = false;
    for (auto j : kept) {
      // Named outputs are observable results, and must remain definitions.
      // Only private intermediates participate in substitution aliases.
      if (!IsIntermediate(nodes_[i].output.name) ||
          !IsIntermediate(nodes_[j].output.name)) {
        continue;
      }
      const auto& a = canonical[i];
      const auto& b = canonical[j];
      if (a.empty() || a.size() != b.size() ||
          nodes_[i].output.indices.size() != nodes_[j].output.indices.size() ||
          nodes_[j].expression.Terms().front().coefficient == 0.0 ||
          a.front().coefficient == 0.0 || b.front().coefficient == 0.0) {
        continue;
      }
      std::map<Index, Index> free_names, bound_names;
      bool matches = true;
      for (std::size_t k = 0; k < a.size() && matches; ++k) {
        if (a[k].tensors.size() != b[k].tensors.size() ||
            a[k].summed_indices != b[k].summed_indices) {
          matches = false;
          break;
        }
        const IndexSet a_bound(
            a[k].summed_indices.begin(), a[k].summed_indices.end());
        const IndexSet b_bound(
            b[k].summed_indices.begin(), b[k].summed_indices.end());
        for (std::size_t operand = 0; operand < a[k].tensors.size() && matches;
             ++operand) {
          const auto& at = a[k].tensors[operand];
          const auto& bt = b[k].tensors[operand];
          if (at.name != bt.name || at.indices.size() != bt.indices.size()) {
            matches = false;
            break;
          }
          for (std::size_t axis = 0; axis < at.indices.size() && matches;
               ++axis) {
            const auto& ai = at.indices[axis];
            const auto& bi = bt.indices[axis];
            if (ai.domain != bi.domain ||
                a_bound.contains(ai) != b_bound.contains(bi)) {
              matches = false;
              break;
            }
            auto& names = a_bound.contains(ai) ? bound_names : free_names;
            if (k == 0) {
              const auto [it, inserted] = names.emplace(ai, bi);
              matches = inserted || it->second == bi;
            } else {
              matches = names.contains(ai) && names.at(ai) == bi;
            }
          }
        }
        if (k != 0) {
          const auto& ar = nodes_[i].expression.Terms();
          const auto& br = nodes_[j].expression.Terms();
          const double scale = ar[0].coefficient * a[0].coefficient /
              (br[0].coefficient * b[0].coefficient);
          const double actual = ar[k].coefficient * a[k].coefficient;
          const double expected = scale * br[k].coefficient * b[k].coefficient;
          // Compare relative coefficients. An absolute tolerance on their
          // cross-products incorrectly identifies small, unequal sums.
          matches = matches &&
              std::abs(actual - expected) <=
                  1.0e-12 * std::max(std::abs(actual), std::abs(expected));
        }
      }
      if (!matches || free_names.size() != nodes_[i].output.indices.size()) {
        continue;
      }
      std::vector<std::size_t> axes;
      for (const auto& index : nodes_[j].output.indices) {
        const auto& source = nodes_[i].output.indices;
        const auto where =
            std::ranges::find_if(source, [&](const auto& candidate) {
              return free_names.contains(candidate) &&
                  free_names.at(candidate) == index;
            });
        if (where == source.end()) {
          matches = false;
          break;
        }
        axes.push_back(static_cast<std::size_t>(where - source.begin()));
      }
      if (!matches) {
        continue;
      }
      const double scale = nodes_[i].expression.Terms().front().coefficient /
          nodes_[j].expression.Terms().front().coefficient *
          a.front().coefficient * b.front().coefficient;
      aliases[{nodes_[i].output.name, nodes_[i].output.indices.size()}] = {
          nodes_[j].output.name, std::move(axes), scale};
      found = true;
      break;
    }
    if (!found) {
      kept.push_back(i);
    }
  }
  ContractionGraph result({}, options_);
  for (auto i : kept) {
    result.nodes_.push_back(nodes_[i]);
  }
  for (auto i : deferred) {
    result.nodes_.push_back(nodes_[i]);
  }
  for (auto& node : result.nodes_) {
    auto terms = node.expression.Terms();
    for (auto& term : terms) {
      for (auto& tensor : term.tensors) {
        const auto found = aliases.find({tensor.name, tensor.indices.size()});
        if (found == aliases.end()) {
          continue;
        }
        const auto& alias = found->second;
        const auto old_indices = tensor.indices;
        tensor.name = alias.name;
        term.coefficient *= alias.scale;
        for (std::size_t axis = 0; axis < alias.axes.size(); ++axis) {
          tensor.indices[axis] = old_indices[alias.axes[axis]];
        }
      }
    }
    node.expression = Expression(std::move(terms));
  }
  return aliases.empty() ? result : result.MergeIntermediates();
}

ContractionGraph ContractionGraph::ExpandPermutations() const {
  ContractionGraph result({}, options_);
  for (const auto& node : nodes_) {
    Expression expanded;
    for (const auto& transform : node.transforms) {
      expanded = expanded +
          transform.coefficient *
              node.expression.RenameIndices(transform.names);
    }
    result.Add(node.output, std::move(expanded));
  }
  return result;
}

ContractionGraph ContractionGraph::InlineIntermediates() const {
  ContractionGraph result({}, options_);
  std::map<std::string, std::pair<Tensor, Expression>> definitions;
  for (const auto& node : nodes_) {
    if (node.transforms != std::vector<IndexTransform>{{}}) {
      throw std::invalid_argument(
          "Expand output permutations before inlining intermediates");
    }
    auto expanded = node.expression.Substitute(definitions);
    if (IsIntermediate(node.output.name)) {
      definitions[node.output.name] = {node.output, std::move(expanded)};
    } else {
      result.nodes_.push_back(
          {node.output, std::move(expanded), node.transforms});
    }
  }
  return result;
}

ContractionGraph ContractionGraph::TopologicalSort() const {
  using Signature = std::pair<std::string, std::size_t>;
  std::map<Signature, std::size_t> definitions;
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    definitions[{nodes_[i].output.name, nodes_[i].output.indices.size()}] = i;
  }
  std::vector<std::set<std::size_t>> dependencies(nodes_.size()),
      consumers(nodes_.size());
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    for (const auto& term : nodes_[i].expression.Terms()) {
      for (const auto& tensor : term.tensors) {
        const auto found =
            definitions.find({tensor.name, tensor.indices.size()});
        if (IsIntermediate(tensor.name) && found != definitions.end()) {
          dependencies[i].insert(found->second);
          consumers[found->second].insert(i);
        }
      }
    }
  }
  std::vector<std::size_t> unmet, uses, ready;
  std::vector<double> sizes;
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    unmet.push_back(dependencies[i].size());
    uses.push_back(consumers[i].size());
    // Tensor extent counts axes, including repeated output slots.
    double extent = 1.0;
    for (const auto& index : nodes_[i].output.indices) {
      extent *= Size({index}, options_);
    }
    sizes.push_back(extent);
    if (unmet.back() == 0) {
      ready.push_back(i);
    }
  }
  const double sentinel = std::accumulate(sizes.begin(), sizes.end(), 0.0);
  auto priority = [&](std::size_t candidate) {
    std::array<double, 4> savings{};
    for (auto dependency : dependencies[candidate]) {
      savings[0] -= uses[dependency] == 1 ? sizes[dependency] : 0;
      savings[1] -= unmet[dependency] == 0 ? sizes[dependency] : 0;
    }
    for (auto consumer : consumers[candidate]) {
      for (auto dependency : dependencies[consumer]) {
        if (dependency != candidate && unmet[dependency] == 0) {
          savings[2] -= sizes[consumer];
        }
      }
      savings[3] -= uses[consumer] == 1 ? sizes[consumer] : 0;
    }
    for (auto& saving : savings) {
      saving = saving == 0.0 ? sentinel : saving + sizes[candidate];
    }
    return savings;
  };
  ContractionGraph result({}, options_);
  for (std::size_t cursor = 0; cursor < ready.size(); ++cursor) {
    std::sort(
        ready.begin() + static_cast<std::ptrdiff_t>(cursor),
        ready.end(),
        [&](auto a, auto b) { return priority(a) < priority(b); });
    const auto selected = ready[cursor];
    result.nodes_.push_back(nodes_[selected]);
    for (auto dependency : dependencies[selected]) {
      --uses[dependency];
    }
    for (auto consumer : consumers[selected]) {
      if (--unmet[consumer] == 0) {
        ready.push_back(consumer);
      }
    }
  }
  if (result.nodes_.size() != nodes_.size()) {
    throw std::runtime_error("Contraction graph contains a dependency cycle");
  }
  return result;
}

ContractionGraph ContractionGraph::Simplify() const {
  return FactorPermutations()
      .OrderContractions()
      .SplitBinary()
      .MergeIntermediates()
      .FactorCommonOperands()
      .TopologicalSort();
}

ContractionGraph ContractionGraph::Expand() const {
  return ExpandPermutations().InlineIntermediates();
}

} // namespace wickqc::equation
