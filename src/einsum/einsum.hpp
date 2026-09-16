#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <map>
#include <numeric>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "equation/graph.hpp"
#include "symbolic/index_domain.hpp"
#include "symbolic/wick.hpp"

namespace wickqc::equation {
class ContractionGraph;
}

namespace wickqc::einsum {

using IndexID = std::uint32_t;

struct Operand {
  std::string variable;
  std::vector<IndexID> indices;
};

struct Term {
  double coefficient = 1.0;
  std::string output;
  std::vector<IndexID> output_indices;
  std::vector<Operand> operands;
  // Text labels are rendering metadata. Contractions refer only to IndexID.
  std::vector<std::string> index_labels;
  bool broadcasts_output = false;
};

struct LoweringOptions {
  std::string intermediate_prefix;
  bool preserve_term_order = false;
  // Private graph values must retain all output axes for later contractions.
  bool materialize_scalar_broadcast = false;
  // A graph can broadcast different orbital spaces at the same tensor rank.
  bool domain_identity_names = false;
  // Numerical consumers use integer IDs and need no NumPy label alphabet.
  bool numeric_indices_only = false;
};

struct NumPyOptions {
  bool initialize = false;
  std::string intermediate_prefix;
  bool separate_constants = true;
  int coefficient_precision = 6;
};

class Program {
 public:
  [[nodiscard]] static Program Lower(
      const equation::TensorEquation& equation,
      const LoweringOptions& options = {});

  [[nodiscard]] const std::vector<Term>& Terms() const noexcept;

 private:
  explicit Program(std::vector<Term> terms);

  std::vector<Term> terms_;
};

[[nodiscard]] std::string RenderNumpy(
    const Program& program,
    const NumPyOptions& options = {});
[[nodiscard]] std::string RenderNumpy(
    const equation::ContractionGraph& graph,
    int coefficient_precision = 6);
// Indent every line, including a final empty line, as in the reference emitter.
[[nodiscard]] std::string AddIndent(
    std::string_view text,
    std::size_t spaces = 4);

namespace einsum_detail {

using symbolic::Index;
using symbolic::IndexDomain;
using symbolic::OrbitalSpace;
using symbolic::Spin;
using symbolic::Tensor;
using symbolic::TensorKind;

struct IndexIdentity {
  std::string name;
  IndexDomain domain;

  auto operator<=>(const IndexIdentity&) const = default;
};

inline IndexIdentity Identity(const Index& index) {
  return {index.name, index.domain};
}

inline bool IsReduction(
    const equation::TensorEquationTerm& term,
    const Index& index) {
  return std::ranges::find(term.reduction_indices, index) !=
      term.reduction_indices.end();
}

inline char OrbitalCode(const IndexDomain& domain) {
  const auto spaces = domain.orbital_spaces;
  if (spaces == static_cast<std::uint8_t>(OrbitalSpace::kInactive)) {
    return 'I';
  }
  if (spaces == static_cast<std::uint8_t>(OrbitalSpace::kActive)) {
    return 'A';
  }
  if (spaces == static_cast<std::uint8_t>(OrbitalSpace::kSingle)) {
    return 'S';
  }
  if (spaces == static_cast<std::uint8_t>(OrbitalSpace::kExternal)) {
    return 'E';
  }
  if (spaces == 0) {
    return 'N';
  }
  throw std::invalid_argument("Einsum lowering requires concrete index spaces");
}

inline std::string DomainSuffix(const IndexDomain& domain) {
  char orbital = OrbitalCode(domain);
  if (domain.spins == 0) {
    return std::string(1, orbital);
  }
  if (domain.spins == static_cast<std::uint8_t>(Spin::kAlpha)) {
    if (orbital == 'N') {
      return "A";
    }
    return std::string(1, static_cast<char>(std::tolower(orbital)));
  }
  if (domain.spins == static_cast<std::uint8_t>(Spin::kBeta)) {
    if (orbital == 'N') {
      return "B";
    }
    return std::string(1, orbital);
  }
  throw std::invalid_argument("Einsum lowering requires concrete spin spaces");
}

inline std::string TensorVariable(
    const Tensor& tensor,
    const std::string& intermediate_prefix) {
  std::string variable = tensor.name;
  if (tensor.kind == TensorKind::kSpinFree || tensor.indices.empty() ||
      (!intermediate_prefix.empty() &&
       tensor.name.starts_with(intermediate_prefix))) {
    return variable;
  }
  for (const auto& index : tensor.indices) {
    variable += DomainSuffix(index.domain);
  }
  return variable;
}

inline std::size_t CanonicalOrdinal(std::string_view name) {
  const auto separator = name.find_last_not_of("0123456789");
  std::size_t ordinal = 0;
  for (const char character : name.substr(separator + 1)) {
    ordinal = ordinal * 10 + static_cast<std::size_t>(character - '0');
  }
  return ordinal;
}

inline char DummyBase(const Index& index) {
  const char orbital = OrbitalCode(index.domain);
  if (orbital == 'I' || orbital == 'S') {
    return 'i';
  }
  if (orbital == 'A') {
    return 'a';
  }
  if (orbital == 'E') {
    return 'r';
  }
  return 'p';
}

inline std::string PreferredLabel(const Index& index) {
  if (index.name.size() == 1) {
    return index.name;
  }
  return std::string(
      1, static_cast<char>(DummyBase(index) + CanonicalOrdinal(index.name)));
}

inline std::uint8_t FirstOrbitalSpace(const Tensor& tensor) {
  return tensor.indices.empty() ? 0
                                : tensor.indices.front().domain.orbital_spaces;
}

inline int FermionOrder(const Tensor& tensor, std::uint8_t occupied_space) {
  const bool annihilation = tensor.kind == TensorKind::kAnnihilation;
  const bool in_occupied_space = !tensor.indices.empty() &&
      (tensor.indices.front().domain.orbital_spaces & occupied_space) != 0;
  return static_cast<int>(annihilation) |
      ((static_cast<int>(annihilation) ^ static_cast<int>(in_occupied_space))
       << 1);
}

inline bool IndexLess(const Index& lhs, const Index& rhs) {
  if (lhs.domain != rhs.domain) {
    return lhs.domain < rhs.domain;
  }
  return lhs.name < rhs.name;
}

inline bool TensorLess(const Tensor& lhs, const Tensor& rhs) {
  std::uint8_t occupied_space =
      std::min(FirstOrbitalSpace(lhs), FirstOrbitalSpace(rhs));
  const auto inactive = static_cast<std::uint8_t>(OrbitalSpace::kInactive);
  const auto active = static_cast<std::uint8_t>(OrbitalSpace::kActive);
  const auto external = static_cast<std::uint8_t>(OrbitalSpace::kExternal);
  if (occupied_space == 0 || occupied_space == external ||
      (occupied_space == active && FirstOrbitalSpace(lhs) == active &&
       FirstOrbitalSpace(rhs) == active)) {
    occupied_space = inactive;
  }
  const int lhs_fermion_order = FermionOrder(lhs, occupied_space);
  const int rhs_fermion_order = FermionOrder(rhs, occupied_space);
  if (lhs_fermion_order != rhs_fermion_order) {
    return lhs_fermion_order < rhs_fermion_order;
  }
  if (lhs.name != rhs.name) {
    return lhs.name < rhs.name;
  }
  if (lhs.kind != rhs.kind) {
    return lhs.kind < rhs.kind;
  }
  return std::lexicographical_compare(
      lhs.indices.begin(),
      lhs.indices.end(),
      rhs.indices.begin(),
      rhs.indices.end(),
      IndexLess);
}

inline bool EquationTermLess(
    const equation::TensorEquationTerm& lhs,
    const equation::TensorEquationTerm& rhs) {
  if (lhs.inputs.size() != rhs.inputs.size()) {
    return lhs.inputs.size() < rhs.inputs.size();
  }
  if (lhs.reduction_indices.size() != rhs.reduction_indices.size()) {
    return lhs.reduction_indices.size() < rhs.reduction_indices.size();
  }
  return std::lexicographical_compare(
      lhs.inputs.begin(),
      lhs.inputs.end(),
      rhs.inputs.begin(),
      rhs.inputs.end(),
      TensorLess);
}

inline std::string ClaimLabel(
    const Index& index,
    std::set<std::string>& used_labels) {
  std::string label = PreferredLabel(index);
  if (label.size() != 1) {
    throw std::invalid_argument(
        "NumPy einsum currently requires one-character index names");
  }
  while (used_labels.contains(label)) {
    ++label.front();
  }
  if (!std::isalpha(static_cast<unsigned char>(label.front()))) {
    throw std::invalid_argument("NumPy einsum exhausted index labels");
  }
  used_labels.insert(label);
  return label;
}

inline void AssignPass(
    const equation::TensorEquationTerm& source,
    bool reductions,
    std::map<IndexIdentity, std::string>& labels,
    std::set<std::string>& used_labels) {
  for (const auto& tensor : source.inputs) {
    for (const auto& index : tensor.indices) {
      const auto identity = Identity(index);
      if (IsReduction(source, index) != reductions ||
          labels.contains(identity)) {
        continue;
      }
      labels.emplace(identity, ClaimLabel(index, used_labels));
    }
  }
}

inline Term LowerTerm(
    const equation::TensorEquationTerm& source,
    const LoweringOptions& options) {
  std::map<IndexIdentity, std::string> labels;
  std::set<std::string> used_labels;
  if (!options.numeric_indices_only) {
    AssignPass(source, false, labels, used_labels);
    AssignPass(source, true, labels, used_labels);
  } else {
    for (const auto& tensor : source.inputs) {
      for (const auto& index : tensor.indices) {
        const auto identity = Identity(index);
        if (!labels.contains(identity)) {
          auto label = std::to_string(labels.size());
          used_labels.insert(label);
          labels.emplace(identity, std::move(label));
        }
      }
    }
  }
  for (const auto& index : source.output.indices) {
    const auto identity = Identity(index);
    if (!labels.contains(identity)) {
      auto label = options.numeric_indices_only
          ? std::to_string(labels.size())
          : ClaimLabel(index, used_labels);
      used_labels.insert(label);
      labels.emplace(identity, std::move(label));
    }
  }

  Term result;
  result.coefficient = source.coefficient;
  result.output = source.output.name;
  result.index_labels.assign(used_labels.begin(), used_labels.end());
  std::map<std::string, IndexID> ids;
  for (IndexID id = 0; id < result.index_labels.size(); ++id) {
    ids.emplace(result.index_labels[id], id);
  }
  auto index_ids = [&](const std::vector<Index>& indices) {
    std::vector<IndexID> result;
    result.reserve(indices.size());
    for (const auto& index : indices) {
      result.push_back(ids.at(labels.at(Identity(index))));
    }
    return result;
  };
  result.output_indices = index_ids(source.output.indices);
  std::set<IndexID> input_labels;
  for (const auto& tensor : source.inputs) {
    auto indices = index_ids(tensor.indices);
    input_labels.insert(indices.begin(), indices.end());
    result.operands.push_back(
        {TensorVariable(tensor, options.intermediate_prefix),
         std::move(indices)});
  }

  Operand identity;
  for (const auto& label : result.output_indices) {
    if (!input_labels.contains(label)) {
      identity.indices.push_back(label);
    }
  }
  // Scalar right-hand sides broadcast directly in NumPy. Identity operands
  // are needed only to supply missing axes of a tensor contraction.
  const bool scalar_rhs = std::ranges::all_of(
      source.inputs,
      [](const Tensor& tensor) { return tensor.indices.empty(); });
  result.broadcasts_output = !identity.indices.empty();
  if (!identity.indices.empty() &&
      (!scalar_rhs || options.materialize_scalar_broadcast)) {
    identity.variable = "ident" + std::to_string(identity.indices.size());
    if (options.domain_identity_names) {
      for (const auto id : identity.indices) {
        const auto axis = std::ranges::find(result.output_indices, id) -
            result.output_indices.begin();
        identity.variable += DomainSuffix(source.output.indices[axis].domain);
      }
    }
    result.operands.push_back(std::move(identity));
  }
  return result;
}

inline std::string FormatCoefficient(double coefficient, int precision) {
  if (coefficient == 1.0) {
    return {};
  }
  std::ostringstream output;
  output << std::setprecision(precision) << coefficient << " * ";
  return output.str();
}

inline std::string JoinIndices(
    const std::vector<IndexID>& indices,
    const std::vector<std::string>& labels) {
  std::string result;
  for (const auto& index : indices) {
    result += labels.at(index);
  }
  return result;
}

} // namespace einsum_detail

inline Program::Program(std::vector<Term> terms) : terms_(std::move(terms)) {}

inline const std::vector<Term>& Program::Terms() const noexcept {
  return terms_;
}

inline Program Program::Lower(
    const equation::TensorEquation& equation,
    const LoweringOptions& options) {
  std::vector<equation::TensorEquationTerm> source_terms = equation.Terms();
  if (!options.preserve_term_order) {
    std::ranges::stable_sort(source_terms, einsum_detail::EquationTermLess);
  }
  std::vector<Term> terms;
  terms.reserve(source_terms.size());
  for (const auto& term : source_terms) {
    terms.push_back(einsum_detail::LowerTerm(term, options));
  }
  return Program(std::move(terms));
}

inline std::string RenderNumpy(
    const Program& program,
    const NumPyOptions& options) {
  for (const auto& term : program.Terms()) {
    for (const auto& label : term.index_labels) {
      if (label.size() != 1 ||
          !std::isalpha(static_cast<unsigned char>(label.front()))) {
        throw std::invalid_argument(
            "NumPy rendering requires lowering with numeric_indices_only=false");
      }
    }
  }
  std::ostringstream output;
  output << std::setprecision(options.coefficient_precision);
  bool initialize = options.initialize;
  const bool broadcast = std::ranges::any_of(
      program.Terms(), [](const auto& term) { return term.broadcasts_output; });
  for (const auto& term : program.Terms()) {
    const bool first_assignment = initialize;
    output << term.output;
    if (initialize) {
      output << " = ";
    } else if (
        broadcast && !options.intermediate_prefix.empty() &&
        term.output.starts_with(options.intermediate_prefix)) {
      output << " = " << term.output << " + ";
    } else {
      output << " += ";
    }
    initialize = false;
    if (term.operands.empty()) {
      output << term.coefficient << '\n';
      continue;
    }
    output << einsum_detail::FormatCoefficient(
        term.coefficient, options.coefficient_precision);
    const auto scalar_count = std::ranges::count_if(
        term.operands,
        [](const Operand& operand) { return operand.indices.empty(); });
    std::size_t scalar_position = 0;
    const bool scalar_only = scalar_count == term.operands.size();
    for (const auto& operand : term.operands) {
      if (!operand.indices.empty() ||
          (!options.separate_constants && !scalar_only)) {
        continue;
      }
      output << operand.variable;
      if (++scalar_position < scalar_count ||
          scalar_count != term.operands.size()) {
        output << " * ";
      }
    }
    if (scalar_count == term.operands.size()) {
      output << '\n';
      continue;
    }
    output << "np.einsum('";
    bool first_operand = true;
    for (std::size_t position = 0; position < term.operands.size();
         ++position) {
      if (options.separate_constants &&
          term.operands[position].indices.empty()) {
        continue;
      }
      if (!first_operand) {
        output << ',';
      }
      first_operand = false;
      output << einsum_detail::JoinIndices(
          term.operands[position].indices, term.index_labels);
    }
    output << "->"
           << einsum_detail::JoinIndices(term.output_indices, term.index_labels)
           << "'";
    for (const auto& operand : term.operands) {
      if (options.separate_constants && operand.indices.empty()) {
        continue;
      }
      output << ", " << operand.variable;
    }
    output << ", optimize=True)";
    // NumPy can return a writable view for a unary permutation or diagonal.
    // An initialized output owns its value before subsequent accumulation.
    if (first_assignment && term.coefficient == 1.0 &&
        term.operands.size() == 1) {
      output << ".copy()";
    }
    output << '\n';
  }
  return output.str();
}

inline std::string AddIndent(std::string_view text, std::size_t spaces) {
  const std::string padding(spaces, ' ');
  std::string result;
  for (std::size_t position = 0;;) {
    const auto next = text.find('\n', position);
    result += padding;
    result += text.substr(
        position, next == std::string_view::npos ? next : next - position);
    result += '\n';
    if (next == std::string_view::npos) {
      return result;
    }
    position = next + 1;
  }
}

namespace numpy_graph_detail {
using equation::ContractionGraph;
using equation::EquationNode;
using symbolic::Index;
using symbolic::Tensor;
using symbolic::TensorKind;

inline std::string Variable(
    const Tensor& tensor,
    const ContractionGraph& graph) {
  std::string name = tensor.name;
  if (graph.IsIntermediate(name) ||
      (tensor.kind != TensorKind::kGeneric &&
       tensor.kind != TensorKind::kDelta)) {
    return name;
  }
  for (const auto& index : tensor.indices) {
    const auto& domain = index.domain;
    std::string code;
    for (const auto& [mask, letter] : std::array<std::pair<unsigned, char>, 4>{
             {{8, 'E'}, {1, 'I'}, {2, 'A'}, {4, 'S'}}}) {
      if ((domain.orbital_spaces & mask) != 0) {
        code += letter;
      }
    }
    if (code.empty()) {
      code = domain.spins == 0 ? "N" : domain.spins == 1 ? "A" : "B";
    } else if (domain.spins == 1) {
      for (auto& letter : code) {
        letter =
            static_cast<char>(std::tolower(static_cast<unsigned char>(letter)));
      }
    }
    name += code;
  }
  return name;
}

template <typename Range>
inline std::string Joined(const Range& values) {
  std::ostringstream output;
  bool first = true;
  for (const auto& value : values) {
    if (!first) {
      output << ", ";
    }
    first = false;
    output << value;
  }
  return output.str();
}

inline bool RequiresEinsum(const EquationNode& node) {
  for (const auto& term : node.expression.Terms()) {
    const auto tensor_count = std::ranges::count_if(
        term.tensors,
        [](const auto& tensor) { return !tensor.indices.empty(); });
    if (tensor_count == 0 && !node.output.indices.empty()) {
      return true;
    }
    if (tensor_count > 2) {
      return true;
    }
    std::map<Index, std::size_t> counts;
    for (const auto& tensor : term.tensors) {
      for (const auto& index : tensor.indices) {
        ++counts[index];
      }
    }
    // A unary reduction/diagonal is not a binary tensordot operation.
    if (tensor_count == 1 &&
        (!term.summed_indices.empty() ||
         std::ranges::any_of(
             counts, [](const auto& count) { return count.second > 1; }) ||
         std::ranges::any_of(node.output.indices, [&](const auto& index) {
           return !counts.contains(index);
         }))) {
      return true;
    }
    if (std::ranges::any_of(node.output.indices, [&](const auto& index) {
          return !counts.contains(index);
        })) {
      return true;
    }
    if (term.summed_indices.empty()) {
      continue;
    }
    const std::set<Index> reduced(
        term.summed_indices.begin(), term.summed_indices.end());
    // Tensordot pairs one occurrence in each operand. A reduction confined to
    // one operand, including an internal trace, needs the general lowering.
    for (const auto& tensor : term.tensors) {
      if (tensor.indices.empty()) {
        continue;
      }
      for (const auto& index : reduced) {
        if (std::ranges::count(tensor.indices, index) != 1) {
          return true;
        }
      }
    }
    for (const auto& [index, count] : counts) {
      if (count > 2 || (count == 2 && !reduced.contains(index))) {
        return true;
      }
    }
    for (const auto& index : node.output.indices) {
      if (!counts.contains(index)) {
        return true;
      }
    }
  }
  return false;
}

struct ScheduledEquation {
  EquationNode node;
  bool first_assignment = true;
};

inline std::vector<ScheduledEquation> ScheduleEmission(
    const ContractionGraph& graph) {
  const auto& nodes = graph.Nodes();
  std::map<std::string, int> definitions;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (graph.IsIntermediate(nodes[i].output.name)) {
      definitions[nodes[i].output.name] = static_cast<int>(i);
    }
  }
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>> partials(
      nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto& terms = nodes[i].expression.Terms();
    if (RequiresEinsum(nodes[i]) || terms.size() <= 1) {
      continue;
    }
    std::vector<int> availability;
    int anchor = -1;
    for (const auto& term : terms) {
      int available = -1;
      for (const auto& tensor : term.tensors) {
        if (definitions.contains(tensor.name)) {
          available = std::max(available, definitions.at(tensor.name));
        }
      }
      availability.push_back(available);
      anchor = anchor < 0 ? available : std::min(anchor, available);
    }
    if (anchor < 0) {
      anchor = static_cast<int>(i);
    }
    for (std::size_t term = 0; term < terms.size(); ++term) {
      partials[availability[term] < 0 ? anchor : availability[term]]
          .emplace_back(i, term);
    }
  }
  std::vector<std::size_t> emitted(nodes.size());
  std::vector<ScheduledEquation> schedule;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (RequiresEinsum(nodes[i]) || nodes[i].expression.Terms().size() <= 1) {
      schedule.push_back({nodes[i], true});
    }
    for (const auto& [source, term] : partials[i]) {
      EquationNode node{
          nodes[source].output,
          symbolic::Expression(nodes[source].expression.Terms()[term])};
      const bool first = emitted[source]++ == 0;
      if (emitted[source] == nodes[source].expression.Terms().size()) {
        node.transforms = nodes[source].transforms;
      }
      schedule.push_back({std::move(node), first});
    }
  }
  return schedule;
}

inline std::string TransposeTo(
    const std::vector<Index>& from,
    const std::vector<Index>& to) {
  std::vector<std::size_t> axes;
  bool identity = true;
  for (const auto& index : to) {
    const auto found = std::ranges::find(from, index);
    if (found == from.end()) {
      throw std::invalid_argument(
          "Graph output index is absent from the contraction result");
    }
    axes.push_back(static_cast<std::size_t>(found - from.begin()));
    identity = identity && axes.back() == axes.size() - 1;
  }
  return identity ? "" : ".transpose(" + Joined(axes) + ")";
}

inline std::string AlignElementwise(
    Tensor tensor,
    std::string value,
    const Tensor& output) {
  // Diagonal extraction moves the retained diagonal axis to the end, as NumPy
  // does; subsequent transposition/broadcasting uses that updated axis list.
  for (;;) {
    std::map<Index, std::size_t> seen;
    bool extracted = false;
    for (std::size_t axis = 0; axis < tensor.indices.size(); ++axis) {
      const auto [it, inserted] = seen.emplace(tensor.indices[axis], axis);
      if (inserted) {
        continue;
      }
      value.insert(0, "np.diagonal(");
      value += ", 0, " + std::to_string(it->second) + ", " +
          std::to_string(axis) + ")";
      const auto diagonal = tensor.indices[axis];
      tensor.indices.erase(
          tensor.indices.begin() + static_cast<std::ptrdiff_t>(axis));
      tensor.indices.erase(
          tensor.indices.begin() + static_cast<std::ptrdiff_t>(it->second));
      tensor.indices.push_back(diagonal);
      extracted = true;
      break;
    }
    if (!extracted) {
      break;
    }
  }
  std::vector<Index> present;
  std::vector<std::string> broadcast;
  for (const auto& index : output.indices) {
    if (std::ranges::find(tensor.indices, index) != tensor.indices.end()) {
      present.push_back(index);
      broadcast.emplace_back(":");
    } else {
      broadcast.emplace_back("None");
    }
  }
  value += TransposeTo(tensor.indices, present);
  if (tensor.indices.size() < output.indices.size()) {
    value += "[" + Joined(broadcast) + "]";
  }
  return value;
}

inline void EmitSimple(
    std::ostream& output,
    const EquationNode& node,
    bool first_assignment,
    const ContractionGraph& graph) {
  bool initialize = graph.IsIntermediate(node.output.name) && first_assignment;
  for (const auto& term : node.expression.Terms()) {
    output << node.output.name;
    if (initialize) {
      output << " = ";
    } else if (
        graph.IsIntermediate(node.output.name) && node.output.indices.empty()) {
      output << " = " << node.output.name << " + ";
    } else {
      output << " += ";
    }
    const bool unit_coefficient = term.coefficient == 1.0;
    std::vector<Tensor> operands;
    for (const auto& tensor : term.tensors) {
      if (!tensor.indices.empty()) {
        operands.push_back(tensor);
      }
    }
    const bool scalar_array =
        initialize && node.output.indices.empty() && operands.empty();
    if (scalar_array) {
      output << "np.asarray(";
    }
    if (term.tensors.empty()) {
      output << term.coefficient
             << (scalar_array ? ", dtype=np.float64)\n" : "\n");
      initialize = false;
      continue;
    }
    if (!unit_coefficient) {
      output << term.coefficient << " * ";
    }
    std::vector<std::string> scalars;
    for (const auto& tensor : term.tensors) {
      if (tensor.indices.empty()) {
        scalars.push_back(tensor.name);
      }
    }
    for (std::size_t i = 0; i < scalars.size(); ++i) {
      output << scalars[i];
      if (i + 1 < scalars.size() || !operands.empty()) {
        output << " * ";
      }
    }
    if (operands.empty()) {
      output << (scalar_array ? ").copy()\n" : "\n");
      initialize = false;
      continue;
    }
    std::vector<Index> axes;
    if (operands.size() == 1) {
      output << Variable(operands.front(), graph);
      axes = operands.front().indices;
    } else if (term.summed_indices.empty()) {
      output << "np.multiply("
             << AlignElementwise(
                    operands[0], Variable(operands[0], graph), node.output)
             << ", "
             << AlignElementwise(
                    operands[1], Variable(operands[1], graph), node.output)
             << ')';
      axes = node.output.indices;
    } else {
      const std::set<Index> reduced(
          term.summed_indices.begin(), term.summed_indices.end());
      std::array<std::vector<std::size_t>, 2> contracted_axes;
      for (std::size_t operand = 0; operand < 2; ++operand) {
        for (const auto& index : reduced) {
          const auto where =
              std::ranges::find(operands[operand].indices, index);
          if (where == operands[operand].indices.end()) {
            throw std::invalid_argument(
                "Binary contraction reduction is missing from an operand");
          }
          contracted_axes[operand].push_back(
              static_cast<std::size_t>(
                  where - operands[operand].indices.begin()));
        }
        for (const auto& index : operands[operand].indices) {
          if (!reduced.contains(index)) {
            axes.push_back(index);
          }
        }
      }
      auto axis_group = [&](std::size_t i) {
        const auto text = Joined(contracted_axes[i]);
        return reduced.size() == 1 ? text : "(" + text + ")";
      };
      output << "np.tensordot(" << Variable(operands[0], graph) << ", "
             << Variable(operands[1], graph) << ", axes=(" << axis_group(0)
             << ", " << axis_group(1) << "))";
    }
    output << TransposeTo(axes, node.output.indices);
    if (operands.size() == 1 && initialize && unit_coefficient &&
        scalars.empty()) {
      output << ".copy()";
    }
    output << '\n';
    initialize = false;
  }
}
} // namespace numpy_graph_detail

inline std::string RenderNumpy(
    const equation::ContractionGraph& graph,
    int coefficient_precision) {
  const auto schedule = numpy_graph_detail::ScheduleEmission(graph);
  std::map<std::string, std::size_t> last_use;
  for (std::size_t i = 0; i < schedule.size(); ++i) {
    for (const auto& term : schedule[i].node.expression.Terms()) {
      for (const auto& tensor : term.tensors) {
        if (graph.IsIntermediate(tensor.name)) {
          last_use[tensor.name] = i;
        }
      }
    }
  }
  std::ostringstream output;
  output << std::setprecision(coefficient_precision);
  int scratch_id = graph.LastIntermediateId();
  for (std::size_t i = 0; i < schedule.size(); ++i) {
    const auto& emission = schedule[i];
    auto node = emission.node;
    const bool intermediate = graph.IsIntermediate(node.output.name);
    if (intermediate && node.expression.Empty()) {
      node.expression = symbolic::Expression(symbolic::Term{0.0, {}, {}});
    }
    const bool use_einsum = numpy_graph_detail::RequiresEinsum(node);
    if (use_einsum) {
      const auto equation = equation::TensorEquation::FromExpression(
          node.expression, node.output);
      output << RenderNumpy(
          Program::Lower(
              equation,
              {graph.Options().intermediate_prefix, true, intermediate, true}),
          {intermediate && emission.first_assignment,
           graph.Options().intermediate_prefix,
           true,
           coefficient_precision});
    } else {
      numpy_graph_detail::EmitSimple(
          output, node, emission.first_assignment, graph);
    }
    if (node.transforms != std::vector<equation::IndexTransform>{{}}) {
      const auto scratch =
          graph.Options().intermediate_prefix + std::to_string(++scratch_id);
      output << scratch << " = " << node.output.name << ".copy()\n";
      for (std::size_t permutation = 0; permutation < node.transforms.size();
           ++permutation) {
        const auto& transform = node.transforms[permutation];
        output << node.output.name;
        if (permutation == 0) {
          if (intermediate && node.output.indices.empty()) {
            output << " = ";
          } else {
            output << (node.output.indices.empty() ? "[...] = " : "[:] = ");
          }
        } else if (
            intermediate && (use_einsum || node.output.indices.empty())) {
          output << " = " << node.output.name << " + ";
        } else {
          output << " += ";
        }
        if (transform.coefficient != 1.0) {
          output << transform.coefficient << " * ";
        }
        output << scratch;
        if (!transform.names.empty()) {
          std::map<std::string, std::size_t> positions;
          for (std::size_t axis = 0; axis < node.output.indices.size();
               ++axis) {
            positions[node.output.indices[axis].name] = axis;
          }
          std::vector<std::size_t> transpose(node.output.indices.size());
          std::iota(transpose.begin(), transpose.end(), 0);
          for (const auto& [from, to] : transform.names) {
            transpose.at(positions.at(to)) = positions.at(from);
          }
          output << ".transpose(" << numpy_graph_detail::Joined(transpose)
                 << ')';
        }
        output << '\n';
      }
      output << scratch << " = None\n";
    }
    std::set<std::string> released;
    for (const auto& term : node.expression.Terms()) {
      for (const auto& tensor : term.tensors) {
        if (last_use.contains(tensor.name) && last_use.at(tensor.name) == i &&
            released.insert(tensor.name).second) {
          output << tensor.name << " = None\n";
        }
      }
    }
  }
  return output.str();
}

} // namespace wickqc::einsum
