#include "einsum/einsum.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "equation/equation.h"
#include "symbolic/wick.h"

namespace wickqc::einsum {
namespace {

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

IndexIdentity Identity(const Index& index) {
  return {index.name, index.domain};
}

bool IsReduction(const equation::TensorEquationTerm& term, const Index& index) {
  return std::ranges::find(term.reduction_indices, index) !=
      term.reduction_indices.end();
}

char OrbitalCode(const IndexDomain& domain) {
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

std::string DomainSuffix(const IndexDomain& domain) {
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

std::string TensorVariable(
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

std::size_t CanonicalOrdinal(std::string_view name) {
  const auto separator = name.find_last_not_of("0123456789");
  std::size_t ordinal = 0;
  for (const char character : name.substr(separator + 1)) {
    ordinal = ordinal * 10 + static_cast<std::size_t>(character - '0');
  }
  return ordinal;
}

char DummyBase(const Index& index) {
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

std::string PreferredLabel(const Index& index) {
  if (index.name.size() == 1) {
    return index.name;
  }
  return std::string(
      1, static_cast<char>(DummyBase(index) + CanonicalOrdinal(index.name)));
}

std::uint8_t FirstOrbitalSpace(const Tensor& tensor) {
  return tensor.indices.empty() ? 0
                                : tensor.indices.front().domain.orbital_spaces;
}

int FermionOrder(const Tensor& tensor, std::uint8_t occupied_space) {
  const bool annihilation = tensor.kind == TensorKind::kAnnihilation;
  const bool in_occupied_space = !tensor.indices.empty() &&
      (tensor.indices.front().domain.orbital_spaces & occupied_space) != 0;
  return static_cast<int>(annihilation) |
      ((static_cast<int>(annihilation) ^ static_cast<int>(in_occupied_space))
       << 1);
}

bool IndexLess(const Index& lhs, const Index& rhs) {
  if (lhs.domain != rhs.domain) {
    return lhs.domain < rhs.domain;
  }
  return lhs.name < rhs.name;
}

bool TensorLess(const Tensor& lhs, const Tensor& rhs) {
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

bool EquationTermLess(
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

std::string ClaimLabel(const Index& index, std::set<std::string>& used_labels) {
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

void AssignPass(
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

Term LowerTerm(
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
  std::map<std::string, IndexId> ids;
  for (IndexId id = 0; id < result.index_labels.size(); ++id) {
    ids.emplace(result.index_labels[id], id);
  }
  auto index_ids = [&](const std::vector<Index>& indices) {
    std::vector<IndexId> result;
    result.reserve(indices.size());
    for (const auto& index : indices) {
      result.push_back(ids.at(labels.at(Identity(index))));
    }
    return result;
  };
  result.output_indices = index_ids(source.output.indices);
  std::set<IndexId> input_labels;
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

std::string FormatCoefficient(double coefficient, int precision) {
  if (coefficient == 1.0) {
    return {};
  }
  std::ostringstream output;
  output << std::setprecision(precision) << coefficient << " * ";
  return output.str();
}

std::string JoinIndices(
    const std::vector<IndexId>& indices,
    const std::vector<std::string>& labels) {
  std::string result;
  for (const auto& index : indices) {
    result += labels.at(index);
  }
  return result;
}

} // namespace

Program::Program(std::vector<Term> terms) : terms_(std::move(terms)) {}

const std::vector<Term>& Program::Terms() const noexcept {
  return terms_;
}

Program Program::Lower(
    const equation::TensorEquation& equation,
    const LoweringOptions& options) {
  std::vector<equation::TensorEquationTerm> source_terms = equation.Terms();
  if (!options.preserve_term_order) {
    std::ranges::stable_sort(source_terms, EquationTermLess);
  }
  std::vector<Term> terms;
  terms.reserve(source_terms.size());
  for (const auto& term : source_terms) {
    terms.push_back(LowerTerm(term, options));
  }
  return Program(std::move(terms));
}

std::string RenderNumpy(const Program& program, const NumpyOptions& options) {
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
    output << FormatCoefficient(
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
      output << JoinIndices(term.operands[position].indices, term.index_labels);
    }
    output << "->" << JoinIndices(term.output_indices, term.index_labels)
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

std::string AddIndent(std::string_view text, std::size_t spaces) {
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

} // namespace wickqc::einsum
