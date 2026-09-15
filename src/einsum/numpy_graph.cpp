#include "einsum/einsum.h"

#include "equation/equation.h"
#include "equation/graph.h"
#include "symbolic/wick.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <iomanip>
#include <map>
#include <numeric>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wickqc::einsum {
namespace {
using equation::ContractionGraph;
using equation::EquationNode;
using symbolic::Index;
using symbolic::Tensor;
using symbolic::TensorKind;

std::string Variable(const Tensor& tensor, const ContractionGraph& graph) {
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
std::string Joined(const Range& values) {
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

bool RequiresEinsum(const EquationNode& node) {
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

std::vector<ScheduledEquation> ScheduleEmission(const ContractionGraph& graph) {
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

std::string TransposeTo(
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

std::string AlignElementwise(
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

void EmitSimple(
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
} // namespace

std::string RenderNumpy(
    const ContractionGraph& graph,
    int coefficient_precision) {
  const auto schedule = ScheduleEmission(graph);
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
    const bool use_einsum = RequiresEinsum(node);
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
      EmitSimple(output, node, emission.first_assignment, graph);
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
          output << ".transpose(" << Joined(transpose) << ')';
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
