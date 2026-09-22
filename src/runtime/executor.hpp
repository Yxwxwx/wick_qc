#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "backend/ndarray.hpp"
#include "einsum/einsum.hpp"
#include "equation/graph.hpp"
#include "runtime/numeric.hpp"
#include "symbolic/wick.hpp"

namespace wickqc::codegen {
class CPPEmitter;
}

namespace wickqc::runtime {

// Compile once after Wick expansion, optionally from graph.Simplify().
// Evaluation performs no symbolic algebra or string einsum parsing. Each call
// owns its outputs and intermediates; caller inputs (including views) stay
// intact. double and complex<double> are supported by every GEMM backend.
class NDArrayExecutor {
 public:
  [[nodiscard]] static NDArrayExecutor Compile(
      const equation::ContractionGraph& graph);
  [[nodiscard]] const std::vector<TensorBinding>& Inputs() const noexcept {
    return inputs_;
  }
  [[nodiscard]] const std::vector<TensorBinding>& Outputs() const noexcept {
    return outputs_;
  }
  [[nodiscard]] std::size_t WorkspaceElements(
      const Dimensions& dimensions) const {
    return runtime::WorkspaceElements(workspace_, dimensions);
  }

  template <typename T>
  [[nodiscard]] TensorMap<T> Evaluate(
      const TensorMap<T>& inputs,
      const Dimensions& dimensions) const;

 private:
  friend class codegen::CPPEmitter;
  enum class Source : std::uint8_t { kInput, kValue, kOnes, kDelta };
  struct Input {
    TensorBinding binding;
    Source source = Source::kInput;
  };
  struct Contraction {
    double coefficient = 1.0;
    std::vector<Input> operands;
    std::vector<std::vector<int>> indices;
    std::vector<int> output_indices;
  };
  struct Transform {
    double coefficient = 1.0;
    std::vector<int> axes;
  };
  struct Assignment {
    TensorBinding output;
    std::vector<Contraction> terms;
    std::vector<Transform> transforms;
    std::vector<std::string> release;
  };
  std::vector<TensorBinding> inputs_;
  std::vector<TensorBinding> outputs_;
  std::vector<Assignment> assignments_;
  std::vector<WorkspaceTerm> workspace_;
};

namespace ndarray_executor_detail {
inline TensorBinding Binding(
    const std::string& name,
    const symbolic::Tensor& tensor) {
  TensorBinding result{name, {}};
  for (const auto& index : tensor.indices) {
    result.domains.push_back(index.domain);
  }
  return result;
}

inline std::vector<int> Labels(const std::vector<einsum::IndexID>& indices) {
  std::vector<int> result;
  for (const auto id : indices) {
    if (id > static_cast<einsum::IndexID>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("NDArray index identifier exceeds int range");
    }
    result.push_back(static_cast<int>(id));
  }
  return result;
}
} // namespace ndarray_executor_detail

inline NDArrayExecutor NDArrayExecutor::Compile(
    const equation::ContractionGraph& graph) {
  NDArrayExecutor result;
  std::map<std::string, TensorBinding> definitions;
  for (const auto& node : graph.Nodes()) {
    if (node.output.kind != symbolic::TensorKind::kGeneric) {
      throw std::invalid_argument(
          "Graph output must be a coefficient tensor: '" + node.output.name +
          "'");
    }
    const auto binding =
        ndarray_executor_detail::Binding(node.output.name, node.output);
    if (!definitions.emplace(binding.name, binding).second) {
      throw std::invalid_argument(
          "Duplicate graph assignment '" + binding.name + "'");
    }
    if (!graph.IsIntermediate(binding.name)) {
      result.outputs_.push_back(binding);
    }
  }
  // TopologicalSort handles graph intermediates; public results can also feed
  // later nodes. Schedule by the actual lowered variable names for both kinds.
  std::map<std::string, TensorBinding> required;
  std::vector<Assignment> pending;
  for (const auto& node : graph.Nodes()) {
    const auto equation =
        equation::TensorEquation::FromExpression(node.expression, node.output);
    const auto program = einsum::Program::Lower(
        equation,
        {graph.Options().intermediate_prefix, true, true, true, true});
    Assignment assignment{
        ndarray_executor_detail::Binding(node.output.name, node.output),
        {},
        {},
        {}};
    std::set<symbolic::Index> output_axes;
    for (const auto& index : node.output.indices) {
      if (!output_axes.insert(index).second || !index.domain.IsConcrete()) {
        throw std::invalid_argument(
            "Invalid output axes for '" + node.output.name + "'");
      }
    }
    for (std::size_t t = 0; t < program.Terms().size(); ++t) {
      const auto& term = program.Terms()[t];
      const auto& tensors = equation.Terms()[t].inputs;
      Contraction contraction{
          term.coefficient,
          {},
          {},
          ndarray_executor_detail::Labels(term.output_indices)};
      for (std::size_t i = 0; i < term.operands.size(); ++i) {
        const auto& operand = term.operands[i];
        Input input;
        if (i >= tensors.size()) {
          input.source = Source::kOnes;
          input.binding.name = operand.variable;
          for (const auto id : operand.indices) {
            const auto axis = std::ranges::find(term.output_indices, id) -
                term.output_indices.begin();
            input.binding.domains.push_back(
                node.output.indices.at(axis).domain);
          }
        } else {
          const auto& tensor = tensors[i];
          input.binding =
              ndarray_executor_detail::Binding(operand.variable, tensor);
          if (tensor.kind == symbolic::TensorKind::kDelta) {
            if (tensor.indices.size() != 2 ||
                tensor.indices[0].domain != tensor.indices[1].domain) {
              throw std::invalid_argument(
                  "Delta '" + operand.variable +
                  "' must have two matching domains");
            }
            input.source = Source::kDelta;
          } else if (definitions.contains(operand.variable)) {
            input.source = Source::kValue;
            if (definitions.at(operand.variable) != input.binding) {
              throw std::invalid_argument(
                  "Graph value domain mismatch for '" + operand.variable + "'");
            }
          } else {
            if (graph.IsIntermediate(tensor.name)) {
              throw std::invalid_argument(
                  "Undefined graph intermediate '" + tensor.name + "'");
            }
            const auto [it, inserted] =
                required.emplace(operand.variable, input.binding);
            if (!inserted && it->second != input.binding) {
              throw std::invalid_argument(
                  "Input domain mismatch for '" + operand.variable + "'");
            }
          }
        }
        contraction.operands.push_back(std::move(input));
        contraction.indices.push_back(
            ndarray_executor_detail::Labels(operand.indices));
      }
      assignment.terms.push_back(std::move(contraction));
    }
    for (const auto& transform : node.transforms) {
      Transform permutation{transform.coefficient, {}};
      if (transform.names.empty()) {
        permutation.axes.resize(node.output.indices.size());
        std::iota(permutation.axes.begin(), permutation.axes.end(), 0);
        assignment.transforms.push_back(std::move(permutation));
        continue;
      }
      std::vector<std::string> renamed;
      for (const auto& index : node.output.indices) {
        const auto found = transform.names.find(index.name);
        renamed.push_back(
            found == transform.names.end() ? index.name : found->second);
      }
      std::set<int> used;
      for (const auto& index : node.output.indices) {
        const auto found = std::ranges::find(renamed, index.name);
        const int axis = static_cast<int>(found - renamed.begin());
        if (found == renamed.end() || !used.insert(axis).second ||
            node.output.indices[axis].domain != index.domain) {
          throw std::invalid_argument(
              "Invalid output permutation for '" + node.output.name + "'");
        }
        permutation.axes.push_back(axis);
      }
      for (const auto& [from, to] : transform.names) {
        const auto exists = [&](const std::string& name) {
          return std::ranges::any_of(
              node.output.indices,
              [&](const auto& index) { return index.name == name; });
        };
        if (!exists(from) || !exists(to)) {
          throw std::invalid_argument(
              "Permutation uses a non-output index for '" + node.output.name +
              "'");
        }
      }
      assignment.transforms.push_back(std::move(permutation));
    }
    pending.push_back(std::move(assignment));
  }
  std::set<std::string> available;
  std::vector<bool> scheduled(pending.size());
  while (result.assignments_.size() < pending.size()) {
    bool progress = false;
    for (std::size_t i = 0; i < pending.size(); ++i) {
      if (scheduled[i]) {
        continue;
      }
      const bool ready =
          std::ranges::all_of(pending[i].terms, [&](const auto& term) {
            return std::ranges::all_of(term.operands, [&](const auto& operand) {
              return operand.source != Source::kValue ||
                  available.contains(operand.binding.name);
            });
          });
      if (ready) {
        available.insert(pending[i].output.name);
        result.assignments_.push_back(std::move(pending[i]));
        scheduled[i] = true;
        progress = true;
      }
    }
    if (!progress) {
      throw std::invalid_argument("NDArray graph contains a dependency cycle");
    }
  }
  std::map<std::string, std::size_t> last_use;
  for (std::size_t i = 0; i < result.assignments_.size(); ++i) {
    const auto& assignment = result.assignments_[i];
    if (graph.IsIntermediate(assignment.output.name)) {
      last_use[assignment.output.name] = i;
    }
    for (const auto& term : assignment.terms) {
      for (const auto& operand : term.operands) {
        if (operand.source == Source::kValue &&
            graph.IsIntermediate(operand.binding.name)) {
          last_use[operand.binding.name] = i;
        }
      }
    }
  }
  for (const auto& [name, last] : last_use) {
    result.assignments_[last].release.push_back(name);
  }
  for (const auto& [name, binding] : required) {
    result.inputs_.push_back(binding);
  }

  // Bound payload without changing contraction order. NDArray reduces unique
  // labels, then contracts left to right. Count dense copies for strided/BLAS
  // packing even when views or TBLIS avoid them. Keeping every assignment live
  // overestimates the release schedule; this is a preflight bound, not an
  // allocation forecast. A per-monomial maximum bounds every contraction.
  using Polynomial = std::map<std::vector<symbolic::IndexDomain>, std::size_t>;
  Polynomial retained, contraction_peak;
  const auto add = [](Polynomial& polynomial,
                      std::vector<symbolic::IndexDomain> axes,
                      std::size_t count) {
    std::ranges::sort(axes);
    polynomial[axes] = memory_detail::Add(polynomial[axes], count);
  };
  for (const auto& assignment : result.assignments_) {
    add(retained, assignment.output.domains, 3);
    // Covers one-element default NDArray temporaries used while forming views
    // and contraction results, including otherwise entirely empty tensors.
    add(retained, {}, 8);
    for (const auto& term : assignment.terms) {
      Polynomial work;
      std::map<int, symbolic::IndexDomain> domains;
      std::map<int, std::size_t> frequency;
      std::vector<std::set<int>> labels;
      for (std::size_t i = 0; i < term.operands.size(); ++i) {
        add(work, term.operands[i].binding.domains, 3);
        labels.emplace_back(term.indices[i].begin(), term.indices[i].end());
        for (const auto label : labels.back()) {
          ++frequency[label];
        }
        for (std::size_t axis = 0; axis < term.indices[i].size(); ++axis) {
          domains.emplace(
              term.indices[i][axis], term.operands[i].binding.domains[axis]);
        }
      }
      const std::set<int> output(
          term.output_indices.begin(), term.output_indices.end());
      const auto record = [&](const std::set<int>& indices) {
        std::vector<symbolic::IndexDomain> axes;
        axes.reserve(indices.size());
        for (const auto label : indices) {
          axes.push_back(domains.at(label));
        }
        add(work, std::move(axes), 3);
      };
      for (auto& indices : labels) {
        std::erase_if(indices, [&](int label) {
          return frequency[label] == 1 && !output.contains(label);
        });
        record(indices);
      }
      if (!labels.empty()) {
        auto current = labels.front();
        for (std::size_t i = 1; i < labels.size(); ++i) {
          std::set<int> future;
          for (std::size_t j = i + 1; j < labels.size(); ++j) {
            future.insert(labels[j].begin(), labels[j].end());
          }
          auto next = current;
          for (const auto label : labels[i]) {
            if (current.contains(label) && !output.contains(label) &&
                !future.contains(label)) {
              next.erase(label);
            } else {
              next.insert(label);
            }
          }
          current = std::move(next);
          record(current);
        }
      }
      for (const auto& [axes, count] : work) {
        contraction_peak[axes] = std::max(contraction_peak[axes], count);
      }
    }
  }
  for (const auto& [axes, count] : contraction_peak) {
    add(retained, axes, count);
  }
  for (auto& [axes, count] : retained) {
    result.workspace_.push_back({count, axes});
  }
  return result;
}

template <typename T>
inline TensorMap<T> NDArrayExecutor::Evaluate(
    const TensorMap<T>& inputs,
    const Dimensions& dimensions) const {
  using Array = NDArray<T>;
  numeric_detail::ValidateInputs(inputs_, inputs, dimensions);
  TensorMap<T> values;
  for (const auto& assignment : assignments_) {
    const auto shape = assignment.output.Shape(dimensions);
    Array value(shape);
    for (const auto& term : assignment.terms) {
      if (term.operands.empty()) {
        value += Array::Full(shape, T(term.coefficient));
        continue;
      }
      std::vector<Array> arrays;
      arrays.reserve(term.operands.size());
      for (const auto& operand : term.operands) {
        const auto& binding = operand.binding;
        switch (operand.source) {
          case Source::kInput:
            arrays.push_back(inputs.at(binding.name));
            break;
          case Source::kValue:
            arrays.push_back(values.at(binding.name));
            break;
          case Source::kOnes:
            arrays.push_back(Array::Ones(binding.Shape(dimensions)));
            break;
          case Source::kDelta: {
            Array delta(binding.Shape(dimensions));
            for (std::size_t i = 0; i < delta.shape()[0]; ++i) {
              delta.At({i, i}) = T{1};
            }
            arrays.push_back(std::move(delta));
            break;
          }
        }
      }
      try {
        const auto contribution =
            Array::Einsum(term.indices, term.output_indices, arrays);
        Array::Copy(contribution, value, {}, T(term.coefficient), T{1});
      } catch (const std::exception& error) {
        throw std::runtime_error(
            "NDArray output '" + assignment.output.name + "': " + error.what());
      }
    }
    Array transformed(shape);
    for (const auto& transform : assignment.transforms) {
      Array::Copy(
          value, transformed, transform.axes, T(transform.coefficient), T{1});
    }
    values.emplace(assignment.output.name, std::move(transformed));
    for (const auto& name : assignment.release) {
      values.erase(name);
    }
  }
  return values;
}

} // namespace wickqc::runtime
