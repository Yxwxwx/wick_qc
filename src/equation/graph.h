#pragma once

#include "symbolic/index_domain.h"
#include "symbolic/wick.h"

#include <map>
#include <string>
#include <vector>

namespace wickqc::equation {

struct IndexTransform {
  double coefficient = 1.0;
  std::map<std::string, std::string> names;
  bool operator==(const IndexTransform&) const = default;
};

struct EquationNode {
  symbolic::Tensor output;
  symbolic::Expression expression;
  std::vector<IndexTransform> transforms = {IndexTransform{}};
};

struct GraphOptions {
  GraphOptions();
  std::map<symbolic::IndexDomain, double> index_scales;
  double multiply_scale = 5.0;
  double add_scale = 3.0;
  std::string intermediate_prefix = "_x";
};

// A graph of symbolic tensor assignments and output-index transformations.
// Each optimization stage is exposed for differential verification.
class ContractionGraph {
 public:
  ContractionGraph() = default;
  explicit ContractionGraph(std::vector<EquationNode> nodes, GraphOptions options = {});

  void Add(symbolic::Tensor output, symbolic::Expression expression);
  [[nodiscard]] const std::vector<EquationNode>& Nodes() const;
  [[nodiscard]] const GraphOptions& Options() const;
  [[nodiscard]] int LastIntermediateId() const;
  [[nodiscard]] bool IsIntermediate(const std::string& name) const;

  [[nodiscard]] ContractionGraph OrderContractions() const;
  [[nodiscard]] ContractionGraph SplitBinary() const;
  [[nodiscard]] ContractionGraph MergeIntermediates() const;
  [[nodiscard]] ContractionGraph FactorCommonOperands() const;
  [[nodiscard]] ContractionGraph FactorPermutations() const;
  [[nodiscard]] ContractionGraph ExpandPermutations() const;
  [[nodiscard]] ContractionGraph InlineIntermediates() const;
  [[nodiscard]] ContractionGraph TopologicalSort() const;
  [[nodiscard]] ContractionGraph Simplify() const;
  [[nodiscard]] ContractionGraph Expand() const;

 private:
  std::vector<EquationNode> nodes_;
  GraphOptions options_;
};

} // namespace wickqc::equation
