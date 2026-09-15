#pragma once

#include "equation/equation.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

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
  [[nodiscard]] static Program Lower(const equation::TensorEquation& equation, const LoweringOptions& options = {});

  [[nodiscard]] const std::vector<Term>& Terms() const noexcept;

 private:
  explicit Program(std::vector<Term> terms);

  std::vector<Term> terms_;
};

[[nodiscard]] std::string RenderNumpy(const Program& program, const NumPyOptions& options = {});
[[nodiscard]] std::string RenderNumpy(const equation::ContractionGraph& graph, int coefficient_precision = 6);
// Indent every line, including a final empty line, as in the reference emitter.
[[nodiscard]] std::string AddIndent(std::string_view text, std::size_t spaces = 4);

} // namespace wickqc::einsum
