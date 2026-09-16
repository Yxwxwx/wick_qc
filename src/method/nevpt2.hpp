#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "einsum/einsum.hpp"
#include "equation/graph.hpp"
#include "symbolic/index_domain.hpp"
#include "symbolic/wick.hpp"

namespace wickqc::method {

// Strongly contracted, spin-free spatial NEVPT2 in its eight outer subspaces.
class SCNEVPT2Generator {
 public:
  SCNEVPT2Generator();

  [[nodiscard]] std::vector<std::pair<std::string, equation::ContractionGraph>>
  Equations(bool sum_outer = false) const;
  [[nodiscard]] std::string GenerateNumpy(bool optimize = false) const;

 private:
  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] symbolic::Tensor ParseTensor(std::string_view text) const;

  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  symbolic::Expression active_hamiltonian_;
  std::vector<std::pair<std::string, std::string>> subspaces_;
};

class ICNEVPT2Generator {
 public:
  ICNEVPT2Generator();

  [[nodiscard]] std::string GenerateNumpy() const;
  // Each block contains the RHS vectors and effective Hamiltonian matrices,
  // before orbital restrictions and the final linear solve.
  [[nodiscard]] std::vector<std::pair<std::string, equation::ContractionGraph>>
  Equations() const;

 private:
  [[nodiscard]] symbolic::Expression Parse(std::string_view text) const;
  [[nodiscard]] symbolic::Tensor ParseTensor(std::string_view text) const;
  [[nodiscard]] symbolic::Expression BuildCommutator(
      std::string_view bra,
      std::string_view ket) const;
  [[nodiscard]] symbolic::Expression BuildRhs(
      std::string_view bra,
      std::string_view ket) const;
  [[nodiscard]] std::string RenderEquation(
      const symbolic::Expression& expression,
      const symbolic::Tensor& output) const;
  [[nodiscard]] static std::string Allocate(const symbolic::Tensor& tensor);
  [[nodiscard]] static std::string Restrict(
      const symbolic::Tensor& tensor,
      bool restrict_active,
      bool strict);
  [[nodiscard]] static std::string Indent(std::string_view text);

  symbolic::IndexRegistry indices_;
  symbolic::SymmetryRegistry symmetries_;
  std::vector<std::pair<std::string, std::string>> subspaces_;
  symbolic::Expression zeroth_order_hamiltonian_;
};

inline SCNEVPT2Generator::SCNEVPT2Generator() {
  indices_.Add(symbolic::OrbitalSpace::kInactive, "mnxyijkl");
  indices_.Add(symbolic::OrbitalSpace::kActive, "mnxyabcdefghpq");
  indices_.Add(symbolic::OrbitalSpace::kExternal, "mnxyrstu");
  symmetries_.Add(
      "w", 4, symbolic::TensorSymmetry::QuantumChemistryPhysicists());
  active_hamiltonian_ =
      Parse("SUM <ab> h[ab] E1[a,b]\n0.5 SUM <abcd> w[abcd] E2[ab,cd]");
  subspaces_ = {
      {"ijrs",
       "gamma[ij] gamma[rs] w[rsij] E1[r,i] E1[s,j]\n"
       "gamma[ij] gamma[rs] w[rsji] E1[s,i] E1[r,j]"},
      {"rsi",
       "SUM <a> gamma[rs] w[rsia] E1[r,i] E1[s,a]\n"
       "SUM <a> gamma[rs] w[sria] E1[s,i] E1[r,a]"},
      {"ijr",
       "SUM <a> gamma[ij] w[raji] E1[r,j] E1[a,i]\n"
       "SUM <a> gamma[ij] w[raij] E1[r,i] E1[a,j]"},
      {"rs", "SUM <ab> gamma[rs] w[rsba] E1[r,b] E1[s,a]"},
      {"ij", "SUM <ab> gamma[ij] w[baij] E1[b,i] E1[a,j]"},
      {"ir",
       "SUM <ab> w[raib] E1[r,i] E1[a,b]\n"
       "SUM <ab> w[rabi] E1[a,i] E1[r,b]\nh[ri] E1[r,i]"},
      {"r",
       "SUM <abc> w[rabc] E1[r,b] E1[a,c]\n"
       "SUM <a> h[ra] E1[r,a]\n-SUM <ab> w[rbba] E1[r,a]"},
      {"i", "SUM <abc> w[baic] E1[b,i] E1[a,c]\nSUM <a> h[ai] E1[a,i]"}};
}

inline symbolic::Expression SCNEVPT2Generator::Parse(
    std::string_view text) const {
  return symbolic::Expression::Parse(text, indices_, symmetries_);
}

inline symbolic::Tensor SCNEVPT2Generator::ParseTensor(
    std::string_view text) const {
  return symbolic::Tensor::Parse(text, indices_, symmetries_);
}

inline std::vector<std::pair<std::string, equation::ContractionGraph>>
SCNEVPT2Generator::Equations(bool sum_outer) const {
  const std::map<std::string, std::pair<symbolic::Tensor, symbolic::Expression>>
      definitions = {
          {"gamma", {ParseTensor("gamma[mn]"), Parse("1\n-0.5 delta[mn]")}}};
  std::vector<std::pair<std::string, equation::ContractionGraph>> blocks;
  for (const auto& [name, source] : subspaces_) {
    const auto ket = Parse(source).Substitute(definitions).Expand().Simplify();
    const auto bra = ket.Conjugate();
    const auto commutator =
        Commutator(active_hamiltonian_, ket).Expand().Simplify();
    const auto norm = (sum_outer ? FullySummedProduct(bra, ket) : bra * ket)
                          .Expand()
                          .AddSpinFreeTransposeSymmetry()
                          .RemoveExternal()
                          .Simplify();
    const auto effective =
        (sum_outer ? FullySummedProduct(bra, commutator) : bra * commutator)
            .Expand()
            .RemoveExternal()
            .AddSpinFreeTransposeSymmetry()
            .Simplify();
    const auto axes = sum_outer ? "" : name;
    equation::ContractionGraph graph;
    graph.Add(ParseTensor("norm[" + axes + "]"), norm);
    graph.Add(ParseTensor("hexp[" + axes + "]"), effective);
    blocks.emplace_back(name, std::move(graph));
  }
  return blocks;
}

inline std::string SCNEVPT2Generator::GenerateNumpy(bool optimize) const {
  std::ostringstream functions;
  for (const auto& [name, equations] : Equations()) {
    const auto& output = equations.Nodes().front().output;
    std::ostringstream body;
    body << "deno = ";
    for (std::size_t axis = 0; axis < output.indices.size(); ++axis) {
      if (axis != 0) {
        body << " + ";
      }
      const bool inactive = output.indices[axis].domain.orbital_spaces ==
          static_cast<unsigned>(symbolic::OrbitalSpace::kInactive);
      body << (inactive ? "(-1) * orbeI[" : "orbeE[");
      for (std::size_t slot = 0; slot < output.indices.size(); ++slot) {
        if (slot != 0) {
          body << ", ";
        }
        body << (slot == axis ? ":" : "None");
      }
      body << ']';
    }
    body << "\n";
    if (optimize) {
      body << "norm = np.zeros_like(deno)\nhexp = np.zeros_like(deno)\n"
           << einsum::RenderNumpy(equations.Simplify()) << '\n';
    } else {
      for (const auto& node : equations.Nodes()) {
        body << node.output.name << " = np.zeros_like(deno)\n"
             << einsum::RenderNumpy(
                    einsum::Program::Lower(
                        equation::TensorEquation::FromExpression(
                            node.expression, node.output)))
             << '\n';
      }
    }
    body << "idx = abs(norm) > 1E-14\n";
    if (output.indices.size() >= 2) {
      body << "grid = np.indices((";
      for (std::size_t axis = 0; axis < output.indices.size(); ++axis) {
        if (axis != 0) {
          body << ", ";
        }
        body
            << (output.indices[axis].domain.orbital_spaces ==
                        static_cast<unsigned>(symbolic::OrbitalSpace::kInactive)
                    ? "ncore"
                    : "nvirt");
      }
      body << "))\n";
      for (std::size_t axis = 1; axis < output.indices.size(); ++axis) {
        if (output.indices[axis - 1].domain == output.indices[axis].domain) {
          body << "idx &= grid[" << axis - 1 << "] <= grid[" << axis << "]\n";
        }
      }
      body << '\n';
    }
    body << "hexp[idx] = deno[idx] + hexp[idx] / norm[idx]\n"
         << "xener = -(norm[idx] / hexp[idx]).sum()\n"
         << "xnorm = norm[idx].sum()\nreturn xnorm, xener\n";
    functions << "def compute_" << name << "():\n";
    std::istringstream lines(body.str());
    for (std::string line; std::getline(lines, line);) {
      functions << "    " << line << '\n';
    }
    functions << "    \n\n";
  }
  return functions.str();
}
namespace ic_nevpt2_detail {

using symbolic::OrbitalSpace;

inline constexpr std::string_view kFullHamiltonian =
    "SUM <mn> h[mn] E1[m,n]\n"
    "-2.0 SUM <mnj> w[mjnj] E1[m,n]\n"
    "+1.0 SUM <mnj> w[mjjn] E1[m,n]\n"
    "0.5 SUM <mnxy> w[mnxy] E2[mn,xy]";

inline std::string Dimension(const symbolic::Index& index) {
  const auto space = index.domain.orbital_spaces;
  if (space == static_cast<std::uint8_t>(OrbitalSpace::kInactive)) {
    return "ncore";
  }
  if (space == static_cast<std::uint8_t>(OrbitalSpace::kActive)) {
    return "ncas";
  }
  if (space == static_cast<std::uint8_t>(OrbitalSpace::kExternal)) {
    return "nvirt";
  }
  throw std::invalid_argument("IC-NEVPT2 output index has no concrete space");
}

inline std::string RenameIndices(
    std::string text,
    const std::map<char, char>& replacements) {
  for (char& character : text) {
    const auto replacement = replacements.find(character);
    if (replacement != replacements.end()) {
      character = replacement->second;
    }
  }
  return text;
}

struct SubspaceNames {
  std::string tensor_indices;
  std::string rhs_indices;
  std::string function_name;
  std::map<char, char> ket_to_bra;
  bool restrict_active = false;
  bool strict = false;
  bool nonorthogonal = false;
};

inline SubspaceNames AnalyzeSubspace(std::string_view tagged_name) {
  SubspaceNames result;
  result.tensor_indices = std::string(tagged_name);
  result.function_name = result.tensor_indices;
  const char suffix = result.tensor_indices.back();
  if (suffix == '+' || suffix == '-') {
    result.tensor_indices.pop_back();
    result.function_name =
        result.tensor_indices + (suffix == '+' ? "_plus" : "_minus");
    result.restrict_active = true;
    result.strict = suffix == '-';
  } else if (suffix == '1' || suffix == '2') {
    result.tensor_indices.pop_back();
    result.function_name = result.tensor_indices;
    result.nonorthogonal = true;
  }

  result.rhs_indices = result.tensor_indices;
  if (result.tensor_indices.size() > 4) {
    const std::size_t active_rank = result.tensor_indices.size() - 4;
    const std::size_t bra_begin = 4 - active_rank;
    constexpr std::size_t ket_begin = 4;
    for (std::size_t offset = 0; offset < active_rank; ++offset) {
      result.ket_to_bra[result.tensor_indices[bra_begin + offset]] =
          result.tensor_indices[ket_begin + offset];
    }
    result.rhs_indices = result.tensor_indices.substr(0, bra_begin) +
        result.tensor_indices.substr(ket_begin);
  }
  return result;
}

} // namespace ic_nevpt2_detail

inline ICNEVPT2Generator::ICNEVPT2Generator() {
  indices_.Add(symbolic::OrbitalSpace::kInactive, "mnxyijkl");
  indices_.Add(symbolic::OrbitalSpace::kActive, "mnxyabcdefghpq");
  indices_.Add(symbolic::OrbitalSpace::kExternal, "mnxyrstu");
  symmetries_.Add(
      "w", 4, symbolic::TensorSymmetry::QuantumChemistryPhysicists());

  const auto diagonal = Parse(
      "SUM <i> orbe[i] E1[i,i]\n"
      "SUM <r> orbe[r] E1[r,r]");
  const auto active_one_body = Parse("SUM <ab> h[ab] E1[a,b]");
  const auto active_two_body = Parse("0.5 SUM <abcd> w[abcd] E2[ab,cd]");
  zeroth_order_hamiltonian_ = diagonal + active_one_body + active_two_body;

  subspaces_ = {
      {"ijrs+", "E1[r,i] E1[s,j]\n+ E1[s,i] E1[r,j]"},
      {"ijrs-", "E1[r,i] E1[s,j]\n- E1[s,i] E1[r,j]"},
      {"rsiap+", "E1[r,i] E1[s,a]\n+ E1[s,i] E1[r,a]"},
      {"rsiap-", "E1[r,i] E1[s,a]\n- E1[s,i] E1[r,a]"},
      {"ijrap+", "E1[r,j] E1[a,i]\n+ E1[r,i] E1[a,j]"},
      {"ijrap-", "E1[r,j] E1[a,i]\n- E1[r,i] E1[a,j]"},
      {"rsabpq+", "E1[r,b] E1[s,a]\n+ E1[s,b] E1[r,a]"},
      {"rsabpq-", "E1[r,b] E1[s,a]\n- E1[s,b] E1[r,a]"},
      {"ijabpq+", "E1[b,i] E1[a,j]\n+ E1[b,j] E1[a,i]"},
      {"ijabpq-", "E1[b,i] E1[a,j]\n- E1[b,j] E1[a,i]"},
      {"irabpq1", "E1[r,i] E1[a,b]"},
      {"irabpq2", "E1[a,i] E1[r,b]"},
      {"rabcpqg", "E1[r,b] E1[a,c]"},
      {"iabcpqg", "E1[b,i] E1[a,c]"},
  };
}

inline symbolic::Expression ICNEVPT2Generator::Parse(
    std::string_view text) const {
  return symbolic::Expression::Parse(text, indices_, symmetries_);
}

inline symbolic::Tensor ICNEVPT2Generator::ParseTensor(
    std::string_view text) const {
  return symbolic::Tensor::Parse(text, indices_, symmetries_);
}

inline symbolic::Expression ICNEVPT2Generator::BuildCommutator(
    std::string_view bra,
    std::string_view ket) const {
  const auto expanded_bra = Parse(bra).Expand().Simplify();
  const auto expanded_ket = Parse(ket).Expand().Simplify();
  return (expanded_bra.Conjugate() *
          Commutator(zeroth_order_hamiltonian_, expanded_ket)
              .Expand()
              .Simplify())
      .Expand()
      .RemoveExternal()
      .AddSpinFreeTransposeSymmetry()
      .Simplify();
}

inline symbolic::Expression ICNEVPT2Generator::BuildRhs(
    std::string_view bra,
    std::string_view ket) const {
  const auto expanded_bra = Parse(bra).Expand().Simplify();
  const auto expanded_ket = Parse(ket).Expand().Simplify();
  return (expanded_bra.Conjugate() * expanded_ket)
      .Expand()
      .AddSpinFreeTransposeSymmetry()
      .RemoveExternal()
      .RemoveInactive()
      .Simplify();
}

inline std::string ICNEVPT2Generator::RenderEquation(
    const symbolic::Expression& expression,
    const symbolic::Tensor& output) const {
  const auto equation =
      equation::TensorEquation::FromExpression(expression, output);
  return einsum::RenderNumpy(einsum::Program::Lower(equation));
}

inline std::string ICNEVPT2Generator::Allocate(const symbolic::Tensor& tensor) {
  std::ostringstream output;
  output << tensor.name << " = np.zeros((";
  for (const auto& index : tensor.indices) {
    output << ic_nevpt2_detail::Dimension(index) << ", ";
  }
  output << "))";
  return output.str();
}

inline std::string ICNEVPT2Generator::Restrict(
    const symbolic::Tensor& tensor,
    bool restrict_active,
    bool strict) {
  std::ostringstream dimensions;
  dimensions << "grid = np.indices((";
  for (std::size_t position = 0; position < tensor.indices.size(); ++position) {
    const auto& index = tensor.indices[position];
    if (!restrict_active &&
        index.domain.orbital_spaces ==
            static_cast<std::uint8_t>(symbolic::OrbitalSpace::kActive)) {
      continue;
    }
    dimensions << ic_nevpt2_detail::Dimension(index);
    if (position + 1 != tensor.indices.size() || position == 0) {
      dimensions << ", ";
    }
  }
  dimensions << "))\n";

  std::ostringstream restrictions;
  bool has_restriction = false;
  for (std::size_t position = 1; position < tensor.indices.size(); ++position) {
    const auto& previous = tensor.indices[position - 1];
    const auto& current = tensor.indices[position];
    if (current.domain != previous.domain) {
      continue;
    }
    if (current.domain.orbital_spaces ==
        static_cast<std::uint8_t>(symbolic::OrbitalSpace::kActive)) {
      const bool adjacent_names =
          current.name.front() == previous.name.front() + 1;
      const bool follows_triple = position >= 2 &&
          previous.name.front() ==
              tensor.indices[position - 2].name.front() + 1;
      const bool starts_triple = position + 1 < tensor.indices.size() &&
          tensor.indices[position + 1].name.front() == current.name.front() + 1;
      if (!adjacent_names || follows_triple || starts_triple) {
        continue;
      }
    }
    restrictions << "idx " << (has_restriction ? "&" : "") << "= grid["
                 << position - 1 << "] <" << (strict ? "" : "=") << " grid["
                 << position << "]\n";
    has_restriction = true;
  }
  return dimensions.str() + restrictions.str();
}

inline std::string ICNEVPT2Generator::Indent(std::string_view text) {
  std::ostringstream output;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t end = text.find('\n', begin);
    output << "    ";
    if (end == std::string_view::npos) {
      output << text.substr(begin) << '\n';
      break;
    }
    output << text.substr(begin, end - begin) << '\n';
    begin = end + 1;
  }
  return output.str();
}

inline std::vector<std::pair<std::string, equation::ContractionGraph>>
ICNEVPT2Generator::Equations() const {
  std::vector<std::pair<std::string, equation::ContractionGraph>> blocks;
  for (std::size_t position = 0; position < subspaces_.size(); ++position) {
    const auto& [key, ket] = subspaces_[position];
    if (key.back() == '2') {
      continue;
    }
    const auto names = ic_nevpt2_detail::AnalyzeSubspace(key);
    const std::vector<std::string> kets = names.nonorthogonal
        ? std::vector<std::string>{ket, subspaces_.at(position + 1).second}
        : std::vector<std::string>{ket};
    equation::ContractionGraph graph;
    for (std::size_t row = 0; row < kets.size(); ++row) {
      const auto bra =
          ic_nevpt2_detail::RenameIndices(kets[row], names.ket_to_bra);
      const auto row_suffix =
          names.nonorthogonal ? std::to_string(row + 1) : "";
      graph.Add(
          ParseTensor("rheq" + row_suffix + "[" + names.rhs_indices + "]"),
          BuildRhs(bra, ic_nevpt2_detail::kFullHamiltonian));
      for (std::size_t column = 0; column < kets.size(); ++column) {
        const auto suffix =
            names.nonorthogonal ? row_suffix + std::to_string(column + 1) : "";
        graph.Add(
            ParseTensor("hexp" + suffix + "[" + names.tensor_indices + "]"),
            BuildCommutator(bra, kets[column]));
      }
    }
    blocks.emplace_back(names.function_name, std::move(graph));
  }
  return blocks;
}

inline std::string ICNEVPT2Generator::GenerateNumpy() const {
  std::ostringstream functions;
  for (std::size_t position = 0; position < subspaces_.size(); ++position) {
    const auto& [tagged_name, ket] = subspaces_[position];
    const auto names = ic_nevpt2_detail::AnalyzeSubspace(tagged_name);
    if (tagged_name.back() == '2') {
      continue;
    }

    const std::string bra =
        ic_nevpt2_detail::RenameIndices(ket, names.ket_to_bra);
    const auto rhs = ParseTensor("rheq[" + names.rhs_indices + "]");
    const auto effective = ParseTensor("hexp[" + names.tensor_indices + "]");

    std::ostringstream body;
    body << Allocate(rhs) << '\n'
         << RenderEquation(
                BuildRhs(bra, ic_nevpt2_detail::kFullHamiltonian), rhs)
         << '\n'
         << Allocate(effective) << '\n'
         << RenderEquation(BuildCommutator(bra, ket), effective) << '\n';

    if (names.nonorthogonal) {
      const std::string& companion_ket = subspaces_.at(position + 1).second;
      const std::string companion_bra =
          ic_nevpt2_detail::RenameIndices(companion_ket, names.ket_to_bra);
      body << "rheq12 = np.zeros(rheq.shape + (2, ))\n"
           << "rheq12[..., 0] = rheq\n\n"
           << Allocate(rhs) << '\n'
           << RenderEquation(
                  BuildRhs(companion_bra, ic_nevpt2_detail::kFullHamiltonian),
                  rhs)
           << '\n'
           << "rheq12[..., 1] = rheq\n\n"
           << "hexp12 = np.zeros(hexp.shape + (2, 2, ))\n"
           << "hexp12[..., 0, 0] = hexp\n\n"
           << Allocate(effective) << '\n'
           << RenderEquation(BuildCommutator(bra, companion_ket), effective)
           << '\n'
           << "hexp12[..., 0, 1] = hexp\n\n"
           << Allocate(effective) << '\n'
           << RenderEquation(BuildCommutator(companion_bra, ket), effective)
           << '\n'
           << "hexp12[..., 1, 0] = hexp\n\n"
           << Allocate(effective) << '\n'
           << RenderEquation(
                  BuildCommutator(companion_bra, companion_ket), effective)
           << '\n'
           << "hexp12[..., 1, 1] = hexp\n\n"
           << "dcas = ncas ** " << names.ket_to_bra.size() << '\n'
           << "xr = rheq12.reshape((-1, dcas * 2))\n"
           << "xh = hexp12.reshape((-1, dcas, dcas, 2, 2))\n"
           << "xh = xh.transpose(0, 1, 3, 2, 4)\n"
           << "xh = xh.reshape((-1, dcas * 2, dcas * 2))\n";
    } else {
      if (names.ket_to_bra.size() == 2 && names.restrict_active) {
        body << "dcas = ncas * (ncas " << (names.strict ? '-' : '+')
             << " 1) // 2 \n";
      } else {
        body << "dcas = ncas ** " << names.ket_to_bra.size() << '\n';
      }
      if (names.tensor_indices.size() - names.ket_to_bra.size() * 2 >= 2) {
        body << Restrict(rhs, names.restrict_active, names.strict)
             << "xr = rheq[idx].reshape((-1, dcas))\n"
             << Restrict(effective, names.restrict_active, names.strict)
             << "xh = hexp[idx].reshape((-1, dcas, dcas))\n\n";
      } else {
        body << "xr = rheq.reshape((-1, dcas))\n\n"
             << "xh = hexp.reshape((-1, dcas, dcas))\n\n";
      }
    }
    body << "return -(np.linalg.solve(xh, xr) * xr).sum()\n";
    functions << "def compute_" << names.function_name << "():\n"
              << Indent(body.str()) << '\n';
  }
  return functions.str() + '\n';
}

} // namespace wickqc::method
