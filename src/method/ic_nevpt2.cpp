#include "method/ic_nevpt2.h"

#include "einsum/einsum.h"
#include "equation/equation.h"
#include "symbolic/index_domain.h"
#include "symbolic/wick.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace wickqc::method {
namespace {

using symbolic::OrbitalSpace;

constexpr std::string_view kFullHamiltonian =
    "SUM <mn> h[mn] E1[m,n]\n"
    "-2.0 SUM <mnj> w[mjnj] E1[m,n]\n"
    "+1.0 SUM <mnj> w[mjjn] E1[m,n]\n"
    "0.5 SUM <mnxy> w[mnxy] E2[mn,xy]";

std::string Dimension(const symbolic::Index& index) {
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

std::string RenameIndices(
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

SubspaceNames AnalyzeSubspace(std::string_view tagged_name) {
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

} // namespace

ICNEVPT2Generator::ICNEVPT2Generator() {
  indices_.Add(OrbitalSpace::kInactive, "mnxyijkl");
  indices_.Add(OrbitalSpace::kActive, "mnxyabcdefghpq");
  indices_.Add(OrbitalSpace::kExternal, "mnxyrstu");
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

symbolic::Expression ICNEVPT2Generator::Parse(std::string_view text) const {
  return symbolic::Expression::Parse(text, indices_, symmetries_);
}

symbolic::Tensor ICNEVPT2Generator::ParseTensor(std::string_view text) const {
  return symbolic::Tensor::Parse(text, indices_, symmetries_);
}

symbolic::Expression ICNEVPT2Generator::BuildCommutator(
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

symbolic::Expression ICNEVPT2Generator::BuildRhs(
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

std::string ICNEVPT2Generator::RenderEquation(
    const symbolic::Expression& expression,
    const symbolic::Tensor& output) const {
  const auto equation =
      equation::TensorEquation::FromExpression(expression, output);
  return einsum::RenderNumpy(einsum::Program::Lower(equation));
}

std::string ICNEVPT2Generator::Allocate(const symbolic::Tensor& tensor) {
  std::ostringstream output;
  output << tensor.name << " = np.zeros((";
  for (const auto& index : tensor.indices) {
    output << Dimension(index) << ", ";
  }
  output << "))";
  return output.str();
}

std::string ICNEVPT2Generator::Restrict(
    const symbolic::Tensor& tensor,
    bool restrict_active,
    bool strict) {
  std::ostringstream dimensions;
  dimensions << "grid = np.indices((";
  for (std::size_t position = 0; position < tensor.indices.size(); ++position) {
    const auto& index = tensor.indices[position];
    if (!restrict_active &&
        index.domain.orbital_spaces ==
            static_cast<std::uint8_t>(OrbitalSpace::kActive)) {
      continue;
    }
    dimensions << Dimension(index);
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
        static_cast<std::uint8_t>(OrbitalSpace::kActive)) {
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

std::string ICNEVPT2Generator::Indent(std::string_view text) {
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

std::vector<std::pair<std::string, equation::ContractionGraph>>
ICNEVPT2Generator::Equations() const {
  std::vector<std::pair<std::string, equation::ContractionGraph>> blocks;
  for (std::size_t position = 0; position < subspaces_.size(); ++position) {
    const auto& [key, ket] = subspaces_[position];
    if (key.back() == '2') {
      continue;
    }
    const auto names = AnalyzeSubspace(key);
    const std::vector<std::string> kets = names.nonorthogonal
        ? std::vector<std::string>{ket, subspaces_.at(position + 1).second}
        : std::vector<std::string>{ket};
    equation::ContractionGraph graph;
    for (std::size_t row = 0; row < kets.size(); ++row) {
      const auto bra = RenameIndices(kets[row], names.ket_to_bra);
      const auto row_suffix =
          names.nonorthogonal ? std::to_string(row + 1) : "";
      graph.Add(
          ParseTensor("rheq" + row_suffix + "[" + names.rhs_indices + "]"),
          BuildRhs(bra, kFullHamiltonian));
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

std::string ICNEVPT2Generator::GenerateNumpy() const {
  std::ostringstream functions;
  for (std::size_t position = 0; position < subspaces_.size(); ++position) {
    const auto& [tagged_name, ket] = subspaces_[position];
    const auto names = AnalyzeSubspace(tagged_name);
    if (tagged_name.back() == '2') {
      continue;
    }

    const std::string bra = RenameIndices(ket, names.ket_to_bra);
    const auto rhs = ParseTensor("rheq[" + names.rhs_indices + "]");
    const auto effective = ParseTensor("hexp[" + names.tensor_indices + "]");

    std::ostringstream body;
    body << Allocate(rhs) << '\n'
         << RenderEquation(BuildRhs(bra, kFullHamiltonian), rhs) << '\n'
         << Allocate(effective) << '\n'
         << RenderEquation(BuildCommutator(bra, ket), effective) << '\n';

    if (names.nonorthogonal) {
      const std::string& companion_ket = subspaces_.at(position + 1).second;
      const std::string companion_bra =
          RenameIndices(companion_ket, names.ket_to_bra);
      body << "rheq12 = np.zeros(rheq.shape + (2, ))\n"
           << "rheq12[..., 0] = rheq\n\n"
           << Allocate(rhs) << '\n'
           << RenderEquation(BuildRhs(companion_bra, kFullHamiltonian), rhs)
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
