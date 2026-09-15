#include "method/sc_nevpt2.h"

#include "einsum/einsum.h"
#include "equation/equation.h"
#include "symbolic/index_domain.h"
#include "symbolic/wick.h"

#include <cstddef>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace wickqc::method {
using symbolic::Expression;
using symbolic::OrbitalSpace;

SCNEVPT2Generator::SCNEVPT2Generator() {
  indices_.Add(OrbitalSpace::kInactive, "mnxyijkl");
  indices_.Add(OrbitalSpace::kActive, "mnxyabcdefghpq");
  indices_.Add(OrbitalSpace::kExternal, "mnxyrstu");
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

Expression SCNEVPT2Generator::Parse(std::string_view text) const {
  return Expression::Parse(text, indices_, symmetries_);
}

symbolic::Tensor SCNEVPT2Generator::ParseTensor(std::string_view text) const {
  return symbolic::Tensor::Parse(text, indices_, symmetries_);
}

std::vector<std::pair<std::string, equation::ContractionGraph>>
SCNEVPT2Generator::Equations(bool sum_outer) const {
  const std::map<std::string, std::pair<symbolic::Tensor, Expression>>
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

std::string SCNEVPT2Generator::GenerateNumpy(bool optimize) const {
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
          static_cast<unsigned>(OrbitalSpace::kInactive);
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
                        static_cast<unsigned>(OrbitalSpace::kInactive)
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
} // namespace wickqc::method
