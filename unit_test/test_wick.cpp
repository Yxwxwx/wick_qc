#include "wick.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {
using wickqc::symbolic::Expression;
using wickqc::symbolic::IndexRegistry;
using wickqc::symbolic::OrbitalSpace;
using wickqc::symbolic::SymmetryRegistry;
using wickqc::symbolic::Tensor;
using wickqc::symbolic::TensorSymmetry;

class WickTest : public testing::Test {
 protected:
  WickTest() {
    indices_.Add(OrbitalSpace::kInactive, "ij");
    indices_.Add(OrbitalSpace::kExternal, "ab");
    indices_.Add(OrbitalSpace::kActive, "pqrs");
    symmetries_.Add("v", 4, TensorSymmetry::QuantumChemistryChemists());
  }
  Expression Parse(std::string_view text) const {
    return Expression::Parse(text, indices_, symmetries_);
  }
  IndexRegistry indices_;
  SymmetryRegistry symmetries_;
};

TEST_F(WickTest, FermionicAnticommutator) {
  const auto actual =
      (Parse("D[p] C[q]") + Parse("C[q] D[p]")).Expand().Simplify();
  EXPECT_TRUE((actual - Parse("delta[pq]")).Simplify().Empty());
}

TEST_F(WickTest, ClosedShellSpinFreeMetric) {
  const auto expected = Parse("2 delta[ab] delta[ij]").Simplify();
  const auto actual = Parse("E1[i,a] E1[b,j]").Expand(0).Simplify();
  EXPECT_TRUE((actual - expected).Simplify().Empty());
}

TEST_F(WickTest, ChemistEightfoldSymmetry) {
  EXPECT_EQ(TensorSymmetry::QuantumChemistryChemists().Elements().size(), 8U);
  EXPECT_TRUE((Parse("v[pqrs]") - Parse("v[qpsr]")).Simplify().Empty());
  EXPECT_TRUE((Parse("v[pqrs]") - Parse("v[rspq]")).Simplify().Empty());
  EXPECT_TRUE((Parse("v[pqrs]") - Parse("v[qprs]")).Simplify().Empty());
}

TEST_F(WickTest, DeltasAndSubstitutionPreserveBoundIndices) {
  const auto expression = Parse("SUM <p> delta[pq] A[p]").Simplify();
  EXPECT_TRUE((expression - Parse("A[q]")).Simplify().Empty());
  const auto definition = Expression::ParseDefinition(
      "X[p] = SUM <q> B[pq] C[q]", indices_, symmetries_);
  const auto substituted =
      Parse("SUM <q> X[q] D[q]").Substitute({{"X", definition}}).Simplify();
  EXPECT_TRUE(
      (substituted - Parse("SUM <pr> B[pr] C[r] D[p]")).Simplify().Empty());
}

TEST_F(WickTest, SerializationRoundTripAndMalformedInput) {
  const auto expression = Parse("SUM <pq> A[pq] E1[p,q]").Expand().Simplify();
  std::stringstream stream;
  expression.Save(stream);
  EXPECT_EQ(Expression::Load(stream), expression);
  EXPECT_THROW(Parse("SUM <p A[p]"), std::invalid_argument);
}

TEST_F(WickTest, GraphDoesNotMergeDifferentLinearCombinations) {
  wickqc::equation::ContractionGraph graph;
  const auto add = [&](std::string_view name, std::string_view expression) {
    graph.Add(Tensor::Parse(name, indices_, symmetries_), Parse(expression));
  };
  // Verified reference defect: x1=A+B, x2=A+2B must give 2A+3B.
  add("_x1[p]", "A[p] + B[p]");
  add("_x2[p]", "A[p] + 2 B[p]");
  add("R[p]", "_x1[p] + _x2[p]");
  const auto expanded = graph.Simplify().Expand();
  ASSERT_EQ(expanded.Nodes().size(), 1U);
  EXPECT_TRUE((expanded.Nodes().front().expression - Parse("2 A[p] + 3 B[p]"))
                  .Simplify()
                  .Empty());
}
} // namespace
