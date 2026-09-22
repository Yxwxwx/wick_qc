#include "runtime_dimensions.hpp"
#include "wick.hpp"

#include <gtest/gtest.h>

#include <cstddef>

namespace {
using wickqc::NDArray;
namespace runtime = wickqc::runtime;
namespace symbolic = wickqc::symbolic;
namespace equation = wickqc::equation;
namespace test = wickqc::test;
using wickqc::method::CCSDOptions;
using wickqc::method::CCSDStep;
using wickqc::method::IntegralConvention;
using wickqc::method::RHFAmplitudes;
using wickqc::method::RHFData;
using wickqc::method::SolveCCSD;
using wickqc::method::SolveMP2;
TEST(SpatialMP2, EnergyAndResidualsMatchDenominatorFormula) {
  using Array = wickqc::NDArray<double>;
  const auto ni = wickqc::test::Dimension("WICKQC_TEST_OCCUPIED");
  const auto ne = wickqc::test::Dimension("WICKQC_TEST_VIRTUAL");
  Array occupied({ni}), external({ne});
  for (std::size_t i = 0; i < ni; ++i) {
    occupied.At({i}) = -1.0 - 0.1 * static_cast<double>(i);
  }
  for (std::size_t a = 0; a < ne; ++a) {
    external.At({a}) = 0.5 + 0.2 * static_cast<double>(a);
  }
  Array v({ni, ni, ne, ne}), t({ne, ne, ni, ni});
  double expected = 0;
  for (std::size_t i = 0; i < ni; ++i) {
    for (std::size_t j = 0; j < ni; ++j) {
      for (std::size_t a = 0; a < ne; ++a) {
        for (std::size_t b = 0; b < ne; ++b) {
          const double integral = 0.02 * static_cast<double>(1 + i + j + a + b);
          const double gap = occupied.At({i}) + occupied.At({j}) -
              external.At({a}) - external.At({b});
          v.At({i, j, a, b}) = integral;
          t.At({a, b, i, j}) = integral / gap;
          expected += integral * integral / gap;
        }
      }
    }
  }
  const wickqc::runtime::TensorMap<double> inputs{
      {"epsI", occupied},
      {"epsE", external},
      {"vIIEE", v},
      {"vEEII", v.TransposeView({2, 3, 0, 1})},
      {"u1EI", Array({ne, ni})},
      {"u1EEII", t}};
  const auto graph =
      wickqc::method::SpatialMPGenerator(2).Equations().Simplify();
  const auto result = wickqc::runtime::NDArrayExecutor::Compile(graph).Evaluate(
      inputs, {{{1, 0}, ni}, {{8, 0}, ne}});
  EXPECT_NEAR(result.at("energy2").Item(), expected, 1e-12);
  EXPECT_LE(result.at("residual1_rank1").Norm(), 1e-12);
  EXPECT_LE(result.at("residual1_rank2").Norm(), 1e-12);
}

TEST(RHFData, RequiredBlocksMatchDenseBinding) {
  const auto ni = test::Dimension("WICKQC_TEST_OCCUPIED");
  const auto ne = test::Dimension("WICKQC_TEST_VIRTUAL");
  const auto n = ni + ne;
  RHFData dense;
  dense.occupied = ni;
  dense.orbital_energies = NDArray<double>::Random({n}, 21);
  dense.fock = NDArray<double>::Random({n, n}, 22);
  dense.chemist_integrals = NDArray<double>::Random({n, n, n, n}, 23);
  const std::vector<runtime::TensorBinding> bindings{
      {"epsI", {{1, 0}}},
      {"fIE", {{1, 0}, {8, 0}}},
      {"vEIEI", {{8, 0}, {1, 0}, {8, 0}, {1, 0}}},
      {"tEI", {{8, 0}, {1, 0}}}};
  const runtime::TensorMap<double> amplitudes{
      {"tEI", NDArray<double>::Random({ne, ni}, 24)}};
  for (auto convention :
       {IntegralConvention::kChemist, IntegralConvention::kPhysicist}) {
    const auto expected = dense.Bind(bindings, amplitudes, convention);
    auto blocks = dense;
    blocks.chemist_integrals = NDArray<double>();
    blocks.block_convention = convention;
    blocks.integral_blocks.emplace("vEIEI", expected.at("vEIEI"));
    const auto actual = blocks.Bind(bindings, amplitudes, convention);
    for (const auto& [name, value] : expected) {
      EXPECT_TRUE(value.AllClose(actual.at(name), 0, 0)) << name;
    }
    const auto other = convention == IntegralConvention::kChemist
        ? IntegralConvention::kPhysicist
        : IntegralConvention::kChemist;
    EXPECT_THROW(
        (void)blocks.Bind(bindings, amplitudes, other), std::invalid_argument);
    blocks.integral_blocks.clear();
    EXPECT_THROW(
        (void)blocks.Bind(bindings, amplitudes, convention),
        std::invalid_argument);
    // An energy-only/empty contract does not require any two-electron block.
    EXPECT_TRUE(blocks.Bind({}, {}, convention).empty());
  }
}

TEST(CCSDDriver, ConstantEnergyDoesNotHideAnUnconvergedResidual) {
  const auto ni = test::Dimension("WICKQC_TEST_OCCUPIED");
  const auto ne = test::Dimension("WICKQC_TEST_VIRTUAL");
  const auto n = ni + ne;
  RHFData data;
  data.occupied = ni;
  data.orbital_energies = NDArray<double>({n});
  data.fock = NDArray<double>({n, n});
  data.block_convention = IntegralConvention::kChemist;
  for (std::size_t i = 0; i < n; ++i) {
    data.fock.At({i, i}) = data.orbital_energies.At({i}) = i < ni ? -1 : 1;
  }
  const runtime::NumericKernel kernel{
      {},
      {{"energy", {}},
       {"residual1", {{8, 0}, {1, 0}}},
       {"residual2", {{8, 0}, {8, 0}, {1, 0}, {1, 0}}}},
      [](const runtime::TensorMap<double>&,
         const runtime::Dimensions& dimensions) {
        const auto occupied = dimensions.at({1, 0}),
                   external = dimensions.at({8, 0});
        return runtime::TensorMap<double>{
            {"energy", NDArray<double>::Full({}, 0)},
            {"residual1", NDArray<double>::Full({external, occupied}, 2)},
            {"residual2",
             NDArray<double>::Full(
                 {external, external, occupied, occupied}, 2)}};
      },
      nullptr};
  const RHFAmplitudes initial{
      NDArray<double>({ne, ni}), NDArray<double>({ne, ne, ni, ni})};
  const auto result = SolveCCSD(
      data,
      kernel,
      -10,
      IntegralConvention::kChemist,
      {2, 1e-10, 1e-8},
      initial);
  EXPECT_FALSE(result.converged);
  EXPECT_EQ(result.iterations, 2U);
  EXPECT_EQ(result.history.size(), 3U);
  EXPECT_DOUBLE_EQ(result.total_energy, -10);
  EXPECT_GT(result.residual_norm, 1);
  EXPECT_DOUBLE_EQ(result.history.back().energy_change, 0);
  EXPECT_DOUBLE_EQ(initial.singles.Norm(), 0);
  EXPECT_DOUBLE_EQ(initial.doubles.Norm(), 0);
  EXPECT_NEAR(result.amplitudes.singles.At({0, 0}), -1, 1e-14);
  EXPECT_NEAR(result.amplitudes.doubles.At({0, 0, 0, 0}), -0.5, 1e-14);
  RHFAmplitudes rounded{initial.singles.Clone(), initial.doubles.Clone()};
  rounded.doubles.At({0, 1, 0, 0}) = 1e-14;
  const auto next = CCSDStep(data, rounded, result.residuals);
  EXPECT_DOUBLE_EQ(
      next.doubles.At({0, 1, 0, 0}), next.doubles.At({1, 0, 0, 0}));
  rounded.doubles.At({0, 1, 0, 0}) = 1e-6;
  EXPECT_THROW(
      (void)CCSDStep(data, rounded, result.residuals), std::invalid_argument);
  EXPECT_THROW(
      (void)SolveCCSD(
          data,
          kernel,
          -10,
          IntegralConvention::kChemist,
          {2, -1, 1e-8},
          initial),
      std::invalid_argument);
  data.orbital_energies.At({ni}) = data.fock.At({ni, ni}) = -1;
  EXPECT_THROW(
      (void)SolveCCSD(
          data, kernel, -10, IntegralConvention::kChemist, {}, initial),
      std::invalid_argument);
}

TEST(MethodMemory, RejectsInsufficientOrUnknownWorkspaceBeforeEvaluation) {
  const auto ni = test::Dimension("WICKQC_TEST_OCCUPIED"),
             ne = test::Dimension("WICKQC_TEST_VIRTUAL");
  RHFData data;
  data.occupied = ni;
  data.orbital_energies = NDArray<double>({ni + ne});
  data.fock = NDArray<double>({ni + ne, ni + ne});
  data.block_convention = IntegralConvention::kChemist;
  for (std::size_t i = 0; i < ni + ne; ++i) {
    data.fock.At({i, i}) = data.orbital_energies.At({i}) = i < ni ? -1 : 1;
  }
  runtime::NumericKernel kernel{
      {},
      {{"energy", {}},
       {"residual1", {{8, 0}, {1, 0}}},
       {"residual2", {{8, 0}, {8, 0}, {1, 0}, {1, 0}}}},
      [](const runtime::TensorMap<double>&,
         const runtime::Dimensions&) -> runtime::TensorMap<double> {
        throw std::logic_error("kernel was evaluated");
      },
      nullptr,
      std::vector<runtime::WorkspaceTerm>{{1024, {}}}};
  const RHFAmplitudes initial{
      NDArray<double>({ne, ni}), NDArray<double>({ne, ne, ni, ni})};
  CCSDOptions options;
  options.memory.limit_bytes = 1;
  EXPECT_THROW(
      (void)SolveCCSD(
          data, kernel, -10, IntegralConvention::kChemist, options, initial),
      std::runtime_error);
  kernel.workspace.reset();
  EXPECT_THROW(
      (void)SolveCCSD(
          data, kernel, -10, IntegralConvention::kChemist, options, initial),
      std::invalid_argument);
  EXPECT_THROW(
      (void)SolveCCSD(
          data, kernel, -10, IntegralConvention::kChemist, {}, initial),
      std::logic_error);
  kernel.workspace = std::vector<runtime::WorkspaceTerm>{{1024, {}}};
  kernel.outputs[0].name = "energy2";
  kernel.outputs[1].name = "residual1_rank1";
  kernel.outputs[2].name = "residual1_rank2";
  EXPECT_THROW(
      (void)SolveMP2(data, kernel, -10, IntegralConvention::kChemist, {1, 0}),
      std::runtime_error);
  const runtime::MemoryBudget overflow{
      std::numeric_limits<std::size_t>::max(),
      std::numeric_limits<std::size_t>::max()};
  EXPECT_THROW((void)overflow.Check(1, "test"), std::overflow_error);
}

TEST(MethodMemory, EmptyReductionStillAccountsForNonemptyOutput) {
  symbolic::IndexRegistry indices;
  indices.Add(symbolic::OrbitalSpace::kInactive, "i");
  indices.Add(symbolic::OrbitalSpace::kExternal, "a");
  equation::ContractionGraph graph;
  graph.Add(
      symbolic::Tensor::Parse("R[a]", indices, {}),
      symbolic::Expression::Parse("SUM <i> A[ai]", indices, {}));
  const auto kernel = runtime::NDArrayExecutor::Compile(graph);
  const auto ne = test::Dimension("WICKQC_TEST_VIRTUAL");
  const runtime::Dimensions dimensions{{{1, 0}, 0}, {{8, 0}, ne}};
  const auto values = kernel.Evaluate(
      runtime::TensorMap<double>{{"AEI", NDArray<double>({ne, 0})}},
      dimensions);
  EXPECT_EQ(values.at("R").shape(), (NDArray<double>::Shape{ne}));
  EXPECT_GE(kernel.WorkspaceElements(dimensions), ne);
  const std::vector<runtime::WorkspaceTerm> huge{{2, {{8, 0}}}};
  EXPECT_THROW(
      (void)runtime::WorkspaceElements(
          huge, {{{8, 0}, std::numeric_limits<std::size_t>::max()}}),
      std::overflow_error);
}
} // namespace
