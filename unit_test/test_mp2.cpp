#include "backend/ndarray.hpp"
#include "method/spatial.hpp"
#include "runtime/executor.hpp"
#include "runtime/numeric.hpp"
#include "runtime_dimensions.hpp"

#include <gtest/gtest.h>

#include <cstddef>

namespace {
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
} // namespace
