#include "runtime_dimensions.hpp"
#include "wick.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {
using Array = wickqc::NDArray<double>;
using wickqc::test::Dimension;

TEST(NDArray, SlicesAreViewsAndClonesOwnData) {
  const auto m = Dimension("WICKQC_TEST_OCCUPIED");
  const auto n = Dimension("WICKQC_TEST_VIRTUAL");
  Array array({m, n});
  for (std::size_t i = 0; i < array.Size(); ++i) {
    array.data()[i] = static_cast<double>(i);
  }
  auto reversed = array.Slice(":,::-1");
  EXPECT_DOUBLE_EQ(reversed.At({0, 0}), array.At({0, n - 1}));
  auto clone = reversed.Clone();
  reversed.At({0, 0}) = -1;
  EXPECT_DOUBLE_EQ(array.At({0, n - 1}), -1);
  EXPECT_DOUBLE_EQ(clone.At({0, 0}), static_cast<double>(n - 1));
  EXPECT_EQ(array.TransposeView({1, 0}).shape(), (Array::Shape{n, m}));
}

TEST(NDArray, DiagonalUnaryReductionAndRuntimeEllipsis) {
  const auto n = Dimension("WICKQC_TEST_OCCUPIED");
  const auto m = Dimension("WICKQC_TEST_VIRTUAL");
  auto array = Array::Random({m, n, n}, 19);
  const auto trace = Array::Einsum("...ii->...", {array});
  const auto total = Array::Einsum("...ii->", {array}).Item();
  double expected_total = 0;
  for (std::size_t batch = 0; batch < m; ++batch) {
    double expected = 0;
    for (std::size_t i = 0; i < n; ++i) {
      expected += array.At({batch, i, i});
    }
    EXPECT_NEAR(trace.At({batch}), expected, 1e-13);
    expected_total += expected;
  }
  EXPECT_NEAR(total, expected_total, 1e-12);
}

TEST(NDArray, BroadcastAndIntegerEinsumAgreeWithLoops) {
  const auto m = Dimension("WICKQC_TEST_OCCUPIED");
  const auto n = Dimension("WICKQC_TEST_VIRTUAL");
  auto a = Array::Random({m, n}, 7);
  auto b = Array::Random({n}, 8);
  const auto c = Array::Einsum(
      std::vector<std::vector<int>>{{1001, -70}, {-70}}, {1001}, {a, b});
  const auto column = Array::Ones({m, 1});
  const auto broadcast = a + column;
  const auto reverse = column + a;
  for (std::size_t i = 0; i < m; ++i) {
    double expected = 0;
    for (std::size_t j = 0; j < n; ++j) {
      expected += a.At({i, j}) * b.At({j});
      EXPECT_DOUBLE_EQ(broadcast.At({i, j}), a.At({i, j}) + 1);
      EXPECT_DOUBLE_EQ(reverse.At({i, j}), a.At({i, j}) + 1);
    }
    EXPECT_NEAR(c.At({i}), expected, 1e-13);
  }
}

TEST(NDArray, InvalidIndicesAndEmptyContraction) {
  const auto n = Dimension("WICKQC_TEST_OCCUPIED");
  const Array array({n, n});
  EXPECT_THROW((void)Array::Einsum("ij->k", {array}), std::invalid_argument);
  EXPECT_THROW(
      (void)Array::Einsum("ii->", {Array({n, n + 1})}), std::invalid_argument);
  EXPECT_THROW((void)array.TransposeView({0, 0}), std::invalid_argument);
  const auto empty = Array::Einsum("ik,jk->ij", {Array({n, 0}), Array({n, 0})});
  EXPECT_EQ(empty.shape(), (Array::Shape{n, n}));
  EXPECT_DOUBLE_EQ(empty.Norm(), 0);
}
} // namespace
