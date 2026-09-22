#include "backend/ndarray.hpp"
#include "runtime_dimensions.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

namespace {
using Array = wickqc::NDArray<double>;
using wickqc::backend::ShouldUseHptt;
using wickqc::backend::TransposeOptions;
using wickqc::backend::transpose_detail::DescribeDenseLayout;
const TransposeOptions kNative{std::nullopt, 1};
// Zero exercises dispatch on small test fixtures; it is not a performance
// recommendation or a production crossover threshold.
const TransposeOptions kExerciseHptt{0, 1};
#if defined(WICKQC_TRANSPOSE_HPTT)
constexpr bool kHpttAvailable = true;
#else
constexpr bool kHpttAvailable = false;
#endif

Array::Shape UnequalShape() {
  const auto n = wickqc::test::Dimension("WICKQC_TEST_OCCUPIED") +
      wickqc::test::Dimension("WICKQC_TEST_ACTIVE");
  return {n, n + 1, n + 2, n + 3};
}

template <typename T>
T Value(double real, double imag) {
  if constexpr (std::is_arithmetic_v<T>) {
    return static_cast<T>(real);
  } else {
    using Real = typename T::value_type;
    return T(static_cast<Real>(real), static_cast<Real>(imag));
  }
}

template <typename T>
void CheckAllPermutations(int threads) {
  using Tensor = wickqc::NDArray<T>;
  Tensor source(UnequalShape());
  for (std::size_t i = 0; i < source.Size(); ++i) {
    source.data()[i] = Value<T>(
        static_cast<double>(i % 31) / 16., static_cast<double>(i % 13) / 8.);
  }
  const auto alpha = Value<T>(1.5, -0.25);
  const auto beta = Value<T>(-0.5, 0.125);
  const auto initial = Value<T>(0.75, -0.375);
  constexpr double tolerance =
      sizeof(typename Tensor::RealType) == sizeof(float) ? 2e-5 : 2e-13;
  std::vector<int> permutation{0, 1, 2, 3};
  std::size_t hptt_calls = 0;
  do {
    const auto view = source.TransposeView(permutation);
    auto reference = Tensor::Full(view.shape(), initial);
    auto actual = reference.Clone(kNative);
    const TransposeOptions options{0, threads};
    Tensor::Copy(source, reference, permutation, alpha, beta, kNative);
    const bool used = wickqc::backend::TryHpttTranspose(
        source.data(),
        source.shape(),
        source.strides(),
        permutation,
        actual.data(),
        actual.strides(),
        alpha,
        beta,
        options);
    const auto layout = DescribeDenseLayout(
        source.shape(), source.strides(), permutation, actual.strides());
    ASSERT_TRUE(layout);
    EXPECT_EQ(used, ShouldUseHptt<T>(*layout, options));
    hptt_calls += used;
    if (!used) {
      Tensor::Copy(source, actual, permutation, alpha, beta, options);
    }
    const auto materialized = source.TransposeCopy(permutation, options);
    const auto c_order = view.ToCOrder(options);
    auto dispatched = Tensor::Full(view.shape(), initial);
    Tensor::Copy(source, dispatched, permutation, alpha, beta, options);
    EXPECT_TRUE(materialized.IsContiguous());
    EXPECT_NE(materialized.data(), source.data());
    for (std::size_t i = 0; i < actual.Size(); ++i) {
      const auto output_index = actual.DecomposeLinearIndex(i);
      std::vector<std::size_t> input_index(permutation.size());
      for (std::size_t axis = 0; axis < permutation.size(); ++axis) {
        input_index[permutation[axis]] = output_index[axis];
      }
      const auto value = source[input_index];
      EXPECT_LE(
          std::abs(reference.data()[i] - (alpha * value + beta * initial)),
          tolerance * (1 + std::abs(value)));
      EXPECT_LE(
          std::abs(actual.data()[i] - reference.data()[i]),
          tolerance * (1 + std::abs(reference.data()[i])));
      EXPECT_LE(
          std::abs(dispatched.data()[i] - reference.data()[i]),
          tolerance * (1 + std::abs(reference.data()[i])));
      EXPECT_LE(std::abs(materialized.data()[i] - value), tolerance);
      EXPECT_LE(std::abs(c_order.data()[i] - value), tolerance);
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  EXPECT_EQ(hptt_calls > 0, kHpttAvailable);
}

TEST(Transpose, RealAndComplexMatchNative) {
  for (const int threads : {1, 2}) {
    CheckAllPermutations<float>(threads);
    CheckAllPermutations<double>(threads);
    CheckAllPermutations<std::complex<float>>(threads);
    CheckAllPermutations<std::complex<double>>(threads);
  }
}

TEST(Transpose, ViewsNeverMaterialize) {
  auto source = Array::Random(UnequalShape(), 17);
  auto view = source.TransposeView({2, 0, 3, 1});
  EXPECT_EQ(view.data(), source.data());
  view.At({1, 0, 2, 3}) = 42;
  EXPECT_DOUBLE_EQ(source.At({0, 3, 1, 2}), 42);
  EXPECT_EQ(source.ToCOrder(kExerciseHptt).data(), source.data());
  const auto materialized = view.ToCOrder(kExerciseHptt);
  EXPECT_NE(materialized.data(), source.data());
  EXPECT_DOUBLE_EQ(materialized.At({1, 0, 2, 3}), 42);
  EXPECT_EQ(view.TransposeView({1, 3, 0, 2}).ToCOrder().data(), source.data());
}

TEST(Transpose, ConservativeHeuristicAndByteBoundary) {
  const Array source(UnequalShape());
  const std::vector<int> permutation{2, 0, 3, 1};
  const Array output(source.TransposeView(permutation).shape());
  const auto layout = DescribeDenseLayout(
      source.shape(), source.strides(), permutation, output.strides());
  ASSERT_TRUE(layout);
  EXPECT_FALSE(ShouldUseHptt<double>(*layout, kNative));
  EXPECT_EQ(ShouldUseHptt<double>(*layout, kExerciseHptt), kHpttAvailable);
  const auto bytes = source.Size() * sizeof(double);
  EXPECT_FALSE(ShouldUseHptt<double>(*layout, {bytes + 1, 1}));
  EXPECT_EQ(ShouldUseHptt<double>(*layout, {bytes, 1}), kHpttAvailable);
  EXPECT_FALSE(ShouldUseHptt<int>(*layout, kExerciseHptt));
  for (const std::vector<int>& simple :
       {std::vector<int>{0, 1, 2, 3}, {2, 3, 0, 1}, {2, 1, 0, 3}}) {
    const Array destination(source.TransposeView(simple).shape());
    const auto description = DescribeDenseLayout(
        source.shape(), source.strides(), simple, destination.strides());
    ASSERT_TRUE(description);
    EXPECT_FALSE(ShouldUseHptt<double>(*description, kExerciseHptt));
  }
  const auto n = source.shape()[0];
  const Array matrix({n, n + 1});
  const Array transposed({n + 1, n});
  const std::vector<int> swap{1, 0};
  const auto rank_two = DescribeDenseLayout(
      matrix.shape(), matrix.strides(), swap, transposed.strides());
  ASSERT_TRUE(rank_two);
  EXPECT_FALSE(ShouldUseHptt<double>(*rank_two, kExerciseHptt));
  EXPECT_THROW(
      (void)source.TransposeCopy(permutation, {0, 0}), std::invalid_argument);
#ifdef _OPENMP
  bool selected_in_parallel = true;
#pragma omp parallel num_threads(2) shared(selected_in_parallel)
  {
#pragma omp single
    selected_in_parallel = ShouldUseHptt<double>(*layout, kExerciseHptt);
  }
  EXPECT_FALSE(selected_in_parallel);
#endif
}

TEST(Transpose, ArbitraryStridesUseNative) {
  auto source = Array::Random(UnequalShape(), 31);
  const std::vector<int> permutation{2, 0, 3, 1};
  const Array output(source.TransposeView(permutation).shape());
  auto padded_strides = source.strides();
  padded_strides[0] += 7;
  auto padded = Array::WithStrides(source.shape(), padded_strides);
  Array::Copy(source, padded, {}, 1., 0., kNative);
  auto broadcast_strides = source.strides();
  broadcast_strides[0] = 0;
  const Array broadcast(source.shape(), broadcast_strides, source.data());
  for (const auto& input :
       {source.Slice("::-1,:,:,:"),
        source.Slice(":,::2,:,:"),
        padded,
        broadcast}) {
    auto actual = input.TransposeCopy(permutation, kNative);
    auto expected = actual.Clone(kNative);
    EXPECT_FALSE(DescribeDenseLayout(
        input.shape(), input.strides(), permutation, actual.strides()));
    EXPECT_FALSE(
        wickqc::backend::TryHpttTranspose(
            input.data(),
            input.shape(),
            input.strides(),
            permutation,
            actual.data(),
            actual.strides(),
            1.,
            0.,
            kExerciseHptt));
    Array::Copy(input, actual, permutation, 1., 0., kExerciseHptt);
    EXPECT_DOUBLE_EQ((actual - expected).Norm(), 0);
  }
  auto destination_strides = output.strides();
  for (auto& stride : destination_strides) {
    stride *= 2;
  }
  auto destination = Array::WithStrides(output.shape(), destination_strides);
  EXPECT_FALSE(DescribeDenseLayout(
      source.shape(), source.strides(), permutation, destination.strides()));
  Array::Copy(source, destination, permutation, 1., 0., kExerciseHptt);
  EXPECT_DOUBLE_EQ(
      (destination - source.TransposeCopy(permutation, kNative)).Norm(), 0);
}

TEST(Transpose, SingletonEmptyScalarAndUnsupportedType) {
  auto shape = UnequalShape();
  shape.insert(shape.begin() + 1, 1);
  const auto source = Array::Random(shape, 39);
  const std::vector<int> permutation{3, 0, 4, 2, 1};
  const auto actual = source.TransposeCopy(permutation, kExerciseHptt);
  EXPECT_DOUBLE_EQ(
      (actual - source.TransposeCopy(permutation, kNative)).Norm(), 0);
  EXPECT_EQ(
      Array({0, shape[0]}).TransposeCopy({1, 0}, kExerciseHptt).Size(), 0U);
  EXPECT_DOUBLE_EQ(
      Array::Full({}, 7.).TransposeCopy({}, kExerciseHptt).Item(), 7.);
  using Integers = wickqc::NDArray<int>;
  const auto integers = Integers::Full(UnequalShape(), 9);
  EXPECT_EQ(integers.TransposeCopy({2, 0, 3, 1}, kExerciseHptt).data()[0], 9);
  const std::vector<std::size_t> too_large{
      static_cast<std::size_t>(std::numeric_limits<int>::max()), 2, 3};
  const std::vector<std::ptrdiff_t> strides{6, 3, 1};
  const std::vector<int> reverse{2, 1, 0};
  EXPECT_FALSE(DescribeDenseLayout(too_large, strides, reverse, strides));
}

TEST(Transpose, ScalingZeroBetaAndOverlapKeepNativeSemantics) {
  const auto source = Array::Random(UnequalShape(), 47);
  const std::vector<int> permutation{2, 0, 3, 1};
  const auto view = source.TransposeView(permutation);
  for (const double beta : {0., 1e-20, -0.5}) {
    auto expected = Array::Full(view.shape(), 1e20);
    expected.data()[0] = std::numeric_limits<double>::quiet_NaN();
    auto actual = expected.Clone(kNative);
    Array::Copy(source, expected, permutation, 1., beta, kNative);
    Array::Copy(source, actual, permutation, 1., beta, kExerciseHptt);
    for (std::size_t i = 0; i < actual.Size(); ++i) {
      if (std::isnan(expected.data()[i])) {
        EXPECT_TRUE(std::isnan(actual.data()[i]));
      } else {
        EXPECT_NEAR(
            actual.data()[i],
            expected.data()[i],
            1e-13 * (1 + std::abs(expected.data()[i])));
      }
    }
  }
  auto reference = source.Clone(kNative);
  auto actual = source.Clone(kNative);
  const Array layout(view.shape());
  Array reference_alias(layout.shape(), layout.strides(), reference.data());
  Array actual_alias(layout.shape(), layout.strides(), actual.data());
  // Preserve the existing sequential copy behavior; HPTT is out-of-place only.
  Array::Copy(reference, reference_alias, permutation, 1., 0., kNative);
  Array::Copy(actual, actual_alias, permutation, 1., 0., kExerciseHptt);
  EXPECT_DOUBLE_EQ((actual - reference).Norm(), 0);
}
} // namespace
