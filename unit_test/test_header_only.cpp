#include "wick.hpp"

#include <gtest/gtest.h>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include "runtime_dimensions.hpp"

namespace wickqc::test {

std::string HeaderOnlyExpression();
double HeaderOnlyTrace(std::size_t size);
decltype(&symbolic::SignedPermutation::Identity) HeaderOnlyIdentityAddress();

TEST(HeaderOnly, MultipleTranslationUnitsShareDefinitions) {
  EXPECT_EQ(
      HeaderOnlyIdentityAddress(), &symbolic::SignedPermutation::Identity);
  symbolic::IndexRegistry indices;
  symbolic::SymmetryRegistry symmetries;
  const auto expression =
      symbolic::Expression::Parse("2 A[] + 3 A[]", indices, symmetries)
          .Simplify();
  std::ostringstream output;
  output << expression;
  EXPECT_EQ(HeaderOnlyExpression(), output.str());
}

TEST(HeaderOnly, UmbrellaIncludesRuntimeAndMethodInterfaces) {
  const auto size = Dimension("WICKQC_TEST_OCCUPIED");
  EXPECT_DOUBLE_EQ(HeaderOnlyTrace(size), static_cast<double>(size));
  const method::SpatialMPGenerator mp(2);
  const auto executor = runtime::NDArrayExecutor::Compile(mp.Equations());
  EXPECT_FALSE(executor.Outputs().empty());
  EXPECT_FALSE(
      codegen::CPPEmitter::Render(executor, "header_only_mp2").empty());
  const runtime::SpatialEvaluator evaluator({method::SpatialFamily::kMP, 2});
  EXPECT_FALSE(evaluator.IsPrecompiled());
  EXPECT_FALSE(evaluator.Outputs().empty());
}

TEST(HeaderOnly, OptionalGeneratedLookupIsExplicit) {
  const auto lookup =
      +[](method::SpatialMethod requested) -> const runtime::NumericKernel* {
    static const runtime::NumericKernel kKernel{{}, {}, nullptr, nullptr};
    return requested == method::SpatialMethod{method::SpatialFamily::kMP, 2}
        ? &kKernel
        : nullptr;
  };
  const method::SpatialMethod method{method::SpatialFamily::kMP, 2};
  const runtime::SpatialEvaluator compiled(
      method, runtime::GenerationPolicy::kPrecompiledOnly, lookup);
  EXPECT_TRUE(compiled.IsPrecompiled());
  const runtime::SpatialEvaluator runtime_evaluator(
      method, runtime::GenerationPolicy::kRuntimeOnly, lookup);
  EXPECT_FALSE(runtime_evaluator.IsPrecompiled());
  EXPECT_THROW(
      (void)runtime::SpatialEvaluator(
          method, runtime::GenerationPolicy::kPrecompiledOnly),
      std::invalid_argument);
  EXPECT_THROW(
      (void)runtime::SpatialEvaluator(
          {method::SpatialFamily::kCC, 2},
          runtime::GenerationPolicy::kPrecompiledOnly,
          lookup),
      std::invalid_argument);
}

} // namespace wickqc::test
