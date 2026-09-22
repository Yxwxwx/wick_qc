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

#if defined(WICKQC_ENABLE_AO2MO)
#include <numbers>
const void* OtherMutex();
double OtherTransform(const ao2mo::Basis&);
TEST(HeaderOnly, IntegralTwoTranslationUnits) {
  // A normalized single s Gaussian, with independently known Coulomb integral.
  ao2mo::Basis basis;
  basis.atm = {2, 20, 1, 0, 0, 0};
  basis.bas = {0, 0, 1, 1, 0, 23, 24, 0};
  basis.env.resize(25);
  basis.env[23] = 1;
  basis.env[24] = CINTgto_norm(0, 1);
  const auto expected = 2 / std::sqrt(std::numbers::pi);
  if (std::abs(OtherTransform(basis) - expected) > 1e-13) {
    FAIL() << "Check 1 failed";
  }
  auto coefficients = std::make_shared<ao2mo::Coefficients<ao2mo::Complex>>();
  coefficients->nao = coefficients->nmo = 1;
  coefficients->alpha = {{0.3, 0.4}};
  coefficients->beta = {{std::sqrt(0.75), 0}};
  const auto requests = ao2mo::ProfileRequests<ao2mo::Complex>(
      ao2mo::Profile::kSpinorDense, coefficients, {0, 1, 1, 0});
  ao2mo::Options options;
  options.workspace = ao2mo::Workspace::kIncore;
  options.audit = ao2mo::AuditMode::kAudit;
  const auto result = ao2mo::Transform(basis, requests, options);
  if (std::abs(result.blocks.at(0).values.at(0) - expected) > 1e-13) {
    FAIL() << "Check 2 failed";
  }
  if (OtherMutex() != &ao2mo::h5::ExecutionMutex()) {
    FAIL() << "Check 3 failed";
  }
  // Check both complex components through the public planning API.
  const auto inf = std::numeric_limits<double>::infinity();
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  for (const ao2mo::Complex invalid :
       {ao2mo::Complex{inf, 0}, {0, inf}, {nan, 0}, {0, nan}}) {
    coefficients->beta[0] = invalid;
    try {
      ao2mo::MakePlan(basis, requests, options);
      FAIL() << "Check 4 failed";
    } catch (const std::invalid_argument&) { // NOLINT(bugprone-empty-catch)
      // Expected rejection is the successful test path.
    }
  }
}
#endif
