#include "method/nevpt2_solver.hpp"
#include "runtime_dimensions.hpp"

#include <gtest/gtest.h>
#include <numeric>

namespace {
using wickqc::NDArray;
using wickqc::method::AssembleICNEVPT2;
using wickqc::method::ICNEVPT2System;
using wickqc::method::kICNEVPT2Subspaces;
using wickqc::method::kSCNEVPT2Subspaces;
using wickqc::method::ReferenceKind;
using wickqc::method::SCNEVPT2Block;
using wickqc::method::SolveICNEVPT2;
using wickqc::method::SolveICNEVPT2Block;
using wickqc::method::SolveSCNEVPT2;
using wickqc::method::SpatialReference;
using wickqc::runtime::TensorMap;

TEST(NEVPT2, SCRestrictionDenominatorAndNormCutoff) {
  const auto n = wickqc::test::Dimension("WICKQC_TEST_VIRTUAL");
  NDArray<double> energies({n});
  for (std::size_t i = 0; i < n; ++i) {
    energies.At({i}) = static_cast<double>(2 + i);
  }
  TensorMap<double> tensors{
      {"norm", NDArray<double>({n, n})}, {"hexp", NDArray<double>({n, n})}};
  double expected = 0, norm = 0;
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t s = 0; s < n; ++s) {
      const double value =
          r == 0 && s == 0 ? 1e-14 : static_cast<double>(1 + r + 2 * s);
      tensors.at("norm").At({r, s}) = value;
      tensors.at("hexp").At({r, s}) = 3 * value;
      if (r <= s && value > 1e-14) {
        norm += value;
        expected -= value / (energies.At({r}) + energies.At({s}) + 3);
      }
    }
  }
  const auto result =
      SCNEVPT2Block("rs", tensors, NDArray<double>({0}), energies);
  EXPECT_NEAR(result.norm, norm, 1e-12);
  EXPECT_NEAR(result.correlation_energy, expected, 1e-12);
  EXPECT_EQ(result.retained, n * (n + 1) / 2 - 1);
  TensorMap<double> single{
      {"norm", NDArray<double>({1})}, {"hexp", NDArray<double>({1})}};
  NDArray<double> core({1});
  core.At({0}) = -2;
  single.at("norm").At({0}) = -1;
  single.at("hexp").At({0}) = -3;
  EXPECT_NEAR(
      SCNEVPT2Block("i", single, core, energies).correlation_energy,
      0.2,
      1e-14);
  single.at("hexp").At({0}) = 2;
  EXPECT_THROW(
      (void)SCNEVPT2Block("i", single, core, energies), std::runtime_error);
  single.at("norm").At({0}) = 1;
  single.at("hexp").At({0}) = std::numeric_limits<double>::max();
  core.At({0}) = -std::numeric_limits<double>::max();
  EXPECT_THROW(
      (void)SCNEVPT2Block("i", single, core, energies), std::runtime_error);
  TensorMap<double> empty{
      {"norm", NDArray<double>({0})}, {"hexp", NDArray<double>({0})}};
  EXPECT_EQ(
      SCNEVPT2Block("i", empty, NDArray<double>({0}), energies)
          .correlation_energy,
      0);
  EXPECT_THROW(
      (void)SCNEVPT2Block("unknown", empty, core, energies),
      std::invalid_argument);
}

TEST(NEVPT2, ICCoupledComponentsStayInterleaved) {
  const auto n = wickqc::test::Dimension("WICKQC_TEST_ACTIVE");
  const wickqc::runtime::Dimensions dimensions{
      {{1, 0}, 1}, {{2, 0}, n}, {{8, 0}, 2}};
  TensorMap<double> tensors;
  for (std::size_t c = 1; c <= 2; ++c) {
    NDArray<double> rhs({1, 2, n, n});
    for (std::size_t x = 0; x < rhs.Size(); ++x) {
      rhs.data()[x] = static_cast<double>(1000 * c + x);
    }
    tensors.emplace("rheq" + std::to_string(c), std::move(rhs));
    for (std::size_t d = 1; d <= 2; ++d) {
      NDArray<double> h({1, 2, n, n, n, n});
      for (std::size_t x = 0; x < h.Size(); ++x) {
        h.data()[x] = static_cast<double>(100000 * c + 10000 * d + x);
      }
      tensors.emplace(
          "hexp" + std::to_string(c) + std::to_string(d), std::move(h));
    }
  }
  const auto system = AssembleICNEVPT2("irabpq", tensors, dimensions);
  const auto active = n * n;
  EXPECT_EQ(system.rhs.shape(), (NDArray<double>::Shape{2, active * 2}));
  for (std::size_t b = 0; b < 2; ++b) {
    for (std::size_t row = 0; row < active * 2; ++row) {
      EXPECT_EQ(
          system.rhs.At({b, row}), 1000 * (row % 2 + 1) + b * active + row / 2);
      for (std::size_t col = 0; col < active * 2; ++col) {
        EXPECT_EQ(
            system.hamiltonian.At({b, row, col}),
            100000 * (row % 2 + 1) + 10000 * (col % 2 + 1) +
                (b * active + row / 2) * active + col / 2);
      }
    }
  }
}

TEST(NEVPT2, ICActiveAndExternalPairRestrictions) {
  const auto n = wickqc::test::Dimension("WICKQC_TEST_ACTIVE");
  const wickqc::runtime::Dimensions dimensions{
      {{1, 0}, 1}, {{2, 0}, n}, {{8, 0}, 2}};
  TensorMap<double> tensors{
      {"rheq", NDArray<double>({2, 2, n, n})},
      {"hexp", NDArray<double>({2, 2, n, n, n, n})}};
  for (auto& [name, tensor] : tensors) {
    std::iota(tensor.data(), tensor.data() + tensor.Size(), 1.0);
  }
  for (const bool strict : {false, true}) {
    const auto system = AssembleICNEVPT2(
        strict ? "rsabpq_minus" : "rsabpq_plus", tensors, dimensions);
    const auto width = n * (strict ? n - 1 : n + 1) / 2;
    EXPECT_EQ(
        system.rhs.shape(), (NDArray<double>::Shape{strict ? 1U : 3U, width}));
    std::size_t batch = 0;
    for (std::size_t r = 0; r < 2; ++r) {
      for (std::size_t s = r + strict; s < 2; ++s) {
        std::size_t row = 0;
        for (std::size_t a = 0; a < n; ++a) {
          for (std::size_t b = a + strict; b < n; ++b, ++row) {
            EXPECT_EQ(
                system.rhs.At({batch, row}),
                tensors.at("rheq").At({r, s, a, b}));
            std::size_t col = 0;
            for (std::size_t p = 0; p < n; ++p) {
              for (std::size_t q = p + strict; q < n; ++q, ++col) {
                EXPECT_EQ(
                    system.hamiltonian.At({batch, row, col}),
                    tensors.at("hexp").At({r, s, a, b, p, q}));
              }
            }
          }
        }
        ++batch;
      }
    }
  }
}

TEST(NEVPT2, ICMinimumNormRankCutoffAndEmptySubspaces) {
  ICNEVPT2System system{NDArray<double>({2, 2, 2}), NDArray<double>({2, 2})};
  // Rank-one system: all exact solutions have x0 + 2*x1 = 1; the minimum
  // norm solution is (1/5, 2/5), yielding -1 correlation energy.
  system.hamiltonian.At({0, 0, 0}) = 1;
  system.hamiltonian.At({0, 0, 1}) = system.hamiltonian.At({0, 1, 0}) = 2;
  system.hamiltonian.At({0, 1, 1}) = 4;
  system.rhs.At({0, 0}) = 1;
  system.rhs.At({0, 1}) = 2;
  system.hamiltonian.At({1, 0, 0}) = 1;
  system.hamiltonian.At({1, 1, 1}) = std::numeric_limits<double>::epsilon();
  system.rhs.At({1, 0}) = system.rhs.At({1, 1}) = 1;
  const auto solved = SolveICNEVPT2Block(system);
  EXPECT_EQ(solved.ranks, (std::vector<std::size_t>{1, 1}));
  EXPECT_NEAR(solved.amplitudes.At({0, 0}), 0.2, 1e-14);
  EXPECT_NEAR(solved.amplitudes.At({0, 1}), 0.4, 1e-14);
  EXPECT_NEAR(solved.amplitudes.At({1, 1}), 0, 1e-14);
  EXPECT_NEAR(solved.correlation_energy, -2, 1e-14);
  for (auto [batch, width] : {std::pair{0U, 2U}, {2U, 0U}, {0U, 0U}}) {
    const ICNEVPT2System empty{
        NDArray<double>({batch, width, width}),
        NDArray<double>({batch, width})};
    EXPECT_EQ(SolveICNEVPT2Block(empty).correlation_energy, 0);
  }
  const TensorMap<double> pair{
      {"rheq", NDArray<double>({2, 2, 1, 1})},
      {"hexp", NDArray<double>({2, 2, 1, 1, 1, 1})}};
  const auto empty_pair = AssembleICNEVPT2(
      "rsabpq_minus", pair, {{{1, 0}, 0}, {{2, 0}, 1}, {{8, 0}, 2}});
  EXPECT_EQ(empty_pair.rhs.shape(), (NDArray<double>::Shape{1, 0}));
  EXPECT_EQ(SolveICNEVPT2Block(empty_pair).correlation_energy, 0);
  system.rhs.At({0, 0}) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)SolveICNEVPT2Block(system), std::invalid_argument);
}

TEST(NEVPT2, MemoryPreflightRunsBeforeAnySubspace) {
  namespace test = wickqc::test;
  namespace runtime = wickqc::runtime;
  const auto nc = test::Dimension("WICKQC_TEST_OCCUPIED");
  const auto na = test::Dimension("WICKQC_TEST_ACTIVE");
  const auto ne = test::Dimension("WICKQC_TEST_VIRTUAL");
  SpatialReference reference;
  reference.kind = ReferenceKind::kCASSCF;
  reference.ncore = nc;
  reference.nactive = na;
  reference.nmo = nc + na + ne;
  reference.electrons = {nc + na, nc + na};
  reference.orbital_ids.resize(reference.nmo);
  std::iota(reference.orbital_ids.begin(), reference.orbital_ids.end(), 0);
  reference.orbital_identity = reference.energy_source = reference.fock_source =
      "test";
  reference.orbital_energies = NDArray<double>({reference.nmo});
  reference.occupations = NDArray<double>({reference.nmo});
  reference.fock_mo = NDArray<double>({reference.nmo, reference.nmo});
  for (std::size_t i = 0; i < nc + na; ++i) {
    reference.occupations.At({i}) = 2;
  }
  for (std::size_t rank = 0; rank < 4; ++rank) {
    reference.rdms[rank] =
        NDArray<double>(NDArray<double>::Shape(2 * (rank + 1), na));
  }
  for (std::size_t i = 0; i < na; ++i) {
    reference.rdms[0].At({i, i}) = 2;
  }
  runtime::NumericKernel kernel{
      {},
      {},
      [](const TensorMap<double>&,
         const runtime::Dimensions&) -> TensorMap<double> {
        throw std::logic_error("kernel was evaluated");
      },
      nullptr,
      std::vector<runtime::WorkspaceTerm>{{1024, {}}}};
  std::map<std::string, runtime::NumericKernel> sc, ic;
  for (auto name : kSCNEVPT2Subspaces) {
    sc.emplace(name, kernel);
  }
  for (auto name : kICNEVPT2Subspaces) {
    ic.emplace(name, kernel);
  }
  EXPECT_THROW(
      (void)SolveSCNEVPT2(sc, {}, reference, {}, {1, 0}), std::runtime_error);
  EXPECT_THROW(
      (void)SolveICNEVPT2(ic, {}, reference, {}, {1, 0}), std::runtime_error);
}
} // namespace
