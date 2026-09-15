#include "backend/ndarray.hpp"
#include "method/rhf_data.h"
#include "method/spatial_method.h"
#include "runtime/spatial_evaluator.h"
#include "runtime/tensor_binding.h"
#include "tensor_io.h"

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {
using Array = wickqc::NDArray<double>;
using wickqc::example::ReadTensor;
using wickqc::example::WriteTensor;
using wickqc::method::IntegralConvention;
using wickqc::method::SpatialFamily;
using wickqc::runtime::GenerationPolicy;
using wickqc::runtime::SpatialEvaluator;
using wickqc::runtime::TensorMap;

double Gap(
    const Array& energies,
    std::size_t occupied,
    std::size_t a,
    std::size_t i) {
  const double gap = energies.At({i}) - energies.At({occupied + a});
  if (!std::isfinite(gap) || gap >= -1e-10) {
    throw std::invalid_argument(
        "Expected negative occupied-virtual orbital-energy gaps");
  }
  return gap;
}

void Evaluate(
    const std::filesystem::path& input,
    const std::filesystem::path& output,
    GenerationPolicy policy,
    IntegralConvention convention) {
  const auto dimensions =
      wickqc::example::ReadDimensions(input / "dimensions.txt");
  const auto nocc = dimensions.at({1, 0}), nvir = dimensions.at({8, 0});
  if (dimensions.at({2, 0}) != 0 || nocc == 0 || nvir == 0 ||
      nocc > std::numeric_limits<std::size_t>::max() - nvir) {
    throw std::invalid_argument(
        "RHF dimensions require nonempty occupied/virtual spaces and no active space");
  }
  const auto nmo = nocc + nvir;
  wickqc::method::RHFData data{
      nocc,
      ReadTensor(input / "eps_mp.bin", {nmo}),
      ReadTensor(input / "fock.bin", {nmo, nmo}),
      ReadTensor(input / "eri_chemist.bin", {nmo, nmo, nmo, nmo})};
  const auto domains = data.Dimensions();
  Array mp2_doubles({nvir, nvir, nocc, nocc});
  for (std::size_t a = 0; a < nvir; ++a) {
    for (std::size_t b = 0; b < nvir; ++b) {
      for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t j = 0; j < nocc; ++j) {
          mp2_doubles.At({a, b, i, j}) =
              data.chemist_integrals.At({nocc + a, i, nocc + b, j}) /
              (Gap(data.orbital_energies, nocc, a, i) +
               Gap(data.orbital_energies, nocc, b, j));
        }
      }
    }
  }
  const SpatialEvaluator mp({SpatialFamily::kMP, 2, convention}, policy);
  const auto mp_result = mp.Evaluate(
      data.Bind(
          mp.Inputs(),
          {{"u1EI", Array({nvir, nocc})}, {"u1EEII", mp2_doubles}},
          convention),
      domains);

  data.orbital_energies = ReadTensor(input / "eps_cc.bin", {nmo});
  const auto t1 = ReadTensor(input / "cc_t1_initial.bin", {nvir, nocc});
  const auto t2 =
      ReadTensor(input / "cc_t2_initial.bin", {nvir, nvir, nocc, nocc});
  const SpatialEvaluator cc({SpatialFamily::kCC, 2, convention}, policy);
  const auto initial = cc.Evaluate(
      data.Bind(cc.Inputs(), {{"tEI", t1}, {"tEEII", t2}}, convention),
      domains);
  const auto& r1 = initial.at("residual1");
  const auto& r2 = initial.at("residual2");
  auto next1 = t1.Clone(), next2 = t2.Clone();
  // E1 projectors give r1=2*R1, r2(abij)=4*R2(abij)-2*R2(abji).
  // Invert that spin metric before Jacobi: t_new = t + R/(eps_occ-eps_vir).
  for (std::size_t a = 0; a < nvir; ++a) {
    for (std::size_t i = 0; i < nocc; ++i) {
      next1.At({a, i}) +=
          r1.At({a, i}) / (2 * Gap(data.orbital_energies, nocc, a, i));
      for (std::size_t b = 0; b < nvir; ++b) {
        for (std::size_t j = 0; j < nocc; ++j) {
          const double residual =
              r2.At({a, b, i, j}) / 3 + r2.At({a, b, j, i}) / 6;
          next2.At({a, b, i, j}) += residual /
              (Gap(data.orbital_energies, nocc, a, i) +
               Gap(data.orbital_energies, nocc, b, j));
        }
      }
    }
  }
  const auto first = cc.Evaluate(
      data.Bind(cc.Inputs(), {{"tEI", next1}, {"tEEII", next2}}, convention),
      domains);
  std::filesystem::create_directories(output);
  WriteTensor(output / "mp2_t2.bin", mp2_doubles);
  WriteTensor(output / "cc_t1_first.bin", next1);
  WriteTensor(output / "cc_t2_first.bin", next2);
  WriteTensor(output / "cc_r1_initial.bin", r1);
  WriteTensor(output / "cc_r2_initial.bin", r2);
  std::ofstream report(output / "energies.json");
  report << std::setprecision(17)
         << "{\"mp2_correlation\":" << mp_result.at("energy2").Item()
         << ",\"mp2_residual1_norm\":" << mp_result.at("residual1_rank1").Norm()
         << ",\"mp2_residual2_norm\":" << mp_result.at("residual1_rank2").Norm()
         << ",\"cc_initial_correlation\":" << initial.at("energy").Item()
         << ",\"cc_first_correlation\":" << first.at("energy").Item() << "}\n";
  report.close();
  if (!report) {
    throw std::runtime_error("Cannot write energy report");
  }
}
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      throw std::invalid_argument(
          "Usage: example_rhf INPUT_DIRECTORY OUTPUT_DIRECTORY {compiled|runtime} {chemist|physicist}");
    }
    const std::string_view mode(argv[3]), convention(argv[4]);
    if ((mode != "compiled" && mode != "runtime") ||
        (convention != "chemist" && convention != "physicist")) {
      throw std::invalid_argument(
          "Unknown execution mode or integral convention");
    }
    Evaluate(
        argv[1],
        argv[2],
        mode == "compiled" ? GenerationPolicy::kPrecompiledOnly
                           : GenerationPolicy::kRuntimeOnly,
        convention == "chemist" ? IntegralConvention::kChemist
                                : IntegralConvention::kPhysicist);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
