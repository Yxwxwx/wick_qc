#include <cstddef>
#include "backend/ndarray.hpp"
#include "method/rhf.hpp"
#include "method/specification.hpp"
#include "precompiled.generated.hpp"
#include "runtime/numeric.hpp"
#include "runtime/spatial.hpp"
#include "tensor_io.hpp"

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
  const auto mp2_doubles =
      wickqc::method::MP2Amplitudes(data, convention).doubles;
  const SpatialEvaluator mp(
      {SpatialFamily::kMP, 2, convention},
      policy,
      wickqc::runtime::FindPrecompiled);
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
  const SpatialEvaluator cc(
      {SpatialFamily::kCC, 2, convention},
      policy,
      wickqc::runtime::FindPrecompiled);
  const auto initial = cc.Evaluate(
      data.Bind(cc.Inputs(), {{"tEI", t1}, {"tEEII", t2}}, convention),
      domains);
  const auto& r1 = initial.at("residual1");
  const auto& r2 = initial.at("residual2");
  const auto [next1, next2] = wickqc::method::CCSDStep(
      data, {t1, t2}, wickqc::method::CCSDResiduals(r1, r2));
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
