#include "precompiled.generated.hpp"
#include "runtime/spatial.hpp"
#include "stored_integrals.hpp"

#include <charconv>
#include <chrono>
#include <iostream>
#include <string_view>

namespace {
using wickqc::NDArray;
using wickqc::method::CCSDOptions;
using wickqc::method::CCSDResult;
using wickqc::method::IntegralConvention;
using wickqc::method::MP2IntegralBinding;
using wickqc::method::SolveCCSD;
using wickqc::method::SolveMP2;
using wickqc::method::SpatialFamily;
using wickqc::method::SpatialMethod;

void Write(hid_t file, const std::string& name, const NDArray<double>& array) {
  const auto values = array.ToCOrder();
  std::array<std::size_t, 4> shape{};
  if (array.Rank() == 4) {
    std::copy(array.shape().begin(), array.shape().end(), shape.begin());
  }
  auto dataset = ao2mo::h5::Matrix<double>(
      file,
      name,
      {array.shape()[0], array.Size() / array.shape()[0]},
      array.Rank() == 4 ? &shape : nullptr);
  ao2mo::h5::Check(H5Dwrite(
      dataset,
      H5T_NATIVE_DOUBLE,
      H5S_ALL,
      H5S_ALL,
      H5P_DEFAULT,
      values.data()));
}

void WriteCC(hid_t file, const char* name, const CCSDResult& result) {
  Write(file, std::string(name) + "/t1", result.amplitudes.singles);
  Write(file, std::string(name) + "/t2", result.amplitudes.doubles);
  Write(file, std::string(name) + "/residual1", result.residuals.singles);
  Write(file, std::string(name) + "/residual2", result.residuals.doubles);
  ao2mo::h5::Handle group(H5Gopen2(file, name, H5P_DEFAULT), H5Gclose);
  ao2mo::h5::DoubleAttribute(
      group, "correlation_energy", result.correlation_energy);
  ao2mo::h5::DoubleAttribute(group, "total_energy", result.total_energy);
  ao2mo::h5::DoubleAttribute(group, "residual_norm", result.residual_norm);
  ao2mo::h5::SizeAttribute(group, "iterations", result.iterations);
  ao2mo::h5::SizeAttribute(group, "converged", result.converged);
  if (result.estimated_peak_bytes) {
    ao2mo::h5::SizeAttribute(
        group, "estimated_peak_bytes", *result.estimated_peak_bytes);
  }
  NDArray<double> history({result.history.size(), 4});
  for (std::size_t i = 0; i < result.history.size(); ++i) {
    const auto& row = result.history[i];
    history.At({i, 0}) = static_cast<double>(row.iteration);
    history.At({i, 1}) = row.correlation_energy;
    history.At({i, 2}) = row.energy_change;
    history.At({i, 3}) = row.residual_norm;
  }
  Write(group, "history", history);
}
} // namespace

int main(int argc, char** argv) {
  try {
    ao2mo::Options options;
    if (argc >= 5 && std::string_view(argv[argc - 2]) == "--memory") {
      const std::string_view value(argv[argc - 1]);
      const auto parsed = std::from_chars(
          value.data(), value.data() + value.size(), options.memory_bytes);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != value.data() + value.size() ||
          options.memory_bytes == 0) {
        throw std::invalid_argument("Invalid memory budget in bytes");
      }
      argc -= 2;
    }
    if (argc < 3 || argc > 8) {
      throw std::invalid_argument(
          "Usage: rhf INPUT.h5 OUTPUT.h5 [chemist|physicist] [stored|incore|outcore] [max_iterations] [compiled|runtime] [dense] [--memory BYTES]");
    }
    const std::string_view notation = argc > 3 ? argv[3] : "chemist";
    const std::string_view workspace = argc > 4 ? argv[4] : "stored";
    const std::string_view mode = argc > 6 ? argv[6] : "compiled";
    if ((notation != "chemist" && notation != "physicist") ||
        (workspace != "stored" && workspace != "incore" &&
         workspace != "outcore") ||
        (mode != "compiled" && mode != "runtime") ||
        (argc == 8 && std::string_view(argv[7]) != "dense")) {
      throw std::invalid_argument("Unknown RHF execution option");
    }
    CCSDOptions solve;
    solve.energy_tolerance = 1e-12;
    solve.residual_tolerance = 1e-10;
    if (argc > 5) {
      const std::string_view count(argv[5]);
      const auto parsed = std::from_chars(
          count.data(), count.data() + count.size(), solve.max_iterations);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != count.data() + count.size()) {
        throw std::invalid_argument("Invalid CCSD iteration limit");
      }
    }
    const auto convention = notation == "chemist"
        ? IntegralConvention::kChemist
        : IntegralConvention::kPhysicist;
    const auto policy = mode == "compiled"
        ? wickqc::runtime::GenerationPolicy::kPrecompiledOnly
        : wickqc::runtime::GenerationPolicy::kRuntimeOnly;
    const SpatialMethod mp_spec{SpatialFamily::kMP, 2, convention},
        cc_spec{SpatialFamily::kCC, 2, convention};
    const wickqc::runtime::SpatialEvaluator mp(
        mp_spec, policy, wickqc::runtime::FindPrecompiled);
    const wickqc::runtime::SpatialEvaluator cc(
        cc_spec, policy, wickqc::runtime::FindPrecompiled);
    using Clock = std::chrono::steady_clock;
    auto t = Clock::now();
    const auto input = ao2mo::ReadReference(argv[1], options.memory_bytes);
    const double read_seconds =
        std::chrono::duration<double>(Clock::now() - t).count();
    auto bindings = cc.Inputs();
    bindings.insert(bindings.end(), mp.Inputs().begin(), mp.Inputs().end());
    bindings.push_back(MP2IntegralBinding(convention));
    options.workspace = workspace == "incore" ? ao2mo::Workspace::kIncore
                                              : ao2mo::Workspace::kOutcore;
    options.scratch_directory =
        std::filesystem::absolute(argv[2]).parent_path();
    t = Clock::now();
    auto data = workspace == "stored"
        ? wickqc::test::StoredIntegrals(argv[1], input).RHF(input)
        : ao2mo::PrepareRHF(input, bindings, cc_spec, options);
    std::size_t integral_bytes = data.chemist_integrals.Rank() == 4
        ? data.chemist_integrals.Size() * sizeof(double)
        : 0;
    for (const auto& [name, tensor] : data.integral_blocks) {
      integral_bytes += tensor.Size() * sizeof(double);
    }
    const auto integral_blocks =
        data.integral_blocks.empty() ? 1 : data.integral_blocks.size();
    if (argc == 8) {
      // Explicit compatibility comparison only; normal execution never asks for
      // pppp.
      ao2mo::Request<double> full;
      full.name = "pppp";
      for (auto& index : full.indices) {
        index = {
            input.integrals.coefficients.at(input.coefficient_family),
            input.integrals.partition.Indices('p')};
      }
      const auto n = input.reference.nmo - input.reference.frozen;
      data.integral_blocks.clear();
      const auto local =
          ao2mo::reference_detail::TransformOptions(input, options);
      (void)ao2mo::MakePlan(input.integrals.basis, std::vector{full}, local);
      auto tensor =
          std::make_shared<NDArray<double>>(NDArray<double>::Shape{n, n, n, n});
      const std::vector buffers{ao2mo::WickBuffer(tensor)};
      (void)ao2mo::Transform(
          input.integrals.basis, std::vector{full}, local, buffers);
      data.chemist_integrals = *tensor;
      data.integral_blocks.clear();
      data.block_convention.reset();
      integral_bytes = tensor->Size() * sizeof(double);
    }
    const double prepare_seconds =
        std::chrono::duration<double>(Clock::now() - t).count();
    const double reference = input.reference.reference_energy;
    std::vector<NDArray<double>> unused_reference{input.reference.occupations};
    unused_reference.insert(
        unused_reference.end(),
        input.reference.rdms.begin(),
        input.reference.rdms.end());
    solve.memory = {
        options.memory_bytes,
        ao2mo::CheckedAdd(
            input.integrals.loading_budget_bytes,
            NDArray<double>::StorageBytes(unused_reference))};
    ao2mo::h5::Handle file(
        H5Fcreate(argv[2], H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    ao2mo::h5::Complete(file, 0);
    ao2mo::h5::StringAttribute(file, "schema", "wickqc.rhf.result.v1");
    ao2mo::h5::StringAttribute(
        file, "input_path", std::filesystem::absolute(argv[1]).string());
    ao2mo::h5::StringAttribute(
        file, "orbital_identity", input.reference.orbital_identity);
    ao2mo::h5::StringAttribute(
        file, "integral_convention", std::string(notation));
    ao2mo::h5::StringAttribute(file, "workspace", std::string(workspace));
    ao2mo::h5::StringAttribute(file, "kernel_mode", std::string(mode));
    ao2mo::h5::DoubleAttribute(file, "reference_energy", reference);
    ao2mo::h5::SizeAttribute(file, "memory_limit_bytes", options.memory_bytes);
    double mp2_seconds = 0;
    {
      t = Clock::now();
      const auto mp2 = SolveMP2(data, mp, reference, convention, solve.memory);
      mp2_seconds = std::chrono::duration<double>(Clock::now() - t).count();
      Write(file, "mp2/t2", mp2.amplitudes.doubles);
      ao2mo::h5::Handle group(H5Gopen2(file, "mp2", H5P_DEFAULT), H5Gclose);
      ao2mo::h5::DoubleAttribute(
          group, "correlation_energy", mp2.correlation_energy);
      ao2mo::h5::DoubleAttribute(group, "total_energy", mp2.total_energy);
      ao2mo::h5::DoubleAttribute(group, "residual_norm", mp2.residual_norm);
      ao2mo::h5::SizeAttribute(
          group, "estimated_peak_bytes", *mp2.estimated_peak_bytes);
      group.Close();
    }
    auto step_options = solve;
    step_options.max_iterations = 0;
    const auto ni = data.occupied, ne = data.orbital_energies.Size() - ni;
    const wickqc::method::RHFAmplitudes guess{
        wickqc::test::ReadTensor(argv[1], "expected/initial/t1", {ni, ne})
            .TransposeView({1, 0}),
        wickqc::test::ReadTensor(
            argv[1], "expected/initial/t2", {ni, ni, ne, ne})
            .TransposeView({2, 3, 0, 1})};
    const auto initial = [&] {
      const auto result =
          SolveCCSD(data, cc, reference, convention, step_options, guess);
      WriteCC(file, "initial", result);
      return result.amplitudes;
    }();
    step_options.max_iterations = 1;
    {
      const auto first =
          SolveCCSD(data, cc, reference, convention, step_options, initial);
      WriteCC(file, "first", first);
    }
    t = Clock::now();
    const auto ccsd =
        SolveCCSD(data, cc, reference, convention, solve, initial);
    const double ccsd_seconds =
        std::chrono::duration<double>(Clock::now() - t).count();
    WriteCC(file, "ccsd", ccsd);
    for (const auto& [name, seconds] : std::map<std::string, double>{
             {"read_seconds", read_seconds},
             {"prepare_seconds", prepare_seconds},
             {"mp2_seconds", mp2_seconds},
             {"ccsd_seconds", ccsd_seconds}}) {
      ao2mo::h5::DoubleAttribute(file, name.c_str(), seconds);
    }
    ao2mo::h5::SizeAttribute(file, "integral_bytes", integral_bytes);
    ao2mo::h5::SizeAttribute(
        file, "integral_blocks", argc == 8 ? 1 : integral_blocks);
    ao2mo::h5::SizeAttribute(
        file, "reference_tensor_bytes", input.reference.TensorBytes());
    ao2mo::h5::SizeAttribute(
        file,
        "amplitude_bytes",
        sizeof(double) *
            (ccsd.amplitudes.singles.Size() + ccsd.amplitudes.doubles.Size()));
    ao2mo::h5::Complete(file, 1);
    file.Close();
    if (!ccsd.converged) {
      std::cerr << "CCSD did not converge after " << ccsd.iterations
                << " updates; residual norm " << ccsd.residual_norm << '\n';
      return 2;
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
