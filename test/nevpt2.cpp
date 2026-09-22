#include "method/nevpt2.hpp"
#include "build_time/kernels.hpp"
#include "method/nevpt2_solver.hpp"
#include "runtime/executor.hpp"
#include "stored_integrals.hpp"

#include <charconv>
#include <chrono>
#include <iostream>

namespace {
using wickqc::NDArray;
namespace runtime = wickqc::runtime;
namespace example = wickqc::example;
using wickqc::method::ICNEVPT2BlockResult;
using wickqc::method::ICNEVPT2Generator;
using wickqc::method::ICNEVPT2System;
using wickqc::method::NEVPT2Method;
using wickqc::method::NEVPT2Result;
using wickqc::method::SCNEVPT2BlockResult;
using wickqc::method::SCNEVPT2Generator;
using wickqc::method::SolveICNEVPT2;
using wickqc::method::SolveSCNEVPT2;
namespace nevpt2_solver_detail = wickqc::method::nevpt2_solver_detail;
using Clock = std::chrono::steady_clock;

ao2mo::h5::Handle Group(hid_t file, const std::string& name) {
  return {
      H5Gcreate2(file, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Gclose};
}

void Write(
    hid_t group,
    const std::string& name,
    const NDArray<double>& tensor) {
  const std::vector<hsize_t> shape(
      tensor.shape().begin(), tensor.shape().end());
  ao2mo::h5::Handle space(
      H5Screate_simple(tensor.Rank(), shape.data(), nullptr), H5Sclose);
  ao2mo::h5::Handle dataset(
      H5Dcreate2(
          group,
          name.c_str(),
          H5T_IEEE_F64LE,
          space,
          H5P_DEFAULT,
          H5P_DEFAULT,
          H5P_DEFAULT),
      H5Dclose);
  const auto contiguous = tensor.ToCOrder();
  if (tensor.Size()) {
    ao2mo::h5::Check(H5Dwrite(
        dataset,
        H5T_NATIVE_DOUBLE,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        contiguous.data()));
  }
  dataset.Close();
}

void WriteSummary(hid_t group, const NEVPT2Result& result) {
  ao2mo::h5::DoubleAttribute(
      group, "correlation_energy", result.correlation_energy);
  ao2mo::h5::DoubleAttribute(group, "total_energy", result.total_energy);
  if (result.estimated_peak_bytes) {
    ao2mo::h5::SizeAttribute(
        group, "estimated_peak_bytes", *result.estimated_peak_bytes);
  }
  ao2mo::h5::DoubleAttribute(
      group, "contraction_seconds", result.contraction_seconds);
  ao2mo::h5::DoubleAttribute(
      group, "assembly_seconds", result.assembly_seconds);
  ao2mo::h5::DoubleAttribute(group, "solve_seconds", result.solve_seconds);
  auto contributions = Group(group, "contributions");
  for (const auto& [name, value] : result.subspace_energies) {
    ao2mo::h5::DoubleAttribute(contributions, name.c_str(), value);
  }
  contributions.Close();
}

template <typename Kernel>
void Run(
    hid_t file,
    const ao2mo::ReferenceInput& input,
    const std::map<std::string, Kernel>& sc_kernels,
    const std::map<std::string, Kernel>& ic_kernels,
    const ao2mo::Options& options,
    bool diagnostics,
    const std::filesystem::path& source,
    bool stored) {
  std::vector<runtime::TensorBinding> bindings;
  for (const auto* kernels : {&sc_kernels, &ic_kernels}) {
    for (const auto& [name, kernel] : *kernels) {
      bindings.insert(
          bindings.end(), kernel.Inputs().begin(), kernel.Inputs().end());
    }
  }
  auto start = Clock::now();
  const auto inputs = stored
      ? wickqc::test::StoredIntegrals(source, input).NEVPT2(input, bindings)
      : ao2mo::PrepareNEVPT2(input, bindings, NEVPT2Method::kIC, options);
  ao2mo::h5::DoubleAttribute(
      file,
      "prepare_seconds",
      std::chrono::duration<double>(Clock::now() - start).count());
  runtime::MemoryBudget memory{
      options.memory_bytes, input.integrals.loading_budget_bytes};
  std::size_t diagnostic_bytes = 0;
  if (diagnostics) {
    // At most one dense serialization copy is live. Restricted/coupled IC
    // matrices fit within the sum of a kernel's unrestricted outputs.
    for (const auto& [name, tensor] : inputs) {
      diagnostic_bytes =
          std::max(diagnostic_bytes, tensor.Size() * sizeof(double));
    }
    for (const auto* kernels : {&sc_kernels, &ic_kernels}) {
      for (const auto& [name, kernel] : *kernels) {
        std::size_t bytes = 0;
        for (const auto& output : kernel.Outputs()) {
          bytes = ao2mo::CheckedAdd(
              bytes,
              ao2mo::CheckedProduct(
                  {nevpt2_solver_detail::Product(
                       output.Shape(input.reference.Dimensions())),
                   sizeof(double)}));
        }
        diagnostic_bytes = std::max(diagnostic_bytes, bytes);
      }
    }
  }
  memory.reserved_bytes =
      ao2mo::CheckedAdd(memory.reserved_bytes, diagnostic_bytes);
  const auto resident =
      nevpt2_solver_detail::ResidentBytes(inputs, input.reference);
  (void)memory.Check(resident, "NEVPT2 input serialization");
  ao2mo::h5::SizeAttribute(file, "resident_tensor_bytes", resident);
  ao2mo::h5::SizeAttribute(file, "diagnostic_copy_bytes", diagnostic_bytes);
  if (diagnostics) {
    auto group = Group(file, "inputs");
    for (const auto& [name, tensor] : inputs) {
      Write(group, name, tensor);
    }
    group.Close();
  }
  auto sc = Group(file, "sc"), ic = Group(file, "ic");
  double output_seconds = 0;
  start = Clock::now();
  const auto sc_result = SolveSCNEVPT2(
      sc_kernels,
      inputs,
      input.reference,
      [&](std::string_view name,
          const runtime::TensorMap<double>& tensors,
          const SCNEVPT2BlockResult& result) {
        const auto output_start = Clock::now();
        auto group = Group(sc, std::string(name));
        ao2mo::h5::DoubleAttribute(group, "norm", result.norm);
        ao2mo::h5::DoubleAttribute(
            group, "correlation_energy", result.correlation_energy);
        ao2mo::h5::SizeAttribute(group, "retained", result.retained);
        if (diagnostics) {
          for (const auto& [key, tensor] : tensors) {
            Write(group, key, tensor);
          }
        }
        group.Close();
        output_seconds +=
            std::chrono::duration<double>(Clock::now() - output_start).count();
      },
      memory);
  ao2mo::h5::DoubleAttribute(
      file,
      "sc_seconds",
      std::chrono::duration<double>(Clock::now() - start).count() -
          output_seconds);
  WriteSummary(sc, sc_result);
  const double sc_output_seconds = output_seconds;
  start = Clock::now();
  const auto ic_result = SolveICNEVPT2(
      ic_kernels,
      inputs,
      input.reference,
      [&](std::string_view name,
          const runtime::TensorMap<double>& tensors,
          const ICNEVPT2System& system,
          const ICNEVPT2BlockResult& result) {
        const auto output_start = Clock::now();
        auto group = Group(ic, std::string(name));
        ao2mo::h5::DoubleAttribute(
            group, "correlation_energy", result.correlation_energy);
        ao2mo::h5::IndexVector(group, "ranks", result.ranks);
        if (diagnostics) {
          for (const auto& [key, tensor] : tensors) {
            Write(group, key, tensor);
          }
          Write(group, "matrix", system.hamiltonian);
          Write(group, "rhs", system.rhs);
          Write(group, "amplitudes", result.amplitudes);
          Write(group, "singular_values", result.singular_values);
        }
        group.Close();
        output_seconds +=
            std::chrono::duration<double>(Clock::now() - output_start).count();
      },
      memory);
  ao2mo::h5::DoubleAttribute(
      file,
      "ic_seconds",
      std::chrono::duration<double>(Clock::now() - start).count() -
          output_seconds + sc_output_seconds);
  ao2mo::h5::DoubleAttribute(file, "diagnostic_seconds", output_seconds);
  WriteSummary(ic, ic_result);
  sc.Close();
  ic.Close();
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
    if (argc < 3 || argc > 6) {
      throw std::invalid_argument(
          "Usage: nevpt2 INPUT.h5 OUTPUT.h5 [stored|incore|outcore] [compiled|runtime] [diagnostics] [--memory BYTES]");
    }
    const std::string_view workspace = argc > 3 ? argv[3] : "stored";
    const std::string_view mode = argc > 4 ? argv[4] : "compiled";
    if ((workspace != "stored" && workspace != "incore" &&
         workspace != "outcore") ||
        (mode != "compiled" && mode != "runtime") ||
        (argc == 6 && std::string_view(argv[5]) != "diagnostics")) {
      throw std::invalid_argument("Unknown NEVPT2 execution option");
    }
    const auto start = Clock::now();
    const auto input = ao2mo::ReadReference(argv[1], options.memory_bytes);
    const double read_seconds =
        std::chrono::duration<double>(Clock::now() - start).count();
    options.workspace = workspace == "incore" ? ao2mo::Workspace::kIncore
                                              : ao2mo::Workspace::kOutcore;
    options.scratch_directory =
        std::filesystem::absolute(argv[2]).parent_path();
    ao2mo::h5::Handle file(
        H5Fcreate(argv[2], H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    ao2mo::h5::Complete(file, 0);
    ao2mo::h5::StringAttribute(file, "schema", "wickqc.nevpt2.result.v1");
    ao2mo::h5::StringAttribute(
        file, "input_path", std::filesystem::absolute(argv[1]).string());
    ao2mo::h5::StringAttribute(
        file, "orbital_identity", input.reference.orbital_identity);
    ao2mo::h5::StringAttribute(file, "workspace", std::string(workspace));
    ao2mo::h5::StringAttribute(file, "kernel_mode", std::string(mode));
    ao2mo::h5::DoubleAttribute(
        file, "reference_energy", input.reference.reference_energy);
    ao2mo::h5::DoubleAttribute(file, "read_seconds", read_seconds);
    ao2mo::h5::SizeAttribute(
        file, "reference_tensor_bytes", input.reference.TensorBytes());
    ao2mo::h5::SizeAttribute(file, "root", input.reference.root);
    ao2mo::h5::SizeAttribute(file, "memory_limit_bytes", options.memory_bytes);
    if (mode == "compiled") {
      std::map<std::string, runtime::NumericKernel> sc, ic;
      for (const auto& entry : example::SCNEVPT2Kernels()) {
        sc.emplace(entry.name, *entry.kernel);
      }
      for (const auto& entry : example::ICNEVPT2Kernels()) {
        ic.emplace(entry.name, *entry.kernel);
      }
      Run(file,
          input,
          sc,
          ic,
          options,
          argc == 6,
          argv[1],
          workspace == "stored");
    } else {
      std::map<std::string, runtime::NDArrayExecutor> sc, ic;
      for (const auto& [name, graph] : SCNEVPT2Generator().Equations()) {
        sc.emplace(name, runtime::NDArrayExecutor::Compile(graph));
      }
      for (const auto& [name, graph] : ICNEVPT2Generator().Equations()) {
        ic.emplace(name, runtime::NDArrayExecutor::Compile(graph));
      }
      Run(file,
          input,
          sc,
          ic,
          options,
          argc == 6,
          argv[1],
          workspace == "stored");
    }
    ao2mo::h5::Complete(file, 1);
    file.Close();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
