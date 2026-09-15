#include "build_time/kernels.h"
#include "runtime/numeric_kernel.h"
#include "runtime/tensor_binding.h"
#include "tensor_io.h"

#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
const wickqc::runtime::NumericKernel& Kernel(std::string_view name) {
  if (name == "ccsd") {
    return wickqc::generated::Kernel_example_ccsd();
  }
  for (const auto& entry : wickqc::example::ICNEVPT2Kernels()) {
    if (name == "ic-nevpt2/" + std::string(entry.name) ||
        name == "fic-nevpt2/" + std::string(entry.name)) {
      return *entry.kernel;
    }
  }
  throw std::invalid_argument(
      "Unknown precompiled kernel: " + std::string(name));
}

void Describe(
    std::span<const wickqc::runtime::TensorBinding> bindings,
    const wickqc::runtime::Dimensions& dimensions) {
  std::cout << '{';
  bool first = true;
  for (const auto& binding : bindings) {
    std::cout << (first ? "" : ",") << '"' << binding.name << "\":[";
    first = false;
    const auto shape = binding.Shape(dimensions);
    for (std::size_t i = 0; i < shape.size(); ++i) {
      std::cout << (i ? "," : "") << shape[i];
    }
    std::cout << ']';
  }
  std::cout << '}';
}
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--list") {
      std::cout << "ccsd\n";
      for (const auto& entry : wickqc::example::ICNEVPT2Kernels()) {
        std::cout << "ic-nevpt2/" << entry.name << '\n';
      }
      return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "--describe") {
      const auto& kernel = Kernel(argv[2]);
      const auto dimensions = wickqc::example::ReadDimensions(argv[3]);
      std::cout << "{\"inputs\":";
      Describe(kernel.Inputs(), dimensions);
      std::cout << ",\"outputs\":";
      Describe(kernel.Outputs(), dimensions);
      std::cout << "}\n";
      return 0;
    }
    if (argc != 4) {
      throw std::invalid_argument(
          "Usage: example_precompiled KERNEL INPUT_DIRECTORY OUTPUT_DIRECTORY\n"
          "       example_precompiled --describe KERNEL DIMENSIONS_FILE\n"
          "       example_precompiled --list");
    }
    const auto& kernel = Kernel(argv[1]);
    const std::filesystem::path input(argv[2]), output(argv[3]);
    const auto dimensions =
        wickqc::example::ReadDimensions(input / "dimensions.txt");
    wickqc::runtime::TensorMap<double> tensors;
    for (const auto& binding : kernel.Inputs()) {
      tensors.emplace(
          binding.name,
          wickqc::example::ReadTensor(
              input / (binding.name + ".bin"), binding.Shape(dimensions)));
    }
    // This executable links only numeric kernels. All dimensions and values
    // above arrive after compilation; Evaluate calls NDArray::Einsum directly.
    const auto result = kernel.Evaluate(tensors, dimensions);
    std::filesystem::create_directories(output);
    for (const auto& binding : kernel.Outputs()) {
      wickqc::example::WriteTensor(
          output / (binding.name + ".bin"), result.at(binding.name));
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
