#include "codegen/cpp_emitter.h"
#include "equation/graph.h"
#include "method/integral_convention.h"
#include "method/spatial_cc.h"
#include "method/spatial_mp.h"
#include "runtime/ndarray_executor.h"

#include <charconv>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <system_error>

int main(int argc, char** argv) {
  try {
    if (argc != 6) {
      throw std::invalid_argument(
          "Usage: generate_cpp {mp|cc} ORDER {chemist|physicist} FUNCTION_SUFFIX OUTPUT.generated.cpp");
    }
    const std::string_view family = argv[1], parameter = argv[2],
                           notation = argv[3];
    int order = 0;
    const auto [end, error] = std::from_chars(
        parameter.data(), parameter.data() + parameter.size(), order);
    if (error != std::errc{} || end != parameter.data() + parameter.size()) {
      throw std::invalid_argument("Method order must be an integer");
    }
    if (notation != "chemist" && notation != "physicist") {
      throw std::invalid_argument(
          "Integral convention must be chemist or physicist");
    }
    const auto convention = notation == "chemist"
        ? wickqc::method::IntegralConvention::kChemist
        : wickqc::method::IntegralConvention::kPhysicist;
    wickqc::equation::ContractionGraph graph;
    if (family == "mp") {
      graph = wickqc::method::SpatialMpGenerator(order, convention).Equations();
    } else if (family == "cc") {
      graph = wickqc::method::SpatialCcGenerator(order, convention).Equations();
    } else {
      throw std::invalid_argument("Method family must be mp or cc");
    }
    const auto program =
        wickqc::runtime::NdArrayExecutor::Compile(graph.Simplify());
    const auto source = wickqc::codegen::CppEmitter::Render(program, argv[4]);
    const std::filesystem::path path(argv[5]);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    const auto temporary = path.string() + ".tmp";
    {
      std::ofstream output(temporary, std::ios::binary);
      output << source;
      output.close();
      if (!output) {
        throw std::runtime_error("Cannot write generated source " + temporary);
      }
    }
    std::filesystem::rename(temporary, path);
    std::cout << family << order << '/' << notation << ": "
              << program.Inputs().size() << " inputs, "
              << program.Outputs().size() << " outputs, " << source.size()
              << " bytes\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
