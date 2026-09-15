#include "method/ccsd.h"
#include "method/ghf.h"
#include "method/ic_nevpt2.h"
#include "method/integral_convention.h"
#include "method/sc_nevpt2.h"
#include "method/spatial_cc.h"
#include "method/spatial_mp.h"
#include "method/uga_ccsd.h"

#include <charconv>
#include <exception>
#include <iostream>
#include <string_view>
#include <system_error>

int main(int argc, char** argv) {
  if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
    std::cout
        << "Usage: example_einsum {ghf|ccsd|spin-orbital-ccsd|uga-ccsd|ic-nevpt2|fic-nevpt2}\n"
           "       example_einsum sc-nevpt2 [--optimize]\n"
           "       example_einsum spatial-mp ORDER [--optimize] [--chemist]\n"
           "       example_einsum spatial-cc {2|3|4} [--optimize] [--chemist]\n"
           "Write the method's NumPy tensor equations to stdout.\n";
    return 0;
  }
  try {
    const std::string_view method = argv[1];
    if (method == "spatial-mp" || method == "spatial-cc") {
      if (argc < 3 || argc > 5) {
        std::cerr
            << "Expected a method order/rank followed by optional --optimize and --chemist\n";
        return 2;
      }
      bool optimize = false, chemist = false;
      for (int i = 3; i < argc; ++i) {
        const std::string_view option = argv[i];
        if (option == "--optimize" && !optimize) {
          optimize = true;
        } else if (option == "--chemist" && !chemist) {
          chemist = true;
        } else {
          std::cerr << "Unknown or repeated option: " << option << '\n';
          return 2;
        }
      }
      const auto convention = chemist
          ? wickqc::method::IntegralConvention::kChemist
          : wickqc::method::IntegralConvention::kPhysicist;
      const std::string_view parameter = argv[2];
      int order = 0;
      const auto converted = std::from_chars(
          parameter.data(), parameter.data() + parameter.size(), order);
      if (converted.ec != std::errc{} ||
          converted.ptr != parameter.data() + parameter.size()) {
        std::cerr << "Method order/rank must be an integer\n";
        return 2;
      }
      std::cout
          << (method == "spatial-mp"
                  ? wickqc::method::SpatialMPGenerator(order, convention)
                        .GenerateNumpy(optimize)
                  : wickqc::method::SpatialCCGenerator(order, convention)
                        .GenerateNumpy(optimize));
      return 0;
    }
    if (method == "sc-nevpt2") {
      if (argc > 3 ||
          (argc == 3 && std::string_view(argv[2]) != "--optimize")) {
        std::cerr << "SC-NEVPT2 accepts only optional --optimize\n";
        return 2;
      }
      std::cout << wickqc::method::SCNEVPT2Generator().GenerateNumpy(argc == 3);
      return 0;
    }
    if (argc != 2) {
      std::cerr << "This method accepts no additional arguments\n";
      return 2;
    }
    if (method == "ghf") {
      std::cout << wickqc::method::GHFGenerator().GenerateNumpy();
    } else if (method == "ccsd") {
      std::cout << wickqc::method::SpatialCCGenerator(
                       2, wickqc::method::IntegralConvention::kChemist)
                       .GenerateNumpy();
    } else if (method == "spin-orbital-ccsd") {
      std::cout << wickqc::method::CCSDGenerator().GenerateNumpy();
    } else if (method == "uga-ccsd") {
      std::cout << wickqc::method::UGACCSDGenerator().GenerateNumpy();
    } else if ((method == "ic-nevpt2" || method == "fic-nevpt2")) {
      std::cout << wickqc::method::ICNEVPT2Generator().GenerateNumpy();
    } else {
      std::cerr << "Unknown method: " << method << '\n';
      return 2;
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
