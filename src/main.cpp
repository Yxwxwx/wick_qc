#include "method/ccsd.h"
#include "method/ghf.h"
#include "method/ic_nevpt2.h"
#include "method/sc_nevpt2.h"
#include "method/spatial_cc.h"
#include "method/spatial_mp.h"
#include "method/uga_ccsd.h"

#include <charconv>
#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
  if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
    std::cout << "Usage: wick_qc {ghf|ccsd|uga-ccsd|ic-nevpt2}\n"
                 "       wick_qc sc-nevpt2 [--optimize]\n"
                 "       wick_qc spatial-mp {2|3|4} [--optimize]\n"
                 "       wick_qc spatial-cc {2|3|4} [--optimize]\n"
                 "Write the method's NumPy tensor equations to stdout.\n";
    return 0;
  }
  try {
    const std::string_view method = argv[1];
    if (method == "spatial-mp" || method == "spatial-cc") {
      if (argc < 3 || argc > 4 ||
          (argc == 4 && std::string_view(argv[3]) != "--optimize")) {
        std::cerr
            << "Expected a method order/rank followed by optional --optimize\n";
        return 2;
      }
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
                  ? wickqc::method::SpatialMpGenerator(order).GenerateNumpy(
                        argc == 4)
                  : wickqc::method::SpatialCcGenerator(order).GenerateNumpy(
                        argc == 4));
      return 0;
    }
    if (method == "sc-nevpt2") {
      if (argc > 3 ||
          (argc == 3 && std::string_view(argv[2]) != "--optimize")) {
        std::cerr << "SC-NEVPT2 accepts only optional --optimize\n";
        return 2;
      }
      std::cout << wickqc::method::ScNevpt2Generator().GenerateNumpy(argc == 3);
      return 0;
    }
    if (argc != 2) {
      std::cerr << "This method accepts no additional arguments\n";
      return 2;
    }
    if (method == "ghf") {
      std::cout << wickqc::method::GhfGenerator().GenerateNumpy();
    } else if (method == "ccsd") {
      std::cout << wickqc::method::CcsdGenerator().GenerateNumpy();
    } else if (method == "uga-ccsd") {
      std::cout << wickqc::method::UgaCcsdGenerator().GenerateNumpy();
    } else if (method == "ic-nevpt2") {
      std::cout << wickqc::method::IcNevpt2Generator().GenerateNumpy();
    } else {
      std::cerr << "Unknown method: " << method << '\n';
      return 2;
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
