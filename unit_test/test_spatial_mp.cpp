#include "method/spatial_mp.h"

#include <exception>
#include <iostream>
#include <string_view>

// Canonical RHF, spin-free spatial MP2--MP4 with intermediate normalization.
// The generator builds F_N, V = H_N - F_N, and the projected amplitude
// equations for |1> (singles/doubles) and |2> (through quadruples).
// E2 = <0|V|1>, E3 = <1|V|1>, E4 = -<2|F_N|2> - E2 <1|1>.
// Output contains every energy and residual through the requested order;
// amplitudes are inputs to these equations and must be solved separately.
int main(int argc, char** argv) {
  constexpr std::string_view kUsage =
      "Usage: test_spatial_mp {2|3|4} [--optimize] [--chemist]\n"
      "Write the complete MP2, MP3, or MP4 NumPy equations to stdout.\n";
  if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
    std::cout << kUsage;
    return 0;
  }
  const std::string_view order = argv[1];
  if (argc > 4 || (order != "2" && order != "3" && order != "4")) {
    std::cerr << kUsage;
    return 2;
  }
  bool optimize = false, chemist = false;
  for (int i = 2; i < argc; ++i) {
    const std::string_view option = argv[i];
    if (option == "--optimize" && !optimize) {
      optimize = true;
    } else if (option == "--chemist" && !chemist) {
      chemist = true;
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  try {
    const auto convention = chemist
        ? wickqc::method::IntegralConvention::kChemist
        : wickqc::method::IntegralConvention::kPhysicist;
    const wickqc::method::SpatialMpGenerator method(
        order.front() - '0', convention);
    std::cout << method.GenerateNumpy(optimize);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
