#include "method/spatial_cc.h"

#include <exception>
#include <iostream>
#include <string_view>

// Closed-shell spin-free CC with T = T1 + ... + Tn and pair-symmetric
// spatial amplitudes. Each Tr carries 1/r! and r excitation operators E1.
// The two-body Hamiltonian's BCH expansion terminates at four commutators.
// Projecting exp(-T) H_N exp(T) onto <0| and the ordered E1 products gives
// the correlation energy and all residual ranks through n.
int main(int argc, char** argv) {
  constexpr std::string_view kUsage =
      "Usage: test_spatial_cc {2|3|4} [--optimize] [--chemist]\n"
      "Ranks: 2=CCSD, 3=CCSDT, 4=CCSDTQ.\n"
      "Write the energy and all residual NumPy equations to stdout.\n";
  if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
    std::cout << kUsage;
    return 0;
  }
  const std::string_view rank = argv[1];
  if (argc > 4 || (rank != "2" && rank != "3" && rank != "4")) {
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
    const wickqc::method::SpatialCcGenerator method(
        rank.front() - '0', convention);
    std::cout << method.GenerateNumpy(optimize);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
