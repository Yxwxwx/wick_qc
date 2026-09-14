#include "method/ic_nevpt2.h"
#include "method/sc_nevpt2.h"

#include <exception>
#include <iostream>
#include <string_view>

// Spin-free spatial NEVPT2 with inactive, active, and external orbitals.
// SC emits norm and effective-Hamiltonian contractions and energy reduction
// for all eight outer subspaces. IC emits RHS vectors, effective Hamiltonian
// matrices, orbital restrictions, and linear solves for all thirteen blocks,
// including the coupled nonorthogonal subspace. Both produce complete compute
// functions using integral, orbital-energy, and active-space RDM inputs.
int main(int argc, char** argv) {
  constexpr std::string_view kUsage =
      "Usage: test_nevpt2_methods sc [--optimize]\n"
      "       test_nevpt2_methods ic\n"
      "Write all SC- or IC-NEVPT2 NumPy compute functions to stdout.\n";
  if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
    std::cout << kUsage;
    return 0;
  }
  const std::string_view method = argv[1];
  if (argc > 3 || (method != "sc" && method != "ic") ||
      (argc == 3 &&
       (method != "sc" || std::string_view(argv[2]) != "--optimize"))) {
    std::cerr << kUsage;
    return 2;
  }
  try {
    if (method == "sc") {
      std::cout << wickqc::method::ScNevpt2Generator().GenerateNumpy(argc == 3);
    } else {
      std::cout << wickqc::method::IcNevpt2Generator().GenerateNumpy();
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
