#include "method/spatial_cc.h"
#include "method/spatial_mp.h"
#include "runtime/ndarray_executor.h"

#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using Array = wickqc::NDArray<double>;
using wickqc::method::IntegralConvention;
using wickqc::runtime::Dimensions;
using wickqc::runtime::NdArrayExecutor;
using wickqc::runtime::TensorMap;
namespace fs = std::filesystem;

// The accompanying Python example writes little-endian IEEE float64 arrays,
// in C order. This small example format is not a general tensor checkpoint API.
Array Read(const fs::path& path, const Array::Shape& shape) {
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
  std::size_t bytes = sizeof(double);
  for (const auto extent : shape) {
    if (extent == 0 ||
        extent > std::numeric_limits<std::size_t>::max() / bytes) {
      throw std::invalid_argument("Invalid input shape for " + path.string());
    }
    bytes *= extent;
  }
  if (fs::file_size(path) != bytes ||
      bytes > static_cast<std::size_t>(
                  std::numeric_limits<std::streamsize>::max())) {
    throw std::invalid_argument("Wrong input size for " + path.string());
  }
  Array array(shape);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(
          reinterpret_cast<char*>(array.data()),
          static_cast<std::streamsize>(bytes))) {
    throw std::runtime_error("Cannot read " + path.string());
  }
  return array;
}

void Write(const fs::path& path, const Array& array) {
  const auto contiguous = array.ToCOrder();
  std::ofstream output(path, std::ios::binary);
  output.write(
      reinterpret_cast<const char*>(contiguous.data()),
      static_cast<std::streamsize>(contiguous.Size() * sizeof(double)));
  if (!output) {
    throw std::runtime_error("Cannot write " + path.string());
  }
}

Array Block(
    const Array& full,
    const wickqc::runtime::TensorBinding& binding,
    std::size_t nocc,
    std::size_t nmo) {
  std::vector<wickqc::NDArraySlice> slices;
  for (const auto domain : binding.domains) {
    if (domain.spins != 0 ||
        (domain.orbital_spaces != 1 && domain.orbital_spaces != 8)) {
      throw std::invalid_argument(
          "RHF example requires inactive/external spin-free axes");
    }
    const bool occupied = domain.orbital_spaces == 1;
    slices.push_back(
        wickqc::NDArraySlice::Range(
            occupied ? 0 : nocc, occupied ? nocc : nmo));
  }
  return full.Slice(slices);
}

TensorMap<double> Bind(
    const NdArrayExecutor& executor,
    const Array& eps,
    const Array& fock,
    const Array& eri,
    const Array& t1,
    const Array& t2,
    std::size_t nocc) {
  TensorMap<double> result;
  for (const auto& binding : executor.Inputs()) {
    if (binding.name == "tEI" || binding.name == "u1EI") {
      result.emplace(binding.name, t1);
    } else if (binding.name == "tEEII" || binding.name == "u1EEII") {
      result.emplace(binding.name, t2);
    } else {
      const Array* source = nullptr;
      if (binding.name.starts_with("eps")) {
        source = &eps;
      } else if (binding.name.starts_with("f")) {
        source = &fock;
      } else if (binding.name.starts_with("v")) {
        source = &eri;
      } else {
        throw std::invalid_argument("Unknown RHF input '" + binding.name + "'");
      }
      result.emplace(binding.name, Block(*source, binding, nocc, eps.Size()));
    }
  }
  return result;
}

double Gap(const Array& eps, std::size_t nocc, std::size_t a, std::size_t i) {
  const double value = eps.At({i}) - eps.At({nocc + a});
  if (!std::isfinite(value) || value >= -1e-10) {
    throw std::invalid_argument(
        "Expected finite, negative occupied-virtual energy gaps");
  }
  return value;
}

void Evaluate(
    const fs::path& input,
    const fs::path& output,
    IntegralConvention convention) {
  std::size_t nocc = 0, nvir = 0;
  std::ifstream dimensions_file(input / "dimensions.txt");
  if (!(dimensions_file >> nocc >> nvir) || nocc == 0 || nvir == 0 ||
      nocc > std::numeric_limits<std::size_t>::max() - nvir) {
    throw std::invalid_argument("Invalid occupied/virtual dimensions");
  }
  const auto nmo = nocc + nvir;
  const Dimensions dimensions{{{1, 0}, nocc}, {{8, 0}, nvir}};
  const auto eps_mp = Read(input / "eps_mp.bin", {nmo});
  const auto eps_cc = Read(input / "eps_cc.bin", {nmo});
  const auto fock = Read(input / "fock.bin", {nmo, nmo});
  const auto chemist = Read(input / "eri_chemist.bin", {nmo, nmo, nmo, nmo});
  const auto eri = convention == IntegralConvention::kChemist
      ? chemist
      : chemist.TransposeView({0, 2, 1, 3});
  const Array zero_singles({nvir, nocc});
  Array mp2_doubles({nvir, nvir, nocc, nocc});
  for (std::size_t a = 0; a < nvir; ++a) {
    for (std::size_t b = 0; b < nvir; ++b) {
      for (std::size_t i = 0; i < nocc; ++i) {
        for (std::size_t j = 0; j < nocc; ++j) {
          mp2_doubles.At({a, b, i, j}) =
              chemist.At({nocc + a, i, nocc + b, j}) /
              (Gap(eps_mp, nocc, a, i) + Gap(eps_mp, nocc, b, j));
        }
      }
    }
  }
  const auto mp = NdArrayExecutor::Compile(
      wickqc::method::SpatialMpGenerator(2, convention).Equations().Simplify());
  const auto mp_result = mp.Evaluate(
      Bind(mp, eps_mp, fock, eri, zero_singles, mp2_doubles, nocc), dimensions);

  // Initial amplitudes are exported once from PySCF, then supplied unchanged
  // to both implementations. All contractions and the update below run in C++.
  const auto t1 = Read(input / "cc_t1_initial.bin", {nvir, nocc});
  const auto t2 = Read(input / "cc_t2_initial.bin", {nvir, nvir, nocc, nocc});
  const auto cc_graph =
      wickqc::method::SpatialCcGenerator(2, convention).Equations();
  const auto cc = NdArrayExecutor::Compile(cc_graph.Simplify());
  const auto initial =
      cc.Evaluate(Bind(cc, eps_cc, fock, eri, t1, t2, nocc), dimensions);
  Array next1 = t1.Clone(), next2 = t2.Clone();
  const auto& r1 = initial.at("residual1");
  const auto& r2 = initial.at("residual2");
  // The E1 projectors give covariant residuals: r1 = 2 R1 and
  // r2(abij) = 4 R2(abij) - 2 R2(abji). Inverting that spin metric gives
  // R2 = r2/3 + swap_ij(r2)/6. Jacobi uses t_new = t + R/(eps_occ-eps_vir).
  for (std::size_t a = 0; a < nvir; ++a) {
    for (std::size_t i = 0; i < nocc; ++i) {
      next1.At({a, i}) += r1.At({a, i}) / (2 * Gap(eps_cc, nocc, a, i));
      for (std::size_t b = 0; b < nvir; ++b) {
        for (std::size_t j = 0; j < nocc; ++j) {
          const double residual =
              r2.At({a, b, i, j}) / 3 + r2.At({a, b, j, i}) / 6;
          next2.At({a, b, i, j}) +=
              residual / (Gap(eps_cc, nocc, a, i) + Gap(eps_cc, nocc, b, j));
        }
      }
    }
  }
  const auto energy = NdArrayExecutor::Compile(
      wickqc::equation::ContractionGraph({cc_graph.Nodes().front()})
          .Simplify());
  const auto first = energy.Evaluate(
      Bind(energy, eps_cc, fock, eri, next1, next2, nocc), dimensions);
  fs::create_directories(output);
  Write(output / "mp2_t2.bin", mp2_doubles);
  Write(output / "cc_t1_first.bin", next1);
  Write(output / "cc_t2_first.bin", next2);
  Write(output / "cc_r1_initial.bin", r1);
  Write(output / "cc_r2_initial.bin", r2);
  std::ofstream report(output / "energies.json");
  report << std::setprecision(17)
         << "{\"mp2_correlation\":" << mp_result.at("energy2").Item()
         << ",\"mp2_residual1_norm\":" << mp_result.at("residual1_rank1").Norm()
         << ",\"mp2_residual2_norm\":" << mp_result.at("residual1_rank2").Norm()
         << ",\"cc_initial_correlation\":" << initial.at("energy").Item()
         << ",\"cc_first_correlation\":" << first.at("energy").Item() << "}\n";
  if (!report) {
    throw std::runtime_error("Cannot write energy report");
  }
  std::cout << output.filename().string()
            << ": E(MP2) = " << std::setprecision(15)
            << mp_result.at("energy2").Item()
            << ", E(CCSD step 1) = " << first.at("energy").Item() << '\n';
}
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "Usage: test_h2o_pyscf INPUT_DIRECTORY OUTPUT_DIRECTORY");
    }
    Evaluate(
        argv[1], fs::path(argv[2]) / "chemist", IntegralConvention::kChemist);
    Evaluate(
        argv[1],
        fs::path(argv[2]) / "physicist",
        IntegralConvention::kPhysicist);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
