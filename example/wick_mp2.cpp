#include <ao2mo.hpp>

#include <iomanip>
#include <iostream>

#include "ao2mo/hdf5_store.hpp"
#include "ao2mo/wick_adapter.hpp"
#include "method/rhf.hpp"
#include "runtime/spatial.hpp"

// A client of the existing wick_qc MP generator; no post-HF equations are
// implemented in the AO2MO library. The amplitudes are the usual first-order
// doubles supplied to wick_qc's existing MP2 evaluator.
int main(int argc, char** argv) {
  if (argc != 2) {
    return 2;
  }
  try {
    auto input = ao2mo::ReadInput(argv[1]);
    const auto& data = std::get<ao2mo::Input<double>>(input);
    const auto n = data.partition.nmo, occupied = data.partition.ncore;
    if (data.partition.ncas || !occupied || occupied >= n) {
      throw std::invalid_argument(
          "MP2 example requires occupied/virtual partition");
    }
    const auto c = data.coefficients.begin()->second;
    ao2mo::Request<double> request;
    request.name = "pppp";
    for (auto& index : request.indices) {
      index = {c, data.partition.Indices('p')};
    }
    auto result = std::make_shared<const ao2mo::Result<double>>(
        ao2mo::Transform(data.basis, std::vector{request}, ao2mo::Options{}));
    auto integrals = ao2mo::ToWick(
        ao2mo::GetBlock(
            result, data.partition, "pppp", ao2mo::Ordering::kChemist),
        512ULL << 20);
    auto file = ao2mo::h5::OpenFile(argv[1], H5F_ACC_RDONLY);
    const auto energies =
        ao2mo::h5::Read<double>(file, "validation/orbital_energies");
    const auto fock = ao2mo::h5::Read<double>(file, "validation/fock");
    using Array = wickqc::NDArray<double>;
    wickqc::method::RHFData rhf{
        occupied, Array({n}, energies), Array({n, n}, fock), integrals};
    const auto virtuals = n - occupied;
    Array t2({virtuals, virtuals, occupied, occupied});
    for (std::size_t a = 0; a < virtuals; ++a) {
      for (std::size_t b = 0; b < virtuals; ++b) {
        for (std::size_t i = 0; i < occupied; ++i) {
          for (std::size_t j = 0; j < occupied; ++j) {
            const auto gap = energies[i] + energies[j] -
                energies[occupied + a] - energies[occupied + b];
            if (gap >= -1e-10) {
              throw std::runtime_error("nonnegative MP denominator");
            }
            t2.At({a, b, i, j}) =
                integrals.At({occupied + a, i, occupied + b, j}) / gap;
          }
        }
      }
    }
    using wickqc::method::IntegralConvention;
    using wickqc::method::SpatialFamily;
    const wickqc::runtime::SpatialEvaluator evaluator(
        {SpatialFamily::kMP, 2, IntegralConvention::kChemist},
        wickqc::runtime::GenerationPolicy::kRuntimeOnly);
    const auto outputs = evaluator.Evaluate(
        rhf.Bind(
            evaluator.Inputs(),
            {{"u1EI", Array({virtuals, occupied})}, {"u1EEII", t2}}),
        rhf.Dimensions());
    std::cout << std::setprecision(17) << outputs.at("energy2").Item() << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
