#include <ao2mo.hpp>

#include <iostream>

#include "ao2mo/wick_adapter.hpp"
#include "method/spin_orbital.hpp"
#include "runtime/executor.hpp"

// Use the existing spin-orbital energy generator and numeric executor. This
// energy expression is valid for complex integrals; the current generator's
// general singles/doubles equations have a separate symmetry limitation.
int main(int argc, char** argv) {
  if (argc != 4) {
    return 2;
  }
  try {
    using T = ao2mo::Complex;
    using Array = wickqc::NDArray<T>;
    auto input = ao2mo::ReadInput(argv[1]);
    const auto& data = std::get<ao2mo::Input<T>>(input);
    const auto& p = data.partition;
    const auto ni = p.ncore, ne = p.nmo - ni;
    if (p.ncas || p.frozen || !ni || !ne) {
      throw std::invalid_argument(
          "spinor MP2 requires occupied/virtual partition");
    }
    ao2mo::Options options;
    const std::string workspace = argv[3];
    if (workspace != "incore" && workspace != "outcore") {
      throw std::invalid_argument("expected incore or outcore");
    }
    options.workspace = workspace == "incore" ? ao2mo::Workspace::kIncore
                                              : ao2mo::Workspace::kOutcore;
    options.scratch_directory = std::filesystem::path(argv[2]).parent_path();
    if (options.scratch_directory.empty()) {
      options.scratch_directory = ".";
    }
    ao2mo::Request<T> request;
    request.name = "cvcv";
    for (int axis = 0; axis < 4; ++axis) {
      request.indices[axis] = {
          data.coefficients.begin()->second, p.Indices(axis % 2 ? 'v' : 'c')};
    }
    const auto result = std::make_shared<const ao2mo::Result<T>>(
        ao2mo::Transform(data.basis, std::vector{request}, options));
    const auto coulomb = ao2mo::ToWick(
        ao2mo::GetBlock(result, p, "IIEE", ao2mo::Ordering::kPhysicist),
        options.memory_bytes);
    auto source = ao2mo::h5::OpenFile(argv[1], H5F_ACC_RDONLY);
    const auto energies =
        ao2mo::h5::Read<double>(source, "validation/orbital_energies");
    const auto fock = ao2mo::h5::Read<T>(source, "validation/fock");
    if (energies.size() != p.nmo ||
        fock.size() != ao2mo::CheckedProduct({p.nmo, p.nmo})) {
      throw std::invalid_argument("invalid orbital energies or Fock matrix");
    }
    Array h({ni, ne}), v({ni, ni, ne, ne}), t2({ne, ne, ni, ni});
    for (std::size_t i = 0; i < ni; ++i) {
      for (std::size_t a = 0; a < ne; ++a) {
        h.At({i, a}) = fock[i * p.nmo + ni + a];
      }
    }
    // Antisymmetrization and first-order amplitudes belong to this Wick client,
    // not the AO2MO backend. Keep the complex phase in both factors.
    for (std::size_t i = 0; i < ni; ++i) {
      for (std::size_t j = 0; j < ni; ++j) {
        for (std::size_t a = 0; a < ne; ++a) {
          for (std::size_t b = 0; b < ne; ++b) {
            const auto gap =
                energies[i] + energies[j] - energies[ni + a] - energies[ni + b];
            if (!std::isfinite(gap) || gap >= -1e-10) {
              throw std::runtime_error(
                  "nonnegative or non-finite MP denominator");
            }
            v.At({i, j, a, b}) =
                coulomb.At({i, j, a, b}) - coulomb.At({i, j, b, a});
            t2.At({a, b, i, j}) = std::conj(v.At({i, j, a, b})) / gap;
          }
        }
      }
    }
    wickqc::method::CCSDGenerator generator;
    wickqc::symbolic::IndexRegistry indices;
    wickqc::symbolic::SymmetryRegistry symmetries;
    wickqc::equation::ContractionGraph graph;
    graph.Add(
        wickqc::symbolic::Tensor::Parse("energy[]", indices, symmetries),
        generator.Energy());
    const auto evaluator = wickqc::runtime::NDArrayExecutor::Compile(graph);
    const wickqc::runtime::Dimensions dimensions{{{1, 0}, ni}, {{8, 0}, ne}};
    auto evaluate = [&](const Array& t1, const Array& doubles) {
      const wickqc::runtime::TensorMap<T> inputs{
          {"hIE", h}, {"vIIEE", v}, {"tEI", t1}, {"tEEII", doubles}};
      return evaluator.Evaluate(inputs, dimensions).at("energy").Item();
    };
    const Array mp2_energy({}, {evaluate(Array({ne, ni}), t2)});
    const auto t1_trial = ao2mo::h5::Read<T>(source, "validation/tEI");
    const auto t2_trial = ao2mo::h5::Read<T>(source, "validation/tEEII");
    const Array trial_energy(
        {},
        {evaluate(
            Array({ne, ni}, t1_trial), Array({ne, ne, ni, ni}, t2_trial))});
    ao2mo::transform_detail::OwnedFile output(argv[2], false, 0);
    auto save = [&](const char* name,
                    const Array& array,
                    std::array<std::size_t, 2> shape) {
      auto dataset = ao2mo::h5::Matrix<T>(output.get(), name, shape);
      ao2mo::h5::Slab(
          dataset, true, {0, 0}, shape, const_cast<T*>(array.data()));
      dataset.Close();
    };
    save("coulomb_IIEE", coulomb, {ni * ni, ne * ne});
    save("vIIEE", v, {ni * ni, ne * ne});
    save("tEEII", t2, {ne * ne, ni * ni});
    save("mp2_energy", mp2_energy, {1, 1});
    save("trial_energy", trial_energy, {1, 1});
    output.Finish();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
