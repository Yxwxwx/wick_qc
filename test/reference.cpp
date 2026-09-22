#include "ao2mo/reference.hpp"

#include <iostream>

namespace {
void Write(
    hid_t file,
    const std::string& name,
    const wickqc::NDArray<double>& array) {
  std::vector<hsize_t> shape(array.shape().begin(), array.shape().end());
  ao2mo::h5::Handle space(
      H5Screate_simple(array.Rank(), shape.data(), nullptr), H5Sclose);
  ao2mo::h5::Handle dataset(
      H5Dcreate2(
          file,
          name.c_str(),
          H5T_IEEE_F64LE,
          space,
          H5P_DEFAULT,
          H5P_DEFAULT,
          H5P_DEFAULT),
      H5Dclose);
  const auto contiguous = array.ToCOrder();
  if (array.Size()) {
    ao2mo::h5::Check(H5Dwrite(
        dataset,
        H5T_NATIVE_DOUBLE,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        contiguous.data()));
  }
}
} // namespace

wickqc::runtime::TensorBinding Binding(
    const std::string& name,
    const std::string& axes) {
  wickqc::runtime::TensorBinding binding{name, {}};
  for (auto axis : axes) {
    binding.domains.push_back(
        {static_cast<std::uint8_t>(
             axis == 'I'       ? 1
                 : axis == 'A' ? 2
                               : 8),
         0});
  }
  return binding;
}

void CheckBindings(
    hid_t file,
    const ao2mo::ReferenceInput& input,
    const ao2mo::Options& options) {
  namespace runtime = wickqc::runtime;
  using wickqc::method::IntegralConvention;
  using wickqc::method::NEVPT2Method;
  using wickqc::method::ReferenceKind;
  using wickqc::method::SpatialFamily;
  std::vector<runtime::TensorBinding> bindings;
  if (input.reference.kind == ReferenceKind::kRHF) {
    for (const std::string axes : {"IEEI", "EEII", "EIEI"}) {
      bindings.push_back(Binding("v" + axes, axes));
    }
    for (const std::string axes : {"II", "IE", "EE"}) {
      bindings.push_back(Binding("f" + axes, axes));
    }
    bindings.push_back(Binding("epsI", "I"));
    bindings.push_back(Binding("epsE", "E"));
    for (auto convention :
         {IntegralConvention::kChemist, IntegralConvention::kPhysicist}) {
      auto prepared = ao2mo::PrepareRHF(
          input, bindings, {SpatialFamily::kCC, 2, convention}, options);
      if (prepared.chemist_integrals.Rank() != 0 ||
          prepared.integral_blocks.size() != 3) {
        throw std::runtime_error(
            "RHF preparation allocated unrequested integrals");
      }
      auto bound = prepared.Bind(bindings, {}, convention);
      prepared = {}; // Bound integral and Fock views must retain their storage.
      for (const auto& [name, array] : bound) {
        Write(
            file,
            (convention == IntegralConvention::kChemist ? "chem_" : "phys_") +
                name,
            array);
      }
    }
    return;
  }
  for (const std::string axes :
       {"AAAA",
        "EAAA",
        "EAIA",
        "EAAI",
        "AAIA",
        "EEIA",
        "EAII",
        "EEAA",
        "AAII",
        "EEII",
        "AAAI",
        "AEAI",
        "EEAI"}) {
    bindings.push_back(Binding("w" + axes, axes));
  }
  for (const std::string axes : {"AA", "AI", "EI", "EA"}) {
    bindings.push_back(Binding("h" + axes, axes));
  }
  bindings.push_back(Binding("orbeI", "I"));
  bindings.push_back(Binding("orbeE", "E"));
  for (std::size_t order = 1; order <= 4; ++order) {
    bindings.push_back(
        Binding("E" + std::to_string(order), std::string(2 * order, 'A')));
  }
  auto bound =
      ao2mo::PrepareNEVPT2(input, bindings, NEVPT2Method::kSC, options);
  for (const auto& [name, array] : bound) {
    Write(file, "sc_" + name, array);
  }
  for (const std::string axes : {"AA", "AI", "EI", "EA"}) {
    bindings.push_back(Binding("f" + axes, axes));
  }
  const auto ic =
      ao2mo::PrepareNEVPT2(input, bindings, NEVPT2Method::kIC, options);
  for (const auto& [name, array] : ic) {
    if (name.starts_with("f")) {
      if (!array.AllClose(bound.at("h" + name.substr(1)), 1e-12, 1e-12)) {
        throw std::runtime_error("IC f must bind to h1eff");
      }
    } else if (!array.AllClose(bound.at(name), 1e-12, 1e-12)) {
      throw std::runtime_error("SC/IC input binding mismatch");
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    return 2;
  }
  try {
    auto input = ao2mo::ReadReference(argv[1], 64ULL << 20);
    bool budget_rejected = false;
    try {
      (void)ao2mo::ReadReference(argv[1], 1);
    } catch (const std::runtime_error&) {
      budget_rejected = true;
    }
    if (!budget_rejected) {
      throw std::runtime_error("Reference read ignored its payload budget");
    }
    const auto& r = input.reference;
    // The public in-memory input follows the same validation contract.
    auto memory = input;
    memory.Validate();
    auto energies = memory.reference.orbital_energies;
    memory = {};
    if (!energies.AllClose(r.orbital_energies, 0, 0)) {
      throw std::runtime_error("Reference lost its NDArray storage owner");
    }
    ao2mo::h5::Handle file(
        H5Fcreate(argv[2], H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    Write(file, "orbital_energies", r.orbital_energies);
    Write(file, "occupations", r.occupations);
    Write(file, "fock_mo", r.fock_mo);
    ao2mo::h5::DoubleAttribute(file, "reference_energy", r.reference_energy);
    ao2mo::h5::SizeAttribute(file, "root", r.root);
    ao2mo::h5::SizeAttribute(
        file, "loading_budget_bytes", input.loading_budget_bytes);
    if (r.kind == wickqc::method::ReferenceKind::kCASSCF) {
      for (std::size_t i = 0; i < r.rdms.size(); ++i) {
        Write(file, "E" + std::to_string(i + 1), r.rdms[i]);
      }
    }
    ao2mo::Options options;
    options.memory_bytes = 64ULL << 20;
    options.workspace =
        argc == 4 ? ao2mo::Workspace::kOutcore : ao2mo::Workspace::kIncore;
    options.scratch_directory = std::filesystem::path(argv[2]).parent_path();
    CheckBindings(file, input, options);
    file.Close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
