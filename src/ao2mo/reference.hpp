#pragma once

#include "ao2mo/input.hpp"
#include "ao2mo/wick_adapter.hpp"
#include "method/reference.hpp"
#include "method/rhf.hpp"

namespace ao2mo {
// Optional integration header: the symbolic/numeric Wick umbrella does not
// acquire libcint or HDF5 dependencies. The same value type accepts memory
// data.
struct ReferenceInput {
  Input<double> integrals;
  wickqc::method::SpatialReference reference;
  std::string coefficient_family = "mo";
  std::size_t loading_budget_bytes = 0;

  void Validate() const {
    reference.Validate();
    const auto& p = integrals.partition;
    if (p.nmo != reference.nmo || p.ncore != reference.ncore ||
        p.ncas != reference.nactive || p.frozen != reference.frozen) {
      throw std::invalid_argument("AO2MO/reference partition mismatch");
    }
    const auto it = integrals.coefficients.find(coefficient_family);
    if (it == integrals.coefficients.end() || !it->second ||
        it->second->nmo != p.nmo ||
        it->second->source_identity != reference.orbital_identity) {
      throw std::invalid_argument(
          "AO2MO/reference coefficient identity mismatch");
    }
    integrals.dressing.Validate(p);
    if (integrals.dressing.orbital_ids != reference.orbital_ids) {
      throw std::invalid_argument("AO2MO/reference orbital order mismatch");
    }
    helper_detail::ValidateH1<double>(integrals.h1e, p.nmo);
  }
};

namespace reference_detail {
inline void Require(hid_t group, const char* name) {
  if (H5Lexists(group, name, H5P_DEFAULT) <= 0) {
    throw std::invalid_argument(
        std::string("Missing reference dataset: ") + name);
  }
}

// Read directly into owned NDArray storage, including E4. No intermediate
// vector doubles the largest RDM's payload. Refuse complex/float32 conversion.
inline wickqc::NDArray<double> Tensor(
    hid_t group,
    const char* name,
    const wickqc::NDArray<double>::Shape& shape,
    std::size_t& budget) {
  Require(group, name);
  auto dataset = h5::OpenDataset(group, name);
  h5::Handle type(H5Dget_type(dataset), H5Tclose);
  if (H5Tget_class(type) != H5T_FLOAT || H5Tget_size(type) != sizeof(double)) {
    throw std::invalid_argument(
        std::string("Reference requires float64: ") + name);
  }
  if (h5::Shape(dataset) != std::vector<hsize_t>(shape.begin(), shape.end())) {
    throw std::invalid_argument(
        std::string("Reference shape mismatch: ") + name);
  }
  std::size_t bytes = sizeof(double);
  for (auto extent : shape) {
    bytes = CheckedProduct({bytes, extent});
  }
  input_detail::Charge(budget, bytes);
  wickqc::NDArray<double> result(shape);
  if (result.Size()) {
    h5::Check(H5Dread(
        dataset,
        H5T_NATIVE_DOUBLE,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        result.data()));
  }
  return result;
}

inline void Expect(
    hid_t group,
    const char* name,
    const std::string& expected,
    std::size_t& budget) {
  if (H5Aexists(group, name) <= 0 ||
      input_detail::Attribute(group, name, budget) != expected) {
    throw std::invalid_argument(
        std::string("Reference metadata mismatch: ") + name);
  }
}
} // namespace reference_detail

inline ReferenceInput ReadReference(
    const std::filesystem::path& path,
    std::size_t memory_bytes = 512ULL << 20) {
  std::lock_guard lock(h5::ExecutionMutex());
  auto input = ReadInput(path, memory_bytes);
  if (!std::holds_alternative<Input<double>>(input)) {
    throw std::invalid_argument(
        "Methods currently require real spatial orbitals");
  }
  ReferenceInput result;
  result.integrals = std::move(std::get<Input<double>>(input));
  auto budget = memory_bytes - result.integrals.loading_budget_bytes;
  auto file = h5::OpenFile(path, H5F_ACC_RDONLY);
  reference_detail::Require(file, "reference");
  h5::Handle group(H5Gopen2(file, "reference", H5P_DEFAULT), H5Gclose);
  using input_detail::Attribute;
  using input_detail::Count;
  using reference_detail::Expect;
  Expect(group, "schema", "wickqc.reference.v1", budget);
  Expect(group, "representation", "real_spatial", budget);
  Expect(
      group, "energy_convention", "total_including_nuclear_and_core", budget);
  if (Count(group, "complete") != 1) {
    throw std::invalid_argument("Incomplete reference export");
  }
  auto& r = result.reference;
  const auto kind = Attribute(group, "kind", budget);
  if (kind != "rhf" && kind != "casscf") {
    throw std::invalid_argument("Unsupported reference kind: " + kind);
  }
  const bool rhf = kind == "rhf";
  r.kind = rhf ? wickqc::method::ReferenceKind::kRHF
               : wickqc::method::ReferenceKind::kCASSCF;
  Expect(
      group,
      "orbital_state",
      rhf ? "canonical" : "core_virtual_canonical",
      budget);
  r.energy_source = Attribute(group, "energy_source", budget);
  r.fock_source = Attribute(group, "fock_source", budget);
  if (r.fock_source.empty()) {
    throw std::invalid_argument("Missing Fock source");
  }
  r.nmo = Count(group, "nmo");
  r.ncore = Count(group, "ncore");
  r.nactive = Count(group, "nactive");
  r.frozen = Count(group, "frozen");
  r.root = Count(group, "root");
  r.root_count = Count(group, "root_count");
  r.electrons = {Count(group, "nalpha"), Count(group, "nbeta")};
  r.reference_energy = input_detail::RealAttribute(group, "reference_energy");
  r.orbital_identity = Attribute(group, "orbital_identity", budget);
  r.orbital_ids = input_detail::Indices(group, "orbital_ids", budget);
  result.coefficient_family = Attribute(group, "coefficient_family", budget);
  const auto found =
      result.integrals.coefficients.find(result.coefficient_family);
  if (found == result.integrals.coefficients.end()) {
    throw std::invalid_argument("Missing reference coefficient family");
  }
  Expect(
      group, "basis_fingerprint", Fingerprint(result.integrals.basis), budget);
  Expect(group, "coefficient_fingerprint", Fingerprint(*found->second), budget);
  for (const char* name : {"orbital_energies", "occupations", "fock_mo"}) {
    reference_detail::Require(group, name);
    auto dataset = h5::OpenDataset(group, name);
    Expect(dataset, "orbital_identity", r.orbital_identity, budget);
  }
  auto h1e = h5::OpenDataset(file, "one_electron/h1e_mo");
  Expect(h1e, "orbital_identity", r.orbital_identity, budget);
  r.orbital_energies =
      reference_detail::Tensor(group, "orbital_energies", {r.nmo}, budget);
  r.occupations =
      reference_detail::Tensor(group, "occupations", {r.nmo}, budget);
  r.fock_mo =
      reference_detail::Tensor(group, "fock_mo", {r.nmo, r.nmo}, budget);
  if (!rhf) {
    reference_detail::Require(group, "rdms");
    h5::Handle rdms(H5Gopen2(group, "rdms", H5P_DEFAULT), H5Gclose);
    Expect(rdms, "convention", "block2.spin_free.normal_ordered.v1", budget);
    Expect(rdms, "orbital_identity", r.orbital_identity, budget);
    if (Count(rdms, "root") != r.root) {
      throw std::invalid_argument("RDM/reference root mismatch");
    }
    const auto active =
        input_detail::Indices(rdms, "active_orbital_ids", budget);
    if (r.ncore > r.orbital_ids.size() ||
        r.nactive > r.orbital_ids.size() - r.ncore ||
        !std::ranges::equal(
            active, std::span(r.orbital_ids).subspan(r.ncore, r.nactive))) {
      throw std::invalid_argument(
          "RDM/reference active orbital order mismatch");
    }
    for (std::size_t i = 0; i < r.rdms.size(); ++i) {
      r.rdms[i] = reference_detail::Tensor(
          rdms,
          ("E" + std::to_string(i + 1)).c_str(),
          wickqc::NDArray<double>::Shape(2 * (i + 1), r.nactive),
          budget);
    }
  }
  result.Validate();
  result.loading_budget_bytes = memory_bytes - budget;
  return result;
}

namespace reference_detail {
inline std::string Axes(const wickqc::runtime::TensorBinding& binding) {
  std::string axes;
  for (const auto domain : binding.domains) {
    if (domain.spins != 0 ||
        (domain.orbital_spaces != 1 && domain.orbital_spaces != 2 &&
         domain.orbital_spaces != 8)) {
      throw std::invalid_argument(
          "Unsupported spatial axes for '" + binding.name + "'");
    }
    axes += domain.orbital_spaces == 1 ? 'I'
        : domain.orbital_spaces == 2   ? 'A'
                                       : 'E';
  }
  return axes;
}

inline wickqc::NDArray<double> Slice(
    const wickqc::NDArray<double>& tensor,
    const Partition& p,
    const std::string& axes,
    bool retained = false) {
  std::vector<wickqc::NDArraySlice> slices;
  for (auto axis : axes) {
    const auto begin = axis == 'I' ? p.frozen
        : axis == 'A'              ? p.ncore
                                   : p.ncore + p.ncas;
    const auto end = axis == 'I' ? p.ncore
        : axis == 'A'            ? p.ncore + p.ncas
                                 : p.nmo;
    const auto offset = retained ? p.frozen : 0;
    slices.push_back(wickqc::NDArraySlice::Range(begin - offset, end - offset));
  }
  return tensor.Slice(slices);
}

inline Options TransformOptions(
    const ReferenceInput& input,
    Options options,
    std::size_t extra = 0) {
  if (options.output != Output::kMemory || options.first_pair_only) {
    throw std::invalid_argument(
        "Method binding requires complete in-memory MO blocks; outcore workspace is supported");
  }
  // Reserve reference tensors while the existing planner accounts for basis,
  // coefficients, AO workspace and directly owned integral outputs.
  const auto resident = CheckedAdd(
      input.reference.TensorBytes(),
      CheckedProduct({input.integrals.h1e.size(), sizeof(double)}));
  input_detail::Charge(options.memory_bytes, CheckedAdd(resident, extra));
  return options;
}

inline std::pair<
    wickqc::runtime::TensorMap<double>,
    std::shared_ptr<const Result<double>>>
TransformArrays(
    const ReferenceInput& input,
    const std::vector<Request<double>>& requests,
    const Options& options) {
  (void)MakePlan(
      input.integrals.basis,
      requests,
      options); // Check before allocating sinks.
  wickqc::runtime::TensorMap<double> arrays;
  std::vector<MemoryBuffer<double>> buffers;
  for (const auto& request : requests) {
    wickqc::NDArray<double>::Shape shape;
    for (const auto& index : request.indices) {
      shape.push_back(index.columns.size());
    }
    if (request.ordering == Ordering::kPhysicist) {
      std::swap(shape[1], shape[2]);
    }
    auto array = std::make_shared<wickqc::NDArray<double>>(std::move(shape));
    if (!arrays.emplace(request.name, *array).second) {
      throw std::invalid_argument(
          "Duplicate integral binding: " + request.name);
    }
    buffers.push_back(WickBuffer(std::move(array)));
  }
  auto result = std::make_shared<const Result<double>>(
      Transform(input.integrals.basis, requests, options, buffers));
  return {std::move(arrays), std::move(result)};
}

inline wickqc::NDArray<double> OwnedView(
    const BlockView<double>& view,
    const wickqc::NDArray<double>& base) {
  if (!base.OwnsData() || !base.IsContiguous() || view.conjugate ||
      view.owner->blocks.at(view.block).Values().data() != base.data()) {
    throw std::logic_error(
        "Integral getter does not refer to its NDArray owner");
  }
  if (std::find(view.shape.begin(), view.shape.end(), 0) != view.shape.end()) {
    return wickqc::NDArray<double>({view.shape.begin(), view.shape.end()});
  }
  const auto start = base.DecomposeLinearIndex(view.offset);
  // Recover the getter's permutation as ordinary Slice/TransposeView calls.
  // Equal strides on singleton axes may admit several equivalent mappings.
  std::vector<int> permutation{0, 1, 2, 3};
  do {
    std::vector<wickqc::NDArraySlice> slices(4);
    bool valid = true;
    for (int axis = 0; axis < 4; ++axis) {
      const auto source = permutation[axis];
      if (view.strides[axis] !=
              static_cast<std::size_t>(base.strides()[source]) ||
          view.shape[axis] > base.shape()[source] - start[source]) {
        valid = false;
        break;
      }
      slices[source] = wickqc::NDArraySlice::Range(
          start[source], start[source] + view.shape[axis]);
    }
    if (valid) {
      return base.Slice(slices).TransposeView(permutation);
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  throw std::logic_error("Integral getter is not a dense slice/permutation");
}
} // namespace reference_detail

// Build only v blocks requested by the method kernel, once before iteration.
// The dense RHFData path remains available; both paths share its Bind contract.
inline wickqc::method::RHFData PrepareRHF(
    const ReferenceInput& input,
    std::span<const wickqc::runtime::TensorBinding> bindings,
    wickqc::method::SpatialMethod method,
    const Options& options = {}) {
  input.Validate();
  using wickqc::method::IntegralConvention;
  if (input.reference.kind != wickqc::method::ReferenceKind::kRHF ||
      (method.family != wickqc::method::SpatialFamily::kMP &&
       method.family != wickqc::method::SpatialFamily::kCC) ||
      (method.convention != IntegralConvention::kChemist &&
       method.convention != IntegralConvention::kPhysicist)) {
    throw std::invalid_argument(
        "MP/CC requires a real closed-shell RHF reference and explicit ERI convention");
  }
  const auto& p = input.integrals.partition;
  const auto coefficients =
      input.integrals.coefficients.at(input.coefficient_family);
  std::vector<Request<double>> requests;
  std::set<std::string> names;
  for (const auto& binding : bindings) {
    auto axes = reference_detail::Axes(binding);
    if (axes.find('A') != std::string::npos) {
      throw std::invalid_argument("MP/CC input cannot have active axes");
    }
    if (axes.size() != 4 || binding.name != "v" + axes ||
        !names.insert(binding.name).second) {
      continue;
    }
    Request<double> request;
    request.name = binding.name;
    if (method.convention == IntegralConvention::kPhysicist) {
      request.ordering = Ordering::kPhysicist;
      std::swap(axes[1], axes[2]);
    }
    for (std::size_t i = 0; i < 4; ++i) {
      request.indices[i] = {coefficients, p.Indices(axes[i])};
    }
    requests.push_back(std::move(request));
  }
  const auto local = reference_detail::TransformOptions(input, options);
  wickqc::method::RHFData result;
  result.occupied = p.ncore - p.frozen;
  result.orbital_energies = input.reference.orbital_energies.Slice(
      {wickqc::NDArraySlice::Range(p.frozen, p.nmo)});
  result.fock = input.reference.fock_mo.Slice(
      {wickqc::NDArraySlice::Range(p.frozen, p.nmo),
       wickqc::NDArraySlice::Range(p.frozen, p.nmo)});
  result.block_convention = method.convention;
  if (!requests.empty()) {
    result.integral_blocks =
        reference_detail::TransformArrays(input, requests, local).first;
  }
  (void)result.Dimensions();
  return result;
}

inline wickqc::runtime::TensorMap<double> PrepareNEVPT2(
    const ReferenceInput& input,
    std::span<const wickqc::runtime::TensorBinding> bindings,
    wickqc::method::NEVPT2Method method,
    const Options& options = {}) {
  input.Validate();
  using wickqc::method::NEVPT2Method;
  if (input.reference.kind != wickqc::method::ReferenceKind::kCASSCF ||
      (method != NEVPT2Method::kSC && method != NEVPT2Method::kIC)) {
    throw std::invalid_argument(
        "SC/IC-NEVPT2 requires a real spin-free CASSCF reference");
  }
  const auto& r = input.reference;
  const auto& p = input.integrals.partition;
  const auto coefficients =
      input.integrals.coefficients.at(input.coefficient_family);
  const auto dimension = r.Dimensions();
  // Our IC generator calls the dressed operator h, while block2's full
  // equations call it f. Both declarations mean h1eff, never the SCF Fock.
  const auto one_electron =
      [method](const std::string& name, const std::string& axes) {
        return name == "h" + axes ||
            (method == NEVPT2Method::kIC && name == "f" + axes);
      };
  // Validate names and domains before integral work; GetBlock checks whether
  // the four base blocks cover each requested ERI permutation.
  for (const auto& binding : bindings) {
    const auto axes = reference_detail::Axes(binding);
    const auto& name = binding.name;
    const bool rdm = axes.size() >= 2 && axes.size() <= 8 &&
        axes.size() % 2 == 0 && axes == std::string(axes.size(), 'A') &&
        name == "E" + std::to_string(axes.size() / 2);
    const bool h = one_electron(name, axes) &&
        (axes == "AA" || axes == "AI" || axes == "EI" || axes == "EA");
    const bool eps = name == "orbe" + axes && (axes == "I" || axes == "E");
    if (!rdm && !h && !eps && (name != "w" + axes || axes.size() != 4)) {
      throw std::invalid_argument("Unsupported NEVPT2 input binding: " + name);
    }
  }
  const auto n = p.nmo - p.frozen;
  const auto local = reference_detail::TransformOptions(
      input, options, CheckedProduct({n, n, sizeof(double)}));
  wickqc::NDArray<double> h1eff;
  {
    const auto core = DressCore<double>(
        input.integrals.basis,
        coefficients,
        p,
        input.integrals.h1e,
        input.integrals.dressing,
        local);
    h1eff = wickqc::NDArray<double>({n, n}, core.h1eff);
  }
  const auto requests =
      ProfileRequests(Profile::kScalarNevpt2, coefficients, p);
  const auto [arrays, transformed] =
      reference_detail::TransformArrays(input, requests, local);
  wickqc::runtime::TensorMap<double> result;
  for (const auto& binding : bindings) {
    const auto axes = reference_detail::Axes(binding);
    const auto& name = binding.name;
    if (name == "w" + axes) {
      const auto view = GetBlock(transformed, p, axes, Ordering::kPhysicist);
      result.emplace(
          name,
          reference_detail::OwnedView(
              view, arrays.at(transformed->blocks[view.block].name)));
    } else if (one_electron(name, axes)) {
      result.emplace(name, reference_detail::Slice(h1eff, p, axes, true));
    } else if (name == "orbe" + axes) {
      result.emplace(
          name, reference_detail::Slice(r.orbital_energies, p, axes));
    } else {
      result.emplace(name, r.rdms[axes.size() / 2 - 1]);
    }
  }
  wickqc::runtime::numeric_detail::ValidateInputs(bindings, result, dimension);
  return result;
}
} // namespace ao2mo
