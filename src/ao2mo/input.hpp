#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <variant>

#include "hdf5_store.hpp"
#include "helper.hpp"
#include "memory.hpp"
#include "provenance.hpp"
#include "types.hpp"

namespace ao2mo {
template <typename T>
struct Input {
  Basis basis;
  std::map<std::string, std::shared_ptr<const Coefficients<T>>> coefficients;
  std::map<std::string, std::vector<std::size_t>> spaces;
  std::vector<Request<T>> requests;
  Partition partition;
  std::vector<T> h1e;
  Dressing dressing;
  std::map<std::string, std::string> provenance;
  // Conservative load-phase payload/metadata reservation, including a file
  // cache and temporary conversions. It is not process RSS.
  std::size_t loading_budget_bytes = 0;
};
using AnyInput = std::variant<Input<double>, Input<Complex>>;
AnyInput ReadInput(
    const std::filesystem::path& path,
    std::size_t memory_bytes = 512ULL << 20);
namespace input_detail {
inline void Charge(std::size_t& budget, std::size_t bytes) {
  if (bytes > budget) {
    throw std::runtime_error("input payload/metadata exceeds memory budget");
  }
  budget -= bytes;
}
// Four pointers cover the node links/alignment of the supported std::map/set
// implementations; allocator bookkeeping and HDF5 internals remain external.
using memory_detail::kNodeBytes;

inline std::string Attribute(
    hid_t object,
    const std::string& name,
    std::size_t& budget) {
  h5::Handle attr(H5Aopen(object, name.c_str(), H5P_DEFAULT), H5Aclose);
  h5::ScalarAttribute(attr);
  h5::Handle type(H5Aget_type(attr), H5Tclose);
  if (H5Tget_class(type) != H5T_STRING) {
    throw std::invalid_argument("expected string attribute: " + name);
  }
  if (H5Tis_variable_str(type)) {
    char* buffer = nullptr;
    h5::Check(H5Aread(attr, type, static_cast<void*>(&buffer)));
    std::unique_ptr<char, decltype(&H5free_memory)> owner(
        buffer, H5free_memory);
    // HDF5's attribute API has no vlen allocation hook. Charge before making
    // our copy; its initial allocation is an external-library temporary.
    Charge(
        budget,
        CheckedProduct({2, CheckedAdd(buffer ? std::strlen(buffer) : 0, 1)}));
    return buffer ? buffer : "";
  }
  const auto length = CheckedAdd(H5Tget_size(type), 1);
  Charge(budget, CheckedProduct({2, length}));
  std::vector<char> buffer(length, 0);
  h5::Check(H5Aread(attr, type, buffer.data()));
  return buffer.data();
}
inline std::vector<std::string> Children(
    hid_t file,
    const char* path,
    std::size_t& budget) {
  h5::Handle group(H5Gopen2(file, path, H5P_DEFAULT), H5Gclose);
  H5G_info_t info{};
  h5::Check(H5Gget_info(group, &info));
  std::vector<std::string> children;
  Charge(budget, CheckedProduct({info.nlinks, sizeof(std::string)}));
  children.reserve(info.nlinks);
  for (hsize_t i = 0; i < info.nlinks; ++i) {
    const auto size = H5Lget_name_by_idx(
        group, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
    h5::Check(size);
    Charge(budget, CheckedAdd(static_cast<std::size_t>(size), 1));
    std::string name(size + 1, '\0');
    h5::Check(H5Lget_name_by_idx(
        group,
        ".",
        H5_INDEX_NAME,
        H5_ITER_INC,
        i,
        name.data(),
        name.size(),
        H5P_DEFAULT));
    name.resize(size);
    children.push_back(std::move(name));
  }
  return children;
}
inline std::vector<std::size_t> Indices(
    hid_t file,
    const std::string& path,
    std::size_t& budget) {
  std::vector<hsize_t> shape;
  const auto values = h5::Read<h5::NativeIndex>(file, path, &shape, &budget);
  if (shape.size() != 1) {
    throw std::invalid_argument("expected rank-one indices");
  }
  std::vector<std::size_t> result;
  Charge(budget, CheckedProduct({values.size(), sizeof(std::size_t)}));
  result.reserve(values.size());
  for (auto v : values) {
    if (v < 0) {
      throw std::invalid_argument("negative index");
    }
    result.push_back(static_cast<std::size_t>(v));
  }
  return result;
}
inline std::size_t Count(hid_t object, const char* name) {
  h5::Handle attribute(H5Aopen(object, name, H5P_DEFAULT), H5Aclose);
  h5::ScalarAttribute(attribute);
  h5::Handle type(H5Aget_type(attribute), H5Tclose);
  if (H5Tget_class(type) != H5T_INTEGER ||
      H5Tget_size(type) > sizeof(h5::NativeIndex)) {
    throw std::invalid_argument("partition counts must be integers");
  }
  h5::NativeIndex value;
  h5::Check(H5Aread(attribute, H5T_NATIVE_LLONG, &value));
  if (value < 0) {
    throw std::invalid_argument("negative partition count");
  }
  return static_cast<std::size_t>(value);
}
inline double RealAttribute(hid_t object, const char* name) {
  h5::Handle attribute(H5Aopen(object, name, H5P_DEFAULT), H5Aclose);
  h5::ScalarAttribute(attribute);
  double value;
  h5::Check(H5Aread(attribute, H5T_NATIVE_DOUBLE, &value));
  if (!std::isfinite(value)) {
    throw std::invalid_argument("non-finite scalar metadata");
  }
  return value;
}
template <typename T>
Input<T> Load(hid_t file, std::size_t& budget) {
  Charge(budget, sizeof(Input<T>));
  Input<T> input;
  std::vector<hsize_t> shape;
  input.basis.atm = h5::Read<int>(file, "/basis/atm", &shape, &budget);
  if (shape.size() != 2 || shape[1] != 6) {
    throw std::invalid_argument("invalid ATM shape");
  }
  input.basis.bas = h5::Read<int>(file, "/basis/bas", &shape, &budget);
  if (shape.size() != 2 || shape[1] != 8) {
    throw std::invalid_argument("invalid BAS shape");
  }
  input.basis.env = h5::Read<double>(file, "/basis/env", &shape, &budget);
  if (shape.size() != 1) {
    throw std::invalid_argument("invalid ENV shape");
  }
  Charge(
      budget,
      CheckedProduct({input.basis.bas.size() / 8 + 1, sizeof(std::size_t)}));
  const auto offsets = input.basis.AoOffsets();
  if (Indices(file, "/basis/ao_loc_sph", budget) != offsets) {
    throw std::invalid_argument("AO offsets mismatch");
  }
  h5::Handle basis_group(H5Gopen2(file, "/basis", H5P_DEFAULT), H5Gclose);
  const auto identity = Attribute(basis_group, "identity", budget);
  Charge(budget, identity.size() + 1);
  input.basis.source_identity = identity;
  for (const auto& id : Children(file, "/coefficients", budget)) {
    Charge(
        budget,
        CheckedAdd(
            sizeof(Coefficients<T>) + 4 * sizeof(void*) +
                kNodeBytes<decltype(input.coefficients)>,
            CheckedProduct({4, id.size() + 1})));
    auto c = std::make_shared<Coefficients<T>>();
    c->source_label = id;
    const auto base = "/coefficients/" + id;
    h5::Handle coefficient_group(
        H5Gopen2(file, base.c_str(), H5P_DEFAULT), H5Gclose);
    if (H5Aexists(coefficient_group, "identity") > 0) {
      c->source_identity = Attribute(coefficient_group, "identity", budget);
    }
    if (Attribute(coefficient_group, "basis_identity", budget) != identity) {
      throw std::invalid_argument(
          "coefficient family belongs to a different AO basis");
    }
    const auto leaf = std::is_same_v<T, double> ? "/real" : "/alpha";
    c->alpha = h5::Read<T>(file, base + leaf, &shape, &budget);
    if (shape.size() != 2 || shape[0] != offsets.back()) {
      throw std::invalid_argument("invalid coefficient shape");
    }
    c->nao = shape[0];
    c->nmo = shape[1];
    if constexpr (std::is_same_v<T, Complex>) {
      const auto alpha_shape = shape;
      c->beta = h5::Read<T>(file, base + "/beta", &shape, &budget);
      if (shape != alpha_shape) {
        throw std::invalid_argument("alpha/beta shape mismatch");
      }
    }
    const auto finite = [](T value) { return std::isfinite(std::abs(value)); };
    if (!std::all_of(c->alpha.begin(), c->alpha.end(), finite) ||
        !std::all_of(c->beta.begin(), c->beta.end(), finite)) {
      throw std::invalid_argument("non-finite input coefficient family");
    }
    input.coefficients.emplace(id, std::move(c));
  }
  input.partition.nmo = input.coefficients.begin()->second->nmo;
  if (H5Lexists(file, "/partition", H5P_DEFAULT) > 0) {
    h5::Handle p(H5Gopen2(file, "/partition", H5P_DEFAULT), H5Gclose);
    input.partition.ncore = Count(p, "ncore");
    input.partition.ncas = Count(p, "ncas");
    input.partition.nmo = Count(p, "nmo");
    if (H5Aexists(p, "frozen") > 0) {
      input.partition.frozen = Count(p, "frozen");
    }
    const auto unit = Attribute(p, "counting_unit", budget);
    if (unit != (std::is_same_v<T, Complex> ? "spinor" : "spatial_orbital")) {
      throw std::invalid_argument("wrong partition counting unit");
    }
    input.partition.Validate(std::is_same_v<T, Complex>);
  }
  if (H5Lexists(file, "/one_electron", H5P_DEFAULT) > 0) {
    h5::Handle group(H5Gopen2(file, "/one_electron", H5P_DEFAULT), H5Gclose);
    input.h1e = h5::Read<T>(group, "h1e_mo", &shape, &budget);
    if (shape !=
        std::vector<hsize_t>{input.partition.nmo, input.partition.nmo}) {
      throw std::invalid_argument("h1e MO shape mismatch");
    }
    input.dressing.state = Attribute(group, "dressing_state", budget);
    input.dressing.operator_label = Attribute(group, "operator_label", budget);
    if (H5Lexists(group, "orbital_ids", H5P_DEFAULT) > 0) {
      input.dressing.orbital_ids = Indices(group, "orbital_ids", budget);
    }
    if (input.dressing.state == "frozen-folded" &&
        (H5Aexists(group, "core_set") <= 0 ||
         H5Aexists(group, "frozen_electronic") <= 0 ||
         H5Lexists(group, "orbital_ids", H5P_DEFAULT) <= 0)) {
      throw std::invalid_argument(
          "folded h1e requires core_set, frozen_electronic and orbital_ids");
    }
    if (H5Aexists(group, "nuclear_repulsion") > 0) {
      input.dressing.nuclear_repulsion =
          RealAttribute(group, "nuclear_repulsion");
    }
    if (H5Aexists(group, "frozen_electronic") > 0) {
      input.dressing.upstream_frozen_electronic =
          RealAttribute(group, "frozen_electronic");
    }
    if (H5Aexists(group, "core_set") > 0) {
      h5::Handle attribute(H5Aopen(group, "core_set", H5P_DEFAULT), H5Aclose);
      h5::Handle space(H5Aget_space(attribute), H5Sclose);
      h5::Handle type(H5Aget_type(attribute), H5Tclose);
      if (H5Tget_class(type) != H5T_INTEGER || H5Tget_size(type) != 8 ||
          H5Tget_sign(type) != H5T_SGN_2) {
        throw std::invalid_argument(
            "upstream core_set must contain int64 identities");
      }
      if (H5Sget_simple_extent_ndims(space) != 1) {
        throw std::invalid_argument("expected rank-one upstream core_set");
      }
      const auto count = H5Sget_simple_extent_npoints(space);
      h5::Check(count);
      Charge(
          budget,
          CheckedProduct(
              {static_cast<std::size_t>(count),
               sizeof(h5::NativeIndex) + sizeof(std::size_t)}));
      std::vector<h5::NativeIndex> indices(count);
      input.dressing.upstream_frozen.reserve(count);
      if (count) {
        h5::Check(H5Aread(attribute, H5T_NATIVE_LLONG, indices.data()));
      }
      for (auto index : indices) {
        if (index < 0) {
          throw std::invalid_argument("negative upstream frozen identity");
        }
        input.dressing.upstream_frozen.push_back(index);
      }
    }
    Charge(
        budget,
        CheckedProduct(
            {CheckedAdd(
                 input.dressing.orbital_ids.size(),
                 input.dressing.upstream_frozen.size()),
             sizeof(std::size_t) + 4 * sizeof(void*)}));
    input.dressing.Validate(input.partition);
    helper_detail::ValidateH1<T>(input.h1e, input.partition.nmo);
  }
  if (H5Lexists(file, "/spaces", H5P_DEFAULT) > 0) {
    for (const auto& name : Children(file, "/spaces", budget)) {
      Charge(
          budget,
          CheckedAdd(kNodeBytes<decltype(input.spaces)>, name.size() + 1));
      input.spaces[name] =
          Indices(file, "/spaces/" + name + "/indices", budget);
    }
  }
  if (H5Lexists(file, "/requests", H5P_DEFAULT) > 0) {
    const auto names = Children(file, "/requests", budget);
    Charge(budget, CheckedProduct({names.size(), sizeof(Request<T>)}));
    input.requests.reserve(names.size());
    for (const auto& name : names) {
      h5::Handle group(
          H5Gopen2(file, ("/requests/" + name).c_str(), H5P_DEFAULT), H5Gclose);
      Request<T> r;
      r.name = Attribute(group, "dataset", budget);
      auto order = Attribute(group, "ordering", budget),
           layout = Attribute(group, "layout", budget);
      if (order != "chemist" && order != "physicist") {
        throw std::invalid_argument("unknown ordering");
      }
      if (layout != "dense" && layout != "s4") {
        throw std::invalid_argument("unknown MO layout");
      }
      r.ordering =
          order == "chemist" ? Ordering::kChemist : Ordering::kPhysicist;
      r.layout = layout == "dense" ? MoLayout::kDense : MoLayout::kS4;
      for (int i = 0; i < 4; ++i) {
        const auto axis = std::to_string(i);
        r.indices[i].coefficients = input.coefficients.at(
            Attribute(group, "coefficient" + axis, budget));
        r.indices[i].columns = Indices(group, "indices" + axis, budget);
      }
      input.requests.push_back(std::move(r));
    }
  }
  if (input.requests.empty()) {
    const auto c = input.coefficients.begin()->second;
    Charge(
        budget,
        CheckedAdd(
            sizeof(Request<T>),
            CheckedProduct({4, c->nmo, sizeof(std::size_t)})));
    Request<T> full;
    for (auto& s : full.indices) {
      s.coefficients = c;
      s.columns.resize(c->nmo);
      std::iota(s.columns.begin(), s.columns.end(), 0);
    }
    input.requests.push_back(std::move(full));
  }
  if (H5Lexists(file, "/provenance", H5P_DEFAULT) > 0) {
    h5::Handle group(H5Gopen2(file, "/provenance", H5P_DEFAULT), H5Gclose);
    const auto count = H5Aget_num_attrs(group);
    h5::Check(count);
    if (count > 128) {
      throw std::invalid_argument("too many source provenance attributes");
    }
    for (int i = 0; i < count; ++i) {
      h5::Handle attribute(
          H5Aopen_by_idx(
              group,
              ".",
              H5_INDEX_NAME,
              H5_ITER_INC,
              i,
              H5P_DEFAULT,
              H5P_DEFAULT),
          H5Aclose);
      const auto length = H5Aget_name(attribute, 0, nullptr);
      h5::Check(length);
      if (length > (64 << 10)) {
        throw std::invalid_argument("source provenance name exceeds 64 KiB");
      }
      Charge(
          budget,
          CheckedAdd(
              kNodeBytes<decltype(input.provenance)>,
              CheckedProduct({2, static_cast<std::size_t>(length) + 1})));
      std::string name(length + 1, '\0');
      h5::Check(H5Aget_name(attribute, name.size(), name.data()));
      name.resize(length);
      input.provenance[name] = Attribute(group, name, budget);
      provenance_detail::Validate(input.provenance);
    }
  }
  return input;
}
} // namespace input_detail
inline AnyInput ReadInput(
    const std::filesystem::path& path,
    std::size_t memory_bytes) {
  std::lock_guard lock(h5::ExecutionMutex());
  const auto limit = memory_bytes;
  input_detail::Charge(memory_bytes, (2ULL << 20) + (64ULL << 10));
  auto file = h5::OpenFile(path, H5F_ACC_RDONLY);
  if (input_detail::Attribute(file, "schema", memory_bytes) !=
      "ao2mo.input.v1") {
    throw std::invalid_argument("unsupported input schema");
  }
  const auto families =
      input_detail::Children(file, "/coefficients", memory_bytes);
  if (families.empty()) {
    throw std::invalid_argument("input has no coefficient families");
  }
  const auto leaf = "/coefficients/" + families.front() + "/real";
  AnyInput input = H5Lexists(file, leaf.c_str(), H5P_DEFAULT) > 0
      ? AnyInput{input_detail::Load<double>(file, memory_bytes)}
      : AnyInput{input_detail::Load<Complex>(file, memory_bytes)};
  std::visit(
      [&](auto& data) {
        input_detail::Charge(
            memory_bytes,
            CheckedAdd(
                input_detail::kNodeBytes<decltype(data.provenance)>,
                CheckedProduct({2, path.native().size() + 4096})));
        data.provenance["input_path"] =
            std::filesystem::absolute(path).string();
        provenance_detail::Validate(data.provenance);
        data.loading_budget_bytes = limit - memory_bytes;
      },
      input);
  return input;
}
} // namespace ao2mo
