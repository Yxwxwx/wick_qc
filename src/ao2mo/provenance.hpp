#pragma once

#include "hdf5_store.hpp"
#include "types.hpp"

#ifdef WICKQC_CBLAS_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <dlfcn.h>
#include <omp.h>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string_view>
extern "C" {
#include <cint_funcs.h>
}

#if defined(OPENBLAS_USE64BITINT) || defined(MKL_ILP64)
#error AO2MO requires LP64 CBLAS
#endif
#ifdef WICKQC_CBLAS_MKL
static_assert(sizeof(MKL_INT) == 4, "AO2MO requires LP64 CBLAS");
#elif defined(WICKQC_CBLAS_OPENBLAS)
static_assert(sizeof(blasint) == 4, "AO2MO requires LP64 CBLAS");
#elif defined(WICKQC_CBLAS_BLIS)
static_assert(sizeof(f77_int) == 4, "AO2MO requires LP64 CBLAS");
#elif defined(WICKQC_CBLAS_NETLIB)
static_assert(sizeof(CBLAS_INT) == 4, "AO2MO requires LP64 CBLAS");
#endif
#ifndef AO2MO_SOURCE_FINGERPRINT
#define AO2MO_SOURCE_FINGERPRINT \
  "unavailable (build without source fingerprint)"
#endif

namespace ao2mo {
namespace provenance_detail {
// Stable non-cryptographic content fingerprint. Never used for cache identity
// or security. Domain, lengths and little-endian scalar bits are explicit.
struct FingerprintState {
  std::uint64_t value = 14695981039346656037ULL;
  void Byte(unsigned char byte) {
    value = (value ^ byte) * 1099511628211ULL;
  }
  void Word(std::uint64_t word) {
    for (int i = 0; i < 8; ++i) {
      Byte(static_cast<unsigned char>(word));
      word >>= 8;
    }
  }
  void Text(std::string_view text) {
    Word(text.size());
    for (unsigned char byte : text) {
      Byte(byte);
    }
  }
  template <typename T>
  void Array(const std::vector<T>& values) {
    Word(values.size());
    for (const auto& v : values) {
      if constexpr (std::is_same_v<T, Complex>) {
        Word(std::bit_cast<std::uint64_t>(v.real()));
        Word(std::bit_cast<std::uint64_t>(v.imag()));
      } else if constexpr (std::is_same_v<T, double>) {
        Word(std::bit_cast<std::uint64_t>(v));
      } else {
        Word(static_cast<std::uint64_t>(v));
      }
    }
  }
  std::string Finish() const {
    std::ostringstream out;
    out << "fnv1a64-le-v1:" << std::hex << std::setw(16) << std::setfill('0')
        << value;
    return out.str();
  }
};
template <typename Function>
std::string LibraryPath(Function function) {
  Dl_info info{};
  return dladdr(reinterpret_cast<const void*>(function), &info) &&
          info.dli_fname
      ? info.dli_fname
      : "unavailable (static link or unresolved loader symbol)";
}
inline std::size_t Validate(
    const std::map<std::string, std::string>& metadata) {
  std::size_t bytes = 0;
  for (const auto& [name, value] : metadata) {
    if (name.empty() || name.find('\0') != std::string::npos ||
        value.find('\0') != std::string::npos) {
      throw std::invalid_argument("invalid provenance attribute");
    }
    bytes = CheckedAdd(bytes, CheckedAdd(name.size(), value.size()));
  }
  if (bytes > (64ULL << 10)) {
    throw std::invalid_argument("source provenance exceeds 64 KiB");
  }
  return bytes;
}
inline h5::Handle Group(hid_t parent, const std::string& name) {
  return {
      H5Gcreate2(parent, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Gclose};
}
} // namespace provenance_detail

inline std::string Fingerprint(const Basis& basis) {
  provenance_detail::FingerprintState state;
  state.Text("ao2mo.basis.real-spherical.v1");
  state.Array(basis.atm);
  state.Array(basis.bas);
  state.Array(basis.env);
  return state.Finish();
}
template <typename T>
std::string Fingerprint(const Coefficients<T>& coefficients) {
  provenance_detail::FingerprintState state;
  state.Text(
      std::is_same_v<T, Complex> ? "ao2mo.coefficients.complex128.v1"
                                 : "ao2mo.coefficients.float64.v1");
  state.Word(coefficients.nao);
  state.Word(coefficients.nmo);
  state.Array(coefficients.alpha);
  state.Array(coefficients.beta);
  return state.Finish();
}

inline std::map<std::string, std::string> RuntimeInfo() {
  std::lock_guard lock(h5::ExecutionMutex());
#ifdef OPENBLAS_VERSION
  if (std::string_view(openblas_get_config()).find("USE64BITINT") !=
      std::string_view::npos) {
    throw std::runtime_error(
        "runtime OpenBLAS uses unsupported ILP64 integers");
  }
#endif
  unsigned major, minor, release;
  h5::Check(H5get_libversion(&major, &minor, &release));
  double a = 3, b = 4, c = 0;
  const Complex x{1, 2}, y{3, 4}, one{1}, zero{};
  Complex z{};
  cblas_dgemm(
      CblasRowMajor,
      CblasNoTrans,
      CblasNoTrans,
      1,
      1,
      1,
      1,
      &a,
      1,
      &b,
      1,
      0,
      &c,
      1);
  cblas_zgemm(
      CblasRowMajor,
      CblasConjTrans,
      CblasNoTrans,
      1,
      1,
      1,
      &one,
      &x,
      1,
      &y,
      1,
      &zero,
      &z,
      1);
  if (c != 12 || std::abs(z - std::conj(x) * y) > 1e-14) {
    throw std::runtime_error("runtime real/complex CBLAS ABI check failed");
  }
  std::map<std::string, std::string> info{
      {"ao2mo_version", "0.1.0"},
      {"ao2mo_source_fingerprint", AO2MO_SOURCE_FINGERPRINT},
      {"compiler", __VERSION__},
      {"cxx_standard", std::to_string(__cplusplus)},
      {"openmp_version", std::to_string(_OPENMP)},
      {"openmp_loaded_path",
       provenance_detail::LibraryPath(&omp_get_max_threads)},
      {"openmp_max_threads", std::to_string(omp_get_max_threads())},
      {"libcint_header_version", CINT_VERSION},
      {"libcint_runtime_version",
       "unavailable (no libcint C API version query)"},
      {"libcint_integer_bytes", std::to_string(sizeof(FINT))},
      {"libcint_loaded_path", provenance_detail::LibraryPath(&int2e_sph)},
      {"hdf5_header_version", H5_VERS_INFO},
      {"hdf5_runtime_version",
       std::to_string(major) + "." + std::to_string(minor) + "." +
           std::to_string(release)},
      {"hdf5_loaded_path", provenance_detail::LibraryPath(&H5get_libversion)},
      {"blas_loaded_path", provenance_detail::LibraryPath(&cblas_dgemm)},
      {"blas_integer_abi", "LP64"},
      {"blas_real_complex_abi_check", "passed"},
      {"complex_bytes", std::to_string(sizeof(Complex))}};
#ifdef OPENBLAS_VERSION
  info["blas_header_version"] = OPENBLAS_VERSION;
  info["blas_runtime_config"] = openblas_get_config();
  info["blas_threads"] = std::to_string(openblas_get_num_threads());
#else
  info["blas_runtime_config"] =
      "unavailable (generic CBLAS has no version query)";
#endif
  return info;
}

namespace provenance_detail {
template <typename T>
void Write(
    hid_t file,
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options,
    const Plan& plan) {
  auto root = Group(file, "provenance");
  h5::StringAttribute(root, "schema", "ao2mo.provenance.v1");
  h5::StringAttribute(root, "basis_fingerprint", Fingerprint(basis));
  h5::StringAttribute(root, "source_basis_identity", basis.source_identity);
  h5::StringAttribute(root, "representation", "real-spherical");
  h5::StringAttribute(
      root,
      "counting_unit",
      std::is_same_v<T, Complex> ? "spinor" : "spatial_orbital");
  h5::StringAttribute(
      root,
      "ao_symmetry",
      options.ao_symmetry == AoSymmetry::kS1 ? "s1" : "s4");
  h5::StringAttribute(
      root,
      "workspace",
      plan.workspace == Workspace::kIncore ? "incore" : "outcore");
  h5::StringAttribute(
      root,
      "screening",
      "libcint env[PTR_EXPCUTOFF]; 0 selects libcint default");
  h5::DoubleAttribute(root, "exponent_cutoff", basis.env.at(PTR_EXPCUTOFF));
  h5::SizeAttribute(root, "threads", options.threads);
  h5::SizeAttribute(root, "memory_bytes", options.memory_bytes);
  h5::SizeAttribute(root, "io_tile_bytes", options.io_tile_bytes);
  h5::SizeAttribute(root, "libcint_optimizer", options.libcint_optimizer);
  auto runtime = Group(root, "runtime");
  for (const auto& [name, value] : RuntimeInfo()) {
    h5::StringAttribute(runtime, name.c_str(), value);
  }
  auto source = Group(root, "source");
  for (const auto& [name, value] : options.provenance) {
    h5::StringAttribute(source, name.c_str(), value);
  }
  auto families = Group(root, "coefficients");
  auto selections = Group(root, "requests");
  std::map<const Coefficients<T>*, std::size_t> ids;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto request = Group(selections, std::to_string(i));
    const auto& r = requests[i];
    h5::StringAttribute(request, "dataset", r.name);
    h5::StringAttribute(request, "selection_order", "chemist");
    const std::array<std::size_t, 4> axes = r.ordering == Ordering::kChemist
        ? std::array<std::size_t, 4>{0, 1, 2, 3}
        : std::array<std::size_t, 4>{0, 2, 1, 3};
    h5::IndexVector(request, "output_axes", axes);
    for (int axis = 0; axis < 4; ++axis) {
      const auto& selection = r.indices[axis];
      const auto& c = *selection.coefficients;
      const auto [position, inserted] = ids.emplace(&c, ids.size());
      const auto id = std::to_string(position->second);
      if (inserted) {
        auto family = Group(families, id);
        h5::StringAttribute(family, "fingerprint", Fingerprint(c));
        h5::StringAttribute(family, "source_identity", c.source_identity);
        h5::StringAttribute(family, "source_label", c.source_label);
        h5::SizeAttribute(family, "nao", c.nao);
        h5::SizeAttribute(family, "nmo", c.nmo);
        family.Close();
      }
      h5::StringAttribute(
          request, ("coefficient" + std::to_string(axis)).c_str(), id);
      h5::IndexVector(
          request,
          ("indices" + std::to_string(axis)).c_str(),
          selection.columns);
    }
    auto dataset = h5::OpenDataset(file, r.name);
    h5::StringAttribute(
        dataset, "provenance", "/provenance/requests/" + std::to_string(i));
    dataset.Close();
    request.Close();
  }
  selections.Close();
  families.Close();
  source.Close();
  runtime.Close();
  root.Close();
}
} // namespace provenance_detail
} // namespace ao2mo
