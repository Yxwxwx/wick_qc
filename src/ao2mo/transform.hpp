#pragma once

#include "types.hpp"

#ifdef WICKQC_CBLAS_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <omp.h>
#include <unistd.h>
extern "C" {
#include <cint_funcs.h>
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "audit.hpp"
#include "hdf5_store.hpp"
#include "memory.hpp"
#include "provenance.hpp"

namespace ao2mo {
namespace transform_detail {
using Clock = std::chrono::steady_clock;
inline double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
inline void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}
inline int BlasInt(std::size_t n) {
  if (n > std::numeric_limits<int>::max()) {
    throw std::overflow_error("dimension exceeds LP64 BLAS range");
  }
  return static_cast<int>(n);
}
inline std::size_t Triangle(std::size_t n) {
  return CheckedProduct({n, CheckedAdd(n, 1)}) / 2;
}
template <typename T>
inline bool Finite(T x) {
  return std::isfinite(std::real(x)) && std::isfinite(std::imag(x));
}
template <typename T>
void CheckFinite(const std::vector<T>& values) {
  Require(
      std::all_of(values.begin(), values.end(), Finite<T>),
      "input or result contains non-finite values");
}

// Allocation payload of int2e_optimizer in libcint 6.1.3 (optimizer.c).
// Do not apply this private-layout estimate to unreviewed versions.
inline std::size_t OptimizerBytes(const Basis& basis) {
  Require(
      std::string_view(CINT_VERSION) == "6.1.3",
      "budgeted optimizer requires reviewed libcint 6.1.3 headers/library");
  const auto ns = basis.bas.size() / BAS_SLOTS;
  std::size_t primitives = 0, contracted = 0, max_l = 0;
  for (std::size_t i = 0; i < ns; ++i) {
    const auto* shell = basis.bas.data() + i * BAS_SLOTS;
    const auto p = static_cast<std::size_t>(shell[NPRIM_OF]);
    primitives = CheckedAdd(primitives, p);
    contracted = CheckedAdd(
        contracted,
        CheckedProduct({p, static_cast<std::size_t>(shell[NCTR_OF])}));
    max_l = std::max(max_l, static_cast<std::size_t>(shell[ANG_OF]));
  }
  const auto l = std::min(max_l, std::size_t{6});
  const auto cart = CheckedProduct({l + 1, l + 2, l + 3}) / 6;
  auto bytes = CheckedAdd(
      sizeof(CINTOpt),
      CheckedProduct({cart, cart, cart, cart, 3, sizeof(FINT)}));
  bytes = CheckedAdd(
      bytes, CheckedProduct({max_l + 1, LMAX1, LMAX1, LMAX1, sizeof(FINT*)}));
  bytes = CheckedAdd(
      bytes, CheckedProduct({ns, 2 * sizeof(FINT*) + sizeof(double*)}));
  bytes = CheckedAdd(bytes, CheckedProduct({primitives, sizeof(double)}));
  bytes = CheckedAdd(
      bytes,
      CheckedProduct({CheckedAdd(primitives, contracted), sizeof(FINT)}));
  if (primitives && primitives <= 2048) {
    bytes = CheckedAdd(bytes, CheckedProduct({ns, ns, sizeof(PairData*)}));
    bytes = CheckedAdd(
        bytes, CheckedProduct({primitives, primitives, sizeof(PairData)}));
  }
  return bytes;
}

// The 6.1.3 int2e path reads optimizer tables without modifying them; cache
// and shell output remain private per worker. Initialization is serial.
struct IntegralProvider {
  struct DeleteOptimizer {
    void operator()(CINTOpt* opt) const {
      CINTdel_optimizer(&opt);
    }
  };
  const Basis& basis;
  std::vector<std::size_t> offsets;
  std::unique_ptr<CINTOpt, DeleteOptimizer> optimizer;
  explicit IntegralProvider(const Basis& b)
      : basis(b), offsets(b.AoOffsets()) {}
  void InitializeOptimizer() {
    CINTOpt* opt = nullptr;
    int2e_optimizer(
        &opt,
        const_cast<int*>(basis.atm.data()),
        BlasInt(basis.atm.size() / ATM_SLOTS),
        const_cast<int*>(basis.bas.data()),
        BlasInt(basis.bas.size() / BAS_SLOTS),
        const_cast<double*>(basis.env.data()));
    optimizer.reset(opt);
    if (!optimizer) {
      throw std::runtime_error("libcint optimizer initialization failed");
    }
  }
  CACHE_SIZE_T Call(double* out, int* shells, double* cache) const {
    return int2e_sph(
        out,
        nullptr,
        shells,
        const_cast<int*>(basis.atm.data()),
        BlasInt(basis.atm.size() / ATM_SLOTS),
        const_cast<int*>(basis.bas.data()),
        BlasInt(basis.bas.size() / BAS_SLOTS),
        const_cast<double*>(basis.env.data()),
        optimizer.get(),
        cache);
  }
  std::size_t CacheSize() const {
    std::size_t size = 0;
    const int ns = BlasInt(offsets.size() - 1);
    for (int i = 0; i < ns; ++i) {
      for (int j = 0; j < ns; ++j) {
        for (int k = 0; k < ns; ++k) {
          for (int l = 0; l < ns; ++l) {
            int shells[4] = {i, j, k, l};
            auto n = Call(nullptr, shells, nullptr);
            Require(n >= 0, "libcint returned invalid cache size");
            size = std::max(size, static_cast<std::size_t>(n));
          }
        }
      }
    }
    return size;
  }
  std::array<std::size_t, 2> Panel(
      int k,
      int l,
      AoSymmetry symmetry,
      int threads,
      std::vector<double>& panel,
      std::vector<std::vector<double>>& work) const {
    const auto n = offsets.back(), nk = offsets[k + 1] - offsets[k];
    const auto nl = offsets[l + 1] - offsets[l];
    std::fill_n(panel.begin(), n * n * nk * nl, 0.0);
    const int ns = static_cast<int>(offsets.size() - 1);
    std::size_t calls = 0, zeros = 0;
    // BLAS is called outside this region, so no nested BLAS/OpenMP teams.
#pragma omp parallel for schedule(dynamic) num_threads(threads) \
    reduction(+ : calls, zeros)
    for (int i = 0; i < ns; ++i) {
      const auto ni = offsets[i + 1] - offsets[i];
      auto& local = work[omp_get_thread_num()];
      for (int j = 0; j < ns; ++j) {
        if (symmetry == AoSymmetry::kS4 && j > i) {
          continue;
        }
        const auto nj = offsets[j + 1] - offsets[j];
        int shells[4] = {i, j, k, l};
        auto* buffer = local.data();
        auto* cache = buffer + ni * nj * nk * nl;
        const auto nonzero = Call(buffer, shells, cache);
        ++calls;
        if (!nonzero) {
          ++zeros;
          continue; // Panel was zero-filled before the call.
        }
        for (std::size_t d = 0; d < nl; ++d) {
          for (std::size_t c = 0; c < nk; ++c) {
            for (std::size_t b = 0; b < nj; ++b) {
              for (std::size_t a = 0; a < ni; ++a) {
                const auto mu = offsets[i] + a, nu = offsets[j] + b;
                const auto value = buffer[a + ni * (b + nj * (c + nk * d))];
                const auto base = (c * nl + d) * n * n;
                panel[base + mu * n + nu] = value;
                if (symmetry == AoSymmetry::kS4 && i != j) {
                  panel[base + nu * n + mu] = value;
                }
              }
            }
          }
        }
      }
    }
    return {calls, zeros};
  }
};

template <typename T>
std::array<std::size_t, 4> Shape(const Request<T>& request) {
  std::array<std::size_t, 4> shape;
  for (int i = 0; i < 4; ++i) {
    shape[i] = request.indices[i].columns.size();
  }
  return shape;
}
template <typename T>
std::array<std::size_t, 2> PairShape(const Request<T>& request) {
  auto s = Shape(request);
  if (request.layout == MoLayout::kS4) {
    return {Triangle(s[0]), Triangle(s[2])};
  }
  if (request.ordering == Ordering::kPhysicist) {
    std::swap(s[1], s[2]);
  }
  return {CheckedProduct({s[0], s[1]}), CheckedProduct({s[2], s[3]})};
}
inline bool Overlap(
    const void* a,
    std::size_t na,
    const void* b,
    std::size_t nb) {
  if (!na || !nb) {
    return false;
  }
  const auto x = reinterpret_cast<std::uintptr_t>(a);
  const auto y = reinterpret_cast<std::uintptr_t>(b);
  return x <= y ? y - x < na : x - y < nb;
}
template <typename T>
std::size_t ValidateBuffers(
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options,
    std::span<const MemoryBuffer<T>> buffers) {
  if (buffers.empty()) {
    return 0;
  }
  Require(
      options.output == Output::kMemory && buffers.size() == requests.size(),
      "caller buffers require memory output and one buffer per request");
  std::size_t bytes = 0;
  // ponytail: O(blocks^2) alias checks; sort intervals if large request sets
  // matter.
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    const auto shape = PairShape(requests[i]);
    const auto size = CheckedProduct({shape[0], shape[1]});
    const auto data = buffers[i].values.data();
    const auto count = CheckedProduct({size, sizeof(T)});
    Require(
        buffers[i].values.size() == size && (!size || data),
        "caller output buffer has wrong size");
    bytes = CheckedAdd(bytes, count);
    for (std::size_t j = 0; j < i; ++j) {
      Require(
          !Overlap(
              data,
              count,
              buffers[j].values.data(),
              CheckedProduct({buffers[j].values.size(), sizeof(T)})),
          "caller output buffers overlap");
    }
    for (const auto& r : requests) {
      for (const auto& selection : r.indices) {
        const auto& c = *selection.coefficients;
        Require(
            !Overlap(
                data,
                count,
                c.alpha.data(),
                CheckedProduct({c.alpha.size(), sizeof(T)})) &&
                !Overlap(
                    data,
                    count,
                    c.beta.data(),
                    CheckedProduct({c.beta.size(), sizeof(T)})),
            "caller output overlaps input coefficients");
      }
    }
    Require(
        !Overlap(
            data,
            count,
            basis.atm.data(),
            CheckedProduct({basis.atm.size(), sizeof(int)})) &&
            !Overlap(
                data,
                count,
                basis.bas.data(),
                CheckedProduct({basis.bas.size(), sizeof(int)})) &&
            !Overlap(
                data,
                count,
                basis.env.data(),
                CheckedProduct({basis.env.size(), sizeof(double)})),
        "caller output overlaps input basis");
  }
  return bytes;
}
template <typename T>
const Selection<T>& First(
    const std::vector<Request<T>>& requests,
    const HalfGroup& group,
    int axis) {
  return requests[group.requests.front()]
      .indices[axis + (group.reversed.front() ? 2 : 0)];
}
template <typename T>
std::vector<T> Select(const Selection<T>& selection, bool beta) {
  const auto& c = *selection.coefficients;
  const auto& data = beta ? c.beta : c.alpha;
  const auto m = selection.columns.size();
  std::vector<T> result(CheckedProduct({c.nao, m}));
  for (std::size_t i = 0; i < c.nao; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      result[i * m + j] = data[i * c.nmo + selection.columns[j]];
    }
  }
  return result;
}

template <typename T>
void Gemm(
    bool adjoint,
    int m,
    int n,
    int k,
    const T* a,
    const T* b,
    T beta,
    T* c) {
  auto transpose = adjoint ? CblasConjTrans : CblasNoTrans;
  if constexpr (std::is_same_v<T, double>) {
    cblas_dgemm(
        CblasRowMajor,
        transpose,
        CblasNoTrans,
        m,
        n,
        k,
        1,
        a,
        adjoint ? m : k,
        b,
        n,
        beta,
        c,
        n);
  } else {
    const Complex one{1};
    cblas_zgemm(
        CblasRowMajor,
        transpose,
        CblasNoTrans,
        m,
        n,
        k,
        &one,
        a,
        adjoint ? m : k,
        b,
        n,
        &beta,
        c,
        n);
  }
}

template <typename T>
struct PairCoefficients {
  std::array<std::vector<T>, 2> left, right;
  int n, p, q;
  PairCoefficients(const Selection<T>& a, const Selection<T>& b)
      : n(BlasInt(a.coefficients->nao)),
        p(BlasInt(a.columns.size())),
        q(BlasInt(b.columns.size())) {
    left[0] = Select(a, false);
    right[0] = Select(b, false);
    if constexpr (std::is_same_v<T, Complex>) {
      left[1] = Select(a, true);
      right[1] = Select(b, true);
    }
  }
  void Contract(const T* matrix, T* temporary, T* result) const {
    if (!p || !q) {
      return;
    }
    constexpr int components = std::is_same_v<T, Complex> ? 2 : 1;
    for (int spin = 0; spin < components; ++spin) {
      Gemm(false, n, q, n, matrix, right[spin].data(), T{}, temporary);
      Gemm(
          true,
          p,
          q,
          n,
          left[spin].data(),
          temporary,
          spin ? T{1} : T{},
          result);
    }
  }
};

class OwnedFile {
 public:
  OwnedFile(
      const std::filesystem::path& path,
      bool temporary,
      std::size_t expected_bytes)
      : temporary_(temporary) {
    auto directory = temporary ? path : path.parent_path();
    if (directory.empty()) {
      directory = ".";
    }
    const auto available = std::filesystem::space(directory).available;
    if (expected_bytes > available) {
      throw std::runtime_error("insufficient disk space for AO2MO data");
    }
    if (temporary) {
      auto pattern = (directory / "ao2mo-half-XXXXXX").string();
      std::vector<char> name(pattern.begin(), pattern.end());
      name.push_back('\0');
      int fd = mkstemp(name.data());
      if (fd < 0) {
        throw std::runtime_error("cannot create unique scratch file");
      }
      close(fd);
      path_ = name.data();
    } else {
      path_ = path;
    }
    try {
      auto access = h5::FileAccess();
      handle_ = h5::Handle(
          H5Fcreate(
              path_.c_str(),
              temporary ? H5F_ACC_TRUNC : H5F_ACC_EXCL,
              H5P_DEFAULT,
              access),
          H5Fclose);
      created_ = true;
      h5::Complete(handle_, 0);
    } catch (...) {
      if (temporary || created_) {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
      }
      throw;
    }
  }
  ~OwnedFile() {
    handle_ = {};
    if (temporary_ || !complete_) {
      std::error_code ec;
      if (created_) {
        std::filesystem::remove(path_, ec);
      }
    }
  }
  hid_t get() const {
    return handle_;
  }
  void Finish() {
    h5::Complete(handle_, 1);
    h5::Check(H5Fflush(handle_, H5F_SCOPE_GLOBAL));
    handle_.Close();
    complete_ = true;
  }

 private:
  std::filesystem::path path_;
  h5::Handle handle_;
  bool temporary_, created_ = false, complete_ = false;
};

template <typename T>
void StoreRow(
    Block<T>& block,
    hid_t dataset,
    const Request<T>& request,
    bool reversed,
    std::size_t pair,
    const std::vector<T>& row,
    std::vector<T>& packed,
    std::size_t tile_elements,
    Statistics& statistics) {
  const auto start = Clock::now();
  double io = 0;
  const auto s = Shape(request);
  const auto left0 = reversed ? s[2] : s[0];
  const auto left1 = reversed ? s[3] : s[1];
  const auto right0 = reversed ? s[0] : s[2];
  const auto right1 = reversed ? s[1] : s[3];
  (void)left0;
  const auto a = pair / left1, b = pair % left1;
  if (request.layout == MoLayout::kS4 && a < b) {
    return;
  }
  auto write = [&](std::size_t r,
                   std::size_t c,
                   std::size_t nr,
                   std::size_t nc,
                   const T* data) {
    if (dataset >= 0) {
      const auto write_start = Clock::now();
      h5::Slab(
          dataset, true, {r, c}, {nr, nc}, const_cast<T*>(data), tile_elements);
      io += Seconds(write_start);
      statistics.final_write_bytes = CheckedAdd(
          statistics.final_write_bytes, CheckedProduct({nr, nc, sizeof(T)}));
    } else {
      for (std::size_t i = 0; i < nr; ++i) {
        std::copy_n(
            data + i * nc,
            nc,
            block.MutableValues().data() + (r + i) * block.pair_shape[1] + c);
      }
      statistics.transpose_bytes = CheckedAdd(
          statistics.transpose_bytes, CheckedProduct({nr, nc, sizeof(T)}));
    }
  };
  if (request.layout == MoLayout::kS4) {
    std::size_t index = 0;
    for (std::size_t c = 0; c < right0; ++c) {
      for (std::size_t d = 0; d <= c; ++d) {
        packed[index++] = row[c * right1 + d];
      }
    }
    statistics.transpose_bytes = CheckedAdd(
        statistics.transpose_bytes, CheckedProduct({index, sizeof(T)}));
    const auto left_pair = Triangle(a) + b;
    if (reversed) {
      write(0, left_pair, index, 1, packed.data());
    } else {
      write(left_pair, 0, 1, index, packed.data());
    }
  } else if (request.ordering == Ordering::kChemist) {
    if (reversed) {
      write(0, pair, right0 * right1, 1, row.data());
    } else {
      write(pair, 0, 1, right0 * right1, row.data());
    }
  } else if (!reversed) {
    // g[a,b,c,d] -> w[a,c,b,d].
    for (std::size_t c = 0; c < right0; ++c) {
      write(a * s[2] + c, b * s[3], 1, right1, row.data() + c * right1);
    }
  } else {
    // First pair was (r,s); row holds g[p,q,r,s].
    for (std::size_t c = 0; c < right0; ++c) {
      for (std::size_t d = 0; d < right1; ++d) {
        write(c * s[2] + a, d * s[3] + b, 1, 1, &row[c * right1 + d]);
      }
    }
  }
  statistics.final_io_seconds += io;
  statistics.io_seconds += io;
  statistics.transpose_seconds += std::max(0.0, Seconds(start) - io);
}
} // namespace transform_detail

inline std::size_t CheckedProduct(std::initializer_list<std::size_t> factors) {
  std::size_t size = 1;
  for (auto n : factors) {
    if (n && size > std::numeric_limits<std::size_t>::max() / n) {
      throw std::overflow_error("AO2MO size multiplication overflow");
    }
    size *= n;
  }
  return size;
}
inline std::size_t CheckedAdd(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b) {
    throw std::overflow_error("AO2MO size addition overflow");
  }
  return a + b;
}

inline std::vector<std::size_t> Basis::AoOffsets() const {
  static_assert(
      sizeof(FINT) == sizeof(int), "this build requires LP64 libcint");
  transform_detail::Require(
      atm.size() % ATM_SLOTS == 0 && bas.size() % BAS_SLOTS == 0,
      "invalid libcint ATM/BAS shape");
  transform_detail::Require(
      !atm.empty() && !bas.empty() && env.size() >= PTR_ENV_START,
      "empty or truncated basis");
  transform_detail::BlasInt(atm.size() / ATM_SLOTS);
  transform_detail::BlasInt(bas.size() / BAS_SLOTS);
  transform_detail::Require(
      bas.size() / BAS_SLOTS <= SHLS_MAX, "basis exceeds libcint shell limit");
  transform_detail::CheckFinite(env);
  transform_detail::Require(
      env[PTR_RANGE_OMEGA] == 0 && env[PTR_F12_ZETA] == 0 &&
          env[PTR_GTG_ZETA] == 0,
      "only full Coulomb operator is supported");
  auto range = [&](int offset, std::size_t count) {
    transform_detail::Require(
        offset >= 0 && static_cast<std::size_t>(offset) <= env.size() &&
            count <= env.size() - static_cast<std::size_t>(offset),
        "libcint env reference out of bounds");
  };
  for (std::size_t i = 0; i < atm.size(); i += ATM_SLOTS) {
    range(atm[i + PTR_COORD], 3);
    range(atm[i + PTR_ZETA], 1);
    range(atm[i + PTR_FRAC_CHARGE], 1);
  }
  std::vector<std::size_t> offsets{0};
  offsets.reserve(bas.size() / BAS_SLOTS + 1);
  for (std::size_t i = 0; i < bas.size(); i += BAS_SLOTS) {
    const auto* b = bas.data() + i;
    transform_detail::Require(
        b[ATOM_OF] >= 0 &&
            static_cast<std::size_t>(b[ATOM_OF]) < atm.size() / ATM_SLOTS,
        "shell atom out of bounds");
    transform_detail::Require(
        b[ANG_OF] >= 0 && b[ANG_OF] <= ANG_MAX && b[NPRIM_OF] > 0 &&
            b[NPRIM_OF] <= NPRIM_MAX && b[NCTR_OF] > 0 &&
            b[NCTR_OF] <= NCTR_MAX,
        "invalid shell dimensions");
    range(b[PTR_EXP], b[NPRIM_OF]);
    range(
        b[PTR_COEFF],
        CheckedProduct(
            {static_cast<std::size_t>(b[NPRIM_OF]),
             static_cast<std::size_t>(b[NCTR_OF])}));
    for (int p = 0; p < b[NPRIM_OF]; ++p) {
      transform_detail::Require(
          env[b[PTR_EXP] + p] > 0, "nonpositive Gaussian exponent");
    }
    offsets.push_back(CheckedAdd(
        offsets.back(),
        CheckedProduct(
            {static_cast<std::size_t>(2 * b[ANG_OF] + 1),
             static_cast<std::size_t>(b[NCTR_OF])})));
  }
  transform_detail::BlasInt(offsets.back());
  return offsets;
}

template <typename T>
Plan MakePlan(
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options) {
  if (options.audit != AuditMode::kRaw) {
    return detail::AuditPlan(basis, requests, options);
  }
  transform_detail::Require(options.threads > 0, "threads must be positive");
  transform_detail::Require(
      options.io_tile_bytes >= sizeof(T),
      "I/O tile must hold at least one value");
  transform_detail::Require(
      options.output != Output::kHdf5 || !options.output_path.empty(),
      "HDF5 output requires a path");
  if constexpr (std::is_same_v<T, Complex>) {
    transform_detail::Require(
        options.ao_symmetry == AoSymmetry::kS1, "spinor supports AO s1 only");
  }
  Plan plan{};
  plan.metadata_bytes =
      memory_detail::MetadataReserve(basis, requests, options);
  if (plan.metadata_bytes > options.memory_bytes) {
    throw std::runtime_error("request/plan metadata exceeds memory budget");
  }
  transform_detail::IntegralProvider provider(basis);
  const auto n = provider.offsets.back();
  auto provenance_bytes = provenance_detail::Validate(options.provenance);
  const auto label_bytes = [](const std::string& label) {
    transform_detail::Require(
        label.find('\0') == std::string::npos,
        "embedded null in source identity");
    return label.size();
  };
  provenance_bytes =
      CheckedAdd(provenance_bytes, label_bytes(basis.source_identity));
  std::set<const Coefficients<T>*> seen;
  std::set<std::string> names;
  std::size_t max_m = 0, max_pair = 0;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    const auto path = std::filesystem::path(r.name)
                          .lexically_normal()
                          .relative_path()
                          .string();
    transform_detail::Require(
        path != "provenance" && !path.starts_with("provenance/"),
        "provenance is a reserved output path");
    transform_detail::Require(
        !path.empty() && path != "." &&
            r.name.find('\0') == std::string::npos &&
            r.name.find("..") == std::string::npos && names.insert(path).second,
        "empty, unsafe or duplicate output dataset name");
    for (const auto& selection : r.indices) {
      transform_detail::Require(
          static_cast<bool>(selection.coefficients), "null coefficients");
      const auto& c = *selection.coefficients;
      transform_detail::Require(
          c.nao == n, "coefficient AO basis dimension mismatch");
      const auto size = CheckedProduct({n, c.nmo});
      transform_detail::Require(
          c.alpha.size() == size, "invalid alpha/real coefficient shape");
      if constexpr (std::is_same_v<T, Complex>) {
        transform_detail::Require(
            c.beta.size() == size, "spinor requires matching alpha and beta");
      } else {
        transform_detail::Require(
            c.beta.empty(), "real coefficients must have one component");
      }
      if (seen.insert(&c).second) {
        provenance_bytes = CheckedAdd(
            provenance_bytes,
            CheckedAdd(
                label_bytes(c.source_label), label_bytes(c.source_identity)));
        transform_detail::CheckFinite(c.alpha);
        transform_detail::CheckFinite(c.beta);
        plan.coefficient_bytes =
            CheckedAdd(plan.coefficient_bytes, memory_detail::NumericBytes(c));
      }
      for (auto col : selection.columns) {
        transform_detail::Require(col < c.nmo, "MO selection out of bounds");
      }
      transform_detail::BlasInt(selection.columns.size());
      max_m = std::max(max_m, selection.columns.size());
    }
    if (r.layout == MoLayout::kS4) {
      transform_detail::Require(
          !r.rank_four_output && std::is_same_v<T, double> &&
              r.ordering == Ordering::kChemist &&
              r.indices[0] == r.indices[1] && r.indices[2] == r.indices[3],
          "MO s4 requires identical ordered real selections within each pair and chemist order");
    }
    const auto ps = transform_detail::PairShape(r);
    plan.output_bytes = CheckedAdd(
        plan.output_bytes, CheckedProduct({ps[0], ps[1], sizeof(T)}));
    const auto s = transform_detail::Shape(r);
    const auto pq = CheckedProduct({s[0], s[1]}),
               rs = CheckedProduct({s[2], s[3]});
    const auto first_flops = [&](bool reverse) {
      const double p = static_cast<double>(s[reverse ? 2 : 0]),
                   q = static_cast<double>(s[reverse ? 3 : 1]),
                   a = static_cast<double>(n);
      return 2 * a * a * a * q + 2 * a * a * p * q;
    };
    const auto second_flops = [&](bool reverse) {
      const double p = static_cast<double>(s[reverse ? 2 : 0]),
                   q = static_cast<double>(s[reverse ? 3 : 1]);
      const double r = static_cast<double>(s[reverse ? 0 : 2]),
                   t = static_cast<double>(s[reverse ? 1 : 3]),
                   a = static_cast<double>(n);
      return p * q * (2 * a * a * t + 2 * a * r * t);
    };
    max_pair = std::max({max_pair, pq, rs});
    if (!pq || !rs) {
      continue;
    }
    bool found = false;
    for (auto& group : plan.groups) {
      for (int reverse = 0;
           reverse < (options.first_pair_only ? 1 : 2) && !found;
           ++reverse) {
        if (transform_detail::First(requests, group, 0) ==
                r.indices[reverse * 2] &&
            transform_detail::First(requests, group, 1) ==
                r.indices[reverse * 2 + 1]) {
          group.requests.push_back(i);
          group.reversed.push_back(reverse != 0);
          group.pass2_flops += second_flops(reverse != 0);
          found = true;
        }
      }
      if (found) {
        break;
      }
    }
    if (!found) {
      // ponytail: greedy direction selection; revisit with measured workloads
      // if globally optimal pair grouping saves material I/O.
      const auto cost = [&](bool reverse) {
        return first_flops(reverse) + second_flops(reverse) +
            static_cast<double>(reverse ? rs : pq) * static_cast<double>(n) *
            static_cast<double>(n) * sizeof(T) * 2;
      };
      const bool reverse = !options.first_pair_only && cost(true) < cost(false);
      const auto pairs = reverse ? rs : pq;
      plan.groups.push_back(
          {{i},
           {reverse},
           pairs,
           CheckedProduct({pairs, n, n, sizeof(T)}),
           first_flops(reverse),
           second_flops(reverse)});
    }
  }
  for (std::size_t i = 1; i < provider.offsets.size(); ++i) {
    plan.max_shell =
        std::max(plan.max_shell, provider.offsets[i] - provider.offsets[i - 1]);
  }
  plan.max_m = max_m;
  plan.max_pair = max_pair;
  if (!plan.groups.empty()) {
    plan.cache_doubles = provider.CacheSize();
  }
  const auto d = plan.max_shell;
  // Live payload maxima, including caller-owned coefficients and basis.
  auto bytes = memory_detail::NumericBytes(basis);
  plan.basis_bytes = bytes;
  // Source labels and their serialization are bounded; index-vector staging
  // is one axis at a time. Full request/plan metadata accounting is separate.
  bytes = CheckedAdd(bytes, CheckedProduct({2, provenance_bytes}));
  if (options.output == Output::kHdf5) {
    bytes = CheckedAdd(
        bytes,
        CheckedAdd(
            64ULL << 10, CheckedProduct({max_m, sizeof(h5::NativeIndex)})));
  }
  if (options.libcint_optimizer && !plan.groups.empty()) {
    plan.optimizer_bytes = transform_detail::OptimizerBytes(basis);
  }
  bytes = CheckedAdd(bytes, plan.optimizer_bytes);
  bytes = CheckedAdd(bytes, CheckedProduct({n, n, d, d, sizeof(double)}));
  bytes = CheckedAdd(
      bytes,
      CheckedProduct(
          {static_cast<std::size_t>(options.threads),
           CheckedAdd(CheckedProduct({d, d, d, d}), plan.cache_doubles),
           sizeof(double)}));
  // Four selected coefficient sets, each with up to two spin components;
  // one AO staging/second-pass row, GEMM intermediate, result, packed buffer,
  // and transposed shell-pair half tile.
  auto elements = CheckedProduct({8, n, max_m});
  elements = CheckedAdd(elements, CheckedProduct({n, n}));
  elements = CheckedAdd(elements, CheckedProduct({n, max_m}));
  elements = CheckedAdd(elements, CheckedProduct({2, max_pair}));
  elements = CheckedAdd(elements, CheckedProduct({max_pair, d, d}));
  bytes = CheckedAdd(bytes, CheckedProduct({elements, sizeof(T)}));
  plan.buffer_bytes = bytes;
  std::size_t largest_half = 0;
  for (const auto& group : plan.groups) {
    largest_half = std::max(largest_half, group.half_bytes);
  }
  const auto resident =
      options.output == Output::kMemory ? plan.output_bytes : 0;
  const auto base = CheckedAdd(
      plan.metadata_bytes,
      CheckedAdd(plan.coefficient_bytes, CheckedAdd(bytes, resident)));
  // HDF5 metadata cache defaults are capped in execution; raw chunk cache off.
  constexpr std::size_t file_cache = 2ULL << 20;
  auto incore_peak = CheckedAdd(base, largest_half);
  if (options.output == Output::kHdf5) {
    incore_peak = CheckedAdd(incore_peak, file_cache);
  }
  plan.workspace = options.workspace == Workspace::kAuto
      ? (incore_peak <= options.memory_bytes ? Workspace::kIncore
                                             : Workspace::kOutcore)
      : options.workspace;
  plan.io_cache_bytes = (options.output == Output::kHdf5 ? file_cache : 0) +
      (plan.workspace == Workspace::kOutcore && !plan.groups.empty()
           ? file_cache
           : 0);
  const auto batch_base = CheckedAdd(base, plan.io_cache_bytes);
  plan.peak_bytes = batch_base;
  if (plan.workspace == Workspace::kIncore) {
    plan.peak_bytes = CheckedAdd(plan.peak_bytes, largest_half);
  }
  if (plan.peak_bytes > options.memory_bytes) {
    throw std::runtime_error(
        "memory budget insufficient; minimum planned bytes=" +
        std::to_string(plan.peak_bytes));
  }
  const auto extra_coefficients = CheckedProduct({4, n, max_m, sizeof(T)});
  std::size_t batch_peak = batch_base, batch_disk = 0;
  for (std::size_t i = 0; i < plan.groups.size(); ++i) {
    const auto half =
        plan.workspace == Workspace::kIncore ? plan.groups[i].half_bytes : 0;
    auto extra = CheckedAdd(
        half,
        plan.batches.empty() || plan.batches.back().empty()
            ? 0
            : extra_coefficients);
    if (plan.batches.empty() ||
        CheckedAdd(batch_peak, extra) > options.memory_bytes) {
      plan.batches.emplace_back();
      batch_peak = batch_base;
      batch_disk = 0;
      extra = half;
    }
    plan.batches.back().push_back(i);
    batch_peak = CheckedAdd(batch_peak, extra);
    batch_disk = CheckedAdd(batch_disk, plan.groups[i].half_bytes);
    plan.peak_bytes = std::max(plan.peak_bytes, batch_peak);
    if (plan.workspace == Workspace::kOutcore) {
      plan.scratch_bytes = std::max(plan.scratch_bytes, batch_disk);
    }
  }
  plan.ao_passes = plan.batches.size();
  if (memory_detail::PlanBytes(plan) >
      memory_detail::PlanBound(requests.size())) {
    throw std::runtime_error(
        "plan container capacity exceeded reserved metadata");
  }
  return plan;
}

inline std::string Describe(const Plan& plan) {
  std::ostringstream out;
  out << "workspace="
      << (plan.workspace == Workspace::kIncore ? "incore" : "outcore")
      << " ao_passes=" << plan.ao_passes << " peak_bytes=" << plan.peak_bytes
      << " coefficient_bytes=" << plan.coefficient_bytes
      << " buffer_bytes=" << plan.buffer_bytes
      << " metadata_bytes=" << plan.metadata_bytes
      << " optimizer_bytes=" << plan.optimizer_bytes
      << " io_cache_bytes=" << plan.io_cache_bytes
      << " output_bytes=" << plan.output_bytes
      << " scratch_bytes=" << plan.scratch_bytes << '\n';
  if (plan.audit_ao_passes) {
    out << "independent_audit_ao_passes=" << plan.audit_ao_passes << '\n';
  }
  for (std::size_t i = 0; i < plan.groups.size(); ++i) {
    const auto& group = plan.groups[i];
    out << "half_group=" << i << " pairs=" << group.pairs
        << " half_bytes=" << group.half_bytes
        << " real_gemm_flops_pass1=" << group.pass1_flops
        << " real_gemm_flops_pass2=" << group.pass2_flops << " requests=";
    for (std::size_t j = 0; j < group.requests.size(); ++j) {
      out << group.requests[j] << (group.reversed[j] ? ":34 " : ":12 ");
    }
    out << '\n';
  }
  for (std::size_t i = 0; i < plan.batches.size(); ++i) {
    out << "ao_pass=" << i << " groups=";
    for (auto group : plan.batches[i]) {
      out << group << ' ';
    }
    out << '\n';
  }
  return out.str();
}

namespace transform_detail {
template <typename T>
struct WorkBuffers {
  std::vector<double> panel;
  std::vector<std::vector<double>> workers;
  std::vector<T> matrix, temporary, values, packed, tile;

  std::size_t Bytes() const {
    auto bytes = CheckedProduct({panel.capacity(), sizeof(double)});
    for (const auto& worker : workers) {
      bytes = CheckedAdd(
          bytes, CheckedProduct({worker.capacity(), sizeof(double)}));
    }
    for (const auto* buffer : {&matrix, &temporary, &values, &packed, &tile}) {
      bytes =
          CheckedAdd(bytes, CheckedProduct({buffer->capacity(), sizeof(T)}));
    }
    return bytes;
  }
  void Allocate(std::size_t n, const Plan& plan, int threads) {
    if (plan.groups.empty() || !panel.empty()) {
      return;
    }
    const auto d = plan.max_shell;
    panel.resize(CheckedProduct({n, n, d, d}));
    workers.resize(threads);
    for (auto& worker : workers) {
      worker.resize(
          CheckedAdd(CheckedProduct({d, d, d, d}), plan.cache_doubles));
    }
    matrix.resize(CheckedProduct({n, n}));
    temporary.resize(CheckedProduct({n, plan.max_m}));
    values.resize(plan.max_pair);
    packed.resize(plan.max_pair);
    tile.resize(CheckedProduct({plan.max_pair, d, d}));
  }
};

template <typename T>
Result<T> ExecuteRaw(
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options,
    std::span<const MemoryBuffer<T>> buffers,
    const Plan& prepared_plan,
    const IntegralProvider& provider,
    WorkBuffers<T>& work) {
  Result<T> result;
  result.plan = prepared_plan;
  result.statistics.metadata_reserve_bytes = result.plan.metadata_bytes;
  const auto& plan = result.plan;
  result.statistics.caller_referenced_bytes = CheckedAdd(
      CheckedAdd(plan.coefficient_bytes, plan.basis_bytes),
      transform_detail::ValidateBuffers(basis, requests, options, buffers));
  const auto n = provider.offsets.back();
  const int ns = static_cast<int>(provider.offsets.size() - 1);
  const bool disk = plan.workspace == Workspace::kOutcore;
  std::unique_ptr<transform_detail::OwnedFile> output;
  std::vector<h5::Handle> datasets;
  const auto output_start = Clock::now();
  if (options.output == Output::kHdf5) {
    output = std::make_unique<transform_detail::OwnedFile>(
        options.output_path, false, plan.output_bytes);
  }
  if (output) {
    h5::StringAttribute(output->get(), "schema", "ao2mo.output.v1");
    h5::StringAttribute(output->get(), "operator", "full_coulomb_int2e_sph");
    h5::StringAttribute(output->get(), "mode", "raw");
  }
  for (const auto& r : requests) {
    Block<T> block;
    block.name = r.name;
    block.shape = transform_detail::Shape(r);
    block.ordering = r.ordering;
    if (r.ordering == Ordering::kPhysicist) {
      std::swap(block.shape[1], block.shape[2]);
    }
    block.layout = r.layout;
    block.pair_shape = transform_detail::PairShape(r);
    if (output) {
      block.file = options.output_path;
      datasets.push_back(
          h5::Matrix<T>(
              output->get(),
              block.name,
              block.pair_shape,
              r.rank_four_output ? &block.shape : nullptr));
      h5::StringAttribute(
          datasets.back(),
          "ordering",
          r.ordering == Ordering::kChemist ? "chemist" : "physicist");
      h5::StringAttribute(
          datasets.back(),
          "layout",
          r.layout == MoLayout::kDense ? "dense" : "s4");
      for (int axis = 0; axis < 4; ++axis) {
        h5::SizeAttribute(
            datasets.back(),
            ("axis" + std::to_string(axis)).c_str(),
            block.shape[axis]);
      }
    } else if (!buffers.empty()) {
      block.external = buffers[result.blocks.size()];
    } else {
      block.values.resize(
          CheckedProduct({block.pair_shape[0], block.pair_shape[1]}));
    }
    result.blocks.push_back(std::move(block));
  }
  if (output) {
    provenance_detail::Write(output->get(), basis, requests, options, plan);
  }
  if (output) {
    const auto elapsed = Seconds(output_start);
    result.statistics.final_io_seconds += elapsed;
    result.statistics.io_seconds += elapsed;
  }
  auto finish_output = [&] {
    if (!output) {
      return;
    }
    const auto start = Clock::now();
    for (auto& dataset : datasets) {
      dataset.Close();
    }
    output->Finish();
    output.reset();
    const auto elapsed = Seconds(start);
    result.statistics.final_io_seconds += elapsed;
    result.statistics.io_seconds += elapsed;
  };
  std::size_t output_capacity = 0;
  for (const auto& block : result.blocks) {
    output_capacity = CheckedAdd(
        output_capacity, CheckedProduct({block.values.capacity(), sizeof(T)}));
  }
  result.statistics.managed_numeric_peak_bytes = output_capacity;
  if (plan.groups.empty()) {
    finish_output();
    return result;
  }
  result.statistics.reused_workspace_bytes = work.Bytes();
  work.Allocate(n, plan, options.threads);
  auto& [panel, workers, matrix, temporary, values, packed, tile] = work;
  const auto base_capacity = CheckedAdd(output_capacity, work.Bytes());
  const auto pair_capacity =
      [](const transform_detail::PairCoefficients<T>& pair) {
        std::size_t bytes = 0;
        for (int spin = 0; spin < 2; ++spin) {
          bytes = CheckedAdd(
              bytes, CheckedProduct({pair.left[spin].capacity(), sizeof(T)}));
          bytes = CheckedAdd(
              bytes, CheckedProduct({pair.right[spin].capacity(), sizeof(T)}));
        }
        return bytes;
      };
  for (const auto& batch : plan.batches) {
    std::unique_ptr<transform_detail::OwnedFile> scratch;
    if (disk) {
      const auto start = Clock::now();
      scratch = std::make_unique<transform_detail::OwnedFile>(
          options.scratch_directory, true, plan.scratch_bytes);
      const auto elapsed = Seconds(start);
      result.statistics.temp_io_seconds += elapsed;
      result.statistics.io_seconds += elapsed;
    }
    std::vector<transform_detail::PairCoefficients<T>> first_pairs;
    std::vector<std::vector<T>> halves;
    std::vector<h5::Handle> half_datasets;
    first_pairs.reserve(batch.size());
    for (auto group_index : batch) {
      const auto& group = plan.groups[group_index];
      first_pairs.emplace_back(
          transform_detail::First(requests, group, 0),
          transform_detail::First(requests, group, 1));
      if (disk) {
        const auto start = Clock::now();
        half_datasets.push_back(
            h5::Matrix<T>(
                scratch->get(),
                "half" + std::to_string(group_index),
                {group.pairs, n * n},
                nullptr,
                true));
        const auto elapsed = Seconds(start);
        result.statistics.temp_io_seconds += elapsed;
        result.statistics.io_seconds += elapsed;
      } else {
        halves.emplace_back(group.half_bytes / sizeof(T));
      }
    }
    auto batch_capacity = base_capacity;
    for (const auto& pair : first_pairs) {
      batch_capacity = CheckedAdd(batch_capacity, pair_capacity(pair));
    }
    for (const auto& half : halves) {
      batch_capacity = CheckedAdd(
          batch_capacity, CheckedProduct({half.capacity(), sizeof(T)}));
    }
    const auto pass1_start = transform_detail::Clock::now();
    ++result.statistics.ao_passes;
    result.statistics.half_transforms += batch.size();
    for (int k = 0; k < ns; ++k) {
      const auto nk = provider.offsets[k + 1] - provider.offsets[k];
      for (int l = 0; l < ns; ++l) {
        if (options.ao_symmetry == AoSymmetry::kS4 && l > k) {
          continue;
        }
        const auto nl = provider.offsets[l + 1] - provider.offsets[l];
        const auto ao_start = Clock::now();
        const auto calls = provider.Panel(
            k, l, options.ao_symmetry, options.threads, panel, workers);
        result.statistics.ao_seconds += Seconds(ao_start);
        result.statistics.shell_calls += calls[0];
        result.statistics.zero_shell_calls += calls[1];
        for (std::size_t gi = 0; gi < batch.size(); ++gi) {
          const auto& group = plan.groups[batch[gi]];
          for (std::size_t a = 0; a < nk; ++a) {
            for (std::size_t b = 0; b < nl; ++b) {
              const auto local = a * nl + b;
              const auto copy_start = Clock::now();
              std::copy_n(panel.data() + local * n * n, n * n, matrix.data());
              result.statistics.transpose_seconds += Seconds(copy_start);
              result.statistics.transpose_bytes = CheckedAdd(
                  result.statistics.transpose_bytes,
                  CheckedProduct({n, n, sizeof(T)}));
              first_pairs[gi].Contract(
                  matrix.data(), temporary.data(), values.data());
              const auto transpose_start = transform_detail::Clock::now();
              for (std::size_t pq = 0; pq < group.pairs; ++pq) {
                tile[pq * nk * nl + local] = values[pq];
              }
              result.statistics.transpose_seconds +=
                  transform_detail::Seconds(transpose_start);
              result.statistics.transpose_bytes = CheckedAdd(
                  result.statistics.transpose_bytes,
                  CheckedProduct({group.pairs, sizeof(T)}));
            }
          }
          const auto io_start = transform_detail::Clock::now();
          if (disk) {
            for (std::size_t a = 0; a < nk; ++a) {
              h5::HalfColumns(
                  half_datasets[gi],
                  (provider.offsets[k] + a) * n + provider.offsets[l],
                  group.pairs,
                  nl,
                  tile.data() + a * nl,
                  nk * nl,
                  1,
                  options.io_tile_bytes / sizeof(T));
            }
            if (options.ao_symmetry == AoSymmetry::kS4 && k != l) {
              for (std::size_t b = 0; b < nl; ++b) {
                h5::HalfColumns(
                    half_datasets[gi],
                    (provider.offsets[l] + b) * n + provider.offsets[k],
                    group.pairs,
                    nk,
                    tile.data() + b,
                    nk * nl,
                    nl,
                    options.io_tile_bytes / sizeof(T));
              }
            }
          } else {
            for (std::size_t pq = 0; pq < group.pairs; ++pq) {
              for (std::size_t a = 0; a < nk; ++a) {
                const auto column =
                    (provider.offsets[k] + a) * n + provider.offsets[l];
                auto* source = tile.data() + (pq * nk + a) * nl;
                std::copy_n(
                    source, nl, halves[gi].data() + pq * n * n + column);
              }
              if (options.ao_symmetry == AoSymmetry::kS4 && k != l) {
                for (std::size_t b = 0; b < nl; ++b) {
                  for (std::size_t a = 0; a < nk; ++a) {
                    matrix[a] = tile[(pq * nk + a) * nl + b];
                  }
                  const auto column =
                      (provider.offsets[l] + b) * n + provider.offsets[k];
                  std::copy_n(
                      matrix.data(),
                      nk,
                      halves[gi].data() + pq * n * n + column);
                }
              }
            }
          }
          const auto elapsed = Seconds(io_start);
          if (disk) {
            result.statistics.temp_io_seconds += elapsed;
            result.statistics.io_seconds += elapsed;
            result.statistics.temp_write_bytes = CheckedAdd(
                result.statistics.temp_write_bytes,
                CheckedProduct(
                    {group.pairs,
                     nk,
                     nl,
                     sizeof(T),
                     options.ao_symmetry == AoSymmetry::kS4 && k != l ? 2ULL
                                                                      : 1ULL}));
          } else {
            result.statistics.transpose_seconds += elapsed;
            // The mirrored s4 panel moves through matrix and then into half.
            result.statistics.transpose_bytes = CheckedAdd(
                result.statistics.transpose_bytes,
                CheckedProduct(
                    {group.pairs,
                     nk,
                     nl,
                     sizeof(T),
                     options.ao_symmetry == AoSymmetry::kS4 && k != l ? 3ULL
                                                                      : 1ULL}));
          }
        }
      }
    }
    result.statistics.pass1_seconds += transform_detail::Seconds(pass1_start);
    if (scratch) {
      const auto start = Clock::now();
      h5::Check(H5Fflush(scratch->get(), H5F_SCOPE_GLOBAL));
      const auto elapsed = Seconds(start);
      result.statistics.temp_io_seconds += elapsed;
      result.statistics.io_seconds += elapsed;
    }
    const auto pass2_start = transform_detail::Clock::now();
    for (std::size_t gi = 0; gi < batch.size(); ++gi) {
      const auto& group = plan.groups[batch[gi]];
      result.statistics.half_reuses += group.requests.size() - 1;
      for (std::size_t ri = 0; ri < group.requests.size(); ++ri) {
        const auto index = group.requests[ri];
        const bool reverse = group.reversed[ri];
        const auto& request = requests[index];
        const auto offset = reverse ? 0 : 2;
        transform_detail::PairCoefficients<T> second(
            request.indices[offset], request.indices[offset + 1]);
        result.statistics.managed_numeric_peak_bytes = std::max(
            result.statistics.managed_numeric_peak_bytes,
            CheckedAdd(batch_capacity, pair_capacity(second)));
        const auto rows_per_read = disk
            ? std::max<std::size_t>(
                  1,
                  std::min(
                      tile.size() / (n * n),
                      options.io_tile_bytes / sizeof(T) / (n * n)))
            : 1;
        for (std::size_t first = 0; first < group.pairs;
             first += rows_per_read) {
          const auto rows = std::min(rows_per_read, group.pairs - first);
          auto* matrices = rows_per_read > 1 ? tile.data() : matrix.data();
          const auto io_start = transform_detail::Clock::now();
          if (disk) {
            h5::Slab(
                half_datasets[gi],
                false,
                {first, 0},
                {rows, n * n},
                matrices,
                options.io_tile_bytes / sizeof(T));
          } else {
            std::copy_n(halves[gi].data() + first * n * n, n * n, matrices);
          }
          const auto elapsed = Seconds(io_start);
          if (disk) {
            result.statistics.temp_io_seconds += elapsed;
            result.statistics.io_seconds += elapsed;
            result.statistics.temp_read_bytes = CheckedAdd(
                result.statistics.temp_read_bytes,
                CheckedProduct({rows, n, n, sizeof(T)}));
          } else {
            result.statistics.transpose_seconds += elapsed;
            result.statistics.transpose_bytes = CheckedAdd(
                result.statistics.transpose_bytes,
                CheckedProduct({rows, n, n, sizeof(T)}));
          }
          for (std::size_t row = 0; row < rows; ++row) {
            second.Contract(
                matrices + row * n * n, temporary.data(), values.data());
            transform_detail::CheckFinite(values);
            transform_detail::StoreRow(
                result.blocks[index],
                output ? static_cast<hid_t>(datasets[index]) : -1,
                request,
                reverse,
                first + row,
                values,
                packed,
                options.io_tile_bytes / sizeof(T),
                result.statistics);
          }
        }
      }
    }
    result.statistics.pass2_seconds += transform_detail::Seconds(pass2_start);
    if (scratch) {
      const auto start = Clock::now();
      for (auto& dataset : half_datasets) {
        dataset.Close();
      }
      scratch->Finish();
      scratch.reset();
      const auto elapsed = Seconds(start);
      result.statistics.temp_io_seconds += elapsed;
      result.statistics.io_seconds += elapsed;
    }
  }
  finish_output();
  return result;
}
} // namespace transform_detail

// Inputs are immutable for this object's lifetime, including through aliases.
// Owns their shared handles and the exact request/options snapshot. Rebuild the
// object when inputs or numerical policy change. No retained half tensors.
template <typename T>
class PreparedTransform {
 public:
  PreparedTransform(
      std::shared_ptr<const Basis> basis,
      std::vector<Request<T>> requests,
      Options options)
      : basis_(RequireBasis(std::move(basis))),
        requests_(std::move(requests)),
        options_(std::move(options)),
        memory_bytes_(options_.memory_bytes),
        provider_(*basis_) {
    std::size_t retained_metadata = 0, plans = 0;
    if (options_.audit != AuditMode::kRaw) {
      plans = 1;
      retained_metadata = memory_detail::PlanBound(requests_.size());
      for (const auto& request : requests_) {
        if (detail::audit_detail::BlockBytes(request)) {
          plans = CheckedAdd(plans, 3);
          retained_metadata = CheckedAdd(
              retained_metadata,
              CheckedProduct({3, memory_detail::PlanBound(1)}));
        }
      }
      if (retained_metadata >= options_.memory_bytes) {
        throw std::runtime_error("prepared plans exceed memory budget");
      }
      options_.memory_bytes -= retained_metadata;
    }
    plan_ = MakePlan(*basis_, requests_, options_);
    auto raw = options_;
    raw.workspace = plan_.workspace;
    if (options_.audit != AuditMode::kRaw) {
      raw = detail::audit_detail::RawOptions(raw);
      raw_plans_.reserve(plans);
      raw_plans_.push_back(MakePlan(*basis_, requests_, raw));
      for (const auto& request : requests_) {
        if (!detail::audit_detail::BlockBytes(request)) {
          continue;
        }
        const auto partner = detail::audit_detail::PartnerOptions(
            request, raw_plans_[0], options_);
        for (int p = 0; p < 3; ++p) {
          raw_plans_.push_back(MakePlan(
              *basis_,
              std::vector{detail::audit_detail::Partner(request, p)},
              partner));
        }
      }
      auto actual = memory_detail::VectorBytes(raw_plans_);
      for (const auto& cached : raw_plans_) {
        actual =
            CheckedAdd(actual, memory_detail::PlanBytes(cached) - sizeof(Plan));
      }
      if (actual > retained_metadata) {
        throw std::runtime_error(
            "prepared plan capacity exceeded metadata reservation");
      }
      plan_.metadata_bytes =
          CheckedAdd(plan_.metadata_bytes, retained_metadata);
      plan_.peak_bytes = CheckedAdd(plan_.peak_bytes, retained_metadata);
    }
    if (plan_.optimizer_bytes) {
      provider_.InitializeOptimizer();
    }
  }
  const Plan& Inspect() const {
    return plan_;
  }
  PreparedTransform(const PreparedTransform&) = delete;
  PreparedTransform& operator=(const PreparedTransform&) = delete;
  PreparedTransform(PreparedTransform&&) = default;
  PreparedTransform& operator=(PreparedTransform&&) = delete;
  // The destination kind is fixed; a fresh HDF5 path may be supplied each run.
  // Prior returned outputs belong to the caller, outside the next run's budget.
  Result<T> Execute(
      std::span<const MemoryBuffer<T>> buffers = {},
      const std::filesystem::path& output_path = {}) {
    std::lock_guard lock(h5::ExecutionMutex());
    auto options = options_;
    if (!output_path.empty()) {
      options.output_path = output_path;
    }
    auto plan = plan_;
    const auto before =
        memory_detail::MetadataReserve(*basis_, requests_, options_);
    const auto after =
        memory_detail::MetadataReserve(*basis_, requests_, options);
    const auto growth = after > before ? after - before : 0;
    const auto extra = CheckedProduct(
        {growth, options.audit == AuditMode::kRaw ? 1ULL : 2ULL});
    plan.metadata_bytes = CheckedAdd(plan.metadata_bytes, extra);
    plan.peak_bytes = CheckedAdd(plan.peak_bytes, extra);
    if (plan.peak_bytes > memory_bytes_) {
      throw std::runtime_error(
          "prepared output path metadata exceeds memory budget");
    }
    if (options.audit == AuditMode::kRaw) {
      return transform_detail::ExecuteRaw(
          *basis_, requests_, options, buffers, plan, provider_, work_);
    }
    std::size_t index = 0;
    auto execute = [&](const auto& requests,
                       const Options& raw,
                       std::span<const MemoryBuffer<T>> sink) {
      auto raw_plan = raw_plans_.at(index++);
      if (index == 1) {
        raw_plan.metadata_bytes = CheckedAdd(raw_plan.metadata_bytes, growth);
        raw_plan.peak_bytes = CheckedAdd(raw_plan.peak_bytes, growth);
      }
      return transform_detail::ExecuteRaw(
          *basis_, requests, raw, sink, raw_plan, provider_, work_);
    };
    return detail::AuditExecute(
        *basis_, requests_, options, buffers, plan, execute);
  }

 private:
  static std::shared_ptr<const Basis> RequireBasis(
      std::shared_ptr<const Basis> basis) {
    transform_detail::Require(static_cast<bool>(basis), "null prepared basis");
    return basis;
  }
  std::shared_ptr<const Basis> basis_;
  std::vector<Request<T>> requests_;
  Options options_;
  std::size_t memory_bytes_;
  Plan plan_;
  transform_detail::IntegralProvider provider_;
  std::vector<Plan> raw_plans_;
  transform_detail::WorkBuffers<T> work_;
};

template <typename T>
Result<T> Transform(
    const Basis& basis,
    const std::vector<Request<T>>& requests,
    const Options& options,
    std::type_identity_t<std::span<const MemoryBuffer<T>>> buffers) {
  std::lock_guard lock(h5::ExecutionMutex());
  if (options.audit != AuditMode::kRaw) {
    return detail::AuditTransform(basis, requests, options, buffers);
  }
  const auto plan = MakePlan(basis, requests, options);
  transform_detail::IntegralProvider provider(basis);
  if (plan.optimizer_bytes) {
    provider.InitializeOptimizer();
  }
  transform_detail::WorkBuffers<T> work;
  return transform_detail::ExecuteRaw(
      basis, requests, options, buffers, plan, provider, work);
}

} // namespace ao2mo
