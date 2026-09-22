#include <gtest/gtest.h>
#include <atomic>
#include <future>
#include <numbers>
#include <numeric>
#include "ao2mo.hpp"
#include "ao2mo/wick_adapter.hpp"
#include "method/spin_orbital.hpp"

namespace {
// Observe the real libcint calls; no substitute integrals or production hooks.
bool observe_integrals = false, inject_asymmetry = false;
std::atomic<std::size_t> queries{0}, optimizer_builds{0}, evaluations{0};
double* last_cache = nullptr;
CINTOpt* last_optimizer = nullptr;
// GNU ld --wrap requires the __real_ / __wrap_ symbol names.
// NOLINTBEGIN(bugprone-reserved-identifier)
extern "C" CINTIntegralFunction __real_int2e_sph;
extern "C" CINTOptimizerFunction __real_int2e_optimizer;
extern "C" void __wrap_int2e_optimizer(
    CINTOpt** opt,
    FINT* atm,
    FINT natm,
    FINT* bas,
    FINT nbas,
    double* env) {
  if (observe_integrals) {
    ++optimizer_builds;
  }
  __real_int2e_optimizer(opt, atm, natm, bas, nbas, env);
}
extern "C" CACHE_SIZE_T __wrap_int2e_sph(
    double* out,
    FINT* dims,
    FINT* shells,
    FINT* atm,
    FINT natm,
    FINT* bas,
    FINT nbas,
    double* env,
    CINTOpt* opt,
    double* cache) {
  if (observe_integrals && !out) {
    ++queries;
  } else if (observe_integrals) {
    ++evaluations;
    last_cache = cache;
    last_optimizer = opt;
  }
  const auto status = __real_int2e_sph(
      out, dims, shells, atm, natm, bas, nbas, env, opt, cache);
  if (inject_asymmetry && out && status) {
    out[0] += 1e-3 * (shells[0] - shells[2]);
  }
  return status;
}

// NOLINTEND(bugprone-reserved-identifier)

class IntegralTest : public testing::Test {
 protected:
  void TearDown() override {
    observe_integrals = false;
    inject_asymmetry = false;
  }
};

TEST_F(IntegralTest, InputSizes) {
  using ao2mo::CheckedProduct;
  if (CheckedProduct({3, 5, 2, 4}) != 120 || CheckedProduct({0, 8}) != 0) {
    FAIL() << "Check 1 failed";
  }
  try {
    CheckedProduct({std::numeric_limits<std::size_t>::max(), 2});
    FAIL() << "Check 2 failed";
  } catch (const std::overflow_error&) { // NOLINT(bugprone-empty-catch)
    // Expected rejection is the successful test path.
  }
  try {
    ao2mo::Basis{}.AoOffsets();
    FAIL() << "Check 3 failed";
  } catch (const std::invalid_argument&) { // NOLINT(bugprone-empty-catch)
    // Expected rejection is the successful test path.
  }
}

TEST_F(IntegralTest, PreparedStorageAndBudget) {
  observe_integrals = true;

  auto basis = std::make_shared<ao2mo::Basis>();
  basis->atm = {2, 20, 1, 0, 0, 0};
  basis->bas = {0, 0, 1, 1, 0, 23, 24, 0};
  basis->env.resize(25);
  basis->env[23] = 1;
  basis->env[24] = CINTgto_norm(0, 1);
  auto c = std::make_shared<ao2mo::Coefficients<double>>();
  c->nao = c->nmo = 1;
  c->alpha = {1};
  ao2mo::Request<double> request;
  request.layout = ao2mo::MoLayout::kS4;
  for (auto& index : request.indices) {
    index = {c, {0}};
  }
  const double expected = 2 / std::sqrt(std::numbers::pi);
  for (auto mode :
       {ao2mo::AuditMode::kRaw,
        ao2mo::AuditMode::kAudit,
        ao2mo::AuditMode::kProjectRoundoff}) {
    ao2mo::Options options;
    options.audit = mode;
    options.libcint_optimizer = true;
    options.workspace = ao2mo::Workspace::kIncore;
    options.scratch_directory = "/nonexistent/prepared-incore";
    auto requests = std::vector{request};
    auto empty = request;
    empty.name = "empty";
    empty.indices[0].columns.clear();
    empty.indices[1].columns.clear();
    requests.push_back(empty);
    ao2mo::PreparedTransform<double> prepared(basis, requests, options);
    const auto prepared_queries = queries.load(),
               built = optimizer_builds.load();
    requests.clear(); // Prepared owns the request snapshot.
    double output = 0;
    const std::array<ao2mo::MemoryBuffer<double>, 2> buffers{
        {{{&output, 1}, {}}, {}}};
    prepared.Execute(buffers);
    const auto cache = last_cache;
    const auto optimizer = last_optimizer;
    auto moved = std::move(prepared);
    const auto result = moved.Execute(buffers);
    if (std::abs(output - expected) > 1e-13 ||
        !result.blocks[1].Values().empty()) {
      throw std::runtime_error(
          "prepared packed/empty analytic output mismatch");
    }
    if (queries != prepared_queries || optimizer_builds != built || !cache ||
        !optimizer || cache != last_cache || optimizer != last_optimizer ||
        !result.statistics.reused_workspace_bytes) {
      throw std::runtime_error("prepared execution rebuilt integral state");
    }
    // Invalid storage must leave both integral state and output untouched.
    try {
      moved.Execute(std::span(buffers).first(1));
      throw std::runtime_error("prepared accepted missing output");
    } catch (const std::invalid_argument&) { // NOLINT(bugprone-empty-catch)
      // Expected rejection is the successful test path.
    }
    moved.Execute(buffers); // Recover after a rejected execution.
    options.memory_bytes = moved.Inspect().peak_bytes - 1;
    try {
      ao2mo::PreparedTransform<double> too_small(
          basis, {request, empty}, options);
      throw std::logic_error("prepared optimizer ignored its memory budget");
    } catch (const std::runtime_error&) { // NOLINT(bugprone-empty-catch)
      // Expected rejection is the successful test path.
    }
    if (optimizer_builds != built) {
      throw std::runtime_error("optimizer allocated before budget check");
    }

    options.libcint_optimizer = false;
    options.output = ao2mo::Output::kHdf5;
    options.output_path = "/tmp/ao2mo-prepared-path.h5";
    options.memory_bytes = 64ULL << 20;
    ao2mo::PreparedTransform<double> sized(basis, {request}, options);
    options.memory_bytes = sized.Inspect().peak_bytes;
    ao2mo::PreparedTransform<double> exact(basis, {request}, options);
    bool path_rejected = false;
    try {
      exact.Execute({}, std::string(2048, 'x'));
    } catch (const std::runtime_error& error) {
      path_rejected =
          std::string(error.what()).find("path metadata") != std::string::npos;
    }
    if (!path_rejected) {
      throw std::runtime_error(
          "prepared destination growth ignored memory budget");
    }
  }

  // vector::size() does not describe retained caller allocations. Exercise
  // numeric capacities and selection/dressing metadata independently.
  ao2mo::Options limited;
  limited.memory_bytes = 1ULL << 20;
  limited.workspace = ao2mo::Workspace::kIncore;
  const auto reject = [](auto&& run) {
    bool rejected = false;
    try {
      run();
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    if (!rejected) {
      throw std::runtime_error("caller vector capacity escaped memory budget");
    }
  };
  auto large_c = std::make_shared<ao2mo::Coefficients<double>>(*c);
  large_c->alpha.reserve(1 << 18);
  auto large_request = request;
  for (auto& index : large_request.indices) {
    index.coefficients = large_c;
  }
  reject([&] { ao2mo::MakePlan(*basis, std::vector{large_request}, limited); });
  auto large_basis = *basis;
  large_basis.env.reserve(1 << 18);
  reject([&] { ao2mo::MakePlan(large_basis, std::vector{request}, limited); });
  auto large_selections = std::vector{request};
  large_selections[0].indices[0].columns.reserve(1 << 18);
  reject([&] { ao2mo::MakePlan(*basis, large_selections, limited); });
  ao2mo::Dressing dressing;
  dressing.operator_label = "analytic test h0";
  dressing.orbital_ids = {0};
  dressing.orbital_ids.reserve(1 << 18);
  const std::array<double, 1> h0{0};
  reject([&] {
    ao2mo::DressCore(
        *basis,
        std::shared_ptr<const ao2mo::Coefficients<double>>(c),
        {0, 0, 1, 0},
        std::span<const double>(h0),
        dressing,
        limited);
  });
  // Three distinct first-pair identities fit jointly at a generous budget;
  // the exact minimum forces three real integral passes. Trace the external
  // calls so the planner's predicted counters are not the sole evidence.
  std::vector<ao2mo::Request<double>> batched;
  for (int i = 1; i <= 3; ++i) {
    auto family = std::make_shared<ao2mo::Coefficients<double>>(*c);
    family->alpha[0] = i;
    auto r = request;
    r.name = "group" + std::to_string(i);
    for (auto& selection : r.indices) {
      selection.coefficients = family;
    }
    batched.push_back(std::move(r));
  }
  for (auto workspace :
       {ao2mo::Workspace::kIncore, ao2mo::Workspace::kOutcore}) {
    ao2mo::Options batch_options;
    batch_options.workspace = workspace;
    const auto joint = ao2mo::MakePlan(*basis, batched, batch_options);
    if (joint.ao_passes != 1 || joint.groups.size() != 3) {
      throw std::runtime_error("generous budget did not share an AO pass");
    }
    const auto extra_group = 4 * sizeof(double) +
        (workspace == ao2mo::Workspace::kIncore ? sizeof(double) : 0);
    batch_options.memory_bytes = joint.peak_bytes - 2 * extra_group;
    const auto before = evaluations.load();
    const auto result = ao2mo::Transform(*basis, batched, batch_options);
    if (evaluations != before + 3 || result.plan.ao_passes != 3 ||
        result.statistics.ao_passes != 3 ||
        result.statistics.half_transforms != 3) {
      throw std::runtime_error(
          "budgeted batches did not execute three integral passes");
    }
    for (std::size_t i = 0; i < batched.size(); ++i) {
      if (std::abs(
              result.blocks[i].Values()[0] -
              expected * std::pow(static_cast<double>(i) + 1.0, 4)) > 1e-12) {
        throw std::runtime_error(
            "budgeted batch disagrees with analytic integral");
      }
    }
  }
}

TEST_F(IntegralTest, RejectCorruptPairSymmetry) {
  inject_asymmetry = true;

  ao2mo::Basis basis;
  basis.atm = {2, 20, 1, 0, 0, 0};
  basis.bas = {0, 0, 1, 1, 0, 23, 25, 0, 0, 0, 1, 1, 0, 24, 26, 0};
  basis.env.resize(27);
  basis.env[23] = 1;
  basis.env[24] = 0.4;
  basis.env[25] = CINTgto_norm(0, 1);
  basis.env[26] = CINTgto_norm(0, 0.4);
  auto c = std::make_shared<ao2mo::Coefficients<ao2mo::Complex>>();
  c->nao = c->nmo = 2;
  c->alpha = {1, 0, 0, 1};
  c->beta.resize(4);
  ao2mo::Request<ao2mo::Complex> request;
  for (auto& index : request.indices) {
    index = {c, {0, 1}};
  }
  const std::vector requests{request};
  ao2mo::detail::audit_detail::ScratchDirectory directory(
      std::filesystem::temp_directory_path(), 64ULL << 20);
  ao2mo::Options options;
  options.io_tile_bytes = 3 * sizeof(ao2mo::Complex);
  options.scratch_directory = directory.path;
  const auto raw = ao2mo::Transform(basis, requests, options);
  if (std::abs(raw.blocks[0].values[3] - raw.blocks[0].values[12]) < 1e-4) {
    throw std::runtime_error("injected asymmetry did not reach raw output");
  }
  for (auto mode :
       {ao2mo::AuditMode::kAudit, ao2mo::AuditMode::kProjectRoundoff}) {
    for (auto workspace :
         {ao2mo::Workspace::kIncore, ao2mo::Workspace::kOutcore}) {
      for (auto output : {ao2mo::Output::kMemory, ao2mo::Output::kHdf5}) {
        for (bool prepared : {false, true}) {
          options.audit = mode;
          options.workspace = workspace;
          options.output = output;
          options.output_path = directory.path / "invalid.h5";
          bool rejected = false;
          try {
            if (prepared) {
              ao2mo::PreparedTransform<ao2mo::Complex> transform(
                  std::make_shared<const ao2mo::Basis>(basis),
                  requests,
                  options);
              transform.Execute();
            } else {
              ao2mo::Transform(basis, requests, options);
            }
          } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            rejected =
                message.find("exceeds roundoff gate") != std::string::npos &&
                message.find("residual=") != std::string::npos &&
                message.find("gate=") != std::string::npos;
          }
          if (!rejected || !std::filesystem::is_empty(directory.path)) {
            throw std::runtime_error(
                "audit accepted corruption or left output/scratch behind");
          }
        }
      }
    }
  }
  // Auto must reconsider raw's incore choice when independent projection
  // copies/partners exceed that budget. No integral execution is needed.
  options.audit = ao2mo::AuditMode::kProjectRoundoff;
  options.output = ao2mo::Output::kHdf5;
  options.memory_bytes = 12ULL << 20;
  for (auto& index : request.indices) {
    index.columns.resize(26);
    for (std::size_t i = 0; i < 26; ++i) {
      index.columns[i] = i % 2;
    }
  }
  options.workspace = ao2mo::Workspace::kAuto;
  if (ao2mo::MakePlan(basis, std::vector{request}, options).workspace !=
      ao2mo::Workspace::kOutcore) {
    throw std::runtime_error("auto audit did not fall back to outcore");
  }
  options.workspace = ao2mo::Workspace::kIncore;
  bool rejected = false;
  try {
    ao2mo::MakePlan(basis, std::vector{request}, options);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error("explicit incore audit exceeded its budget");
  }
}

// Two normalized s Gaussians at the same center. Four different coefficient
// families exercise rectangular transformations without external reference
// data.
template <typename T>
ao2mo::Input<T> AnalyticInput(bool general) {
  ao2mo::Input<T> data;
  data.basis.atm = {2, 20, 1, 0, 0, 0};
  data.basis.bas = {0, 0, 1, 1, 0, 23, 25, 0, 0, 0, 1, 1, 0, 24, 26, 0};
  data.basis.env.resize(27);
  data.basis.env[23] = 1;
  data.basis.env[24] = 0.4;
  data.basis.env[25] = CINTgto_norm(0, 1);
  data.basis.env[26] = CINTgto_norm(0, 0.4);
  data.partition = {1, 1, 4, 0};
  const std::array<std::size_t, 4> sizes = general
      ? std::array<std::size_t, 4>{3, 5, 2, 4}
      : std::array<std::size_t, 4>{4, 4, 4, 4};
  ao2mo::Request<T> request;
  for (std::size_t family = 0; family < (general ? 4 : 1); ++family) {
    auto c = std::make_shared<ao2mo::Coefficients<T>>();
    c->nao = 2;
    c->nmo = sizes[family];
    c->alpha.resize(2 * c->nmo);
    if constexpr (std::is_same_v<T, ao2mo::Complex>) {
      c->beta.resize(2 * c->nmo);
    }
    for (std::size_t i = 0; i < c->alpha.size(); ++i) {
      const auto value = static_cast<double>(i + 1) / (c->nmo + family + 3);
      if constexpr (std::is_same_v<T, ao2mo::Complex>) {
        c->alpha[i] = {value, 0.13 * static_cast<double>(i % 3)};
        c->beta[i] = {0.17 * value, -0.11 * static_cast<double>((i + 1) % 4)};
      } else {
        c->alpha[i] = value;
      }
    }
    data.coefficients.emplace("c" + std::to_string(family), c);
    for (std::size_t axis = 0; axis < 4; ++axis) {
      if (!general || axis == family) {
        request.indices[axis].coefficients = c;
        request.indices[axis].columns.resize(c->nmo);
        std::iota(
            request.indices[axis].columns.begin(),
            request.indices[axis].columns.end(),
            0);
      }
    }
  }
  data.requests.push_back(request);
  auto physicist = request;
  physicist.name = "physicist";
  physicist.ordering = ao2mo::Ordering::kPhysicist;
  data.requests.push_back(physicist);
  return data;
}

template <typename T>
void CheckAnalytic(
    const ao2mo::Input<T>& data,
    const ao2mo::Result<T>& result) {
  const auto& indices = data.requests.front().indices;
  const auto pair = [&](std::size_t left,
                        std::size_t right,
                        std::size_t a,
                        std::size_t b,
                        std::size_t p,
                        std::size_t q) {
    const auto& c = *indices[left].coefficients;
    const auto& d = *indices[right].coefficients;
    if constexpr (std::is_same_v<T, ao2mo::Complex>) {
      return std::conj(c.alpha[a * c.nmo + p]) * d.alpha[b * d.nmo + q] +
          std::conj(c.beta[a * c.nmo + p]) * d.beta[b * d.nmo + q];
    } else {
      return c.alpha[a * c.nmo + p] * d.alpha[b * d.nmo + q];
    }
  };
  const std::array<double, 2> exponents{1, 0.4};
  const auto integral =
      [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
        const auto p = exponents[a] + exponents[b],
                   q = exponents[c] + exponents[d];
        double norm = 1;
        for (auto i : {a, b, c, d}) {
          norm *= std::pow(2 * exponents[i] / std::numbers::pi, 0.75);
        }
        return norm * 2 * std::pow(std::numbers::pi, 2.5) /
            (p * q * std::sqrt(p + q));
      };
  const auto n0 = indices[0].columns.size(), n1 = indices[1].columns.size(),
             n2 = indices[2].columns.size(), n3 = indices[3].columns.size();
  for (std::size_t i = 0; i < n0; ++i)
    for (std::size_t j = 0; j < n1; ++j)
      for (std::size_t k = 0; k < n2; ++k)
        for (std::size_t l = 0; l < n3; ++l) {
          T expected{};
          for (std::size_t a = 0; a < 2; ++a)
            for (std::size_t b = 0; b < 2; ++b)
              for (std::size_t c = 0; c < 2; ++c)
                for (std::size_t d = 0; d < 2; ++d) {
                  expected += pair(0, 1, a, b, i, j) * pair(2, 3, c, d, k, l) *
                      integral(a, b, c, d);
                }
          EXPECT_NEAR(
              std::abs(
                  result.blocks[0].Values()[((i * n1 + j) * n2 + k) * n3 + l] -
                  expected),
              0,
              1e-11);
          EXPECT_NEAR(
              std::abs(
                  result.blocks[1].Values()[((i * n2 + k) * n1 + j) * n3 + l] -
                  expected),
              0,
              1e-11);
        }
}

template <typename T>
void CheckStorage(
    const ao2mo::Input<T>& data,
    const std::filesystem::path& output) {
  const auto check_capacity = [](const auto& result) {
    if (ao2mo::CheckedAdd(
            ao2mo::CheckedAdd(
                result.statistics.managed_numeric_peak_bytes,
                result.plan.optimizer_bytes),
            ao2mo::CheckedAdd(
                result.statistics.caller_referenced_bytes,
                result.statistics.metadata_reserve_bytes)) >
        result.plan.peak_bytes) {
      throw std::runtime_error(
          "numerical capacity and metadata reservation exceed plan");
    }
  };
  ao2mo::Options options;
  options.io_tile_bytes = 3 * sizeof(T);
  options.workspace = ao2mo::Workspace::kIncore;
  options.scratch_directory = "/nonexistent/ao2mo-incore-must-not-touch";
  const auto memory = ao2mo::Transform(data.basis, data.requests, options);
  check_capacity(memory);
  CheckAnalytic(data, memory);
  options.workspace = ao2mo::Workspace::kOutcore;
  options.scratch_directory = output.parent_path();
  const auto disk_half = ao2mo::Transform(data.basis, data.requests, options);
  check_capacity(disk_half);
  auto first = std::async(std::launch::async, [&] {
    return ao2mo::Transform(data.basis, data.requests, options);
  });
  auto second = std::async(std::launch::async, [&] {
    return ao2mo::Transform(data.basis, data.requests, options);
  });
  const auto concurrent_first = first.get(), concurrent_second = second.get();
  for (std::size_t b = 0; b < memory.blocks.size(); ++b) {
    if (concurrent_first.blocks[b].values !=
        concurrent_second.blocks[b].values) {
      throw std::runtime_error("concurrent host executions disagree");
    }
  }
  for (std::size_t b = 0; b < memory.blocks.size(); ++b) {
    if (memory.blocks[b].values.size() != disk_half.blocks[b].values.size()) {
      throw std::runtime_error("size mismatch");
    }
    for (std::size_t i = 0; i < memory.blocks[b].values.size(); ++i) {
      if (std::abs(memory.blocks[b].values[i] - disk_half.blocks[b].values[i]) >
          1e-11) {
        throw std::runtime_error("memory outputs disagree");
      }
    }
  }
  options.output = ao2mo::Output::kHdf5;
  options.output_path = output;
  const auto disk = ao2mo::Transform(data.basis, data.requests, options);
  check_capacity(disk);
  ao2mo::h5::Handle file(
      H5Fopen(output.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  for (std::size_t b = 0; b < memory.blocks.size(); ++b) {
    const auto values = ao2mo::h5::Read<T>(file, disk.blocks[b].name);
    if (values.size() != memory.blocks[b].values.size()) {
      throw std::runtime_error("disk shape mismatch");
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (std::abs(values[i] - memory.blocks[b].values[i]) > 1e-11) {
        throw std::runtime_error("disk output disagrees");
      }
    }
  }
  options.output = ao2mo::Output::kMemory;
  for (auto mode :
       {ao2mo::AuditMode::kRaw,
        ao2mo::AuditMode::kAudit,
        ao2mo::AuditMode::kProjectRoundoff}) {
    for (auto workspace :
         {ao2mo::Workspace::kIncore, ao2mo::Workspace::kOutcore}) {
      options.audit = mode;
      options.workspace = workspace;
      options.libcint_optimizer = true;
      ao2mo::PreparedTransform<T> prepared(
          std::make_shared<const ao2mo::Basis>(data.basis),
          data.requests,
          options);
      std::vector<ao2mo::MemoryBuffer<T>> buffers;
      std::vector<std::weak_ptr<std::vector<T>>> owners;
      for (const auto& block : memory.blocks) {
        auto storage =
            std::make_shared<std::vector<T>>(block.values.size(), T{123});
        owners.push_back(storage);
        buffers.push_back({*storage, storage});
      }
      auto direct = prepared.Execute(buffers);
      direct = {};
      direct = prepared.Execute(buffers);
      check_capacity(direct);
      if (!direct.statistics.reused_workspace_bytes ||
          !prepared.Inspect().optimizer_bytes) {
        throw std::runtime_error("prepared workspace not reused");
      }
      if (direct.statistics.caller_referenced_bytes !=
          memory.statistics.caller_referenced_bytes +
              memory.plan.output_bytes) {
        throw std::runtime_error("caller output memory not counted");
      }
      for (std::size_t b = 0; b < direct.blocks.size(); ++b) {
        const auto values = direct.blocks[b].Values();
        if (!direct.blocks[b].values.empty() ||
            values.data() != buffers[b].values.data()) {
          throw std::runtime_error("caller output was copied");
        }
        for (std::size_t i = 0; i < values.size(); ++i) {
          if (std::abs(values[i] - memory.blocks[b].values[i]) > 1e-11) {
            throw std::runtime_error("caller output value mismatch");
          }
        }
      }
      buffers.clear();
      for (const auto& owner : owners) {
        if (owner.expired()) {
          throw std::runtime_error("caller output lost ownership");
        }
      }
      direct = {};
      for (const auto& owner : owners) {
        if (!owner.expired()) {
          throw std::runtime_error("caller output leaked ownership");
        }
      }
      options.output = ao2mo::Output::kHdf5;
      options.output_path = output.string() + ".prepared";
      ao2mo::PreparedTransform<T> disk_prepared(
          std::make_shared<const ao2mo::Basis>(data.basis),
          data.requests,
          options);
      for (int run = 0; run < 2; ++run) {
        const auto path = options.output_path.string() + std::to_string(run);
        const auto disk_result = disk_prepared.Execute({}, path);
        check_capacity(disk_result);
        {
          auto prepared_file = ao2mo::h5::OpenFile(path, H5F_ACC_RDONLY);
          for (std::size_t b = 0; b < memory.blocks.size(); ++b) {
            const auto actual =
                ao2mo::h5::Read<T>(prepared_file, memory.blocks[b].name);
            for (std::size_t i = 0; i < actual.size(); ++i) {
              if (std::abs(actual[i] - memory.blocks[b].values[i]) > 1e-11) {
                throw std::runtime_error("prepared disk result mismatch");
              }
            }
          }
        }
        std::filesystem::remove(path);
      }
      options.output = ao2mo::Output::kMemory;
    }
  }
  options.libcint_optimizer = false;
  options.audit = ao2mo::AuditMode::kRaw;
  options.workspace = ao2mo::Workspace::kIncore;
  auto rejects = [&](const std::vector<ao2mo::Request<T>>& requests,
                     const std::vector<ao2mo::MemoryBuffer<T>>& buffers) {
    try {
      ao2mo::Transform(data.basis, requests, options, buffers);
    } catch (const std::invalid_argument&) {
      return;
    }
    throw std::runtime_error("invalid caller output accepted");
  };
  std::vector<T> sentinel(memory.blocks[0].values.size(), T{123});
  std::vector<ao2mo::MemoryBuffer<T>> invalid(
      data.requests.size(), {sentinel, {}});
  rejects(data.requests, invalid); // Overlapping outputs.
  invalid.resize(1);
  rejects(data.requests, invalid); // Missing output.
  auto single = std::vector{data.requests[0]};
  invalid[0].values = invalid[0].values.first(sentinel.size() - 1);
  rejects(single, invalid); // Wrong extent.
  invalid[0].values = sentinel;
  options.output = ao2mo::Output::kHdf5;
  rejects(single, invalid);
  options.output = ao2mo::Output::kMemory;
  for (auto& index : single[0].indices) {
    index.columns.resize(1);
  }
  invalid[0].values = {
      const_cast<T*>(single[0].indices[0].coefficients->alpha.data()), 1};
  rejects(single, invalid); // Coefficient alias.
  if (!std::all_of(sentinel.begin(), sentinel.end(), [](T value) {
        return value == T{123};
      })) {
    throw std::runtime_error("preflight changed caller output");
  }
  for (auto mode :
       {ao2mo::AuditMode::kAudit, ao2mo::AuditMode::kProjectRoundoff}) {
    for (auto workspace :
         {ao2mo::Workspace::kIncore, ao2mo::Workspace::kOutcore}) {
      options.audit = mode;
      options.workspace = workspace;
      const auto audited = ao2mo::Transform(data.basis, data.requests, options);
      check_capacity(audited);
      for (std::size_t b = 0; b < memory.blocks.size(); ++b) {
        for (std::size_t i = 0; i < memory.blocks[b].values.size(); ++i) {
          if (std::abs(
                  audited.blocks[b].values[i] - memory.blocks[b].values[i]) >
              1e-11) {
            throw std::runtime_error("audit memory output disagrees with raw");
          }
        }
      }
    }
  }
  options.audit = ao2mo::AuditMode::kRaw;
  options.workspace = ao2mo::Workspace::kIncore;
  options.memory_bytes = memory.plan.peak_bytes - 1;
  bool rejected = false;
  try {
    ao2mo::Transform(data.basis, data.requests, options);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error("incore accepted insufficient budget");
  }
}

TEST_F(IntegralTest, Storage) {
  ao2mo::detail::audit_detail::ScratchDirectory directory(
      std::filesystem::temp_directory_path(), 64ULL << 20);
  CheckStorage(AnalyticInput<double>(true), directory.path / "real.h5");
  CheckStorage(
      AnalyticInput<ao2mo::Complex>(true), directory.path / "complex.h5");
}

template <typename T>
void CheckGetters(
    const ao2mo::Input<T>& data,
    const std::filesystem::path& output) {
  constexpr bool spinor = std::is_same_v<T, ao2mo::Complex>;
  const auto coefficients = data.coefficients.begin()->second;
  ao2mo::Options options;
  options.scratch_directory = output.parent_path();
  options.threads = 4;
  options.io_tile_bytes = 7 * sizeof(T);
  const auto profile =
      spinor ? ao2mo::Profile::kSpinorDense : ao2mo::Profile::kScalarMrci;
  auto requests = ao2mo::ProfileRequests(profile, coefficients, data.partition);
  std::vector<ao2mo::MemoryBuffer<T>> buffers;
  for (const auto& request : requests) {
    std::vector<std::size_t> shape;
    shape.reserve(request.indices.size());
    for (const auto& index : request.indices) {
      shape.push_back(index.columns.size());
    }
    if (request.ordering == ao2mo::Ordering::kPhysicist) {
      std::swap(shape[1], shape[2]);
    }
    auto array = std::make_shared<wickqc::NDArray<T>>(shape);
    buffers.push_back(ao2mo::WickBuffer(std::move(array)));
  }
  auto result = std::make_shared<const ao2mo::Result<T>>(
      ao2mo::Transform(data.basis, requests, options, buffers));
  for (std::size_t b = 0; b < buffers.size(); ++b) {
    if (result->blocks[b].Values().data() != buffers[b].values.data() ||
        !result->blocks[b].values.empty()) {
      throw std::runtime_error("NDArray sink was copied");
    }
  }
  buffers.clear(); // Result must retain the actual NDArray owners.
  ao2mo::Request<T> full;
  full.name = "pppp";
  for (auto& index : full.indices) {
    index.coefficients = coefficients;
    index.columns.resize(coefficients->nmo);
    std::iota(index.columns.begin(), index.columns.end(), 0);
  }
  const auto dense = ao2mo::Transform(data.basis, std::vector{full}, options);
  const auto n = coefficients->nmo;
  auto check = [&](const std::shared_ptr<const ao2mo::Result<T>>& owner,
                   const std::string& key,
                   ao2mo::Ordering order) {
    const auto view = ao2mo::GetBlock(owner, data.partition, key, order);
    const auto array = ao2mo::ToWick(view, 32ULL << 20);
    const auto p = data.partition.Indices(key[0]),
               q = data.partition.Indices(key[1]);
    const auto r = data.partition.Indices(key[2]),
               s = data.partition.Indices(key[3]);
    if (view.shape !=
        std::array<std::size_t, 4>{p.size(), q.size(), r.size(), s.size()}) {
      throw std::runtime_error("getter shape mismatch: " + key);
    }
    std::size_t linear = 0;
    for (auto i : p) {
      for (auto j : q) {
        for (auto k : r) {
          for (auto l : s) {
            const auto source = order == ao2mo::Ordering::kChemist
                ? ((i * n + j) * n + k) * n + l
                : ((i * n + k) * n + j) * n + l;
            if (std::abs(
                    array.data()[linear++] - dense.blocks[0].values[source]) >
                1e-11) {
              throw std::runtime_error(
                  "getter value/offset/conjugation mismatch: " + key);
            }
          }
        }
      }
    }
    const auto& block = owner->blocks[view.block];
    bool refused = false;
    try {
      view.Data();
    } catch (const std::logic_error&) {
      refused = true;
    }
    if (refused != (view.conjugate || !block.file.empty())) {
      throw std::runtime_error("unsafe pointer from disk/conjugated getter");
    }
    if (linear) {
      refused = false;
      try {
        view.Materialize(linear * sizeof(T) - 1);
      } catch (const std::runtime_error&) {
        refused = true;
      }
      if (!refused) {
        throw std::runtime_error("materialization ignored its budget");
      }
    }
    return view.conjugate;
  };
  const std::string letters = spinor ? "cavp" : "cav";
  for (char a : letters) {
    for (char b : letters) {
      for (char c : letters) {
        for (char d : letters) {
          const std::string key{a, b, c, d};
          auto alias = key;
          for (char& x : alias) {
            x = std::string("IAEP")[std::string("cavp").find(x)];
          }
          for (auto order :
               {ao2mo::Ordering::kChemist, ao2mo::Ordering::kPhysicist}) {
            check(result, key, order);
            check(result, alias, order);
          }
        }
      }
    }
  }
  requests = ao2mo::ProfileRequests(
      spinor ? ao2mo::Profile::kSpinorNevpt2 : ao2mo::Profile::kScalarNevpt2,
      coefficients,
      data.partition);
  // Only the four legal complex mappings, stated independently of the
  // getter.
  constexpr int permutations[4][4] = {
      {0, 1, 2, 3}, {2, 3, 0, 1}, {1, 0, 3, 2}, {3, 2, 1, 0}};
  for (auto destination : {ao2mo::Output::kMemory, ao2mo::Output::kHdf5}) {
    options.output = destination;
    options.workspace = ao2mo::Workspace::kOutcore;
    options.output_path = output.string() + ".getters.h5";
    options.scratch_directory = output.parent_path();
    auto sparse = std::make_shared<const ao2mo::Result<T>>(
        ao2mo::Transform(data.basis, requests, options));
    bool saw_conjugation = false;
    for (const auto& request : requests) {
      auto chem = request.name.substr(request.name.find_last_of('/') + 1);
      if (request.ordering == ao2mo::Ordering::kPhysicist) {
        std::swap(chem[1], chem[2]);
      }
      for (const auto& perm : permutations) {
        std::string key{
            chem[perm[0]], chem[perm[1]], chem[perm[2]], chem[perm[3]]};
        saw_conjugation |= check(sparse, key, ao2mo::Ordering::kChemist);
        std::swap(key[1], key[2]);
        saw_conjugation |= check(sparse, key, ao2mo::Ordering::kPhysicist);
      }
    }
    if (spinor && !saw_conjugation) {
      throw std::runtime_error(
          "test did not exercise a conjugated sparse view");
    }
    bool missing = false;
    try {
      ao2mo::GetBlock(
          sparse, data.partition, "vvvv", ao2mo::Ordering::kChemist);
    } catch (const std::out_of_range&) {
      missing = true;
    }
    if (!missing) {
      throw std::runtime_error("uncovered block was silently provided");
    }
    auto view = ao2mo::GetBlock(
        sparse, data.partition, "aaaa", ao2mo::Ordering::kChemist);
    const auto before = view.Materialize(32ULL << 20);
    std::weak_ptr<const ao2mo::Result<T>> weak = sparse;
    sparse.reset();
    if (weak.expired() || view.Materialize(32ULL << 20) != before) {
      throw std::runtime_error("view lost its owner");
    }
    view = {};
    if (!weak.expired()) {
      throw std::runtime_error("view leaked its owner");
    }
    if (destination == ao2mo::Output::kHdf5) {
      std::filesystem::remove(options.output_path);
    }
  }
  try {
    ao2mo::GetBlock<T>({}, data.partition, "aaaa", ao2mo::Ordering::kChemist);
    throw std::runtime_error("null owner accepted");
    // NOLINTNEXTLINE(bugprone-empty-catch)
  } catch (const std::invalid_argument&) {
    // Expected rejection is the successful test path.
  }
}

TEST_F(IntegralTest, Getters) {
  ao2mo::detail::audit_detail::ScratchDirectory directory(
      std::filesystem::temp_directory_path(), 64ULL << 20);
  CheckGetters(AnalyticInput<double>(false), directory.path / "real.h5");
  CheckGetters(
      AnalyticInput<ao2mo::Complex>(false), directory.path / "complex.h5");
}

TEST_F(IntegralTest, SpinorWickSymmetry) {
  ao2mo::Basis basis;
  basis.atm = {2, 20, 1, 0, 0, 0};
  basis.bas = {0, 0, 1, 1, 0, 23, 24, 0};
  basis.env.resize(25);
  basis.env[23] = 1;
  basis.env[24] = CINTgto_norm(0, 1);
  auto c = std::make_shared<ao2mo::Coefficients<ao2mo::Complex>>();
  c->nao = 1;
  c->nmo = 4;
  c->alpha = {1, {0, 1}, 1, {1, 1}};
  c->beta.resize(4);
  const ao2mo::Partition partition{2, 0, 4, 0};
  const auto requests = ao2mo::ProfileRequests<ao2mo::Complex>(
      ao2mo::Profile::kSpinorDense, c, partition);
  auto result = std::make_shared<const ao2mo::Result<ao2mo::Complex>>(
      ao2mo::Transform(basis, requests, ao2mo::Options{}));
  const auto v = ao2mo::ToWick(
      ao2mo::GetBlock(result, partition, "pppp", ao2mo::Ordering::kPhysicist),
      1ULL << 20);
  const auto residual = v.At({2, 0, 3, 1}) - v.At({3, 0, 2, 1});
  const auto expression = wickqc::method::CCSDGenerator(false)
                              .Parse("v[aibj] - v[biaj]")
                              .Simplify();
  const bool simplified_to_zero = expression.Terms().empty();
  // Preserve the documented complex spin-orbital generator limitation.
  EXPECT_TRUE(simplified_to_zero);
  EXPECT_GT(std::abs(residual), 1e-10);
}

} // namespace
