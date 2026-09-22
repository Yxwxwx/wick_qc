#include <ao2mo.hpp>

#include <cstdint>

#include <gtest/gtest.h>

// Fail a real HDF5 call once, without adding production fault switches.
enum class Operation : std::uint8_t {
  kNone,
  kWrite,
  kAttribute,
  kDataset,
  kAttrClose,
  kGroup,
  kFlush,
  kFile
};
Operation operation = Operation::kNone;
int remaining = -1, observed = 0;
bool injected = false;
bool Fail(Operation current) {
  if (operation != current) {
    return false;
  }
  ++observed;
  if (--remaining != 0) {
    return false;
  }
  injected = true;
  return true;
}
// GNU ld --wrap requires the __real_ / __wrap_ symbol names.
// NOLINTBEGIN(bugprone-reserved-identifier)
extern "C" herr_t __real_H5Dwrite(
    hid_t,
    hid_t,
    hid_t,
    hid_t,
    hid_t,
    const void*);
extern "C" herr_t __wrap_H5Dwrite(
    hid_t d,
    hid_t t,
    hid_t m,
    hid_t f,
    hid_t p,
    const void* b) {
  return Fail(Operation::kWrite) ? -1 : __real_H5Dwrite(d, t, m, f, p, b);
}
extern "C" herr_t __real_H5Awrite(hid_t, hid_t, const void*);
extern "C" herr_t __wrap_H5Awrite(hid_t a, hid_t t, const void* b) {
  return Fail(Operation::kAttribute) ? -1 : __real_H5Awrite(a, t, b);
}
extern "C" herr_t __real_H5Dclose(hid_t);
extern "C" herr_t __wrap_H5Dclose(hid_t h) {
  return Fail(Operation::kDataset) ? -1 : __real_H5Dclose(h);
}
extern "C" herr_t __real_H5Aclose(hid_t);
extern "C" herr_t __wrap_H5Aclose(hid_t h) {
  return Fail(Operation::kAttrClose) ? -1 : __real_H5Aclose(h);
}
extern "C" herr_t __real_H5Gclose(hid_t);
extern "C" herr_t __wrap_H5Gclose(hid_t h) {
  return Fail(Operation::kGroup) ? -1 : __real_H5Gclose(h);
}
extern "C" herr_t __real_H5Fflush(hid_t, H5F_scope_t);
extern "C" herr_t __wrap_H5Fflush(hid_t h, H5F_scope_t scope) {
  return Fail(Operation::kFlush) ? -1 : __real_H5Fflush(h, scope);
}
extern "C" herr_t __real_H5Fclose(hid_t);
extern "C" herr_t __wrap_H5Fclose(hid_t h) {
  return Fail(Operation::kFile) ? -1 : __real_H5Fclose(h);
}

// NOLINTEND(bugprone-reserved-identifier)

TEST(Hdf5, FailureRecovery) {
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
  for (auto& index : request.indices) {
    index = {c, {0}};
  }
  const std::vector requests{request};
  ao2mo::detail::audit_detail::ScratchDirectory directory(
      std::filesystem::temp_directory_path(), 64ULL << 20);
  ao2mo::Options options;
  options.scratch_directory = directory.path;
  options.output_path = directory.path / "result.h5";
  const auto objects = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL);
  std::size_t failures = 0;
  const std::array operations{
      Operation::kWrite,
      Operation::kAttribute,
      Operation::kDataset,
      Operation::kAttrClose,
      Operation::kGroup,
      Operation::kFlush,
      Operation::kFile};
  auto arm = [](Operation current, int ordinal) {
    operation = current;
    remaining = ordinal;
    observed = 0;
    injected = false;
  };
  // Enumerate every data-write, dataset/group/file-close and flush on the
  // successful path; first/last attributes cover the shared attribute API.
  auto sweep = [&](auto&& execute, auto&& check_failure) {
    for (auto current : operations) {
      arm(current, -1);
      execute();
      const int count = observed;
      operation = Operation::kNone;
      std::filesystem::remove(options.output_path);
      for (int ordinal = 1; ordinal <= count; ++ordinal) {
        if ((current == Operation::kAttribute ||
             current == Operation::kAttrClose) &&
            ordinal != 1 && ordinal != count) {
          continue;
        }
        arm(current, ordinal);
        bool rejected = false;
        try {
          execute();
        } catch (const std::runtime_error& error) {
          rejected = std::string(error.what()).find("HDF5 operation failed") !=
              std::string::npos;
        }
        operation = Operation::kNone;
        if (!injected || !rejected) {
          throw std::runtime_error(
              "HDF5 failure was not propagated: operation=" +
              std::to_string(static_cast<int>(current)) +
              " ordinal=" + std::to_string(ordinal));
        }
        check_failure();
        if (!std::filesystem::is_empty(directory.path) ||
            H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL) != objects) {
          throw std::runtime_error("HDF5 failure leaked a file or handle");
        }
        ++failures;
      }
    }
  };
  for (auto mode :
       {ao2mo::AuditMode::kRaw,
        ao2mo::AuditMode::kAudit,
        ao2mo::AuditMode::kProjectRoundoff}) {
    for (auto workspace :
         {ao2mo::Workspace::kIncore, ao2mo::Workspace::kOutcore}) {
      for (auto output : {ao2mo::Output::kMemory, ao2mo::Output::kHdf5}) {
        for (bool prepared : {false, true}) {
          options.audit = mode;
          options.workspace = workspace;
          options.output = output;
          ao2mo::PreparedTransform<double> transform(basis, requests, options);
          sweep(
              [&] {
                if (prepared) {
                  transform.Execute();
                } else {
                  ao2mo::Transform(*basis, requests, options);
                }
              },
              [] {});
        }
      }
    }
  }

  options.audit = ao2mo::AuditMode::kRaw;
  options.workspace = ao2mo::Workspace::kIncore;
  options.output = ao2mo::Output::kHdf5;
  const ao2mo::Partition partition{0, 1, 1, 0};
  ao2mo::OneElectron<double> h;
  h.h1e = h.h1eff = {-1};
  h.dressing.operator_label = "analytic fixture";
  auto attach = [&] {
    const auto saved = operation;
    operation = Operation::kNone;
    const auto result = ao2mo::Transform(*basis, requests, options);
    operation = saved;
    ao2mo::WriteHelperMetadata(result, partition, h);
  };
  sweep(attach, [&] {
    // This API extends a caller-owned file. It retains the ERI and marks
    // it incomplete after any mutation; a preflight failure leaves it intact.
    auto file = ao2mo::h5::OpenFile(options.output_path, H5F_ACC_RDONLY);
    const auto eri = ao2mo::h5::Read<double>(file, "eri_mo");
    if (eri.size() != 1 || !std::isfinite(eri[0])) {
      throw std::runtime_error("helper failure damaged its caller's ERI");
    }
    ao2mo::h5::Handle marker(H5Aopen(file, "complete", H5P_DEFAULT), H5Aclose);
    int complete = -1;
    ao2mo::h5::Check(H5Aread(marker, H5T_NATIVE_INT, &complete));
    if (complete != 0 &&
        (complete != 1 || H5Lexists(file, "one_electron", H5P_DEFAULT))) {
      throw std::runtime_error("failed helper retained a complete marker");
    }
    marker.Close();
    file.Close();
    std::filesystem::remove(options.output_path);
  });
  attach();
  auto result = ao2mo::Result<double>{};
  result.blocks.emplace_back();
  result.blocks.front().file = options.output_path;
  bool rejected = false;
  try {
    ao2mo::WriteHelperMetadata(result, partition, h);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  auto file = ao2mo::h5::OpenFile(options.output_path, H5F_ACC_RDONLY);
  ao2mo::h5::Handle marker(H5Aopen(file, "complete", H5P_DEFAULT), H5Aclose);
  int complete = 0;
  ao2mo::h5::Check(H5Aread(marker, H5T_NATIVE_INT, &complete));
  if (!rejected || complete != 1 ||
      ao2mo::h5::Read<double>(file, "one_electron/h1e") != h.h1e) {
    throw std::runtime_error(
        "duplicate helper attachment changed an existing result");
  }
  EXPECT_GT(failures, 0U);
}
