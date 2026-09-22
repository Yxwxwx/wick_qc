#include <ao2mo.hpp>

#include <charconv>
#include <chrono>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace {
std::size_t Number(std::string_view value) {
  std::size_t n;
  auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), n);
  if (error != std::errc{} || end != value.data() + value.size()) {
    throw std::invalid_argument("invalid nonnegative integer option");
  }
  return n;
}
} // namespace
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
      for (const auto& [name, value] : ao2mo::RuntimeInfo()) {
        std::cout << name << '=' << value << '\n';
      }
      return 0;
    }
    if (argc < 3) {
      throw std::invalid_argument(
          "usage: wickqc_integrals inspect INPUT [--output-path OUTPUT] | transform INPUT OUTPUT [--workspace incore|outcore|auto] "
          "[--memory BYTES] [--threads N] [--scratch DIR] [--ao-symmetry s1|s4]");
    }
    const std::string_view command = argv[1];
    ao2mo::Options options;
    options.output = ao2mo::Output::kHdf5;
    options.output_path = "inspect-only";
    std::optional<ao2mo::Profile> profile;
    int pos = 3;
    if (command == "transform") {
      if (argc < 4) {
        throw std::invalid_argument("missing output filename");
      }
      options.output = ao2mo::Output::kHdf5;
      options.output_path = argv[3];
      pos = 4;
    } else if (command != "inspect") {
      throw std::invalid_argument("unknown command");
    }
    for (; pos < argc; pos += 2) {
      if (pos + 1 == argc) {
        throw std::invalid_argument("missing option value");
      }
      const std::string_view key = argv[pos], value = argv[pos + 1];
      if (key == "--mode") {
        if (value == "raw") {
          options.audit = ao2mo::AuditMode::kRaw;
        } else if (value == "audit") {
          options.audit = ao2mo::AuditMode::kAudit;
        } else if (value == "project-roundoff") {
          options.audit = ao2mo::AuditMode::kProjectRoundoff;
        } else {
          throw std::invalid_argument("unknown audit mode");
        }
      } else if (key == "--preset") {
        if (value == "scalar-nevpt2") {
          profile = ao2mo::Profile::kScalarNevpt2;
        } else if (value == "scalar-mrci") {
          profile = ao2mo::Profile::kScalarMrci;
        } else if (value == "spinor-dense") {
          profile = ao2mo::Profile::kSpinorDense;
        } else if (value == "spinor-nevpt2") {
          profile = ao2mo::Profile::kSpinorNevpt2;
        } else {
          throw std::invalid_argument("unknown helper preset");
        }
      } else if (key == "--workspace") {
        if (value == "incore") {
          options.workspace = ao2mo::Workspace::kIncore;
        } else if (value == "outcore") {
          options.workspace = ao2mo::Workspace::kOutcore;
        } else if (value == "auto") {
          options.workspace = ao2mo::Workspace::kAuto;
        } else {
          throw std::invalid_argument("unknown workspace");
        }
      } else if (key == "--ao-symmetry") {
        if (value == "s1") {
          options.ao_symmetry = ao2mo::AoSymmetry::kS1;
        } else if (value == "s4") {
          options.ao_symmetry = ao2mo::AoSymmetry::kS4;
        } else {
          throw std::invalid_argument("unknown AO symmetry");
        }
      } else if (key == "--optimizer") {
        if (value != "on" && value != "off") {
          throw std::invalid_argument("optimizer must be on or off");
        }
        options.libcint_optimizer = value == "on";
      } else if (key == "--output-path" && command == "inspect") {
        options.output_path = value;
      } else if (key == "--io-tile") {
        options.io_tile_bytes = Number(value);
      } else if (key == "--memory") {
        options.memory_bytes = Number(value);
      } else if (key == "--threads") {
        auto count = Number(value);
        if (count > 1024) {
          throw std::invalid_argument("threads exceeds 1024");
        }
        options.threads = static_cast<int>(count);
      } else if (key == "--scratch") {
        options.scratch_directory = value;
      } else {
        throw std::invalid_argument("unknown option: " + std::string(key));
      }
    }
    auto input = ao2mo::ReadInput(argv[2], options.memory_bytes);
    std::visit(
        [&](auto& data) {
          using T = typename std::decay_t<decltype(data.h1e)>::value_type;
          options.provenance = std::move(data.provenance);
          const auto primary = profile
              ? data.coefficients.begin()->second
              : std::shared_ptr<const ao2mo::Coefficients<T>>{};
          const auto requests = profile
              ? ao2mo::ProfileRequests(*profile, primary, data.partition)
              : std::move(data.requests);
          // Requests retain exactly the coefficient owners used below. Release
          // unrelated imported families and descriptive spaces before
          // execution.
          data.coefficients.clear();
          data.spaces.clear();
          decltype(data.requests){}.swap(data.requests);
          if (!profile) {
            decltype(data.h1e){}.swap(data.h1e);
            data.dressing = {};
          }
          std::cout << "input_loading_budget_bytes="
                    << data.loading_budget_bytes << '\n';
          auto transform_options = options;
          const auto matrices = profile
              ? ao2mo::helper_detail::ResidentBytes<T>(data.partition)
              : 0;
          const auto helper_metadata = profile
              ? ao2mo::helper_detail::MetadataBytes(
                    data.partition, data.dressing)
              : 0;
          const auto resident = ao2mo::CheckedAdd(matrices, helper_metadata);
          if (resident >= options.memory_bytes) {
            throw std::runtime_error("helper matrices exceed budget");
          }
          transform_options.memory_bytes -= resident;
          auto core_options = options;
          const auto retained_requests =
              profile ? ao2mo::memory_detail::RequestBytes(requests) : 0;
          if (retained_requests >= core_options.memory_bytes) {
            throw std::runtime_error("helper requests exceed budget");
          }
          core_options.memory_bytes -= retained_requests;
          if (profile) {
            data.dressing.Validate(data.partition);
            ao2mo::helper_detail::ValidateH1<T>(data.h1e, data.partition.nmo);
          }
          const auto start = std::chrono::steady_clock::now();
          if (command == "inspect" || profile) {
            const auto plan =
                ao2mo::MakePlan(data.basis, requests, transform_options);
            const auto metadata_peak = profile
                ? ao2mo::helper_detail::MetadataPeak<T>(
                      plan, data.partition, data.dressing)
                : 0;
            if (metadata_peak > options.memory_bytes) {
              throw std::runtime_error(
                  "helper metadata workspace exceeds budget");
            }
            if (command == "inspect") {
              std::size_t peak = std::max(
                  ao2mo::CheckedAdd(plan.peak_bytes, resident), metadata_peak);
              std::size_t core_passes = 0;
              if (profile) {
                peak = std::max(
                    peak,
                    ao2mo::CheckedAdd(
                        ao2mo::CheckedProduct(
                            {5,
                             data.partition.nmo,
                             data.partition.nmo,
                             sizeof(T)}),
                        ao2mo::CheckedAdd(
                            ao2mo::CheckedAdd(
                                retained_requests,
                                ao2mo::helper_detail::CoreMetadataBytes(
                                    data.partition,
                                    data.dressing,
                                    core_options)),
                            ao2mo::helper_detail::ReferencedBytes(
                                data.basis, *primary))));
                const auto core_plans = ao2mo::MakeCorePlans(
                    data.basis,
                    primary,
                    data.partition,
                    core_options,
                    data.dressing);
                for (std::size_t i = 0; i < core_plans.size(); ++i) {
                  std::cout << "core_orbital=" << i << ' '
                            << ao2mo::Describe(core_plans[i]);
                  core_passes += core_plans[i].ao_passes;
                  peak = std::max(
                      peak,
                      ao2mo::CheckedAdd(
                          core_plans[i].peak_bytes, retained_requests));
                }
              }
              std::cout << "eri_plan " << ao2mo::Describe(plan);
              std::cout << "core_ao_passes=" << core_passes
                        << " total_ao_passes="
                        << core_passes + plan.ao_passes + plan.audit_ao_passes
                        << " helper_resident_matrix_bytes=" << matrices
                        << " total_planned_peak_bytes=" << peak << '\n';
            }
          }
          if (command == "transform") {
            std::optional<ao2mo::OneElectron<T>> core;
            double core_seconds = 0;
            if (profile) {
              const auto core_start = std::chrono::steady_clock::now();
              core = ao2mo::DressCore(
                  data.basis,
                  primary,
                  data.partition,
                  std::span<const T>(data.h1e),
                  data.dressing,
                  core_options);
              core->planned_peak_bytes = ao2mo::CheckedAdd(
                  core->planned_peak_bytes, retained_requests);
              core->statistics.metadata_reserve_bytes = ao2mo::CheckedAdd(
                  core->statistics.metadata_reserve_bytes, retained_requests);
              core_seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - core_start)
                                 .count();
            }
            auto result =
                ao2mo::Transform(data.basis, requests, transform_options);
            if (core) {
              try {
                ao2mo::WriteHelperMetadata(result, data.partition, *core);
              } catch (...) {
                // Transform created this path exclusively in this invocation.
                std::error_code error;
                std::filesystem::remove(transform_options.output_path, error);
                throw;
              }
            }
            const auto elapsed = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
            std::cout << "transform_seconds=" << elapsed << '\n';
            std::cout << ao2mo::Describe(result.plan);
            auto s = result.statistics;
            if (core) {
              s = ao2mo::helper_detail::HelperStatistics(
                  result, data.partition, *core);
              std::cout
                  << "core_seconds=" << core_seconds
                  << " core_ao_passes=" << core->statistics.ao_passes
                  << " core_shell_calls=" << core->statistics.shell_calls
                  << " total_planned_peak_bytes="
                  << std::max(
                         {core->planned_peak_bytes,
                          ao2mo::CheckedAdd(result.plan.peak_bytes, resident),
                          ao2mo::helper_detail::MetadataPeak<T>(
                              result.plan, data.partition, data.dressing)})
                  << '\n';
            }
            std::cout << "ao_passes=" << s.ao_passes
                      << " shell_calls=" << s.shell_calls
                      << " half_reuses=" << s.half_reuses
                      << " half_transforms=" << s.half_transforms
                      << " zero_shell_calls=" << s.zero_shell_calls
                      << " ao_seconds=" << s.ao_seconds
                      << " pass1_seconds=" << s.pass1_seconds
                      << " pass2_seconds=" << s.pass2_seconds
                      << " transpose_seconds=" << s.transpose_seconds
                      << " transpose_bytes=" << s.transpose_bytes
                      << " io_seconds=" << s.io_seconds
                      << " temp_io_seconds=" << s.temp_io_seconds
                      << " final_io_seconds=" << s.final_io_seconds
                      << " audit_io_seconds=" << s.audit_io_seconds
                      << " audit_seconds=" << s.audit_seconds
                      << " temp_read_bytes=" << s.temp_read_bytes
                      << " temp_write_bytes=" << s.temp_write_bytes
                      << " final_write_bytes=" << s.final_write_bytes
                      << " audit_read_bytes=" << s.audit_read_bytes
                      << " audit_write_bytes=" << s.audit_write_bytes
                      << " managed_numeric_peak_bytes="
                      << s.managed_numeric_peak_bytes
                      << " caller_referenced_bytes="
                      << s.caller_referenced_bytes
                      << " metadata_reserve_bytes=" << s.metadata_reserve_bytes
                      << '\n';
            for (const auto& audit : result.audit) {
              std::cout << "audit_block=" << audit.name
                        << " raw_residuals=" << audit.raw_residuals[0] << ','
                        << audit.raw_residuals[1] << ','
                        << audit.raw_residuals[2]
                        << " roundoff_gate=" << audit.roundoff_gate
                        << " correction=" << audit.correction << '\n';
            }
          }
        },
        input);
  } catch (const std::exception& e) {
    std::cerr << "ao2mo: " << e.what() << '\n';
    return 1;
  }
}
