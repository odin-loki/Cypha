#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/infer_cpu.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"

namespace cypha::intelligence {

/// Build a profile by running batch infer on ``native_parity.bin`` rows.
IntelligenceProfiler profile_from_reference_fixture(const std::filesystem::path& repo_root,
                                                    const std::filesystem::path& cypha_path =
                                                        {});

/// Load ``reference.cypha`` + ``native_parity.bin`` F_field for single-row infer tests.
cypha::CyphaInferModel load_reference_model_from_fixture(const std::filesystem::path& repo_root);

/// First ``x`` row from ``native_parity.bin`` (length ``model.d_latent``).
std::vector<double> reference_fixture_first_input(const std::filesystem::path& repo_root);

/// Export full Paper II–III diagnostics JSON (profile matrix, κ, navigation loss, failure modes).
nlohmann::json intelligence_profile_report_json(const IntelligenceProfiler& profiler);

}  // namespace cypha::intelligence
