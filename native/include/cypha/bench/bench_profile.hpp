#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace cypha::bench {

constexpr int kVisionInputDimThreshold = 256;

using ProfileJson = nlohmann::json;

/// Load everyday / fallback profile JSON (mirrors ``load_profile.load_profile``).
ProfileJson load_profile(const std::filesystem::path& path = {});

bool uses_regimes(const ProfileJson& profile);
std::string select_classification_regime(int input_dim);
ProfileJson regime_params(const ProfileJson& profile, const std::string& regime);
ProfileJson architecture_params(const ProfileJson& profile, const std::string* regime = nullptr);
ProfileJson classification_params(const ProfileJson* profile = nullptr,
                                  const std::string* regime = nullptr);
ProfileJson regression_params(const ProfileJson* profile = nullptr);
ProfileJson cyphalm_params(const ProfileJson* profile = nullptr);

/// ``profiles_index.json`` registry.
ProfileJson load_profiles_index();

/// Resolve CyphaLM profile path (mirrors ``load_cyphalm_profile.resolve_cyphalm_profile_path``).
std::filesystem::path resolve_cyphalm_profile_path(const std::string& name = {});

/// Load CyphaLM profile JSON, stripping ``_meta`` keys.
ProfileJson load_cyphalm_profile_file(const std::filesystem::path& path = {});

}  // namespace cypha::bench
