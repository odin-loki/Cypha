#include "cypha/bench/bench_profile.hpp"

#include "cypha/bench/bench_paths.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace cypha::bench {

namespace fs = std::filesystem;

namespace {

const std::unordered_map<std::string, std::string> kCyphalmAliases = {
    {"llm", "profiles/cyphalm_llm.json"},
    {"default", "profiles/cyphalm_llm.json"},
    {"d04", "profiles/cyphalm_d04_gutenberg.json"},
    {"gutenberg", "profiles/cyphalm_d04_gutenberg.json"},
    {"d17", "profiles/cyphalm_d17_wikitext.json"},
    {"wikitext", "profiles/cyphalm_d17_wikitext.json"},
    {"d17_hybrid", "profiles/cyphalm_d17_hybrid.json"},
    {"hybrid", "profiles/cyphalm_d17_hybrid.json"},
};

std::string lower_copy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

fs::path profile_path_from_env() {
    if (const char* raw = std::getenv("CYPHA_BENCH_PROFILE_PATH")) {
        if (*raw != '\0') return fs::path(raw);
    }
    if (const char* raw = std::getenv("CYPHA_BENCH_PROFILE_JSON")) {
        if (*raw != '\0') return fs::path(raw);
    }
    return {};
}

ProfileJson read_json_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open profile: " + path.string());
    ProfileJson j;
    in >> j;
    return j;
}

std::string resolve_regime(const ProfileJson& profile, const std::string& regime) {
    static const std::unordered_set<std::string> kValid = {"tabular", "vision", "regression"};
    if (kValid.count(regime)) return regime;
    const std::string def = profile.value("default_regime", "tabular");
    return kValid.count(def) ? def : "tabular";
}

}  // namespace

ProfileJson load_profile(const fs::path& path) {
    const fs::path env = profile_path_from_env();
    const fs::path everyday = config_dir() / "everyday_profile.json";
    const fs::path fallback = config_dir() / "profiled_medium.json";

    const fs::path* candidates[] = {&path, &env, &everyday, &fallback};
    for (const fs::path* p : candidates) {
        if (p->empty() || !fs::is_regular_file(*p)) continue;
        return read_json_file(*p);
    }
    throw std::runtime_error("No Cypha profile JSON found");
}

bool uses_regimes(const ProfileJson& profile) {
    return profile.contains("regimes") && profile["regimes"].is_object() && !profile["regimes"].empty();
}

std::string select_classification_regime(int input_dim) {
    return input_dim >= kVisionInputDimThreshold ? "vision" : "tabular";
}

ProfileJson regime_params(const ProfileJson& profile, const std::string& regime) {
    const std::string r = resolve_regime(profile, regime);
    if (uses_regimes(profile)) {
        const auto& regimes = profile["regimes"];
        if (regimes.contains(r)) return regimes[r];
        const std::string def = profile.value("default_regime", "tabular");
        if (regimes.contains(def)) return regimes[def];
        return ProfileJson::object();
    }
    if (r == "regression") return profile.value("regression_difregressor", ProfileJson::object());
    return profile.value("classification_cyphadif", ProfileJson::object());
}

ProfileJson architecture_params(const ProfileJson& profile, const std::string* regime) {
    ProfileJson arch = profile.value("architecture", ProfileJson::object());
    if (regime != nullptr) {
        const ProfileJson overrides = regime_params(profile, *regime);
        for (const char* key : {"replay_ratio", "ood_sigma"}) {
            if (overrides.contains(key)) arch[key] = overrides[key];
        }
    }
    return arch;
}

ProfileJson classification_params(const ProfileJson* profile, const std::string* regime) {
    ProfileJson p = profile ? *profile : load_profile();
    if (!uses_regimes(p)) return p.value("classification_cyphadif", ProfileJson::object());
    std::string r = regime ? *regime : p.value("default_regime", "tabular");
    if (r != "tabular" && r != "vision") r = "tabular";
    return regime_params(p, r);
}

ProfileJson regression_params(const ProfileJson* profile) {
    ProfileJson p = profile ? *profile : load_profile();
    return regime_params(p, "regression");
}

ProfileJson cyphalm_params(const ProfileJson* profile) {
    ProfileJson p = profile ? *profile : load_profile();
    return p.value("cyphalm", ProfileJson::object());
}

ProfileJson load_profiles_index() {
    const fs::path path = config_dir() / "profiles_index.json";
    if (!fs::is_regular_file(path)) return ProfileJson::object();
    return read_json_file(path);
}

fs::path resolve_cyphalm_profile_path(const std::string& name) {
    std::string key = name;
    if (key.empty()) {
        if (const char* env = std::getenv("CYPHALM_PROFILE")) key = env;
        else if (const char* env = std::getenv("CYPHA_LM_PROFILE")) key = env;
    }
    if (!key.empty()) {
        const std::string lk = lower_copy(key);
        const auto it = kCyphalmAliases.find(lk);
        if (it != kCyphalmAliases.end()) return config_dir() / it->second;
        fs::path p(key);
        if (fs::is_regular_file(p)) return p;
        if (fs::is_regular_file(config_dir() / key)) return config_dir() / key;
        if (fs::is_regular_file(profiles_dir() / key)) return profiles_dir() / key;
    }
    const fs::path legacy = config_dir() / "cyphalm_profile.json";
    if (fs::is_regular_file(legacy)) return legacy;
    return config_dir() / kCyphalmAliases.at("llm");
}

ProfileJson load_cyphalm_profile_file(const fs::path& path) {
    const fs::path p = path.empty() ? resolve_cyphalm_profile_path() : path;
    if (!fs::is_regular_file(p)) return ProfileJson::object();
    ProfileJson data = read_json_file(p);
    ProfileJson out = ProfileJson::object();
    for (auto it = data.begin(); it != data.end(); ++it) {
        const std::string& key = it.key();
        if (!key.empty() && key.front() == '_') continue;
        out[it.key()] = it.value();
    }
    return out;
}

}  // namespace cypha::bench
