#include "cypha/bench/bench_paths.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>

namespace cypha::bench {

namespace fs = std::filesystem;

namespace {

fs::path repo_root_from_source() {
    const fs::path native = fs::path(__FILE__).parent_path().parent_path().parent_path();
    return native.parent_path();
}

fs::path env_path(const char* key) {
    if (const char* raw = std::getenv(key)) {
        if (*raw != '\0') return fs::path(raw);
    }
    return {};
}

}  // namespace

fs::path find_repo_root(const fs::path& start) {
    if (const fs::path env = env_path("CYPHA_REPO_ROOT"); !env.empty()) {
        return fs::absolute(env);
    }
    fs::path cur = fs::absolute(start);
    for (int i = 0; i < 12 && !cur.empty(); ++i) {
        if (fs::is_directory(cur / "fixtures") && fs::is_directory(cur / "bench") &&
            fs::is_directory(cur / "native")) {
            return cur;
        }
        cur = cur.parent_path();
    }
    return repo_root_from_source();
}

fs::path repo_root() { return find_repo_root(fs::current_path()); }

fs::path bench_root() { return repo_root() / "bench"; }

fs::path results_dir() { return bench_root() / "results"; }

fs::path config_dir() { return bench_root() / "config"; }

fs::path profiles_dir() { return config_dir() / "profiles"; }

fs::path tables_dir() { return bench_root() / "report" / "tables"; }

fs::path data_dir() { return bench_root() / "data"; }

fs::path config_file(const std::string& rel_path) { return config_dir() / rel_path; }

int bench_scale(int default_value, int fast_value) {
    if (bench_env_truthy("CYPHA_BENCH_FAST")) {
        if (fast_value >= 0) return fast_value;
        return std::max(default_value / 5, 1);
    }
    return default_value;
}

bool bench_env_truthy(const char* key) {
    if (const char* raw = std::getenv(key)) {
        if (*raw == '\0') return false;
        if (raw[0] == '1') return true;
        if (raw[0] == 't' || raw[0] == 'T') return true;
        if (raw[0] == 'y' || raw[0] == 'Y') return true;
    }
    return false;
}

bool bench_overnight_enabled() { return bench_env_truthy("CYPHA_BENCH_OVERNIGHT"); }

int bench_full_n_train() {
    int n = 300000;
    if (const char* raw = std::getenv("CYPHA_BENCH_FULL_N_TRAIN")) {
        try {
            n = std::max(1, std::stoi(raw));
        } catch (...) {
        }
    }
    return n;
}

}  // namespace cypha::bench
