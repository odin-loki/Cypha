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
        if (fs::is_directory(cur / "parity_fixtures")) return cur;
        if (fs::is_directory(cur / "cypha_bench") && fs::is_directory(cur / "native")) return cur;
        cur = cur.parent_path();
    }
    return repo_root_from_source();
}

fs::path repo_root() { return find_repo_root(fs::current_path()); }

fs::path bench_root() { return repo_root() / "cypha_bench"; }

fs::path config_dir() { return bench_root() / "config"; }

fs::path profiles_dir() { return config_dir() / "profiles"; }

fs::path tables_dir() { return bench_root() / "report" / "tables"; }

fs::path data_dir() { return bench_root() / "data"; }

fs::path config_file(const std::string& rel_path) { return config_dir() / rel_path; }

int bench_scale(int default_value, int fast_value) {
    if (const char* raw = std::getenv("CYPHA_BENCH_FAST")) {
        if (*raw == '1' || (raw[0] == 't' || raw[0] == 'T')) {
            if (fast_value >= 0) return fast_value;
            return std::max(default_value / 5, 1);
        }
    }
    return default_value;
}

}  // namespace cypha::bench
