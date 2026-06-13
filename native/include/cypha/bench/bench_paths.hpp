#pragma once

#include <filesystem>
#include <string>

namespace cypha::bench {

/// Repo root: walk upward from ``start`` (``fixtures/`` + ``bench/`` + ``native/``), else compile-time path.
std::filesystem::path find_repo_root(const std::filesystem::path& start = std::filesystem::current_path());

std::filesystem::path repo_root();
std::filesystem::path bench_root();
std::filesystem::path config_dir();
std::filesystem::path profiles_dir();
std::filesystem::path tables_dir();
std::filesystem::path data_dir();

std::filesystem::path config_file(const std::string& rel_path);

/// Mirror legacy bench path scaling (honours ``CYPHA_BENCH_FAST``).
int bench_scale(int default_value, int fast_value = -1);

}  // namespace cypha::bench
