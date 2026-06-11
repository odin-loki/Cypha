#pragma once

#include <filesystem>
#include <vector>

namespace cypha::bench {

std::filesystem::path figures_dir();

/// Emit figure data JSON, PNG bar charts, and optional CSV under ``cypha_bench/report/figures/``.
std::vector<std::filesystem::path> generate_figure_data();

}  // namespace cypha::bench
