#pragma once

#include "cypha/bench/bench_report_json.hpp"

#include <filesystem>
#include <optional>

namespace cypha::bench {

/// Render ``figure_data_v1`` JSON to a PNG bar chart (stb_image_write).
std::optional<std::filesystem::path> render_figure_png(const ProfileJson& figure,
                                                       const std::filesystem::path& out_path);

}  // namespace cypha::bench
