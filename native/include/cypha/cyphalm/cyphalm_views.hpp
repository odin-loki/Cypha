#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cypha::cyphalm {

/// One training segment under a multi-view schedule (mirrors ``cypha_views.runner``).
struct ViewTrainSegment {
    int macro_index = 0;
    std::string view_name;
    std::vector<int> ids;
    bool reset_before = false;
};

std::vector<int> view_identity(const std::vector<int>& ids);
std::vector<int> view_reverse(const std::vector<int>& ids);
std::vector<int> view_rotate_start(const std::vector<int>& ids, int offset);

std::vector<std::vector<int>> split_blocks(const std::vector<int>& ids, int block_size);
std::vector<std::vector<int>> block_shuffle_blocks(const std::vector<int>& ids, int block_size,
                                                   std::uint64_t seed);

/// Resolve preset name to ordered view names (``same_order`` repeats ``forward`` *train_epochs*).
std::vector<std::string> resolve_view_schedule(const std::string& name, int train_epochs);

/// Expand token ids into view segments for training.
std::vector<ViewTrainSegment> iter_view_segments(const std::vector<int>& ids,
                                                 const std::vector<std::string>& view_names,
                                                 int block_size, std::uint64_t seed);

}  // namespace cypha::cyphalm
