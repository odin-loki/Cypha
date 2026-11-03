#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cypha::cyphalm {

/// Per-view rules for fast/slow memory carry (mirrors ``cypha_views.types.MemoryPolicy``).
struct ViewMemoryPolicy {
    bool reset_fast = true;
    bool carry_slow = false;
    bool carry_dif = true;
    bool carry_gria_bias = true;
};

/// One presentation of the corpus (mirrors ``cypha_views.types.ViewSpec``).
struct ViewSpec {
    std::string name;
    std::string view_id;
    std::string transform_name;
    ViewMemoryPolicy memory_policy;
};

/// Ordered views across training epochs (mirrors ``cypha_views.types.ViewSchedule``).
struct ViewSchedule {
    std::vector<ViewSpec> views;
    std::uint64_t seed = 42;
};

/// One training segment under a multi-view schedule.
struct ViewTrainSegment {
    int macro_index = 0;
    std::string view_name;
    std::vector<int> ids;
    bool reset_before = false;
};

/// Yield item from ``iter_view_epochs`` (view spec + segment metadata).
struct ViewEpochItem {
    ViewSpec view_spec;
    int epoch_idx = 0;
    std::vector<int> segment_ids;
    bool reset_before = false;
};

std::vector<int> view_identity(const std::vector<int>& ids);
std::vector<int> view_reverse(const std::vector<int>& ids);
std::vector<int> view_rotate_start(const std::vector<int>& ids, int offset);

std::vector<std::vector<int>> split_blocks(const std::vector<int>& ids, int block_size);
std::vector<std::vector<int>> split_blocks_by_delimiter(const std::vector<int>& ids,
                                                        int delimiter_id);
std::vector<std::vector<int>> block_shuffle_blocks(const std::vector<int>& ids, int block_size,
                                                   std::uint64_t seed);

/// Resolve preset name to ordered view names (``same_order`` repeats ``forward`` *train_epochs*).
std::vector<std::string> resolve_view_schedule(const std::string& name, int train_epochs);

/// Build a full ``ViewSchedule`` with memory policies (mirrors ``cypha_views.schedule.resolve_schedule``).
ViewSchedule resolve_view_schedule_struct(const std::string& name_or_list, std::uint64_t seed,
                                          int train_epochs);

ViewSpec make_view_spec(const std::string& name);

/// Expand token ids into view segments for training (legacy flat API).
std::vector<ViewTrainSegment> iter_view_segments(const std::vector<int>& ids,
                                                 const std::vector<std::string>& view_names,
                                                 int block_size, std::uint64_t seed);

/// Iterator equivalent to Python ``cypha_views.runner.iter_view_epochs``.
std::vector<ViewEpochItem> iter_view_epochs(const std::vector<int>& ids,
                                            const ViewSchedule& schedule,
                                            std::optional<int> char_newline_id = std::nullopt,
                                            int block_size = 512);

}  // namespace cypha::cyphalm
