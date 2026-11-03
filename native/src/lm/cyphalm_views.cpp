#include "cypha/cyphalm/cyphalm_views.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace cypha::cyphalm {

namespace {

const std::unordered_set<std::string> kBlockSegmentedViews = {"forward", "block_shuffle"};

std::string view_transform_name(const std::string& name) {
    if (name == "forward") return "identity";
    if (name == "block_shuffle") return "block_shuffle";
    if (name == "rotated") return "rotate_start";
    if (name == "backward" || name == "reverse") return "reverse";
    return name;
}

ViewMemoryPolicy memory_policy_for_view(const std::string& name) {
    if (name == "forward") {
        return ViewMemoryPolicy{false, true, true, true};
    }
    if (name == "block_shuffle") {
        return ViewMemoryPolicy{true, false, true, true};
    }
    return ViewMemoryPolicy{true, false, true, true};
}

std::uint64_t transform_seed(const ViewSchedule& schedule, int epoch_idx) {
    return schedule.seed + static_cast<std::uint64_t>(epoch_idx);
}

int rotate_offset(const std::vector<int>& ids) {
    return ids.empty() ? 0 : static_cast<int>(ids.size()) / 4;
}

std::vector<int> apply_transform(const std::string& transform_name, const std::vector<int>& ids,
                                 std::uint64_t seed, int offset) {
    if (transform_name == "identity") return view_identity(ids);
    if (transform_name == "reverse") return view_reverse(ids);
    if (transform_name == "rotate_start") return view_rotate_start(ids, offset);
    if (transform_name == "block_shuffle") {
        std::vector<int> flat;
        const auto blocks = block_shuffle_blocks(ids, 512, seed);
        for (const auto& block : blocks) {
            flat.insert(flat.end(), block.begin(), block.end());
        }
        return flat;
    }
    throw std::runtime_error("unknown view transform: " + transform_name);
}

std::vector<std::vector<int>> block_segments_for_view(const ViewSpec& view_spec,
                                                      const std::vector<int>& ids,
                                                      const ViewSchedule& schedule, int epoch_idx,
                                                      std::optional<int> char_newline_id,
                                                      int block_size) {
    const std::uint64_t seed = transform_seed(schedule, epoch_idx);
    const int offset = rotate_offset(ids);

    if (view_spec.name == "block_shuffle") {
        return block_shuffle_blocks(ids, block_size, seed);
    }

    const auto transformed =
        apply_transform(view_spec.transform_name, ids, seed, offset);
    if (char_newline_id.has_value()) {
        return split_blocks_by_delimiter(transformed, *char_newline_id);
    }
    return split_blocks(transformed, block_size);
}

}  // namespace

std::vector<int> view_identity(const std::vector<int>& ids) {
    return ids;
}

std::vector<int> view_reverse(const std::vector<int>& ids) {
    return std::vector<int>(ids.rbegin(), ids.rend());
}

std::vector<int> view_rotate_start(const std::vector<int>& ids, int offset) {
    if (ids.empty()) return {};
    const int n = static_cast<int>(ids.size());
    int k = offset % n;
    if (k < 0) k += n;
    if (k == 0) return ids;
    std::vector<int> out;
    out.reserve(ids.size());
    out.insert(out.end(), ids.begin() + k, ids.end());
    out.insert(out.end(), ids.begin(), ids.begin() + k);
    return out;
}

std::vector<std::vector<int>> split_blocks(const std::vector<int>& ids, int block_size) {
    std::vector<std::vector<int>> blocks;
    if (ids.empty()) return blocks;
    const int bs = std::max(1, block_size);
    for (int start = 0; start < static_cast<int>(ids.size()); start += bs) {
        const int end = std::min(start + bs, static_cast<int>(ids.size()));
        blocks.emplace_back(ids.begin() + start, ids.begin() + end);
    }
    return blocks;
}

std::vector<std::vector<int>> split_blocks_by_delimiter(const std::vector<int>& ids,
                                                        int delimiter_id) {
    std::vector<std::vector<int>> blocks;
    if (ids.empty()) return blocks;

    std::vector<int> boundaries{0};
    for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
        if (ids[static_cast<std::size_t>(i)] == delimiter_id) {
            const int nxt = i + 1;
            if (nxt < static_cast<int>(ids.size()) &&
                std::find(boundaries.begin(), boundaries.end(), nxt) == boundaries.end()) {
                boundaries.push_back(nxt);
            }
        }
    }

    for (std::size_t i = 0; i < boundaries.size(); ++i) {
        const int start = boundaries[i];
        const int end = (i + 1 < boundaries.size()) ? boundaries[i + 1]
                                                    : static_cast<int>(ids.size());
        if (start < end) {
            blocks.emplace_back(ids.begin() + start, ids.begin() + end);
        }
    }
    return blocks;
}

std::vector<std::vector<int>> block_shuffle_blocks(const std::vector<int>& ids, int block_size,
                                                   std::uint64_t seed) {
    auto blocks = split_blocks(ids, block_size);
    std::vector<int> order(static_cast<std::size_t>(blocks.size()));
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::mt19937_64 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<std::vector<int>> out;
    out.reserve(blocks.size());
    for (int idx : order) {
        out.push_back(std::move(blocks[static_cast<std::size_t>(idx)]));
    }
    return out;
}

std::vector<std::string> resolve_view_schedule(const std::string& name, int train_epochs) {
    if (name == "same_order") {
        const int n = std::max(1, train_epochs);
        return std::vector<std::string>(static_cast<std::size_t>(n), "forward");
    }
    if (name == "schedule_a") return {"forward", "block_shuffle"};
    if (name == "schedule_b") return {"forward", "block_shuffle", "rotated"};
    if (name == "schedule_c") return {"forward", "block_shuffle", "backward"};
    if (name.empty() || name == "forward") return {"forward"};
    return {name};
}

ViewSpec make_view_spec(const std::string& name) {
    ViewSpec spec;
    spec.name = name;
    spec.view_id = name;
    spec.transform_name = view_transform_name(name);
    spec.memory_policy = memory_policy_for_view(name);
    return spec;
}

ViewSchedule resolve_view_schedule_struct(const std::string& name_or_list, std::uint64_t seed,
                                          int train_epochs) {
    ViewSchedule schedule;
    schedule.seed = seed;
    std::vector<std::string> view_names;
    if (name_or_list == "same_order") {
        const int n = std::max(1, train_epochs);
        view_names.assign(static_cast<std::size_t>(n), "forward");
    } else if (name_or_list == "schedule_a") {
        view_names = {"forward", "block_shuffle"};
    } else if (name_or_list == "schedule_b") {
        view_names = {"forward", "block_shuffle", "rotated"};
    } else if (name_or_list == "schedule_c") {
        view_names = {"forward", "block_shuffle", "backward"};
    } else if (name_or_list.empty() || name_or_list == "forward") {
        view_names = {"forward"};
    } else {
        view_names = {name_or_list};
    }
    schedule.views.reserve(view_names.size());
    for (const auto& name : view_names) {
        schedule.views.push_back(make_view_spec(name));
    }
    return schedule;
}

std::vector<ViewTrainSegment> iter_view_segments(const std::vector<int>& ids,
                                                 const std::vector<std::string>& view_names,
                                                 int block_size, std::uint64_t seed) {
    ViewSchedule schedule;
    schedule.seed = seed;
    for (const auto& name : view_names) {
        schedule.views.push_back(make_view_spec(name));
    }
    const auto epochs = iter_view_epochs(ids, schedule, std::nullopt, block_size);
    std::vector<ViewTrainSegment> out;
    out.reserve(epochs.size());
    for (const auto& item : epochs) {
        ViewTrainSegment seg;
        seg.macro_index = item.epoch_idx;
        seg.view_name = item.view_spec.name;
        seg.ids = item.segment_ids;
        seg.reset_before = item.reset_before;
        out.push_back(std::move(seg));
    }
    return out;
}

std::vector<ViewEpochItem> iter_view_epochs(const std::vector<int>& ids,
                                            const ViewSchedule& schedule,
                                            std::optional<int> char_newline_id, int block_size) {
    std::vector<ViewEpochItem> out;
    if (ids.size() < 2) return out;

    const int bs = std::max(1, block_size);
    for (int epoch_idx = 0; epoch_idx < static_cast<int>(schedule.views.size()); ++epoch_idx) {
        const ViewSpec& view_spec = schedule.views[static_cast<std::size_t>(epoch_idx)];

        if (kBlockSegmentedViews.count(view_spec.name) > 0) {
            const auto segments =
                block_segments_for_view(view_spec, ids, schedule, epoch_idx, char_newline_id, bs);
            const bool reset = view_spec.memory_policy.reset_fast;
            for (const auto& segment : segments) {
                if (segment.empty()) continue;
                out.push_back(ViewEpochItem{view_spec, epoch_idx, segment, reset});
            }
            continue;
        }

        const std::uint64_t seed = transform_seed(schedule, epoch_idx);
        const int offset = rotate_offset(ids);
        auto segment = apply_transform(view_spec.transform_name, ids, seed, offset);
        if (!segment.empty()) {
            out.push_back(
                ViewEpochItem{view_spec, epoch_idx, std::move(segment), view_spec.memory_policy.reset_fast});
        }
    }
    return out;
}

}  // namespace cypha::cyphalm
