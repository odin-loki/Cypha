#include "cypha/cyphalm/cyphalm_views.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace cypha::cyphalm {

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

std::vector<ViewTrainSegment> iter_view_segments(const std::vector<int>& ids,
                                                 const std::vector<std::string>& view_names,
                                                 int block_size, std::uint64_t seed) {
    std::vector<ViewTrainSegment> out;
    if (ids.size() < 2) return out;

    for (int macro = 0; macro < static_cast<int>(view_names.size()); ++macro) {
        const std::string& view = view_names[static_cast<std::size_t>(macro)];
        const std::uint64_t view_seed = seed + static_cast<std::uint64_t>(macro);

        if (view == "forward") {
            const auto blocks = split_blocks(ids, block_size);
            for (const auto& block : blocks) {
                if (block.size() < 2) continue;
                out.push_back(ViewTrainSegment{macro, view, block, false});
            }
        } else if (view == "block_shuffle") {
            const auto blocks = block_shuffle_blocks(ids, block_size, view_seed);
            for (const auto& block : blocks) {
                if (block.size() < 2) continue;
                out.push_back(ViewTrainSegment{macro, view, block, true});
            }
        } else if (view == "rotated" || view == "rotate_start") {
            const int offset = static_cast<int>(ids.size()) / 4;
            auto segment = view_rotate_start(ids, offset);
            if (segment.size() >= 2) {
                out.push_back(ViewTrainSegment{macro, view, std::move(segment), true});
            }
        } else if (view == "backward" || view == "reverse") {
            auto segment = view_reverse(ids);
            if (segment.size() >= 2) {
                out.push_back(ViewTrainSegment{macro, view, std::move(segment), true});
            }
        } else if (view == "identity") {
            if (ids.size() >= 2) {
                out.push_back(ViewTrainSegment{macro, view, ids, false});
            }
        } else {
            throw std::runtime_error("unknown view name: " + view);
        }
    }
    return out;
}

}  // namespace cypha::cyphalm
