#include "cypha/cyphalm/view_embedding.hpp"

#include <cmath>
#include <random>

namespace cypha::cyphalm {

ViewEmbedding::ViewEmbedding(int n_slots, int d_view, std::uint64_t seed, bool learnable)
    : n_slots_(std::max(1, n_slots)), d_view_(d_view), learnable_(learnable) {
    table_.assign(static_cast<std::size_t>(n_slots_ * d_view_), 0.0);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 0.02);
    for (auto& v : table_) v = nd(rng);
}

int ViewEmbedding::slot_for_view(const std::string& view_name) const {
    if (view_name == "forward") return 0 % n_slots_;
    if (view_name == "block_shuffle") return 1 % n_slots_;
    if (view_name == "rotated" || view_name == "rotate_start") return 2 % n_slots_;
    if (view_name == "backward" || view_name == "reverse") return 3 % n_slots_;
    return static_cast<int>(std::hash<std::string>{}(view_name) & 0x7FFFFFFF) % n_slots_;
}

std::vector<double> ViewEmbedding::forward(int slot) const {
    const int idx = ((slot % n_slots_) + n_slots_) % n_slots_;
    std::vector<double> out(static_cast<std::size_t>(d_view_));
    const std::size_t off = static_cast<std::size_t>(idx * d_view_);
    for (int i = 0; i < d_view_; ++i) {
        out[static_cast<std::size_t>(i)] = table_[off + static_cast<std::size_t>(i)];
    }
    return out;
}

void ViewEmbedding::update(int slot, const double* grad_view, int grad_len, double lr) {
    if (!learnable_ || lr <= 0.0 || !grad_view) return;
    const int idx = ((slot % n_slots_) + n_slots_) % n_slots_;
    const std::size_t off = static_cast<std::size_t>(idx * d_view_);
    const int n = std::min(grad_len, d_view_);
    double norm = 0.0;
    for (int i = 0; i < n; ++i) norm += grad_view[i] * grad_view[i];
    norm = std::sqrt(norm);
    double scale = 1.0;
    if (norm > 0.05) scale = 0.05 / norm;
    for (int i = 0; i < n; ++i) {
        table_[off + static_cast<std::size_t>(i)] -= lr * scale * grad_view[i];
    }
}

}  // namespace cypha::cyphalm
