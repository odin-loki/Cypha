#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cypha::cyphalm {

/// Per-view vector concatenated to GRIA input (``ViewEmbedding``).
class ViewEmbedding {
 public:
    ViewEmbedding(int n_slots, int d_view, std::uint64_t seed, bool learnable);

    int slot_for_view(const std::string& view_name) const;
    std::vector<double> forward(int slot) const;
    void update(int slot, const double* grad_view, int grad_len, double lr);

    const std::vector<double>& table() const { return table_; }
    void set_table(std::vector<double> t) { table_ = std::move(t); }

 private:
    int n_slots_;
    int d_view_;
    bool learnable_;
    std::vector<double> table_;  // row-major [n_slots, d_view]
};

}  // namespace cypha::cyphalm
