#pragma once

#include <cstdint>
#include <vector>

namespace cypha::cyphalm {

/// CyphaDIF-style compressive memory: pool SSM state every N tokens, LLR-weighted retrieve.
class CompressiveMemory {
public:
    CompressiveMemory(std::uint32_t slot_dim, std::uint32_t max_slots, double kappa0, double alpha0,
                      double beta0, std::uint32_t seed);

    std::uint32_t slot_dim() const { return slot_dim_; }
    std::uint32_t num_slots() const { return static_cast<std::uint32_t>(slots_.size()); }

    void maybe_store(std::uint32_t token_index, const double* pooled, std::uint32_t dim);
    /// H13: store with TD-error priority; evicts lowest-priority slot when full.
    void maybe_store_priority(std::uint32_t token_index, const double* pooled, std::uint32_t dim,
                              double priority);
    std::vector<double> retrieve(const double* query, std::uint32_t query_len) const;
    /// Softmax attention over slot means (dot-product). Writes ``slot_dim`` into ``out``.
    /// Returns false if no filled slots.
    bool soft_attend(const double* query, std::uint32_t query_len, double* out) const;
    /// Copy slot ``i`` mean into ``out`` (length ``slot_dim``). Returns false if OOB/empty.
    bool slot_mean(std::uint32_t i, double* out) const;
    void reset();

    void set_compress_interval(std::uint32_t n) { compress_interval_ = n; }
    std::uint32_t compress_interval() const { return compress_interval_; }
    void set_priority_replay(bool on) { priority_replay_ = on; }
    bool priority_replay() const { return priority_replay_; }
    std::uint32_t replay_slot_count() const { return static_cast<std::uint32_t>(slots_.size()); }

private:
    struct Slot {
        std::vector<double> mean;
        double kappa = 1.0;
        double alpha = 2.0;
        double beta = 1.0;
        std::uint32_t count = 0;
        double priority = 1.0;
    };

    std::uint32_t slot_dim_;
    std::uint32_t max_slots_;
    std::uint32_t compress_interval_ = 64;
    bool priority_replay_ = false;
    double kappa0_;
    double alpha0_;
    double beta0_;

    std::vector<Slot> slots_;

    double prior_log_prob(const double* x, std::uint32_t dim) const;
    double slot_log_prob(const Slot& slot, const double* x, std::uint32_t dim) const;
    static std::vector<double> softmax_llr(const std::vector<double>& llrs);
};

}  // namespace cypha::cyphalm
