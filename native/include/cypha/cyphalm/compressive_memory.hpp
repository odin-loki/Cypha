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
    std::vector<double> retrieve(const double* query, std::uint32_t query_len) const;
    void reset();

    void set_compress_interval(std::uint32_t n) { compress_interval_ = n; }
    std::uint32_t compress_interval() const { return compress_interval_; }

private:
    struct Slot {
        std::vector<double> mean;
        double kappa = 1.0;
        double alpha = 2.0;
        double beta = 1.0;
        std::uint32_t count = 0;
    };

    std::uint32_t slot_dim_;
    std::uint32_t max_slots_;
    std::uint32_t compress_interval_ = 64;
    double kappa0_;
    double alpha0_;
    double beta0_;

    std::vector<Slot> slots_;

    double prior_log_prob(const double* x, std::uint32_t dim) const;
    double slot_log_prob(const Slot& slot, const double* x, std::uint32_t dim) const;
    static std::vector<double> softmax_llr(const std::vector<double>& llrs);
};

}  // namespace cypha::cyphalm
