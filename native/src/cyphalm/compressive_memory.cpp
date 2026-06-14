#include "cypha/cyphalm/compressive_memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

namespace cypha::cyphalm {

CompressiveMemory::CompressiveMemory(std::uint32_t slot_dim, std::uint32_t max_slots, double kappa0,
                                     double alpha0, double beta0, std::uint32_t /*seed*/)
    : slot_dim_(slot_dim),
      max_slots_(max_slots),
      kappa0_(kappa0),
      alpha0_(alpha0),
      beta0_(beta0) {}

void CompressiveMemory::reset() { slots_.clear(); }

double CompressiveMemory::prior_log_prob(const double* x, std::uint32_t dim) const {
    double lp = 0.0;
    for (std::uint32_t d = 0; d < dim; ++d) {
        const double v = x[d];
        const double var = beta0_ * (kappa0_ + 1.0) / (kappa0_ * std::max(alpha0_ - 1.0, 1e-6));
        lp += -0.5 * std::log(2.0 * 3.14159265358979323846 * var) - 0.5 * v * v / var;
    }
    return lp;
}

double CompressiveMemory::slot_log_prob(const Slot& slot, const double* x, std::uint32_t dim) const {
    if (slot.count == 0) return prior_log_prob(x, dim);
    double lp = 0.0;
    for (std::uint32_t d = 0; d < dim; ++d) {
        const double mu = slot.mean[d];
        const double v = x[d];
        const double eff_kappa = slot.kappa + static_cast<double>(slot.count);
        const double var =
            slot.beta * (eff_kappa + 1.0) / (eff_kappa * std::max(slot.alpha - 1.0, 1e-6));
        lp += -0.5 * std::log(2.0 * 3.14159265358979323846 * var) -
              0.5 * (v - mu) * (v - mu) / var;
    }
    return lp;
}

std::vector<double> CompressiveMemory::softmax_llr(const std::vector<double>& llrs) {
    if (llrs.empty()) return {};
    double m = -std::numeric_limits<double>::infinity();
    for (double z : llrs) m = std::max(m, z);
    std::vector<double> probs(llrs.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < llrs.size(); ++i) {
        probs[i] = std::exp(llrs[i] - m);
        sum += probs[i];
    }
    if (sum <= 0.0) {
        const double u = 1.0 / static_cast<double>(llrs.size());
        return std::vector<double>(llrs.size(), u);
    }
    for (double& p : probs) p /= sum;
    return probs;
}

void CompressiveMemory::maybe_store(std::uint32_t token_index, const double* pooled,
                                    std::uint32_t dim) {
    maybe_store_priority(token_index, pooled, dim, 1.0);
}

void CompressiveMemory::maybe_store_priority(std::uint32_t token_index, const double* pooled,
                                             std::uint32_t dim, double priority) {
    if (compress_interval_ == 0) return;
    if (token_index == 0 || token_index % compress_interval_ != 0) return;
    if (dim != slot_dim_) return;

    const double pri = std::max(priority, 1e-6);
    std::size_t idx = 0;
    if (priority_replay_ && slots_.size() >= max_slots_) {
        idx = 0;
        double min_pri = slots_[0].priority;
        for (std::size_t i = 1; i < slots_.size(); ++i) {
            if (slots_[i].priority < min_pri) {
                min_pri = slots_[i].priority;
                idx = i;
            }
        }
        if (pri <= min_pri) {
            return;
        }
    } else if (slots_.size() < max_slots_) {
        Slot s;
        s.mean.assign(dim, 0.0);
        s.kappa = kappa0_;
        s.alpha = alpha0_;
        s.beta = beta0_;
        s.count = 0;
        s.priority = pri;
        slots_.push_back(std::move(s));
        idx = slots_.size() - 1;
    } else {
        idx = slots_.size() - 1;
    }

    Slot& slot = slots_[idx];
    slot.priority = std::max(slot.priority, pri);
    if (slot.count == 0) {
        slot.mean.assign(pooled, pooled + dim);
        slot.count = 1;
        return;
    }
    const double n = static_cast<double>(slot.count);
    for (std::uint32_t d = 0; d < dim; ++d)
        slot.mean[d] = (slot.mean[d] * n + pooled[d]) / (n + 1.0);
    slot.count += 1;
    slot.kappa += 1.0;
}

std::vector<double> CompressiveMemory::retrieve(const double* query, std::uint32_t query_len) const {
    std::vector<double> bias(slot_dim_, 0.0);
    if (slots_.empty() || query_len == 0) return bias;

    const std::uint32_t dim = std::min(query_len, slot_dim_);
    const double prior_lp = prior_log_prob(query, dim);
    std::vector<double> llrs;
    llrs.reserve(slots_.size());
    for (const Slot& slot : slots_) {
        const double lp = slot_log_prob(slot, query, dim);
        llrs.push_back(lp - prior_lp);
    }
    std::vector<double> scores = llrs;
    if (priority_replay_) {
        for (std::size_t s = 0; s < slots_.size(); ++s) {
            scores[s] += std::log(slots_[s].priority + 1e-6);
        }
    }
    const std::vector<double> weights = softmax_llr(scores);
    for (std::size_t s = 0; s < slots_.size(); ++s) {
        for (std::uint32_t d = 0; d < slot_dim_; ++d) bias[d] += weights[s] * slots_[s].mean[d];
    }
    return bias;
}

}  // namespace cypha::cyphalm
