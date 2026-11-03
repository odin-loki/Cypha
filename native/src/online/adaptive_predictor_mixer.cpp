#include "cypha/online/adaptive_predictor_mixer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cypha::cyphalm {

AdaptivePredictorMixer::AdaptivePredictorMixer(AdaptiveMixerConfig cfg) : cfg_(cfg) {
    if (cfg_.vocab_size <= 0) {
        throw std::runtime_error("AdaptivePredictorMixer: invalid vocab_size");
    }
    cfg_.max_order = std::max(0, cfg_.max_order);
    cfg_.knn_k = std::max(1, cfg_.knn_k);
    cfg_.knn_store = std::max(cfg_.knn_k, cfg_.knn_store);
    cfg_.knn_feat_dim = std::max(8, cfg_.knn_feat_dim);

    slots_.clear();
    slots_.push_back({ExpertKind::Neural, -1});
    for (int o = 0; o <= cfg_.max_order; ++o) {
        slots_.push_back({ExpertKind::Ngram, o});
    }
    if (cfg_.use_match_expert) slots_.push_back({ExpertKind::Match, -1});
    if (cfg_.use_knn_expert) slots_.push_back({ExpertKind::Knn, -1});
    if (cfg_.use_bias_expert) slots_.push_back({ExpertKind::Bias, -1});

    scores_.assign(slots_.size(), 0.0);
    scores_[0] = cfg_.neural_prior;
    tables_.assign(static_cast<std::size_t>(cfg_.max_order + 1), {});
    bias_.assign(static_cast<std::size_t>(cfg_.vocab_size), 0.0);
    expert_lp_.assign(slots_.size(), std::vector<double>(static_cast<std::size_t>(cfg_.vocab_size)));
    weights_.assign(slots_.size(), 0.0);
    mix_p_.assign(static_cast<std::size_t>(cfg_.vocab_size), 0.0);
    knn_feat_scratch_.assign(static_cast<std::size_t>(cfg_.knn_feat_dim), 0.0f);
    history_.clear();
    knn_store_.clear();
}

void AdaptivePredictorMixer::reset() {
    std::fill(scores_.begin(), scores_.end(), 0.0);
    if (!scores_.empty()) scores_[0] = cfg_.neural_prior;
    for (auto& t : tables_) t.clear();
    std::fill(bias_.begin(), bias_.end(), 0.0);
    history_.clear();
    knn_store_.clear();
}

std::uint64_t AdaptivePredictorMixer::context_key(int order) const {
    if (order <= 0 || history_.empty()) return 0;
    const int use = std::min(order, static_cast<int>(history_.size()));
    std::uint64_t key = 0;
    const int start = static_cast<int>(history_.size()) - use;
    for (int i = 0; i < use; ++i) {
        key = key * 257ull + (static_cast<std::uint64_t>(history_[static_cast<std::size_t>(start + i)]) + 1ull);
    }
    key = key * 16ull + static_cast<std::uint64_t>(order);
    return key;
}

double AdaptivePredictorMixer::context_count(int order) const {
    if (order < 0 || order >= static_cast<int>(tables_.size())) return 0.0;
    const std::uint64_t key = context_key(order);
    const auto& table = tables_[static_cast<std::size_t>(order)];
    const auto it = table.find(key);
    if (it == table.end()) return 0.0;
    double total = 0.0;
    for (float c : it->second) total += static_cast<double>(c);
    return total;
}

void AdaptivePredictorMixer::fill_ngram_log_probs(int order, std::vector<double>& out) const {
    out.assign(static_cast<std::size_t>(cfg_.vocab_size), 0.0);
    if (order < 0 || order >= static_cast<int>(tables_.size())) return;
    const std::uint64_t key = context_key(order);
    const auto& table = tables_[static_cast<std::size_t>(order)];
    const auto it = table.find(key);
    const double V = static_cast<double>(cfg_.vocab_size);
    const double alpha = cfg_.laplace;
    double total = 0.0;
    if (it != table.end()) {
        for (float c : it->second) total += static_cast<double>(c);
    }
    const double denom = total + alpha * V;
    for (int k = 0; k < cfg_.vocab_size; ++k) {
        double c = 0.0;
        if (it != table.end() && k < static_cast<int>(it->second.size())) {
            c = static_cast<double>(it->second[static_cast<std::size_t>(k)]);
        }
        out[static_cast<std::size_t>(k)] = std::log((c + alpha) / denom + cfg_.log_eps);
    }
}

bool AdaptivePredictorMixer::find_match(std::uint32_t* next_out) const {
    const int n = static_cast<int>(history_.size());
    if (n < cfg_.match_min_len + 1) return false;
    // Vote across the longest matches found (up to 8 continuations).
    std::vector<int> votes(static_cast<std::size_t>(cfg_.vocab_size), 0);
    int best_len = 0;
    int n_hits = 0;
    const int max_len = std::min(n - 1, 32);
    for (int len = max_len; len >= cfg_.match_min_len; --len) {
        if (best_len > 0 && len < best_len) break;  // only collect at the longest length
        const int suf = n - len;
        for (int i = 0; i + len < n; ++i) {
            bool ok = true;
            for (int j = 0; j < len; ++j) {
                if (history_[static_cast<std::size_t>(i + j)] !=
                    history_[static_cast<std::size_t>(suf + j)]) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            best_len = len;
            const std::uint32_t nxt = history_[static_cast<std::size_t>(i + len)];
            if (nxt < static_cast<std::uint32_t>(cfg_.vocab_size)) {
                votes[static_cast<std::size_t>(nxt)] += 1;
                ++n_hits;
                if (n_hits >= 8) break;
            }
        }
        if (n_hits > 0) break;
    }
    if (n_hits == 0) return false;
    int best_sym = 0;
    int best_v = votes[0];
    for (int k = 1; k < cfg_.vocab_size; ++k) {
        if (votes[static_cast<std::size_t>(k)] > best_v) {
            best_v = votes[static_cast<std::size_t>(k)];
            best_sym = k;
        }
    }
    if (next_out) *next_out = static_cast<std::uint32_t>(best_sym);
    return true;
}

void AdaptivePredictorMixer::fill_match_log_probs(std::vector<double>& out) const {
    out.assign(static_cast<std::size_t>(cfg_.vocab_size), std::log(1.0 / cfg_.vocab_size));
    std::uint32_t best_next = 0;
    if (!find_match(&best_next)) return;

    // Peaked distribution on matched continuation; residual uniform.
    const double peak = 0.85;
    const double rest = (1.0 - peak) / static_cast<double>(cfg_.vocab_size - 1);
    for (int k = 0; k < cfg_.vocab_size; ++k) {
        const double p = (static_cast<std::uint32_t>(k) == best_next) ? peak : rest;
        out[static_cast<std::size_t>(k)] = std::log(p + cfg_.log_eps);
    }
}

void AdaptivePredictorMixer::knn_features(std::vector<float>& out) const {
    out.assign(static_cast<std::size_t>(cfg_.knn_feat_dim), 0.0f);
    // Rolling hashed bag of recent tokens (order-aware via position phases).
    const int n = static_cast<int>(history_.size());
    const int use = std::min(n, 16);
    for (int i = 0; i < use; ++i) {
        const std::uint32_t tok = history_[static_cast<std::size_t>(n - use + i)];
        const std::uint32_t h = tok * 2654435761u + static_cast<std::uint32_t>(i) * 97u;
        const int idx = static_cast<int>(h % static_cast<std::uint32_t>(cfg_.knn_feat_dim));
        out[static_cast<std::size_t>(idx)] += 1.0f;
        const int idx2 = static_cast<int>((h / 7u) % static_cast<std::uint32_t>(cfg_.knn_feat_dim));
        out[static_cast<std::size_t>(idx2)] += 0.5f;
    }
    double norm = 0.0;
    for (float v : out) norm += static_cast<double>(v) * static_cast<double>(v);
    norm = std::sqrt(norm) + 1e-8;
    for (float& v : out) v = static_cast<float>(static_cast<double>(v) / norm);
}

bool AdaptivePredictorMixer::knn_nearest_dist2(float* dist2_out) const {
    if (static_cast<int>(knn_store_.size()) < std::max(4, cfg_.knn_k / 2)) return false;
    std::vector<float> q;
    knn_features(q);
    float best = 1e30f;
    for (const auto& e : knn_store_) {
        float d2 = 0.0f;
        for (int i = 0; i < cfg_.knn_feat_dim; ++i) {
            const float d = q[static_cast<std::size_t>(i)] - e.feat[static_cast<std::size_t>(i)];
            d2 += d * d;
        }
        best = std::min(best, d2);
    }
    if (dist2_out) *dist2_out = best;
    return static_cast<double>(best) <= cfg_.knn_max_dist2;
}

void AdaptivePredictorMixer::fill_knn_log_probs(std::vector<double>& out) const {
    out.assign(static_cast<std::size_t>(cfg_.vocab_size), std::log(1.0 / cfg_.vocab_size));
    float nearest = 0.0f;
    if (!knn_nearest_dist2(&nearest)) return;

    std::vector<float> q;
    knn_features(q);
    struct Hit {
        float dist;
        std::uint32_t next;
    };
    std::vector<Hit> hits;
    hits.reserve(knn_store_.size());
    for (const auto& e : knn_store_) {
        float d2 = 0.0f;
        for (int i = 0; i < cfg_.knn_feat_dim; ++i) {
            const float d = q[static_cast<std::size_t>(i)] - e.feat[static_cast<std::size_t>(i)];
            d2 += d * d;
        }
        hits.push_back({d2, e.next});
    }
    const int k = std::min(cfg_.knn_k, static_cast<int>(hits.size()));
    std::partial_sort(hits.begin(), hits.begin() + k, hits.end(),
                      [](const Hit& a, const Hit& b) { return a.dist < b.dist; });

    std::vector<double> counts(static_cast<std::size_t>(cfg_.vocab_size), cfg_.laplace);
    double total = cfg_.laplace * static_cast<double>(cfg_.vocab_size);
    for (int i = 0; i < k; ++i) {
        const double w = 1.0 / (1e-3 + static_cast<double>(hits[static_cast<std::size_t>(i)].dist));
        counts[hits[static_cast<std::size_t>(i)].next] += w;
        total += w;
    }
    for (int s = 0; s < cfg_.vocab_size; ++s) {
        out[static_cast<std::size_t>(s)] =
            std::log(counts[static_cast<std::size_t>(s)] / total + cfg_.log_eps);
    }
    (void)nearest;
}

void AdaptivePredictorMixer::fill_bias_log_probs(std::vector<double>& out) const {
    out.resize(static_cast<std::size_t>(cfg_.vocab_size));
    double max_b = bias_[0];
    for (double b : bias_) max_b = std::max(max_b, b);
    double z = 0.0;
    for (int k = 0; k < cfg_.vocab_size; ++k) {
        out[static_cast<std::size_t>(k)] = std::exp(bias_[static_cast<std::size_t>(k)] - max_b);
        z += out[static_cast<std::size_t>(k)];
    }
    for (int k = 0; k < cfg_.vocab_size; ++k) {
        out[static_cast<std::size_t>(k)] =
            std::log(out[static_cast<std::size_t>(k)] / z + cfg_.log_eps);
    }
}

bool AdaptivePredictorMixer::expert_active(int expert_idx) const {
    if (expert_idx < 0 || expert_idx >= static_cast<int>(slots_.size())) return false;
    const auto& slot = slots_[static_cast<std::size_t>(expert_idx)];
    switch (slot.kind) {
        case ExpertKind::Neural:
            return true;
        case ExpertKind::Ngram:
            return context_count(slot.order) >= static_cast<double>(cfg_.min_context_count);
        case ExpertKind::Match:
            // Only when a real longest-match continuation exists (not uniform fallback).
            return find_match(nullptr);
        case ExpertKind::Knn:
            return knn_nearest_dist2(nullptr);
        case ExpertKind::Bias:
            return static_cast<int>(history_.size()) >= cfg_.bias_min_steps;
    }
    return false;
}

void AdaptivePredictorMixer::expert_log_probs(int expert_idx, std::vector<double>& out) const {
    const auto& slot = slots_[static_cast<std::size_t>(expert_idx)];
    switch (slot.kind) {
        case ExpertKind::Neural:
            break;
        case ExpertKind::Ngram:
            fill_ngram_log_probs(slot.order, out);
            break;
        case ExpertKind::Match:
            fill_match_log_probs(out);
            break;
        case ExpertKind::Knn:
            fill_knn_log_probs(out);
            break;
        case ExpertKind::Bias:
            fill_bias_log_probs(out);
            break;
    }
}

void AdaptivePredictorMixer::predict(const double* neural_log_probs, std::vector<double>& mix_log_probs) {
    const int V = cfg_.vocab_size;
    const int K = static_cast<int>(scores_.size());

    std::fill(weights_.begin(), weights_.end(), 0.0);
    double max_s = scores_[0];
    for (int i = 0; i < K; ++i) {
        if (expert_active(i)) max_s = std::max(max_s, scores_[static_cast<std::size_t>(i)]);
    }
    double z = 0.0;
    for (int i = 0; i < K; ++i) {
        if (!expert_active(i)) continue;
        const double w = std::exp(scores_[static_cast<std::size_t>(i)] - max_s);
        weights_[static_cast<std::size_t>(i)] = w;
        z += w;
    }
    if (!(z > 0.0)) {
        weights_[0] = 1.0;
        z = 1.0;
    }
    for (int i = 0; i < K; ++i) weights_[static_cast<std::size_t>(i)] /= z;

    // Warm neural safety: keep a floor so cold/noisy experts cannot dominate.
    const double floor = std::clamp(cfg_.neural_weight_floor, 0.0, 0.99);
    if (floor > 0.0 && weights_[0] < floor) {
        const double rest = 1.0 - floor;
        double other = 1.0 - weights_[0];
        if (other > 1e-12) {
            const double scale = rest / other;
            for (int i = 1; i < K; ++i) weights_[static_cast<std::size_t>(i)] *= scale;
        }
        weights_[0] = floor;
    }

    for (int k = 0; k < V; ++k) {
        expert_lp_[0][static_cast<std::size_t>(k)] = neural_log_probs[k];
    }
    for (int i = 1; i < K; ++i) {
        expert_log_probs(i, expert_lp_[static_cast<std::size_t>(i)]);
    }

    std::fill(mix_p_.begin(), mix_p_.end(), 0.0);
    mix_log_probs.resize(static_cast<std::size_t>(V));
    for (int i = 0; i < K; ++i) {
        if (weights_[static_cast<std::size_t>(i)] <= 0.0) continue;
        const auto& lp = expert_lp_[static_cast<std::size_t>(i)];
        double max_lp = lp[0];
        for (int k = 1; k < V; ++k) max_lp = std::max(max_lp, lp[static_cast<std::size_t>(k)]);
        double sum = 0.0;
        for (int k = 0; k < V; ++k) {
            mix_log_probs[static_cast<std::size_t>(k)] =
                std::exp(lp[static_cast<std::size_t>(k)] - max_lp);
            sum += mix_log_probs[static_cast<std::size_t>(k)];
        }
        const double w = weights_[static_cast<std::size_t>(i)];
        for (int k = 0; k < V; ++k) {
            mix_p_[static_cast<std::size_t>(k)] +=
                w * (mix_log_probs[static_cast<std::size_t>(k)] / sum);
        }
    }
    for (int k = 0; k < V; ++k) {
        mix_log_probs[static_cast<std::size_t>(k)] =
            std::log(mix_p_[static_cast<std::size_t>(k)] + cfg_.log_eps);
    }
}

void AdaptivePredictorMixer::observe(std::uint32_t symbol) {
    if (symbol >= static_cast<std::uint32_t>(cfg_.vocab_size)) return;
    const int V = cfg_.vocab_size;
    const int K = static_cast<int>(scores_.size());

    const double p_mix = mix_p_[symbol];
    for (int i = 0; i < K; ++i) {
        // Only experts that participated in the last predict() update scores.
        if (weights_[static_cast<std::size_t>(i)] <= 0.0) continue;
        const auto& lp = expert_lp_[static_cast<std::size_t>(i)];
        double max_lp = lp[0];
        for (int k = 1; k < V; ++k) max_lp = std::max(max_lp, lp[static_cast<std::size_t>(k)]);
        double sum = 0.0;
        double py = 0.0;
        for (int k = 0; k < V; ++k) {
            const double e = std::exp(lp[static_cast<std::size_t>(k)] - max_lp);
            sum += e;
            if (static_cast<std::uint32_t>(k) == symbol) py = e;
        }
        py /= sum;
        scores_[static_cast<std::size_t>(i)] += cfg_.mixer_lr * (py - p_mix);
    }

    // Store kNN entry for current context → observed next symbol.
    if (cfg_.use_knn_expert) {
        KnnEntry e;
        knn_features(e.feat);
        e.next = symbol;
        knn_store_.push_back(std::move(e));
        while (static_cast<int>(knn_store_.size()) > cfg_.knn_store) knn_store_.pop_front();
    }

    for (int order = 0; order <= cfg_.max_order; ++order) {
        const std::uint64_t key = context_key(order);
        auto& counts = tables_[static_cast<std::size_t>(order)][key];
        if (counts.size() < static_cast<std::size_t>(V)) {
            counts.assign(static_cast<std::size_t>(V), 0.0f);
        }
        counts[symbol] += 1.0f;
    }

    if (cfg_.use_bias_expert) {
        // SSE-style: push bias toward residual vs mix.
        for (int k = 0; k < V; ++k) {
            const double target = (static_cast<std::uint32_t>(k) == symbol) ? 1.0 : 0.0;
            bias_[static_cast<std::size_t>(k)] +=
                cfg_.bias_lr * (target - mix_p_[static_cast<std::size_t>(k)]);
        }
    }

    history_.push_back(symbol);
    const std::size_t cap = 4096;
    if (history_.size() > cap) {
        history_.erase(history_.begin(), history_.end() - (cap / 2));
    }
}

}  // namespace cypha::cyphalm
