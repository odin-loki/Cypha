#include "cypha/cyphalm/hp_backend.hpp"

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/infinigram.hpp"
#include "hp/shard_merge.hpp"  // Predictor::reset_stream_state

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>

namespace cypha::cyphalm {

namespace {

constexpr double kLogEps = 1e-300;
constexpr double kLog2 = 0.6931471805599453;

double bit_log_prob(int p12, int bit) {
    const double p1 = static_cast<double>(p12) / 4096.0;
    const double p = bit ? p1 : (1.0 - p1);
    return std::log(std::max(p, kLogEps));
}

/// Ensemble members score on worker threads unless CYPHA_HP_ENSEMBLE_THREADS=0.
bool ensemble_threads_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("CYPHA_HP_ENSEMBLE_THREADS");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}

bool use_legacy_byte_log_probs() {
    const char* v = std::getenv("CYPHA_HP_LEGACY_BYTE_LOGPROBS");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

int greedy_bit(int p12) {
    const double p1 = static_cast<double>(p12) / 4096.0;
    const double p0 = 1.0 - p1;
    return (p1 >= p0) ? 1 : 0;
}

int sample_bit(int p12, double temperature, double (*rng01)()) {
    if (temperature <= 1e-6 || rng01 == nullptr) {
        return greedy_bit(p12);
    }
    const double p1 = static_cast<double>(p12) / 4096.0;
    const double p0 = 1.0 - p1;
    const double log_p0 = std::log(std::max(p0, kLogEps)) / temperature;
    const double log_p1 = std::log(std::max(p1, kLogEps)) / temperature;
    const double mx = std::max(log_p0, log_p1);
    const double w0 = std::exp(log_p0 - mx);
    const double w1 = std::exp(log_p1 - mx);
    const double r = rng01();
    return (r < w0 / (w0 + w1 + kLogEps)) ? 0 : 1;
}

/// Turns learning off for the duration of a scoring call when frozen scoring
/// is on, and restores the previous setting.
class ScoringScope {
 public:
    ScoringScope(hp::Predictor& p, bool frozen) : p_(p), prev_(p.learning()), active_(frozen) {
        if (active_) p_.set_learning(false);
    }
    ~ScoringScope() {
        if (active_) p_.set_learning(prev_);
    }
    ScoringScope(const ScoringScope&) = delete;
    ScoringScope& operator=(const ScoringScope&) = delete;

 private:
    hp::Predictor& p_;
    bool prev_;
    bool active_;
};

}  // namespace

hp::Config hp_config_from_cyphalm(int table_bits, int mixer_lr, bool gria) {
    hp::Config cfg;
    cfg.table_bits = table_bits;
    cfg.mixer_lr = mixer_lr;
    cfg.gria = gria;
    cfg.normalize();
    return cfg;
}

hp::Config hp_config_from_cyphalm(const CyphaLMConfig& c) {
    hp::Config cfg = hp_config_from_cyphalm(hp_effective_table_bits(c), c.hp_mixer_lr, c.hp_gria);
    cfg.cm_drop = c.hp_cm_drop;
    cfg.cm_bits_cap = c.hp_cm_bits_cap;
    cfg.gate_drop = c.hp_gate_drop;
    cfg.mixer_skip = c.hp_mixer_skip;
    cfg.match_bits_cap = c.hp_match_bits_cap;
    cfg.pool_slots = c.hp_pool_slots;
    cfg.pool_bits_cap = c.hp_pool_bits_cap;
    cfg.hebb_bits_cap = c.hp_hebb_bits_cap;
    cfg.match_drop = c.hp_match_drop;
    return cfg;
}

HpSequenceBackend::HpSequenceBackend(hp::Config cfg)
    : cfg_(cfg), pred_(std::make_unique<hp::Predictor>(cfg)) {}

double HpSequenceBackend::byte_log_prob_on_pred_(std::uint8_t byte) const {
    ScoringScope scoring(*pred_, frozen_scoring_);
    hp::PredictorUndoStack undo;
    hp::UndoFrame& frame = undo.push_frame();
    double log_p = 0.0;
    {
        hp::UndoRecorderScope scope(frame);
        log_p = byte_log_prob(*pred_, static_cast<int>(byte));
    }
    undo.pop_frame(*pred_);
    return log_p;
}

void HpSequenceBackend::compact_for_serve() {
    serve_compact_ = true;
}

void HpSequenceBackend::prune_cold_slots(int min_total) {
    if (min_total > 0) {
        pred_->prune_cold_hash_slots(min_total);
    }
}

void HpSequenceBackend::reset() {
    pred_ = std::make_unique<hp::Predictor>(cfg_);
    log_probs_buf_.clear();
    members_.clear();
    self_weight_ = 1.0;
    last_valid_ = ig_valid_ = false;
}

std::unique_ptr<hp::Predictor> HpSequenceBackend::predictor_snapshot() const {
    auto snap = std::make_unique<hp::Predictor>(cfg_);
    snap->copy_state_from(*pred_);
    return snap;
}

std::vector<double> HpSequenceBackend::byte_log_probs_bit_tree(hp::Predictor& pred, int vocab_size) {
    const int n = std::max(1, std::min(vocab_size, 256));
    std::vector<double> out(static_cast<std::size_t>(n),
                            std::log(1.0 / static_cast<double>(n)));
    hp::PredictorUndoStack undo;
    expand_bit_tree_dfs(n, 0, 0, 0.0, pred, undo, out);
    return out;
}

void HpSequenceBackend::consume_byte_on(hp::Predictor& pred, std::uint8_t byte) {
    for (int i = 7; i >= 0; --i) {
        const int bit = (static_cast<int>(byte) >> i) & 1;
        (void)pred.predict();
        pred.update(bit);
    }
}

double HpSequenceBackend::byte_log_prob(hp::Predictor& snap, int byte) {
    double log_p = 0.0;
    for (int i = 7; i >= 0; --i) {
        const int p12 = snap.predict();
        const int bit = (byte >> i) & 1;
        log_p += bit_log_prob(p12, bit);
        snap.update(bit);
    }
    return log_p;
}

bool HpSequenceBackend::branch_reaches_vocab(int vocab_size, int prefix, int depth, int bit) {
    const int n = std::max(1, std::min(vocab_size, 256));
    const int next_prefix = (prefix << 1) | bit;
    const int remaining = 7 - depth;
    const int lo = next_prefix << remaining;
    if (lo >= n) {
        return false;
    }
    const int hi = ((next_prefix + 1) << remaining) - 1;
    return hi >= 0;
}

void HpSequenceBackend::expand_bit_tree_dfs(int vocab_size, int depth, int prefix,
                                            double log_p_nats, hp::Predictor& node,
                                            hp::PredictorUndoStack& undo,
                                            std::vector<double>& out_log_nats) {
    if (depth == 8) {
        if (prefix >= 0 && prefix < static_cast<int>(out_log_nats.size())) {
            out_log_nats[static_cast<std::size_t>(prefix)] = log_p_nats;
        }
        return;
    }
    const bool take0 = branch_reaches_vocab(vocab_size, prefix, depth, 0);
    const bool take1 = branch_reaches_vocab(vocab_size, prefix, depth, 1);
    if (!take0 && !take1) {
        return;
    }
    // One predict() gives both children's bit probability. update() also reads
    // predict()'s scratch (mixer inputs, layer-1 outputs), which the bit-0 subtree
    // overwrites, so the bit-1 child re-predicts before its update. Leaves
    // (depth 7) need no update at all: 382 predicts + 254 updates per call
    // instead of 510 + 510, with identical log-probs.
    hp::UndoFrame& pframe = undo.push_frame();
    int p12 = 0;
    {
        hp::UndoRecorderScope scope(pframe);
        p12 = node.predict();
    }
    for (int bit = 0; bit <= 1; ++bit) {
        if (!(bit ? take1 : take0)) {
            continue;
        }
        const double child_log = log_p_nats + bit_log_prob(p12, bit);
        const int next_prefix = (prefix << 1) | bit;
        if (depth == 7) {
            // Leaf: the byte's probability is complete; no state to advance.
            expand_bit_tree_dfs(vocab_size, 8, next_prefix, child_log, node, undo, out_log_nats);
            continue;
        }
        hp::UndoFrame& frame = undo.push_frame();
        {
            hp::UndoRecorderScope scope(frame);
            if (bit == 1 && take0) {
                (void)node.predict();
            }
            node.update(bit);
            expand_bit_tree_dfs(vocab_size, depth + 1, next_prefix, child_log, node, undo,
                                out_log_nats);
        }
        undo.pop_frame(node);
    }
    undo.pop_frame(node);
}

std::vector<double> HpSequenceBackend::next_byte_log_probs_bit_tree(int vocab_size) {
    ScoringScope scoring(*pred_, frozen_scoring_);
    const int n = std::max(1, std::min(vocab_size, 256));
    if (log_probs_buf_.size() != static_cast<std::size_t>(n)) {
        log_probs_buf_.assign(static_cast<std::size_t>(n),
                              std::log(1.0 / static_cast<double>(n)));
    } else {
        const double uniform = std::log(1.0 / static_cast<double>(n));
        for (double& v : log_probs_buf_) {
            v = uniform;
        }
    }
    hp::PredictorUndoStack undo;
    expand_bit_tree_dfs(n, 0, 0, 0.0, *pred_, undo, log_probs_buf_);
    return std::vector<double>(log_probs_buf_.begin(),
                               log_probs_buf_.begin() + static_cast<std::size_t>(n));
}

std::vector<double> HpSequenceBackend::next_byte_log_probs_assign_reuse(int vocab_size) const {
    const int n = std::max(1, std::min(vocab_size, 256));
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    hp::Predictor scratch(cfg_);
    for (int b = 0; b < n; ++b) {
        scratch.copy_state_from(*pred_);
        out[static_cast<std::size_t>(b)] = byte_log_prob(scratch, b);
    }
    return out;
}

std::vector<double> HpSequenceBackend::next_byte_log_probs_legacy(int vocab_size) const {
    const int n = std::max(1, std::min(vocab_size, 256));
    if (log_probs_buf_.size() != static_cast<std::size_t>(n)) {
        log_probs_buf_.resize(static_cast<std::size_t>(n));
    }
    for (int b = 0; b < n; ++b) {
        hp::Predictor snap(cfg_);
        snap.copy_state_from(*pred_);
        log_probs_buf_[static_cast<std::size_t>(b)] = byte_log_prob(snap, b);
    }
    return std::vector<double>(log_probs_buf_.begin(),
                               log_probs_buf_.begin() + static_cast<std::size_t>(n));
}

std::vector<double> HpSequenceBackend::next_byte_log_probs(int vocab_size) {
    std::vector<double> base = scored_log_probs_(vocab_size);
    if (!ig_) return base;
    last_final_ = infinigram_mix_(base);
    ig_valid_ = true;
    return last_final_;
}

void HpSequenceBackend::set_infinigram(std::shared_ptr<const InfiniGram> ig, double eta) {
    ig_ = std::move(ig);
    ig_eta_ = eta;
    ig_w_.assign(kIgBuckets, {0.8, 0.1, 0.1});
    ig_valid_ = false;
}

std::vector<double> HpSequenceBackend::infinigram_weights() const {
    std::vector<double> out;
    for (const auto& w : ig_w_) out.insert(out.end(), w.begin(), w.end());
    return out;
}

void HpSequenceBackend::set_infinigram_weights(const std::vector<double>& w) {
    if (w.size() != static_cast<std::size_t>(kIgBuckets) * 3) {
        throw std::invalid_argument("set_infinigram_weights: need 3 weights per bucket");
    }
    ig_w_.assign(kIgBuckets, {0.0, 0.0, 0.0});
    for (std::size_t i = 0; i < w.size(); ++i) ig_w_[i / 3][i % 3] = w[i];
    ig_valid_ = false;
}

std::vector<double> HpSequenceBackend::infinigram_mix_(const std::vector<double>& base) {
    constexpr std::size_t kCtx = 256;
    std::uint8_t ctx[kCtx];
    const std::size_t len = pred_->recent_bytes(ctx, kCtx);
    const InfiniGram::Result r = ig_->query(ctx, len, static_cast<int>(kCtx));
    InfiniGram::Result rr = r;
    for (int m = r.n; rr.total < 16 && m > 0;) {  // back off to a well-attested suffix
        m /= 2;
        rr = ig_->query(ctx, len, m, m);
    }
    const std::size_t v = base.size();
    for (auto& p : ig_p_) p.assign(v, 0.0);
    double pmax = 0.0;
    for (std::size_t b = 0; b < v; ++b) pmax = std::max(pmax, ig_p_[0][b] = std::exp(base[b]));
    auto fill = [&](const InfiniGram::Result& q, std::vector<double>& out) {
        double tot = 0.0;
        for (std::size_t b = 0; b < v; ++b) tot += q.count[b];
        if (tot <= 0.0) {
            out = ig_p_[0];  // no evidence in-vocabulary: defer to the model
            return;
        }
        for (std::size_t b = 0; b < v; ++b) out[b] = q.count[b] / tot;
    };
    fill(r, ig_p_[1]);
    fill(rr, ig_p_[2]);
    const int nb = r.n == 0 ? 0 : std::min(7, 1 + static_cast<int>(std::log2(static_cast<double>(r.n))));
    const int cb = r.total <= 1 ? 0 : r.total <= 3 ? 1 : r.total <= 15 ? 2 : 3;
    // Model confidence (low / high) x whether the model's and the longest
    // match's top bytes agree.
    std::size_t top_m = 0, top_i = 0;
    for (std::size_t b = 1; b < v; ++b) {
        if (ig_p_[0][b] > ig_p_[0][top_m]) top_m = b;
        if (ig_p_[1][b] > ig_p_[1][top_i]) top_i = b;
    }
    const int hb = (pmax < 0.3 ? 0 : pmax < 0.6 ? 2 : pmax < 0.9 ? 4 : 6) + (top_m == top_i ? 1 : 0);
    ig_bucket_ = (nb * 4 + cb) * 8 + hb;
    const auto& w = ig_w_[static_cast<std::size_t>(ig_bucket_)];
    std::vector<double> out(v);
    for (std::size_t b = 0; b < v; ++b) {
        const double p = w[0] * ig_p_[0][b] + w[1] * ig_p_[1][b] + w[2] * ig_p_[2][b];
        out[b] = std::log(std::max(p, 1e-300));
    }
    return out;
}

std::vector<double> HpSequenceBackend::scored_log_probs_(int vocab_size) {
    if (members_.empty()) {
        return use_legacy_byte_log_probs() ? next_byte_log_probs_legacy(vocab_size)
                                           : next_byte_log_probs_bit_tree(vocab_size);
    }
    // Members score on worker threads while this model scores here. Scoring
    // restores each predictor's state, and each thread records its own undo.
    std::vector<std::vector<double>> member_lp(members_.size());
    std::vector<std::thread> workers;
    const bool threaded = ensemble_threads_enabled();
    for (std::size_t i = 0; i < members_.size(); ++i) {
        auto job = [this, i, vocab_size, &member_lp] {
            member_lp[i] = members_[i].backend->next_byte_log_probs(vocab_size);
        };
        if (threaded) workers.emplace_back(job);
        else job();
    }
    std::vector<double> own = use_legacy_byte_log_probs() ? next_byte_log_probs_legacy(vocab_size)
                                                          : next_byte_log_probs_bit_tree(vocab_size);
    for (auto& w : workers) w.join();
    std::vector<double> mix = mix_with_members_(own, member_lp);
    // Kept until the next byte is consumed: observe_next_byte reuses the mix,
    // and the weight update needs every model's distribution.
    last_own_ = std::move(own);
    last_member_lp_ = std::move(member_lp);
    last_mix_ = mix;
    last_valid_ = true;
    return mix;
}

std::vector<double> HpSequenceBackend::ensemble_weights() const {
    std::vector<double> w{self_weight_};
    for (const auto& m : members_) w.push_back(m.weight);
    return w;
}

void HpSequenceBackend::update_ensemble_weights_(std::uint8_t byte) {
    // d(-log p_mix(y))/dw_i = -(log p_i(y) - E_mix[log p_i]); multiplicative step.
    auto grad = [&](const std::vector<double>& lp) {
        double e = 0.0;
        for (std::size_t b = 0; b < lp.size(); ++b) e += std::exp(last_mix_[b]) * lp[b];
        return lp[byte] - e;
    };
    std::vector<double> w = ensemble_weights();
    w[0] *= std::exp(ens_eta_ * std::clamp(grad(last_own_), -20.0, 20.0));
    for (std::size_t i = 0; i < members_.size(); ++i) {
        w[i + 1] *= std::exp(ens_eta_ * std::clamp(grad(last_member_lp_[i]), -20.0, 20.0));
    }
    double z = 0.0;
    for (double& v : w) z += (v = std::max(v, 1e-4));
    self_weight_ = w[0] / z;
    for (std::size_t i = 0; i < members_.size(); ++i) members_[i].weight = w[i + 1] / z;
}

void HpSequenceBackend::add_ensemble_member(std::unique_ptr<HpSequenceBackend> member, double weight) {
    if (!member) throw std::invalid_argument("add_ensemble_member: null member");
    double total = weight;
    for (const auto& m : members_) total += m.weight;
    if (!(weight > 0.0) || !(total < 1.0)) {
        throw std::invalid_argument("add_ensemble_member: weights must be > 0 and sum below 1");
    }
    member->set_frozen_scoring(frozen_scoring_);
    member->set_learning(pred_->learning());
    members_.push_back(Member{std::move(member), weight});
    self_weight_ = 1.0 - total;
    last_valid_ = ig_valid_ = false;
}

std::vector<hp::Predictor*> HpSequenceBackend::all_predictors() {
    std::vector<hp::Predictor*> out{pred_.get()};
    for (auto& m : members_) {
        const auto sub = m.backend->all_predictors();
        out.insert(out.end(), sub.begin(), sub.end());
    }
    return out;
}

void HpSequenceBackend::reset_stream(bool keep_history) {
    last_valid_ = ig_valid_ = false;
    pred_->reset_stream_state(keep_history);
    for (auto& m : members_) m.backend->reset_stream(keep_history);
}

void HpSequenceBackend::set_serve_adaptation(int num, int den, int skip) {
    pred_->set_serve_adaptation(num, den, skip);
    for (auto& m : members_) m.backend->set_serve_adaptation(num, den, skip);
}

std::vector<double> HpSequenceBackend::mix_with_members_(const std::vector<double>& own,
                                                         const std::vector<std::vector<double>>& member_lp) {
    const double w_self = self_weight_;
    std::vector<double> mix(own.size());
    for (std::size_t b = 0; b < own.size(); ++b) mix[b] = w_self * own[b];
    for (std::size_t i = 0; i < members_.size(); ++i) {
        const std::vector<double>& lp = member_lp[i];
        for (std::size_t b = 0; b < mix.size() && b < lp.size(); ++b) mix[b] += members_[i].weight * lp[b];
    }
    double mx = -std::numeric_limits<double>::infinity();
    for (double v : mix) mx = std::max(mx, v);
    double z = 0.0;
    for (double v : mix) z += std::exp(v - mx);
    const double log_z = mx + std::log(z);
    for (double& v : mix) v -= log_z;
    return mix;
}

std::vector<double> HpSequenceBackend::ensemble_log_probs_(int vocab_size) const {
    // Scoring restores all state; const_cast keeps the const serve API.
    return const_cast<HpSequenceBackend*>(this)->next_byte_log_probs(vocab_size);
}

double HpSequenceBackend::log_prob_byte(std::uint8_t byte) const {
    if (!members_.empty() || ig_) return ensemble_log_probs_(256)[byte];
    return byte_log_prob_on_pred_(byte);
}

std::uint8_t HpSequenceBackend::serve_greedy_next_byte() const {
    if (!members_.empty() || ig_) {
        const auto lp = ensemble_log_probs_(256);
        return static_cast<std::uint8_t>(std::max_element(lp.begin(), lp.end()) - lp.begin());
    }
    ScoringScope scoring(*pred_, frozen_scoring_);
    hp::PredictorUndoStack undo;
    hp::UndoFrame& frame = undo.push_frame();
    int byte = 0;
    {
        hp::UndoRecorderScope scope(frame);
        hp::Predictor& snap = *pred_;
        for (int i = 7; i >= 0; --i) {
            const int p12 = snap.predict();
            const int bit = greedy_bit(p12);
            byte = (byte << 1) | bit;
            snap.update(bit);
        }
    }
    undo.pop_frame(*pred_);
    return static_cast<std::uint8_t>(byte);
}

std::uint8_t HpSequenceBackend::sample_next_byte(double (*rng01)()) const {
    return serve_sample_next_byte(1.0, rng01);
}

std::uint8_t HpSequenceBackend::serve_sample_next_byte(double temperature,
                                                       double (*rng01)()) const {
    if (rng01 == nullptr || temperature <= 1e-6) {
        return serve_greedy_next_byte();
    }
    if (!members_.empty() || ig_) {
        const auto lp = ensemble_log_probs_(256);
        double mx = -std::numeric_limits<double>::infinity();
        for (double v : lp) mx = std::max(mx, v / temperature);
        std::vector<double> w(lp.size());
        double z = 0.0;
        for (std::size_t b = 0; b < lp.size(); ++b) z += (w[b] = std::exp(lp[b] / temperature - mx));
        double r = rng01() * z;
        for (std::size_t b = 0; b < w.size(); ++b) {
            r -= w[b];
            if (r <= 0.0) return static_cast<std::uint8_t>(b);
        }
        return static_cast<std::uint8_t>(w.size() - 1);
    }
    ScoringScope scoring(*pred_, frozen_scoring_);
    hp::PredictorUndoStack undo;
    hp::UndoFrame& frame = undo.push_frame();
    int byte = 0;
    {
        hp::UndoRecorderScope scope(frame);
        hp::Predictor& snap = *pred_;
        for (int i = 7; i >= 0; --i) {
            const int p12 = snap.predict();
            const int bit = sample_bit(p12, temperature, rng01);
            byte = (byte << 1) | bit;
            snap.update(bit);
        }
    }
    undo.pop_frame(*pred_);
    return static_cast<std::uint8_t>(byte);
}

void HpSequenceBackend::consume_byte(std::uint8_t byte) {
    for (int i = 7; i >= 0; --i) {
        const int bit = (static_cast<int>(byte) >> i) & 1;
        (void)pred_->predict();
        pred_->update(bit);
    }
    for (auto& m : members_) m.backend->consume_byte(byte);
    if (last_valid_ && ens_eta_ > 0.0 && pred_->learning()) update_ensemble_weights_(byte);
    if (ig_valid_ && ig_eta_ > 0.0 && pred_->learning() && byte < ig_p_[0].size()) {
        // Exponentiated gradient on the mixture's log loss for this bucket.
        auto& w = ig_w_[static_cast<std::size_t>(ig_bucket_)];
        const double pm = w[0] * ig_p_[0][byte] + w[1] * ig_p_[1][byte] + w[2] * ig_p_[2][byte];
        double z = 0.0;
        for (int e = 0; e < 3; ++e) {
            const double g = std::clamp(ig_eta_ * (ig_p_[e][byte] / std::max(pm, 1e-12) - 1.0), -2.0, 2.0);
            z += (w[static_cast<std::size_t>(e)] = std::max(1e-4, w[static_cast<std::size_t>(e)] * std::exp(g)));
        }
        for (double& x : w) x /= z;
    }
    last_valid_ = ig_valid_ = false;
}

double HpSequenceBackend::observe_next_byte(std::uint8_t next) {
    if (ig_) {
        const double lp = ig_valid_ ? last_final_[next] : next_byte_log_probs(256)[next];
        consume_byte(next);
        return -lp;
    }
    if (!members_.empty()) {
        const double lp = last_valid_ ? last_mix_[next] : next_byte_log_probs(256)[next];
        consume_byte(next);
        return -lp;
    }
    double log_p = 0.0;
    for (int i = 7; i >= 0; --i) {
        const int p12 = pred_->predict();
        const int bit = (static_cast<int>(next) >> i) & 1;
        log_p += bit_log_prob(p12, bit);
        pred_->update(bit);
    }
    return -log_p;
}

double HpSequenceBackend::observe_stream_bits(const std::uint8_t* bytes, std::size_t len) {
    if (bytes == nullptr || len == 0) {
        return 0.0;
    }
    double bits = 0.0;
    for (std::size_t k = 0; k < len; ++k) {
        bits += observe_next_byte(bytes[k]) / kLog2;
    }
    return bits;
}

}  // namespace cypha::cyphalm
