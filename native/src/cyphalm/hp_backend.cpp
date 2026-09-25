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
    cfg.lr1_scale = c.hp_lr1_scale;
    cfg.mixer_scale = c.hp_mixer_scale;
    cfg.mixer_skip_l1 = c.hp_mixer_skip_l1;
    cfg.extra_cms = c.hp_extra_cms;
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
        served_valid_ = false;
    }
}

void HpSequenceBackend::reset() {
    pred_ = std::make_unique<hp::Predictor>(cfg_);
    log_probs_buf_.clear();
    members_.clear();
    self_weight_ = 1.0;
    last_valid_ = ig_valid_ = nn_valid_ = ss_valid_ = served_valid_ = false;
    serve_rate_ = {0, 0, 0};  // the new predictor runs at its trained rates
    ++settings_epoch_;
    // The attached stages restart with it: start mixing weights, experts at
    // their initial state (no history to prime from), an empty session.
    ig_w_ = ig_w0_;
    reset_neural_weights_();
    prime_neural_();
    set_session_cache(ss_on_, ss_eta_, ss_window_);
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
                                            std::vector<double>& out_log_nats,
                                            double prune_log) {
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
        if (depth < 7 && child_log < prune_log) {
            // Pruned: this subtree's mass is spread evenly over its bytes, so
            // the distribution stays normalised without expanding it.
            const int shift = 7 - depth;
            const int first = next_prefix << shift;
            const int last = std::min(static_cast<int>(out_log_nats.size()), (next_prefix + 1) << shift);
            if (first < last) {
                const double each = child_log - std::log(static_cast<double>(last - first));
                for (int b = first; b < last; ++b) out_log_nats[static_cast<std::size_t>(b)] = each;
            }
            continue;
        }
        if (depth == 7) {
            // Leaf: the byte's probability is complete; no state to advance.
            expand_bit_tree_dfs(vocab_size, 8, next_prefix, child_log, node, undo, out_log_nats, prune_log);
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
                                out_log_nats, prune_log);
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
    expand_bit_tree_dfs(n, 0, 0, 0.0, *pred_, undo, log_probs_buf_, prune_log_);
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
    std::vector<double> out = scored_log_probs_(vocab_size);
    if (!mixed_()) return out;
    if (ig_) {
        out = infinigram_mix_(out);
        ig_valid_ = true;
    }
    if (ss_on_ && out.size() == 256) {
        out = session_mix_(out);
        ss_valid_ = true;
    }
    if (!nn_.empty() && out.size() == 256) {
        out = neural_mix_(out);
        nn_valid_ = true;
    }
    // Served until the next byte: every stage ran (session and neural need
    // the full 256-byte vocabulary).
    served_ = out;
    served_valid_ = out.size() == 256 || (!ss_on_ && nn_.empty());
    served_epoch_ = settings_epoch_;
    return out;
}

void HpSequenceBackend::set_session_cache(bool on, double eta, std::size_t window) {
    ss_on_ = on;
    ss_eta_ = eta;
    ss_window_ = window;
    ss_w_.assign(kSsBuckets, 0.97);
    ss_hist_.clear();
    ss_len_ = 0;
    ss_ig_.reset();
    ss_built_ = ss_indexed_ = 0;
    ss_valid_ = served_valid_ = false;
}

void HpSequenceBackend::truncate_session(std::size_t n) {
    if (n >= ss_len_) return;
    const std::size_t front = ss_len_ - ss_hist_.size();  // bytes already out of the window
    ss_hist_.resize(n > front ? n - front : 0);
    ss_len_ = n;
    if (ss_built_ > n) {  // the index saw bytes that are gone: rebuild later
        ss_ig_.reset();
        ss_built_ = ss_indexed_ = 0;
    }
    ss_valid_ = served_valid_ = false;
}

void HpSequenceBackend::session_grow_() {
    // First at 1 KiB, then when the new bytes equal the indexed length
    // (doubling), at most window / 16 apart.
    const std::size_t cap = ss_window_ > 0 ? std::max<std::size_t>(ss_window_ / 16, 1) : std::size_t{1} << 16;
    const std::size_t step = ss_indexed_ == 0 ? 1024 : std::min(ss_indexed_, cap);
    if (ss_len_ - ss_built_ < step) return;
    if (ss_window_ > 0 && ss_hist_.size() > ss_window_) {
        ss_hist_.erase(ss_hist_.begin(), ss_hist_.end() - static_cast<std::ptrdiff_t>(ss_window_));
    }
    ss_ig_ = std::make_shared<const InfiniGram>(ss_hist_.data(), ss_hist_.size());
    ss_built_ = ss_len_;
    ss_indexed_ = ss_hist_.size();
    ++ss_builds_;
}

std::vector<double> HpSequenceBackend::session_mix_(const std::vector<double>& base) {
    const std::size_t h = ss_hist_.size();
    ss_bucket_ = -1;
    if (!ss_ig_) return base;
    constexpr std::size_t kCtx = 256;
    const std::size_t len = std::min(kCtx, h);
    const InfiniGram::Result r = ss_ig_->query(ss_hist_.data() + h - len, len, static_cast<int>(kCtx));
    // Only long repeats: shorter ones are what hp's own match models and
    // online learning already capture.
    if (r.n < 16 || r.total == 0) return base;
    const int nb = std::min(7, static_cast<int>(std::log2(static_cast<double>(r.n))));
    const int cb = r.total <= 1 ? 0 : r.total <= 3 ? 1 : r.total <= 15 ? 2 : 3;
    ss_bucket_ = nb * 4 + cb;
    ss_pin_.resize(256);
    ss_p_.resize(256);
    for (std::size_t b = 0; b < 256; ++b) {
        ss_pin_[b] = std::exp(base[b]);
        ss_p_[b] = static_cast<double>(r.count[b]) / static_cast<double>(r.total);
    }
    const double w = ss_w_[static_cast<std::size_t>(ss_bucket_)];
    std::vector<double> out(256);
    for (std::size_t b = 0; b < 256; ++b) out[b] = std::log(std::max(w * ss_pin_[b] + (1.0 - w) * ss_p_[b], 1e-300));
    return out;
}

void HpSequenceBackend::add_neural(std::shared_ptr<const ByteNeuralExpert> nn) {
    if (!nn) return;
    nn_.push_back({std::move(nn), {}});
    reset_neural_weights_();
    prime_neural_();
}

void HpSequenceBackend::reset_neural_weights_() {
    const std::size_t k = nn_.size();
    if (k == 0) {
        nn_w_.clear();
        return;
    }
    nn_w_.assign(kNnBuckets * (k + 1), 0.3 / static_cast<double>(k));
    for (int b = 0; b < kNnBuckets; ++b) nn_w_[static_cast<std::size_t>(b) * (k + 1)] = 0.7;
}

void HpSequenceBackend::set_neural(std::shared_ptr<const ByteNeuralExpert> nn, double eta) {
    nn_.clear();
    nn_w_.clear();
    nn_eta_ = eta;
    nn_valid_ = served_valid_ = false;
    add_neural(std::move(nn));
}

void HpSequenceBackend::prime_neural_() {
    constexpr std::size_t kPrime = 512;
    std::uint8_t ctx[kPrime];
    const std::size_t len = nn_.empty() ? 0 : pred_->recent_bytes(ctx, kPrime);
    for (auto& s : nn_) {
        s.state = s.model->initial_state();
        if (nn_adapt_ > 0.0) s.model->init_adaptation(s.state);
        for (std::size_t i = 0; i < len; ++i) s.model->step(s.state, ctx[i]);  // priming does not adapt
    }
    nn_valid_ = served_valid_ = false;
}

std::vector<double> HpSequenceBackend::neural_mix_(const std::vector<double>& base) {
    const std::size_t k = nn_.size();
    nn_pin_.resize(256);
    std::size_t top_m = 0, top_n = 0;
    const auto& lp0 = nn_[0].state.log_p;
    for (std::size_t b = 0; b < 256; ++b) {
        nn_pin_[b] = std::exp(base[b]);
        if (nn_pin_[b] > nn_pin_[top_m]) top_m = b;
        if (lp0[b] > lp0[top_n]) top_n = b;
    }
    const int conf = std::min(7, static_cast<int>(nn_pin_[top_m] * 8.0));
    nn_bucket_ = conf * 2 + (top_m == top_n ? 1 : 0);
    const double* w = &nn_w_[static_cast<std::size_t>(nn_bucket_) * (k + 1)];
    std::vector<double> out(256);
    for (std::size_t b = 0; b < 256; ++b) {
        double p = w[0] * nn_pin_[b];
        for (std::size_t i = 0; i < k; ++i) p += w[i + 1] * std::exp(nn_[i].state.log_p[b]);
        out[b] = std::log(std::max(p, 1e-300));
    }
    return out;
}

void HpSequenceBackend::set_infinigram(std::shared_ptr<const InfiniGram> ig, double eta) {
    ig_ = std::move(ig);
    ig_eta_ = eta;
    ig_w_.assign(kIgBuckets, {0.8, 0.1, 0.1});
    ig_w0_ = ig_w_;
    ig_valid_ = served_valid_ = false;
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
    ig_w0_ = ig_w_;  // start weights: reset() returns to them
    ig_valid_ = served_valid_ = false;
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

HpSequenceBackend::MixingState HpSequenceBackend::mixing_state() const {
    MixingState s;
    s.ensemble = ensemble_weights();
    s.ig = ig_w_;
    s.session = ss_w_;
    s.neural = nn_w_;
    for (const auto& m : members_) s.members.push_back(m.backend->mixing_state());
    return s;
}

void HpSequenceBackend::set_mixing_state(const MixingState& s) {
    if (s.ensemble.size() == members_.size() + 1) {
        self_weight_ = s.ensemble[0];
        for (std::size_t i = 0; i < members_.size(); ++i) members_[i].weight = s.ensemble[i + 1];
    }
    if (s.ig.size() == ig_w_.size()) ig_w_ = s.ig;
    if (s.session.size() == ss_w_.size()) ss_w_ = s.session;
    if (s.neural.size() == nn_w_.size()) nn_w_ = s.neural;
    for (std::size_t i = 0; i < members_.size() && i < s.members.size(); ++i) {
        members_[i].backend->set_mixing_state(s.members[i]);
    }
    // The served distribution and the stage caches were mixed with the old weights.
    last_valid_ = ig_valid_ = nn_valid_ = ss_valid_ = served_valid_ = false;
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
    member->set_mixing_learning(mix_learning_);
    members_.push_back(Member{std::move(member), weight});
    self_weight_ = 1.0 - total;
    last_valid_ = ig_valid_ = nn_valid_ = ss_valid_ = served_valid_ = false;
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
    last_valid_ = ig_valid_ = nn_valid_ = ss_valid_ = served_valid_ = false;
    pred_->reset_stream_state(keep_history);
    for (auto& m : members_) m.backend->reset_stream(keep_history);
    prime_neural_();  // the LSTM re-reads whatever history the predictor kept
    if (!keep_history) set_session_cache(ss_on_, ss_eta_, ss_window_);  // a new session
}

void HpSequenceBackend::set_serve_adaptation(int num, int den, int skip) {
    if (serve_rate_ != std::array<int, 3>{num, den, skip}) {
        serve_rate_ = {num, den, skip};
        ++settings_epoch_;
    }
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

const std::vector<double>& HpSequenceBackend::served_log_probs_() const {
    // A distribution scored under other learning / serve-rate settings is
    // stale here (non-frozen scoring trains inside the hypothetical byte).
    if (!served_valid_ || served_.size() != 256 || served_epoch_ != settings_epoch_) {
        // Scoring restores all state; const_cast keeps the const serve API.
        (void)const_cast<HpSequenceBackend*>(this)->next_byte_log_probs(256);
    }
    return served_;
}

double HpSequenceBackend::log_prob_byte(std::uint8_t byte) const {
    if (mixed_()) return served_log_probs_()[byte];
    return byte_log_prob_on_pred_(byte);
}

std::uint8_t HpSequenceBackend::serve_greedy_next_byte() const {
    if (mixed_()) {
        const auto& lp = served_log_probs_();
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
    if (mixed_()) {
        const auto& lp = served_log_probs_();
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
    // Mixing weights learn from a scored byte while learning is on, unless
    // frozen (set_mixing_learning).
    const bool learn_mix = pred_->learning() && mix_learning_;
    if (last_valid_ && ens_eta_ > 0.0 && learn_mix) update_ensemble_weights_(byte);
    if (ig_valid_ && ig_eta_ > 0.0 && learn_mix && byte < ig_p_[0].size()) {
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
    if (ss_on_) {
        if (ss_valid_ && ss_bucket_ >= 0 && ss_eta_ > 0.0 && learn_mix) {
            double& w = ss_w_[static_cast<std::size_t>(ss_bucket_)];
            const double pa = ss_pin_[byte], pb = ss_p_[byte];
            const double p = std::max(w * pa + (1.0 - w) * pb, 1e-12);
            w = std::clamp(w + ss_eta_ * (pa - pb) / p, 0.01, 0.99);
        }
        ss_hist_.push_back(byte);
        ++ss_len_;
        // A byte read under an undo recorder (hp::StreamRewind lookahead) is
        // speculative: truncate_session takes it back, and a rebuild over it
        // would be thrown away with it (and redone for every candidate).
        if (hp::UndoRecorderScope::active() == nullptr) session_grow_();
    }
    if (!nn_.empty()) {
        if (nn_valid_ && nn_eta_ > 0.0 && learn_mix) {
            // Exponentiated gradient on the mixture's log loss for this bucket.
            const std::size_t k = nn_.size();
            double* w = &nn_w_[static_cast<std::size_t>(nn_bucket_) * (k + 1)];
            std::vector<double> pe(k + 1);
            pe[0] = nn_pin_[byte];
            for (std::size_t i = 0; i < k; ++i) pe[i + 1] = std::exp(nn_[i].state.log_p[byte]);
            double pm = 0.0;
            for (std::size_t i = 0; i <= k; ++i) pm += w[i] * pe[i];
            pm = std::max(pm, 1e-12);
            double z = 0.0;
            for (std::size_t i = 0; i <= k; ++i) {
                const double g = std::clamp(nn_eta_ * (pe[i] / pm - 1.0), -2.0, 2.0);
                z += (w[i] = std::max(1e-4, w[i] * std::exp(g)));
            }
            for (std::size_t i = 0; i <= k; ++i) w[i] /= z;
        }
        const float lr = pred_->learning() ? static_cast<float>(nn_adapt_) : 0.0f;
        for (auto& s : nn_) {
            s.state.adapt_lr = lr;
            s.model->step(s.state, byte);
        }
    }
    last_valid_ = ig_valid_ = nn_valid_ = ss_valid_ = served_valid_ = false;
}

double HpSequenceBackend::observe_next_byte(std::uint8_t next) {
    if (mixed_()) {
        // The distribution served at this position, whatever the settings
        // since: the loss of what was scored, and the stage updates in
        // consume_byte read that same scoring.
        const double lp = served_valid_ && next < served_.size() ? served_[next] : next_byte_log_probs(256)[next];
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
