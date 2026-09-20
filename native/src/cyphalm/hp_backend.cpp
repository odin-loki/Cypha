#include "cypha/cyphalm/hp_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

namespace cypha::cyphalm {

namespace {

constexpr double kLogEps = 1e-300;
constexpr double kLog2 = 0.6931471805599453;

double bit_log_prob(int p12, int bit) {
    const double p1 = static_cast<double>(p12) / 4096.0;
    const double p = bit ? p1 : (1.0 - p1);
    return std::log(std::max(p, kLogEps));
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

}  // namespace

hp::Config hp_config_from_cyphalm(int table_bits, int mixer_lr, bool gria) {
    hp::Config cfg;
    cfg.table_bits = table_bits;
    cfg.mixer_lr = mixer_lr;
    cfg.gria = gria;
    cfg.normalize();
    return cfg;
}

HpSequenceBackend::HpSequenceBackend(hp::Config cfg)
    : cfg_(cfg), pred_(std::make_unique<hp::Predictor>(cfg)) {}

double HpSequenceBackend::byte_log_prob_on_pred_(std::uint8_t byte) const {
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
    for (int bit = 0; bit <= 1; ++bit) {
        if (!branch_reaches_vocab(vocab_size, prefix, depth, bit)) {
            continue;
        }
        hp::UndoFrame& frame = undo.push_frame();
        {
            hp::UndoRecorderScope scope(frame);
            const int p12 = node.predict();
            const double child_log = log_p_nats + bit_log_prob(p12, bit);
            node.update(bit);
            const int next_prefix = (prefix << 1) | bit;
            expand_bit_tree_dfs(vocab_size, depth + 1, next_prefix, child_log, node, undo,
                                out_log_nats);
        }
        undo.pop_frame(node);
    }
}

std::vector<double> HpSequenceBackend::next_byte_log_probs_bit_tree(int vocab_size) {
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
    if (use_legacy_byte_log_probs()) {
        return next_byte_log_probs_legacy(vocab_size);
    }
    return next_byte_log_probs_bit_tree(vocab_size);
}

double HpSequenceBackend::log_prob_byte(std::uint8_t byte) const {
    return byte_log_prob_on_pred_(byte);
}

std::uint8_t HpSequenceBackend::serve_greedy_next_byte() const {
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
    if (rng01 == nullptr) {
        return serve_greedy_next_byte();
    }
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
}

double HpSequenceBackend::observe_next_byte(std::uint8_t next) {
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
