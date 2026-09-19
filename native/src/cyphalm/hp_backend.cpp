#include "cypha/cyphalm/hp_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace cypha::cyphalm {

namespace {

constexpr double kLogEps = 1e-300;
constexpr double kLog2 = 0.6931471805599453;
constexpr int kByteBitDepth = 8;

double bit_log_prob(int p12, int bit) {
    const double p1 = static_cast<double>(p12) / 4096.0;
    const double p = bit ? p1 : (1.0 - p1);
    return std::log(std::max(p, kLogEps));
}

bool use_legacy_byte_log_probs() {
    const char* v = std::getenv("CYPHA_HP_LEGACY_BYTE_LOGPROBS");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
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
    : cfg_(cfg),
      pred_(std::make_unique<hp::Predictor>(cfg)),
      scratch_(std::make_unique<hp::Predictor>(cfg)) {
    init_dfs_ckpts_();
}

void HpSequenceBackend::reset() {
    pred_ = std::make_unique<hp::Predictor>(cfg_);
    scratch_ = std::make_unique<hp::Predictor>(cfg_);
    dfs_ckpts_.clear();
    init_dfs_ckpts_();
}

void HpSequenceBackend::init_dfs_ckpts_() {
#if !defined(CYPHA_HP_PROFILE_GATE24) && !defined(CYPHA_HP_PROFILE_CHAMP)
    dfs_ckpts_.reserve(static_cast<std::size_t>(kByteBitDepth + 1));
    for (int d = 0; d <= kByteBitDepth; ++d) {
        dfs_ckpts_.push_back(std::make_unique<hp::Predictor>(cfg_));
    }
#endif
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

void HpSequenceBackend::expand_bit_tree_dfs(int vocab_size, int depth, int prefix, double log_p_nats,
                                            hp::Predictor& node,
                                            std::vector<double>& out_log_nats) const {
    if (depth == kByteBitDepth) {
        if (prefix >= 0 && prefix < static_cast<int>(out_log_nats.size())) {
            out_log_nats[static_cast<std::size_t>(prefix)] = log_p_nats;
        }
        return;
    }
    const int p12 = node.predict();
    *dfs_ckpts_[static_cast<std::size_t>(depth)] = node;
    for (int bit = 0; bit <= 1; ++bit) {
        if (!branch_reaches_vocab(vocab_size, prefix, depth, bit)) {
            continue;
        }
        node.update(bit);
        const int next_prefix = (prefix << 1) | bit;
        expand_bit_tree_dfs(vocab_size, depth + 1, next_prefix,
                            log_p_nats + bit_log_prob(p12, bit), node, out_log_nats);
        node = *dfs_ckpts_[static_cast<std::size_t>(depth)];
    }
}

std::vector<double> HpSequenceBackend::next_byte_log_probs_bit_tree(int vocab_size) const {
    const int n = std::max(1, std::min(vocab_size, 256));
    std::vector<double> out(static_cast<std::size_t>(n),
                            std::log(1.0 / static_cast<double>(n)));
    *scratch_ = *pred_;
    expand_bit_tree_dfs(n, 0, 0, 0.0, *scratch_, out);
    return out;
}

std::vector<double> HpSequenceBackend::next_byte_log_probs_legacy(int vocab_size) const {
    const int n = std::max(1, std::min(vocab_size, 256));
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    for (int b = 0; b < n; ++b) {
        hp::Predictor snap = hp::Predictor::clone_from(*pred_, cfg_);
        out[static_cast<std::size_t>(b)] = byte_log_prob(snap, b);
    }
    return out;
}

std::vector<double> HpSequenceBackend::next_byte_log_probs(int vocab_size) const {
    if (use_legacy_byte_log_probs()) {
        return next_byte_log_probs_legacy(vocab_size);
    }
#if defined(CYPHA_HP_PROFILE_GATE24) || defined(CYPHA_HP_PROFILE_CHAMP)
    // gate24/champ: bit-tree DFS needs O(depth) predictor checkpoints; use legacy
    // 256-clone until undo stack lands (see CYPHALM_HP_ALGORITHM_PROFILE.md).
    return next_byte_log_probs_legacy(vocab_size);
#else
    return next_byte_log_probs_bit_tree(vocab_size);
#endif
}

double HpSequenceBackend::log_prob_byte(std::uint8_t byte) const {
    *scratch_ = *pred_;
    return byte_log_prob(*scratch_, static_cast<int>(byte));
}

std::uint8_t HpSequenceBackend::sample_next_byte(double (*rng01)()) const {
    if (rng01 == nullptr) {
        return 0;
    }
    *scratch_ = *pred_;
    int byte = 0;
    for (int i = 7; i >= 0; --i) {
        const int p12 = scratch_->predict();
        const double p1 = static_cast<double>(p12) / 4096.0;
        const double p0 = 1.0 - p1;
        const double r = rng01();
        const int bit = (r < p0 / (p0 + p1 + kLogEps)) ? 0 : 1;
        byte = (byte << 1) | bit;
        scratch_->update(bit);
    }
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
