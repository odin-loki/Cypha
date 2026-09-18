#include "cypha/cyphalm/hp_backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cypha::cyphalm {

namespace {

constexpr double kLogEps = 1e-300;

double bit_log_prob(int p12, int bit) {
    const double p1 = static_cast<double>(p12) / 4096.0;
    const double p = bit ? p1 : (1.0 - p1);
    return std::log(std::max(p, kLogEps));
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
      scratch_(std::make_unique<hp::Predictor>(cfg)) {}

void HpSequenceBackend::reset() {
    pred_ = std::make_unique<hp::Predictor>(cfg_);
    scratch_ = std::make_unique<hp::Predictor>(cfg_);
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

void HpSequenceBackend::consume_byte(std::uint8_t byte) {
    for (int i = 7; i >= 0; --i) {
        const int bit = (static_cast<int>(byte) >> i) & 1;
        (void)pred_->predict();
        pred_->update(bit);
    }
}

std::vector<double> HpSequenceBackend::next_byte_log_probs(int vocab_size) const {
    const int n = std::max(1, std::min(vocab_size, 256));
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    for (int b = 0; b < n; ++b) {
        scratch_ = std::make_unique<hp::Predictor>(*pred_);
        out[static_cast<std::size_t>(b)] = byte_log_prob(*scratch_, b);
    }
    return out;
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

}  // namespace cypha::cyphalm
