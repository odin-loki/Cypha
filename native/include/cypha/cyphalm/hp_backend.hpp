#pragma once

/// Cypha adapter for odin-loki/CompressionAlgorithm hp::Predictor.
/// Integer-exact context mixing lives in hp/; this layer exposes byte-level
/// log-probabilities for the CyphaLM public API (BPC path uses double log_probs).

#include <cstdint>
#include <memory>
#include <vector>

#include "hp/predictor.hpp"

namespace cypha::cyphalm {

/// Wraps hp::Predictor for byte/token sequence modelling inside Cypha.
class HpSequenceBackend {
 public:
    explicit HpSequenceBackend(hp::Config cfg);

    void reset();

    /// Consume one byte (token id) through the hp bit path (predict + update per bit).
    void consume_byte(std::uint8_t byte);

    /// P(next_byte | history including bytes consumed so far). Does not advance main state.
    std::vector<double> next_byte_log_probs(int vocab_size) const;

    /// Cross-entropy loss in nats for observing `next` after current history.
    double observe_next_byte(std::uint8_t next);

    const hp::Predictor& predictor() const { return *pred_; }
    hp::Predictor& predictor() { return *pred_; }

    int table_bits() const { return cfg_.table_bits; }
    int mixer_lr() const { return cfg_.mixer_lr; }
    bool gria_enabled() const { return cfg_.gria; }

 private:
    hp::Config cfg_;
    std::unique_ptr<hp::Predictor> pred_;
    mutable std::unique_ptr<hp::Predictor> scratch_;

    static double byte_log_prob(hp::Predictor& snap, int byte);
};

hp::Config hp_config_from_cyphalm(int table_bits, int mixer_lr, bool gria);

}  // namespace cypha::cyphalm
