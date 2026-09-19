#pragma once

/// Cypha adapter for odin-loki/CompressionAlgorithm hp::Predictor.
/// Integer-exact context mixing lives in hp/; this layer exposes byte-level
/// log-probabilities for the CyphaLM public API (BPC path uses double log_probs).
///
/// RAM note: holds two ``hp::Predictor`` instances (``pred_`` + ``scratch_``).
/// ``next_byte_log_probs()`` expands an MSB-first bit prefix tree (≤255 fork clones
/// vs 256×8 bit steps on the legacy per-byte clone path). BPC / train loss should
/// use ``observe_next_byte`` / ``observe_stream_bits`` (O(8) bits per byte, no fan-out).

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

    /// MSB prefix-tree fan-out (default ``next_byte_log_probs`` implementation).
    std::vector<double> next_byte_log_probs_bit_tree(int vocab_size) const;

    /// Legacy 256× deep-clone path (``CYPHA_HP_LEGACY_BYTE_LOGPROBS=1`` or parity tests).
    std::vector<double> next_byte_log_probs_legacy(int vocab_size) const;

    /// log P(single byte | current history); one fork clone + 8 bit steps (train / top-1).
    double log_prob_byte(std::uint8_t byte) const;

    /// Sample one byte MSB-first (8 bit steps on a single fork clone). Does not advance main.
    std::uint8_t sample_next_byte(double (*rng01)()) const;

    /// Cross-entropy loss in nats for observing `next` after current history.
    double observe_next_byte(std::uint8_t next);

    /// Cumulative cross-entropy in **bits** over ``bytes[0..len)`` (hp compress encode path).
    /// Each byte is predict→score→update on the main predictor; no vocab clone fan-out.
    double observe_stream_bits(const std::uint8_t* bytes, std::size_t len);

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

    static bool branch_reaches_vocab(int vocab_size, int prefix, int depth, int bit);
    static void expand_bit_tree_dfs(const hp::Config& cfg, int vocab_size, int depth, int prefix,
                                    double log_p_nats, hp::Predictor& node,
                                    std::vector<double>& out_log_nats);
};

hp::Config hp_config_from_cyphalm(int table_bits, int mixer_lr, bool gria);

}  // namespace cypha::cyphalm
