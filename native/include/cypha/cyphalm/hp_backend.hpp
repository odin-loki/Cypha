#pragma once

/// Cypha adapter for odin-loki/CompressionAlgorithm hp::Predictor.
/// Integer-exact context mixing lives in hp/; this layer exposes byte-level
/// log-probabilities for the CyphaLM public API (BPC path uses double log_probs).
///
/// RAM note: holds live ``pred_`` plus optional ``scratch_`` for single-byte fork
/// scoring. MSB bit-tree uses delta undo on ``scratch_`` (one fork copy), not
/// depth-indexed full Predictor snapshots.
/// ``next_byte_log_probs()`` defaults to bit-tree joint scoring; legacy 256-clone path:
/// ``CYPHA_HP_LEGACY_BYTE_LOGPROBS=1``. BPC / train use ``observe_stream_bits`` (no fan-out).

#include <cstdint>
#include <memory>
#include <vector>

#include "hp/predictor.hpp"
#include "hp/undo.hpp"

namespace cypha::cyphalm {

/// Wraps hp::Predictor for byte/token sequence modelling inside Cypha.
class HpSequenceBackend {
 public:
    explicit HpSequenceBackend(hp::Config cfg);

    void reset();

    /// Consume one byte (token id) through the hp bit path (predict + update per bit).
    void consume_byte(std::uint8_t byte);

    /// P(next_byte | history including bytes consumed so far). Does not advance main state.
    std::vector<double> next_byte_log_probs(int vocab_size);

    /// MSB prefix-tree fan-out with delta undo backtracking (default inference path).
    std::vector<double> next_byte_log_probs_bit_tree(int vocab_size);

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

    /// Drop ``scratch_`` to cut RSS on serve paths (lazy recreate).
    void compact_for_serve();

    /// Lossy: reset cold hash slots (see ``hp::Predictor::prune_cold_hash_slots``).
    void prune_cold_slots(int min_total);

    bool serve_compact() const { return serve_compact_; }

    int table_bits() const { return cfg_.table_bits; }
    int mixer_lr() const { return cfg_.mixer_lr; }
    bool gria_enabled() const { return cfg_.gria; }

 private:
    hp::Config cfg_;
    std::unique_ptr<hp::Predictor> pred_;
    mutable std::unique_ptr<hp::Predictor> scratch_;

    static double byte_log_prob(hp::Predictor& snap, int byte);

    static bool branch_reaches_vocab(int vocab_size, int prefix, int depth, int bit);
    static void expand_bit_tree_dfs(int vocab_size, int depth, int prefix, double log_p_nats,
                                    hp::Predictor& node, hp::PredictorUndoStack& undo,
                                    std::vector<double>& out_log_nats);
    void ensure_scratch_() const;

    bool serve_compact_ = false;
};

hp::Config hp_config_from_cyphalm(int table_bits, int mixer_lr, bool gria);

}  // namespace cypha::cyphalm
