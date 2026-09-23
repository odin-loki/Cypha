#pragma once

/// Cypha adapter for odin-loki/CompressionAlgorithm hp::Predictor.
/// Integer-exact context mixing lives in hp/; this layer exposes byte-level
/// log-probabilities for the CyphaLM public API (BPC path uses double log_probs).
///
/// **Train vs serve:** BPC / train loss use ``observe_next_byte`` / ``observe_stream_bits``
/// (O(8) bits per byte, no vocab fan-out). Serve / generation use ``serve_*`` helpers:
/// ``serve_advance_byte`` advances live context; ``serve_next_byte_log_probs`` and
/// ``serve_greedy_next_byte`` score on a scratch fork without train bookkeeping.
///
/// RAM note: holds one live ``pred_`` for context advance. MSB bit-tree and
/// single-byte fork scoring use delta undo on ``pred_`` directly (no standing
/// scratch twin). Legacy 256-fork path allocates ephemeral forks per byte.
/// ``Predictor::copy_state_from`` / ``assign_from_`` must copy all ctx-chain models
/// (parity tests and any reuse path).
/// ``next_byte_log_probs()`` defaults to bit-tree joint scoring; legacy path:
/// ``CYPHA_HP_LEGACY_BYTE_LOGPROBS=1``.

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

    /// Train / compress: consume one byte on the live predictor (predict + update per bit).
    void consume_byte(std::uint8_t byte);

    /// Serve alias — same as ``consume_byte`` (online hp context advance, no loss fan-out).
    void serve_advance_byte(std::uint8_t byte) { consume_byte(byte); }

    /// P(next_byte | history). Does not advance main state. Default: bit-tree with delta undo.
    std::vector<double> next_byte_log_probs(int vocab_size);

    /// Serve alias for ``next_byte_log_probs`` (explicit generation surface).
    std::vector<double> serve_next_byte_log_probs(int vocab_size) {
        return next_byte_log_probs(vocab_size);
    }

    /// MSB prefix-tree fan-out with delta undo backtracking (default inference path).
    std::vector<double> next_byte_log_probs_bit_tree(int vocab_size);

    /// Legacy 256× fork path (``CYPHA_HP_LEGACY_BYTE_LOGPROBS=1`` or parity tests).
    std::vector<double> next_byte_log_probs_legacy(int vocab_size) const;

    /// Loop-local ``copy_state_from`` reuse (parity vs legacy; not production default).
    std::vector<double> next_byte_log_probs_assign_reuse(int vocab_size) const;

    /// log P(single byte | current history); undo on live ``pred_`` (no standing scratch).
    double log_prob_byte(std::uint8_t byte) const;

    /// Serve: O(8) greedy next byte on scratch fork (no 256-way fan-out).
    std::uint8_t serve_greedy_next_byte() const;

    /// Sample one byte MSB-first (8 bit steps on a single fork clone). Does not advance main.
    std::uint8_t sample_next_byte(double (*rng01)()) const;

    /// Serve: temperature-scaled bit sampling (temperature <= 0 → greedy).
    std::uint8_t serve_sample_next_byte(double temperature, double (*rng01)()) const;

    /// Train: cross-entropy loss in nats for observing ``next`` after current history.
    double observe_next_byte(std::uint8_t next);

    /// Train: cumulative cross-entropy in **bits** over ``bytes[0..len)`` (hp compress encode).
    double observe_stream_bits(const std::uint8_t* bytes, std::size_t len);

    const hp::Predictor& predictor() const { return *pred_; }
    hp::Predictor& predictor() { return *pred_; }

    /// Serve hint: legacy API to drop optional scratch (no-op when single-predictor).
    void compact_for_serve();

    /// Lossy: reset cold hash slots (see ``hp::Predictor::prune_cold_hash_slots``).
    void prune_cold_slots(int min_total);

    bool serve_compact() const { return serve_compact_; }

    /// Frozen scoring: distributions (bit tree, greedy, sampling, log_prob_byte)
    /// are computed from the model as it stands at the byte boundary, with no
    /// learning inside the hypothetical byte. Faster (no update work to record
    /// and undo); still a normalised distribution. Off = hp's compression
    /// semantics (each hypothetical bit trains the model before the next).
    void set_frozen_scoring(bool on) { frozen_scoring_ = on; }
    bool frozen_scoring() const { return frozen_scoring_; }

    /// Online learning on/off for subsequent bytes (``hp::Predictor::set_learning``).
    void set_learning(bool on) { pred_->set_learning(on); }
    bool learning() const { return pred_->learning(); }

    int table_bits() const { return cfg_.table_bits; }
    int mixer_lr() const { return cfg_.mixer_lr; }
    bool gria_enabled() const { return cfg_.gria; }
    const hp::Config& hp_config() const { return cfg_; }

    /// Copy live predictor state (beam / fork scoring).
    std::unique_ptr<hp::Predictor> predictor_snapshot() const;

    /// Bit-tree joint log P(next byte) on ``pred``; restores ``pred`` via undo.
    static std::vector<double> byte_log_probs_bit_tree(hp::Predictor& pred, int vocab_size);

    /// Advance ``pred`` by one byte (8 bit updates).
    static void consume_byte_on(hp::Predictor& pred, std::uint8_t byte);

 private:
    hp::Config cfg_;
    std::unique_ptr<hp::Predictor> pred_;
    /// Reused output buffer for ``next_byte_log_probs`` (avoids per-call alloc).
    mutable std::vector<double> log_probs_buf_;

    static double byte_log_prob(hp::Predictor& snap, int byte);

    static bool branch_reaches_vocab(int vocab_size, int prefix, int depth, int bit);
    static void expand_bit_tree_dfs(int vocab_size, int depth, int prefix, double log_p_nats,
                                    hp::Predictor& node, hp::PredictorUndoStack& undo,
                                    std::vector<double>& out_log_nats);
    double byte_log_prob_on_pred_(std::uint8_t byte) const;

    bool serve_compact_ = false;
    bool frozen_scoring_ = false;
};

hp::Config hp_config_from_cyphalm(int table_bits, int mixer_lr, bool gria);

struct CyphaLMConfig;
/// Full mapping: effective table bits, mixer lr, GRIA and the lossy mixer knobs.
hp::Config hp_config_from_cyphalm(const CyphaLMConfig& cfg);

}  // namespace cypha::cyphalm
