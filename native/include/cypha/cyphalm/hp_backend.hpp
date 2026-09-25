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
/// ``CYPHA_HP_LEGACY_BYTE_LOGPROBS=1``. ``CYPHA_HP_TREE_REPREDICT=1`` makes
/// the bit tree predict again before every bit-1 update, as it used to
/// (reference for parity tests; same distributions, slower).

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cypha/cyphalm/neural_expert.hpp"
#include "hp/predictor.hpp"
#include "hp/undo.hpp"

#include <array>

namespace cypha::cyphalm {
class InfiniGram;
}

namespace cypha::cyphalm {

/// Wraps hp::Predictor for byte/token sequence modelling inside Cypha.
class HpSequenceBackend {
 public:
    explicit HpSequenceBackend(hp::Config cfg);

    /// A cold model: an untrained predictor, no ensemble members, and every
    /// attached stage back to its start: ∞-gram and neural mixing weights
    /// (as attached / loaded), neural experts at their initial state (and
    /// output layers un-adapted), an empty session cache. Two models reset
    /// alike score alike whatever each read before (codec symmetry).
    void reset();

    /// Train / compress: consume one byte on the live predictor (predict + update per bit).
    void consume_byte(std::uint8_t byte);

    /// Serve alias — same as ``consume_byte`` (online hp context advance, no loss fan-out).
    void serve_advance_byte(std::uint8_t byte) { consume_byte(byte); }

    /// P(next_byte | history). Does not advance main state. Default: bit-tree with delta undo.
    /// Always scores. When a mixing stage is active (``is_composite``) the
    /// result is also kept as the served distribution of this position:
    /// ``observe_next_byte``, ``log_prob_byte``, ``serve_greedy_next_byte``
    /// and ``serve_sample_next_byte`` reuse it instead of re-scoring the whole
    /// mixture. Consuming a byte, rewinding (``invalidate_scoring_cache``) or
    /// changing a stage drops it; the last three also re-score after
    /// ``set_learning`` or ``set_serve_adaptation`` changed a setting.
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
    /// Composite models: the entry of the full served distribution.
    double log_prob_byte(std::uint8_t byte) const;

    /// Serve: O(8) greedy next byte on scratch fork (no 256-way fan-out).
    /// Composite models: the served distribution's argmax.
    std::uint8_t serve_greedy_next_byte() const;

    /// Sample one byte MSB-first (8 bit steps on a single fork clone). Does not advance main.
    std::uint8_t sample_next_byte(double (*rng01)()) const;

    /// Serve: temperature-scaled bit sampling (temperature <= 0 → greedy).
    /// Composite models: sampled from the served distribution ^ (1 / temperature).
    std::uint8_t serve_sample_next_byte(double temperature, double (*rng01)()) const;

    /// True when any mixing stage is active (ensemble members, ∞-gram,
    /// neural experts, session cache): the served distribution is then not
    /// the primary predictor's alone, and every serve path uses the full mix.
    bool is_composite() const { return mixed_(); }

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
    /// Speed: bit-tree subtrees whose probability falls below ``min_prob`` are
    /// not expanded; their mass is spread evenly over their bytes (still a
    /// normalised distribution). 0 = exact.
    void set_tree_prune(double min_prob) {
        prune_log_ = min_prob > 0.0 ? std::log(min_prob) : -1e300;
        for (auto& m : members_) m.backend->set_tree_prune(min_prob);
        last_valid_ = ig_valid_ = nn_valid_ = ss_valid_ = served_valid_ = false;
    }

    void set_frozen_scoring(bool on) {
        frozen_scoring_ = on;
        last_valid_ = ig_valid_ = nn_valid_ = ss_valid_ = served_valid_ = false;
        for (auto& m : members_) m.backend->set_frozen_scoring(on);
    }
    bool frozen_scoring() const { return frozen_scoring_; }

    /// Online learning on/off for subsequent bytes (``hp::Predictor::set_learning``).
    void set_learning(bool on) {
        if (on != pred_->learning()) ++settings_epoch_;
        pred_->set_learning(on);
        for (auto& m : members_) m.backend->set_learning(on);
    }
    bool learning() const { return pred_->learning(); }

    /// Serve-time ensemble: another pretrained hp model advanced alongside this
    /// one. Next-byte distributions are mixed geometrically,
    /// log p = (1 - sum w_i) log p_self + sum w_i log p_i - log Z (sum w_i < 1).
    /// Every serve / generation path fans out to members; ``observe_*`` scores
    /// the mixture. Training a model with members attached is not supported
    /// (``reset()`` drops them).
    void add_ensemble_member(std::unique_ptr<HpSequenceBackend> member, double weight);
    std::size_t ensemble_size() const { return members_.size(); }
    /// Adapt the mixing weights online (exponentiated gradient on the mix's log
    /// loss) at this rate, whenever a scored byte is consumed with learning on.
    /// 0 = fixed weights.
    void set_ensemble_learning_rate(double eta) { ens_eta_ = eta; }
    /// Current weights: [self, member 0, member 1, ...].
    std::vector<double> ensemble_weights() const;
    /// Every learned mixing weight: the ensemble's (``ensemble_weights``),
    /// the ∞-gram, session and neural buckets, and each member's own.
    /// Generation takes it before a request and restores it after
    /// (``DecodeParams::restore_mixing``), so what one prompt taught the
    /// weights does not carry into the next request.
    struct MixingState {
        std::vector<double> ensemble;
        std::vector<std::array<double, 3>> ig;
        std::vector<double> session, neural;
        std::vector<MixingState> members;
        bool operator==(const MixingState&) const = default;
    };
    MixingState mixing_state() const;
    /// Back to ``s``; a stage whose shape changed since keeps its weights.
    void set_mixing_state(const MixingState& s);
    /// Mixing-weight learning (default on). Off: no stage's weights move
    /// (ensemble, ∞-gram, session, neural; members too), while the models,
    /// the session text and the experts' output layers still learn as set.
    /// Measures the start-weight mixture (``cyphalm_lm_quality --freeze-mixing``).
    void set_mixing_learning(bool on) {
        mix_learning_ = on;
        for (auto& m : members_) m.backend->set_mixing_learning(on);
    }
    bool mixing_learning() const { return mix_learning_; }
    /// Worker threads (default on; ``CYPHA_HP_ENSEMBLE_THREADS=0`` makes the
    /// default off), members too. On: members score on a persistent pool
    /// while this model scores, and when a byte is read the neural experts
    /// step and the members read it there too, unless an undo recorder
    /// (hp::StreamRewind) is active on the calling thread: recorders are per
    /// thread, so the members then read it here. Results are the same bit
    /// for bit either way; off runs everything on the calling thread.
    void set_parallel(bool on);
    bool parallel() const { return parallel_; }
    /// Forget the last scored distributions, this model's and every member's
    /// (after rewinding state).
    void invalidate_scoring_cache() {
        last_valid_ = ig_valid_ = nn_valid_ = ss_valid_ = served_valid_ = false;
        for (auto& m : members_) m.backend->invalidate_scoring_cache();
    }

    /// ∞-gram expert (InfiniGram over the pretraining corpus): the served
    /// distribution becomes w0 p_model + w1 p_longest + w2 p_reliable, where
    /// p_longest counts the bytes after every corpus occurrence of the
    /// longest matching context suffix and p_reliable those after the longest
    /// suffix seen at least 16 times. Weights are learned online (exponentiated
    /// gradient) per (match length, count, model confidence, whether the
    /// model and the longest match agree on the top byte) bucket while
    /// learning is on. Shared and read-only: many models can use one index.
    void set_infinigram(std::shared_ptr<const InfiniGram> ig, double eta = 0.3);
    bool has_infinigram() const { return static_cast<bool>(ig_); }
    /// The per-bucket mixing weights (bucket-major, 3 per bucket), to save
    /// weights learned on held-out text and start from them later.
    std::vector<double> infinigram_weights() const;
    void set_infinigram_weights(const std::vector<double>& w);
    /// Neural experts (pretrained byte LSTM / Transformer, ``ByteNeuralExpert``):
    /// after the ∞-gram mix the served distribution becomes the linear mix
    /// w_0 p + sum_i w_i p_nn_i, weights learned online (exponentiated
    /// gradient, rate ``eta``) per (model confidence, top-byte agreement with
    /// the first expert) bucket while learning is on. Experts read every
    /// consumed byte; adding one primes it with the recent history.
    void add_neural(std::shared_ptr<const ByteNeuralExpert> nn);
    /// Replace all neural experts by ``nn`` (null: none).
    void set_neural(std::shared_ptr<const ByteNeuralExpert> nn, double eta = 0.1);
    /// Mixing-weight rate for every attached expert (one rate for all).
    void set_neural_learning_rate(double eta) { nn_eta_ = eta; }
    double neural_learning_rate() const { return nn_eta_; }
    /// Dynamic evaluation of the experts' output layers: per-byte SGD at
    /// ``lr`` while learning is on (0 = frozen experts). Resets the adapted
    /// layers.
    void set_neural_adaptation(double lr) {
        nn_adapt_ = lr;
        prime_neural_();
    }
    bool has_neural() const { return !nn_.empty(); }
    std::size_t neural_count() const { return nn_.size(); }
    /// Per bucket: [model, expert 0, expert 1, ...].
    std::vector<double> neural_weights() const { return nn_w_; }
    /// Expert states, to restore after rewinding the predictors.
    std::vector<ByteNeuralExpert::State> neural_states() const {
        std::vector<ByteNeuralExpert::State> out;
        for (const auto& s : nn_) out.push_back(s.state);
        return out;
    }
    void set_neural_states(const std::vector<ByteNeuralExpert::State>& st) {
        for (std::size_t i = 0; i < nn_.size() && i < st.size(); ++i) {
            nn_[i].state = st[i];
            nn_[i].primed = false;
        }
        nn_valid_ = served_valid_ = false;
    }
    /// Experts primed so far (diagnostics, tests). Priming reads the last
    /// 512 bytes of history into an expert's initial state; an expert that
    /// was primed on the same bytes (and adaptation setting) and has read
    /// nothing since already holds that state and is not primed again
    /// (``reset_stream``, ``reset``, adding experts, ``set_neural_adaptation``).
    std::size_t neural_primes() const { return nn_primes_; }
    /// Session cache: an ∞-gram index over the last ``window`` bytes this
    /// stream has read (prompt, conversation, document; 0 = all of them),
    /// rebuilt just in time as it grows: at 1 KiB, then whenever the bytes
    /// read since the last build equal the indexed length (doubling), at
    /// most ``window`` / 16 bytes apart (64 KiB for 0). So a rebuild costs at
    /// most ``window`` bytes and all of them cost O(16) per byte read. Bytes
    /// read while an undo recorder is active (hp::StreamRewind lookahead)
    /// are speculative: they are matched against but never trigger a
    /// rebuild, so ``truncate_session`` back over them keeps the index. The
    /// longest-match next-byte counts are mixed in as w p + (1 - w) p_sess,
    /// w learned online per (match length, count) bucket. Sets up an empty
    /// session (also what ``reset_stream(false)`` and ``reset`` do).
    static constexpr std::size_t kSessionWindow = std::size_t{1} << 20;
    void set_session_cache(bool on, double eta = 0.02, std::size_t window = kSessionWindow);
    bool has_session_cache() const { return ss_on_; }
    /// Bytes read so far in this session (not capped by the window), and
    /// rewinding to an earlier count (generation lookahead).
    std::size_t session_size() const { return ss_len_; }
    void truncate_session(std::size_t n);
    /// Session bytes held (<= window + window / 16) and indexed (<= window),
    /// and how many times the index was built (tests, diagnostics).
    std::size_t session_held() const { return ss_hist_.size(); }
    std::size_t session_indexed() const { return ss_indexed_; }
    std::size_t session_builds() const { return ss_builds_; }
    /// This predictor and every member's (for ``hp::StreamRewind``).
    std::vector<hp::Predictor*> all_predictors();
    /// New stream on every model (``hp::Predictor::reset_stream_state``).
    /// Without the history the session cache starts empty too (text, index
    /// and weights); with it, the session is kept like the byte history.
    void reset_stream(bool keep_history);
    /// Per-table occupancy fold (``hp::Predictor::fold_auto``), members too.
    /// Throws std::invalid_argument unless 0 < ``max_occupancy`` < 1.
    std::size_t fold_auto(double max_occupancy) {
        if (!(max_occupancy > 0.0 && max_occupancy < 1.0))
            throw std::invalid_argument("fold_auto: max_occupancy must be in (0, 1)");
        last_valid_ = ig_valid_ = nn_valid_ = ss_valid_ = served_valid_ = false;
        std::size_t freed = pred_->fold_auto(max_occupancy);
        for (auto& m : members_) freed += m.backend->fold_auto(max_occupancy);
        return freed;
    }
    /// Fold this model's and every ensemble member's tables to the caps and
    /// drops in ``target`` (``hp::Predictor::fold_tables``).
    void fold_tables(const hp::Config& target) {
        pred_->fold_tables(target);
        cfg_ = pred_->config();
        for (auto& m : members_) m.backend->fold_tables(target);
        last_valid_ = ig_valid_ = nn_valid_ = ss_valid_ = served_valid_ = false;
    }
    /// Serve mixer rate on every model (``hp::Predictor::set_serve_adaptation``).
    void set_serve_adaptation(int num, int den, int skip);

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
    /// ``reference``: predict again before every bit-1 update that follows a
    /// taken bit 0 (the old schedule; CYPHA_HP_TREE_REPREDICT=1).
    static void expand_bit_tree_dfs(int vocab_size, int depth, int prefix, double log_p_nats,
                                    hp::Predictor& node, hp::PredictorUndoStack& undo,
                                    std::vector<double>& out_log_nats,
                                    double prune_log = -1e300, bool reference = false);
    double byte_log_prob_on_pred_(std::uint8_t byte) const;

    bool serve_compact_ = false;
    double prune_log_ = -1e300;  // set_tree_prune
    bool frozen_scoring_ = false;
    hp::PredictorUndoStack undo_;  // bit-tree frames, kept so their buffers are reused

    // Worker threads (set_parallel), shared out by claiming (hp_backend.cpp).
    class WorkerPool;
    std::shared_ptr<WorkerPool> pool_;
    bool parallel_ = true;
    /// job(i) for i in [0, n) on the pool while this thread runs here().
    void run_parallel_(std::size_t n, const std::function<void(std::size_t)>& job,
                       const std::function<void()>& here);

    struct Member {
        std::unique_ptr<HpSequenceBackend> backend;
        double weight = 0.0;
    };
    std::vector<Member> members_;
    double self_weight_ = 1.0;   // 1 - sum of member weights
    double ens_eta_ = 0.0;
    bool mix_learning_ = true;   // set_mixing_learning
    // Last scored distributions, for the weight update in consume_byte.
    bool last_valid_ = false;
    std::vector<double> last_own_, last_mix_;
    std::vector<std::vector<double>> last_member_lp_;
    std::vector<double> ens_p_;  // exp(last_mix_), for the weight update
    void update_ensemble_weights_(std::uint8_t byte);

    // Model (+ ensemble) distribution before the ∞-gram mix.
    std::vector<double> scored_log_probs_(int vocab_size);
    std::vector<double> infinigram_mix_(const std::vector<double>& base);
    std::shared_ptr<const InfiniGram> ig_;
    double ig_eta_ = 0.3;
    static constexpr int kIgBuckets = 8 * 4 * 8;
    std::vector<std::array<double, 3>> ig_w_;
    std::vector<std::array<double, 3>> ig_w0_;  // start weights (reset)
    bool ig_valid_ = false;
    int ig_bucket_ = 0;
    std::vector<double> ig_p_[3];       // model, longest, reliable (probabilities)
    // The last query's context and match length: its n + 1 bounds the next
    // query when the next context extends this one (InfiniGram::query hint).
    std::array<std::uint8_t, 256> ig_prev_ctx_{};
    std::size_t ig_prev_len_ = 0;
    int ig_prev_n_ = -1;                // -1: no previous query
    // Served distribution (after every mixing stage) at this position, kept
    // while ``served_valid_``: dropped with the stage caches above.
    bool mixed_() const { return !members_.empty() || ig_ || !nn_.empty() || ss_on_; }
    std::vector<double> served_;
    bool served_valid_ = false;
    std::uint64_t settings_epoch_ = 0;  // bumped when learning or the serve rate changes
    std::uint64_t served_epoch_ = 0;    // settings_epoch_ when served_ was scored
    std::array<int, 3> serve_rate_{0, 0, 0};  // last set_serve_adaptation
    /// ``served_`` if it is valid for the current settings, else a fresh score.
    const std::vector<double>& served_log_probs_() const;
    // Session cache.
    std::vector<double> session_mix_(const std::vector<double>& base);
    void session_grow_();               // trim to the window and rebuild when due
    bool ss_on_ = false;
    double ss_eta_ = 0.02;
    std::size_t ss_window_ = kSessionWindow;
    std::vector<std::uint8_t> ss_hist_; // the last session bytes (session_size() - size() dropped)
    std::size_t ss_len_ = 0;            // session bytes read
    std::shared_ptr<const InfiniGram> ss_ig_;
    std::size_t ss_built_ = 0;          // ss_len_ when ss_ig_ was built
    std::size_t ss_indexed_ = 0;        // bytes indexed by ss_ig_ (0: none)
    std::size_t ss_builds_ = 0;
    static constexpr int kSsBuckets = 8 * 4;
    std::vector<double> ss_w_;
    bool ss_valid_ = false;
    int ss_bucket_ = -1;                // -1: no session evidence this byte
    std::vector<double> ss_pin_, ss_p_;
    // Neural expert.
    std::vector<double> neural_mix_(const std::vector<double>& base);
    void prime_neural_();
    void reset_neural_weights_();       // start weights for the attached experts
    struct NnSlot {
        std::shared_ptr<const ByteNeuralExpert> model;
        ByteNeuralExpert::State state;
        // ``state`` is what priming on ``primed_ctx`` (adaptation
        // ``primed_adapt``) built, and no byte was read since.
        bool primed = false;
        double primed_adapt = 0.0;
        std::vector<std::uint8_t> primed_ctx;
    };
    std::vector<NnSlot> nn_;
    std::size_t nn_primes_ = 0;
    double nn_eta_ = 0.1;
    double nn_adapt_ = 0.0;
    static constexpr int kNnBuckets = 16;
    std::vector<double> nn_w_;          // kNnBuckets x (1 + experts)
    bool nn_valid_ = false;
    int nn_bucket_ = 0;
    std::vector<double> nn_pin_;        // probabilities before the neural mix
    /// Ensemble: geometric mix of this model's ``own`` log probs with members'.
    std::vector<double> mix_with_members_(const std::vector<double>& own,
                                          const std::vector<std::vector<double>>& member_lp);
};

hp::Config hp_config_from_cyphalm(int table_bits, int mixer_lr, bool gria);

struct CyphaLMConfig;
/// Full mapping: effective table bits, mixer lr, GRIA and the lossy mixer knobs.
hp::Config hp_config_from_cyphalm(const CyphaLMConfig& cfg);

}  // namespace cypha::cyphalm
