#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/bpe_tokenizer.hpp"
#include "cypha/cyphalm/cyphalm_alpha_spectrum.hpp"
#include "cypha/cyphalm/cellai_ssm.hpp"
#include "cypha/cyphalm/char_lstm.hpp"
#include "cypha/cyphalm/compressive_memory.hpp"
#include "cypha/cyphalm/context_bank.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_dif.hpp"
#include "cypha/cyphalm/cyphalm_ewc_regularizer.hpp"
#include "cypha/cyphalm/embed_table.hpp"
#include "cypha/cyphalm/gria_lowrank.hpp"
#include "cypha/cyphalm/hebbian_stack.hpp"
#include "cypha/cyphalm/hierarchical_ssm.hpp"
#include "cypha/cyphalm/ngram_fusion.hpp"
#include "cypha/cyphalm/algebraic_fingerprint.hpp"
#include "cypha/cyphalm/axiom_activation.hpp"
#include "cypha/cyphalm/ca_state_cell.hpp"
#include "cypha/cyphalm/gria_gated_mixture.hpp"
#include "cypha/cyphalm/mdl_forget.hpp"
#include "cypha/cyphalm/nig_state_cell.hpp"
#include "cypha/cyphalm/reversible_ssm_cell.hpp"
#include "cypha/cyphalm/selective_ssm.hpp"
#include "cypha/cyphalm/view_embedding.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_guided_loss.hpp"
#include "cypha/rpsm/rpsm_sequence_layer.hpp"
#include "cypha/som/discriminative_feedback.hpp"
#include "cypha/som/gng_expert.hpp"
#include "cypha/som/gria_controller.hpp"

namespace cypha::intelligence {
class IntelligenceProfiler;
class EpistemicThreshold;
}  // namespace cypha::intelligence

namespace cypha::cyphalm {
class LmIntelligenceMonitor;
}  // namespace cypha::cyphalm

namespace cypha {
namespace cyphalm {

class CyphaLMModel;
CyphaLMModel load_cyphalm_model(const std::string& json_path);

struct PredictNextOutput {
    std::vector<double> log_probs;
    std::vector<std::uint32_t> top_k_tokens;
    std::vector<double> top_k_probs;
    double epistemic_var = 0.0;
    double aleatoric_var = 0.0;
};

struct TrainStepMetrics {
    double loss = 0.0;
    double epistemic_var = 0.0;
    double aleatoric_var = 0.0;
    int active_experts = 0;
    double alpha_gria = 0.0;
    double profile_guided_loss = 0.0;
    double ewc_penalty = 0.0;
    double free_energy_penalty = 0.0;
};

/// Phase 3 follow-up (docs/reports/HIDDEN_DIM_SCALE_PLAN.md, "history-buffer sampling
/// fix"): full detail behind ``lstm_hidden_d_eff_report()`` so callers can tell a
/// statistically well-powered measurement from an under-sampled one without re-deriving
/// it from ``lstm_hidden`` and a hardcoded history-buffer constant. All fields are -1.0
/// (or 0 for the int fields) when there's no LSTM head or fewer than 4 history rows have
/// been observed yet -- the same "unavailable" contract as the legacy scalar accessor.
struct LstmHiddenDEffReport {
    /// Width-normalized participation ratio in [0, 1] -- ``raw / n_dims``. This is the
    /// value historically exported as ``lstm_hidden_d_eff``; kept for continuity with
    /// existing consumers/baselines.
    double normalized = -1.0;
    /// Unnormalized effective-dimension count in [0, n_dims] -- ``(Σλ)² / Σλ²`` before
    /// the width normalization. Bounded above by ``min(n_samples, n_dims)`` in practice.
    double raw = -1.0;
    /// ``n_samples / n_dims`` actually used for this specific measurement -- the
    /// statistical-power signal Finding 2 (Phase 3, 2026-07-11) was missing. Values well
    /// below 1.0 mean the covariance estimate is underdetermined and ``normalized``
    /// should be read with caution (or not at all).
    double sample_ratio = -1.0;
    int n_samples = 0;
    int n_dims = 0;
};

/// Unified native CyphaLM stack (Tier 0–2–4 integration point).
class CyphaLMModel {
 public:
    explicit CyphaLMModel(CyphaLMConfig cfg);
    ~CyphaLMModel();

    CyphaLMModel(const CyphaLMModel&) = delete;
    CyphaLMModel& operator=(const CyphaLMModel&) = delete;
    CyphaLMModel(CyphaLMModel&&) noexcept = default;
    CyphaLMModel& operator=(CyphaLMModel&&) noexcept = default;

    static CyphaLMModel from_json_npz(const std::string& json_path);

    void save(const std::string& base_path) const;

    const CyphaLMConfig& config() const { return cfg_; }

    void reset_context();

    PredictNextOutput predict_next(std::uint32_t token_id);
    /// Re-blend cached hybrid GRIA/LSTM logits without advancing recurrent state.
    PredictNextOutput repredict_hybrid_blend(double blend_logit) const;
    TrainStepMetrics train_step(std::uint32_t token_id, std::uint32_t next_token_id,
                                cypha::intelligence::IntelligenceProfiler* profiler = nullptr,
                                LmIntelligenceMonitor* monitor = nullptr);
    std::vector<double> forward_log_probs(std::uint32_t token_id);

    void train_sequence(const std::vector<int>& ids, int n_steps, int epochs,
                        cypha::intelligence::IntelligenceProfiler* profiler = nullptr);
    void train_sequence_views(const std::vector<int>& ids,
                              cypha::intelligence::IntelligenceProfiler* profiler = nullptr);
    double eval_bpc(const std::vector<int>& ids, int n_eval,
                    cypha::intelligence::IntelligenceProfiler* profiler = nullptr);

    /// Full 7-stat profile pass over eval tokens (separate from BPC scoring).
    void accumulate_intelligence_profile(const std::vector<int>& ids, int n_steps,
                                         cypha::intelligence::IntelligenceProfiler& profiler);

    std::vector<std::uint32_t> encode_text(const std::string& text) const;
    std::string decode_tokens(const std::vector<std::uint32_t>& ids) const;
    bool has_bpe_tokenizer() const { return bpe_ != nullptr; }

    double hybrid_blend_logit() const { return hybrid_blend_logit_; }
    double hybrid_gria_weight() const;
    void set_hybrid_blend_logit(double logit) { hybrid_blend_logit_ = logit; }

    AlphaSpectrumSnapshot alpha_spectrum_snapshot() const;
    nlohmann::json compression_profile() const;

    /// Phase-5 CellAI probe (state norms, decay rates, routing connectivity).
    nlohmann::json ssm_diagnostic_report(const std::vector<int>& token_ids, int max_steps);

    CellAISSM* active_ssm();
    const CellAISSM* active_ssm() const;
    std::vector<double> embed_vector(std::uint32_t token_id) const;
    const std::vector<double>& field_vector() const { return field_x_; }
    double ssm_projection_rms() const;
    bool has_gria_routing() const { return gria_ != nullptr; }

    /// Low-rank GRIA head when routing is enabled (nullptr otherwise).
    const GRIALowRank* gria_routing() const { return gria_.get(); }

    /// Snapshot char-LSTM embed, recurrent, and lm_head weights for EWC (no-op without LSTM head).
    void ewc_snapshot();

    /// Current EWC quadratic penalty over snapshotted LSTM weights (0 without snapshot/LSTM).
    double ewc_penalty() const;

    CyphaLMEwcRegularizer& ewc_regularizer() { return ewc_.lstm_part(); }
    const CyphaLMEwcRegularizer& ewc_regularizer() const { return ewc_.lstm_part(); }

    HybridEwcRegularizer& hybrid_ewc_regularizer() { return ewc_; }
    const HybridEwcRegularizer& hybrid_ewc_regularizer() const { return ewc_; }

    /// Char-LSTM head when ``context_mode == CharLstm`` (nullptr otherwise).
    const CharLSTMHead* char_lstm() const { return lstm_.get(); }

    const cypha::intelligence::KappaTrajectoryState& kappa_trajectory_state() const {
        return kappa_trajectory_state_;
    }
    std::uint32_t train_step_count() const { return step_count_; }

    /// Public accessor for the participation-ratio D_eff computed directly over the actual
    /// LSTM hidden-state history (``lstm_h_history_rows_``, width ``lstm_->hidden``) --
    /// distinct from the GRIA-field ``d_eff`` already exported by the intelligence profile
    /// (which is measured over the fixed ``field_dim``-wide GRIA field and is therefore
    /// structurally blind to ``lstm_hidden``). Returns -1.0 if there's no LSTM head or fewer
    /// than 4 hidden-state rows have been observed yet (see ``lstm_hidden_d_eff()``).
    double lstm_hidden_d_eff_report() const { return lstm_hidden_d_eff(); }

    /// Full raw/normalized/sample-ratio detail behind ``lstm_hidden_d_eff_report()`` --
    /// see ``LstmHiddenDEffReport`` for field semantics (Phase 3 follow-up).
    LstmHiddenDEffReport lstm_hidden_d_eff_detail() const;

    friend void save_cyphalm_model(const CyphaLMModel& model, const std::string& base_path);

 private:
    CyphaLMConfig cfg_;
    std::unique_ptr<EmbedTable> embed_;
    std::unique_ptr<CellAISSM> ssm_;
    std::unique_ptr<HierarchicalSSM> hierarchical_ssm_;
    std::unique_ptr<HebbianStack> hebbian_stack_;
    std::unique_ptr<BpeTokenizer> bpe_;
    std::unique_ptr<GRIALowRank> gria_;
    std::unique_ptr<CharLSTMHead> lstm_;
    std::unique_ptr<SelectiveSSM> selective_;
    std::unique_ptr<CompressiveMemory> memory_;
    std::unique_ptr<ContextBank> context_bank_;
    std::unique_ptr<NigStateCell> nig_state_cell_;
    std::unique_ptr<ReversibleSSMCell> reversible_cell_;
    std::unique_ptr<NgramFusion> ngram_fusion_;
    std::unique_ptr<cypha::som::GNGExpertManager> gng_;
    std::unique_ptr<cypha::som::GRIAController> gria_controller_;
    std::unique_ptr<cypha::som::DiscriminativeFeedback> discriminative_feedback_;

    std::unique_ptr<CyphaDIF> dif_;
    std::unique_ptr<ViewEmbedding> view_emb_;
    std::unique_ptr<cypha::rpsm::RpsmSequenceLayer> rpsm_layer_;
    std::unique_ptr<cypha::intelligence::IntelligenceProfiler> train_profiler_;
    std::vector<double> rpsm_log_probs_;

    std::vector<double> proj_ssm_;
    std::vector<double> proj_dif_;
    std::vector<double> proj_embed_;
    std::vector<double> proj_ngram_;
    std::vector<double> proj_ngram_embed_;
    DIFPredictOutput last_dif_out_;
    std::vector<double> field_x_;
    std::vector<double> gria_in_;
    std::vector<double> lstm_h_;
    std::vector<double> lstm_c_;
    std::vector<std::vector<double>> embed_history_;
    std::vector<double> token_counts_;
    std::vector<std::uint32_t> token_history_;
    std::unordered_map<std::uint64_t, std::vector<double>> ngram_count_table_;
    std::vector<double> last_e_;
    std::vector<double> last_ctx_;
    std::vector<double> last_ssm_h_fast_;
    std::vector<std::vector<double>> bptt_buffer_;
    std::vector<std::vector<double>> bptt_slow_buffer_;
    // §14 (RPSM_UPGRADE_PLAN.md §13.6/§13.7(a)): token ids for the RPSM BPTT window currently
    // in flight, in the same chronological order `RpsmSequenceLayer` fills its own window cache
    // -- zipped with `rpsm_layer_->bptt_window_input_grads()` at flush time so each cached
    // step's *own* token gets the corresponding embedding-gradient correction.
    std::vector<std::uint32_t> rpsm_bptt_token_ids_;
    int gria_d_in_ = 160;
    int last_gng_bmu_ = 0;
    int current_view_slot_ = 0;
    std::uint32_t step_count_ = 0;
    cypha::intelligence::KappaTrajectoryState kappa_trajectory_state_;
    double hybrid_blend_logit_ = 0.0;
    double last_mean_alpha_ = 0.5;
    double last_profile_tau_ = 0.65;
    double last_profile_r_eu_ = 0.70;
    double last_train_loss_ = 0.0;
    CharLSTMCache hybrid_lstm_cache_;
    bool hybrid_lstm_has_cache_{false};
    // Perf: reused across train_step calls via CharLSTMHead::backward_step's out-param overload
    // (see docs/reports/PERFORMANCE_PROFILE_2026-07-12.md) so the hybrid-mode LSTM backward pass
    // doesn't re-allocate its ~1.5MB gradient buffers (dWx/dWh/dE/dWy) on every character step.
    // Safe to reuse: CyphaLMModel::train_step runs on a single thread per model instance (this is
    // not the CyphaLMBatch multi-thread batch path), and the buffer is fully overwritten by
    // backward_step before being read by apply_grads immediately below it.
    CharLSTMGrad hybrid_lstm_grad_scratch_;
    // Perf (2026-07-12, part 2, docs/reports/PERFORMANCE_PROFILE_2026-07-12.md "Follow-up"):
    // bptt_ssm_update()'s own allocation-churn fix, same pattern/rationale as
    // hybrid_lstm_grad_scratch_ above -- these were previously fresh-allocated `std::vector`
    // locals every single train_step call (bptt_ssm_update runs once per call whenever BPTT-on-
    // SSM is active); reused via `.assign()`/`.resize()` in place instead. Safe for the same
    // reason as hybrid_lstm_grad_scratch_: CyphaLMModel::train_step runs on a single thread per
    // model instance, and every element is fully overwritten before being read each call.
    std::vector<double> bptt_grad_core_scratch_;
    std::vector<double> bptt_grad_field_scratch_;
    std::vector<double> bptt_delta_rows_scratch_;
    std::vector<double> bptt_inv_v_scratch_;
    std::vector<double> bptt_grad_ctx_scratch_;
    std::vector<double> last_hybrid_log_g_;
    std::vector<double> last_hybrid_log_l_;
    HybridEwcRegularizer ewc_;
    // Phase 3 follow-up (2026-07-12, docs/reports/HIDDEN_DIM_SCALE_PLAN.md "Finding 2"):
    // this used to be a fixed `kLstmHiddenHistoryMax = 48` regardless of `lstm_hidden`,
    // which meant the sample/dims ratio feeding `lstm_hidden_d_eff()`'s participation-
    // ratio estimate *shrank* as `lstm_hidden` grew (0.375 at hidden=128, down to 0.094
    // at hidden=512 -- a 4x more severely underdetermined covariance estimate at exactly
    // the widths this metric is supposed to compare). `lstm_h_history_max_` is now sized
    // from `cfg_.lstm_hidden` at construction time (see `init_components()`) so the
    // sample count tracks hidden width instead of a constant. Kept as a per-instance
    // member (not `static constexpr`) since it depends on `cfg_`, which is only known at
    // runtime. `std::deque` (not `std::vector`) so the ring-buffer's `pop_front()` stays
    // O(1) instead of O(history rows) now that the row count itself scales with
    // `lstm_hidden` and can reach several hundred to ~1000+ rows -- this is called once
    // per training step (`append_lstm_hidden_history`, cyphalm_model.cpp), so an O(rows)
    // per-step cost would have compounded across a 300k-step production run.
    int lstm_h_history_max_ = 48;
    std::deque<std::vector<double>> lstm_h_history_rows_;

    void init_components();
    void append_lstm_hidden_history(const std::vector<double>& h);
    double lstm_hidden_d_eff() const;
    void record_embedding(const std::vector<double>& e);
    void record_token_history(std::uint32_t token_id);
    std::vector<double> ngram_count_log_prior() const;
    std::uint64_t ngram_context_key() const;
    void observe_ngram_count(std::uint32_t next_token_id);
    std::vector<double> ngram_embedding_vector() const;
    std::vector<double> build_gria_input(const std::vector<double>& field,
                                         const DIFPredictOutput* dif_out);
    double hybrid_forget_gate_scale(const DIFPredictOutput& dif_out) const;
    std::vector<double> augment_gria_input(const std::vector<double>& v) const;
    std::vector<double> gria_input_core(const std::vector<double>& field,
                                        const DIFPredictOutput* dif_out) const;
    int ssm_context_dim() const;
    std::vector<double> ssm_step(const std::vector<double>& e);
    void apply_hebbian_hooks(std::vector<double>& ctx);
    void fill_top_k(const std::vector<double>& log_probs, PredictNextOutput& out, int k = 5) const;
    std::vector<double> project_field(const std::vector<double>& ctx);
    void bptt_ssm_update(std::uint32_t next_token_id);
    void apply_lstm_ewc(TrainStepMetrics& m, const CharLSTMGrad& grads);
    void apply_hybrid_ewc(TrainStepMetrics& m, const HybridEwcGradStub& grads);
    TrainStepMetrics train_step_rpsm(std::uint32_t token_id, std::uint32_t next_token_id,
                                     cypha::intelligence::IntelligenceProfiler* profiler = nullptr,
                                     LmIntelligenceMonitor* monitor = nullptr);
    void train_sequence_rpsm(const std::vector<int>& ids, int n_steps, int epochs,
                             cypha::intelligence::IntelligenceProfiler* profiler = nullptr);
    void rpsm_embed_backprop(std::uint32_t token_id);
    void rpsm_embed_backprop_from_grad(std::uint32_t token_id, const std::vector<double>& field_grad_raw);
    void rpsm_bptt_embed_flush();
    void set_view_slot(int slot) { current_view_slot_ = slot; }
    void refresh_laplace_prior();

    /// Paper IV self-correcting loop (opt-in via `cfg_.use_self_correcting_loop`, see its
    /// doc comment in `cyphalm_config.hpp`): re-blend the hybrid GRIA/LSTM logits at a wider
    /// deliberation setting when live r_eu exceeds `threshold`, keeping whichever pass has
    /// higher confidence. Returns `initial` unchanged when the flag is off, when
    /// `context_mode != Hybrid`, or when r_eu does not exceed the threshold.
    PredictNextOutput self_correct_if_needed(const PredictNextOutput& initial,
                                             cypha::intelligence::EpistemicThreshold& threshold) const;

    friend CyphaLMModel load_cyphalm_model(const std::string& json_path);
};

using CyphaLMNative = CyphaLMModel;

}  // namespace cyphalm
}  // namespace cypha
