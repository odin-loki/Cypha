#pragma once

/// CyphaLM sequence model — backed by hp (Hutter Prize integer-exact context mixer).
/// Vendored from odin-loki/CompressionAlgorithm hp/ tree.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/bpe_tokenizer.hpp"
#include "cypha/cyphalm/cyphalm_alpha_spectrum.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_ewc_regularizer.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_guided_loss.hpp"

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

struct LstmHiddenDEffReport {
    double normalized = -1.0;
    double raw = -1.0;
    double sample_ratio = -1.0;
    int n_samples = 0;
    int n_dims = 0;
};

/// Forward declarations for legacy API stubs (Hybrid GRIA+LSTM removed; hp is the LLM core).
class CellAISSM;
class GRIALowRank;
class CharLSTMHead;
class PGMCell;

/// Unified CyphaLM stack — hp context-mixing compressor as the sequence/LLM algorithm.
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
    void reset_optim_state();

    PredictNextOutput predict_next(std::uint32_t token_id);
    PredictNextOutput repredict_hybrid_blend(double blend_logit) const;
    TrainStepMetrics adapt_after_predict(std::uint32_t next_token_id, double lr_scale = 1.0,
                                         cypha::intelligence::IntelligenceProfiler* profiler = nullptr,
                                         LmIntelligenceMonitor* monitor = nullptr);
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

    /// Mean −log₂ P(byte) via ``observe_next_byte`` (hp compress encode math, no 256-clone path).
    /// Scores every byte including the first; online-adapts like ``hp c``.
    double eval_bpc_compress_equivalent(const std::vector<int>& ids, int n_eval);

    /// Alias for ``eval_bpc_compress_equivalent`` (harness / reporting name).
    double compress_equivalent_bpc(const std::vector<int>& ids, int n_eval) {
        return eval_bpc_compress_equivalent(ids, n_eval);
    }

    void accumulate_intelligence_profile(const std::vector<int>& ids, int n_steps,
                                         cypha::intelligence::IntelligenceProfiler& profiler);

    std::vector<std::uint32_t> encode_text(const std::string& text) const;
    std::string decode_tokens(const std::vector<std::uint32_t>& ids) const;
    bool has_bpe_tokenizer() const { return bpe_ != nullptr; }

    double hybrid_blend_logit() const { return 0.0; }
    double hybrid_gria_weight() const { return 1.0; }
    void set_hybrid_blend_logit(double) {}

    std::vector<double> hidden_knn_log_probs() const { return {}; }

    AlphaSpectrumSnapshot alpha_spectrum_snapshot() const;
    nlohmann::json compression_profile() const;

    nlohmann::json ssm_diagnostic_report(const std::vector<int>& token_ids, int max_steps);

    CellAISSM* active_ssm() { return nullptr; }
    const CellAISSM* active_ssm() const { return nullptr; }
    std::vector<double> embed_vector(std::uint32_t token_id) const;
    const std::vector<double>& field_vector() const { return field_stub_; }
    double ssm_projection_rms() const { return 0.0; }
    bool has_gria_routing() const { return false; }
    const GRIALowRank* gria_routing() const { return nullptr; }

    void ewc_snapshot() {}
    double ewc_penalty() const { return 0.0; }
    CyphaLMEwcRegularizer& ewc_regularizer() { return ewc_stub_; }
    const CyphaLMEwcRegularizer& ewc_regularizer() const { return ewc_stub_; }
    HybridEwcRegularizer& hybrid_ewc_regularizer() { return hybrid_ewc_stub_; }
    const HybridEwcRegularizer& hybrid_ewc_regularizer() const { return hybrid_ewc_stub_; }

    const CharLSTMHead* char_lstm() const { return nullptr; }
    const PGMCell* pgm_cell() const { return nullptr; }
    PGMCell* pgm_cell() { return nullptr; }

    const cypha::intelligence::KappaTrajectoryState& kappa_trajectory_state() const {
        return kappa_trajectory_state_;
    }
    std::uint32_t train_step_count() const { return step_count_; }

    double lstm_hidden_d_eff_report() const { return -1.0; }
    LstmHiddenDEffReport lstm_hidden_d_eff_detail() const { return {}; }

    const HpSequenceBackend& hp_backend() const { return *hp_; }
    HpSequenceBackend& hp_backend() { return *hp_; }

    friend void save_cyphalm_model(const CyphaLMModel& model, const std::string& base_path);

 private:
    CyphaLMConfig cfg_;
    std::unique_ptr<HpSequenceBackend> hp_;
    std::unique_ptr<BpeTokenizer> bpe_;
    PredictNextOutput last_predict_out_;
    std::vector<double> field_stub_;
    cypha::intelligence::KappaTrajectoryState kappa_trajectory_state_;
    CyphaLMEwcRegularizer ewc_stub_;
    HybridEwcRegularizer hybrid_ewc_stub_;
    std::uint32_t step_count_ = 0;
    double last_train_loss_ = 0.0;

    void init_components();
    void fill_top_k(const std::vector<double>& log_probs, PredictNextOutput& out, int k = 5) const;
    void remember_last_predict(const PredictNextOutput& out);
    std::uint8_t token_to_byte(std::uint32_t token_id) const;

    friend CyphaLMModel load_cyphalm_model(const std::string& json_path);
};

using CyphaLMNative = CyphaLMModel;

}  // namespace cyphalm
}  // namespace cypha
