#pragma once

#include <cstdint>
#include <memory>
#include <string>
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
#include "cypha/cyphalm/embed_table.hpp"
#include "cypha/cyphalm/gria_lowrank.hpp"
#include "cypha/cyphalm/hebbian_stack.hpp"
#include "cypha/cyphalm/hierarchical_ssm.hpp"
#include "cypha/cyphalm/ngram_fusion.hpp"
#include "cypha/cyphalm/selective_ssm.hpp"
#include "cypha/cyphalm/view_embedding.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/rpsm/rpsm_sequence_layer.hpp"
#include "cypha/som/discriminative_feedback.hpp"
#include "cypha/som/gng_expert.hpp"
#include "cypha/som/gria_controller.hpp"

namespace cypha::intelligence {
class IntelligenceProfiler;
}  // namespace cypha::intelligence

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
    TrainStepMetrics train_step(std::uint32_t token_id, std::uint32_t next_token_id);
    std::vector<double> forward_log_probs(std::uint32_t token_id);

    void train_sequence(const std::vector<int>& ids, int n_steps, int epochs);
    void train_sequence_views(const std::vector<int>& ids);
    double eval_bpc(const std::vector<int>& ids, int n_eval,
                    cypha::intelligence::IntelligenceProfiler* profiler = nullptr);

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
    std::vector<double> last_e_;
    std::vector<double> last_ctx_;
    std::vector<std::vector<double>> bptt_buffer_;
    int gria_d_in_ = 160;
    int last_gng_bmu_ = 0;
    int current_view_slot_ = 0;
    std::uint32_t step_count_ = 0;
    double hybrid_blend_logit_ = 0.0;
    CharLSTMCache hybrid_lstm_cache_;
    bool hybrid_lstm_has_cache_{false};
    std::vector<double> last_hybrid_log_g_;
    std::vector<double> last_hybrid_log_l_;

    void init_components();
    void record_embedding(const std::vector<double>& e);
    std::vector<double> ngram_embedding_vector() const;
    std::vector<double> build_gria_input(const std::vector<double>& field,
                                         const DIFPredictOutput* dif_out);
    std::vector<double> augment_gria_input(const std::vector<double>& v) const;
    std::vector<double> gria_input_core(const std::vector<double>& field,
                                        const DIFPredictOutput* dif_out) const;
    int ssm_context_dim() const;
    std::vector<double> ssm_step(const std::vector<double>& e);
    void apply_hebbian_hooks(std::vector<double>& ctx);
    void fill_top_k(const std::vector<double>& log_probs, PredictNextOutput& out, int k = 5) const;
    std::vector<double> project_field(const std::vector<double>& ctx);
    void bptt_ssm_update(std::uint32_t next_token_id);
    void set_view_slot(int slot) { current_view_slot_ = slot; }
    void refresh_laplace_prior();

    friend CyphaLMModel load_cyphalm_model(const std::string& json_path);
};

using CyphaLMNative = CyphaLMModel;

}  // namespace cyphalm
}  // namespace cypha
