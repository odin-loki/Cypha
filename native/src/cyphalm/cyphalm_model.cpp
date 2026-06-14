#include "cypha/cyphalm/cyphalm_intelligence_hook.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cellai_ssm.hpp"
#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/hybrid_blend.hpp"
#include "cypha/cyphalm/npz_util.hpp"
#include "cypha/cyphalm/cyphalm_views.hpp"
#include "cypha/cyphalm/ssm_diagnose.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_guided_loss.hpp"

namespace cypha {
namespace cyphalm {

namespace {

constexpr double kLog2 = 0.6931471805599453;
constexpr double kLogEps = 1e-12;

bool uses_ssm(ContextMode m) {
    return m != ContextMode::CharLstm;
}

bool uses_gria(ContextMode m) {
    return m != ContextMode::CharLstm;
}

bool uses_rpsm(const CyphaLMConfig& cfg) {
    return cfg.use_rpsm_layer || cfg.context_mode == ContextMode::Rpsm;
}

bool uses_lstm(ContextMode m) {
    return m == ContextMode::Hybrid || m == ContextMode::CharLstm;
}

void init_proj(std::vector<double>& proj, int rows, int cols, std::uint64_t seed, double scale) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, scale);
    proj.assign(static_cast<std::size_t>(rows * cols), 0.0);
    for (auto& v : proj) v = nd(rng);
}

void init_proj_from_rng(std::vector<double>& proj, int rows, int cols, std::mt19937_64& rng,
                        double scale) {
    std::normal_distribution<double> nd(0.0, scale);
    proj.assign(static_cast<std::size_t>(rows * cols), 0.0);
    for (auto& v : proj) v = nd(rng);
}

std::vector<double> resize_to_field(const std::vector<double>& v, int field_dim) {
    std::vector<double> out(static_cast<std::size_t>(field_dim), 0.0);
    const int n = std::min(field_dim, static_cast<int>(v.size()));
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = v[static_cast<std::size_t>(i)];
    }
    return out;
}

std::vector<double> matvec(const std::vector<double>& m, int rows, int cols,
                           const std::vector<double>& x) {
    std::vector<double> out(static_cast<std::size_t>(rows), 0.0);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int c = 0; c < cols; ++c) {
            acc += m[static_cast<std::size_t>(r * cols + c)] * x[static_cast<std::size_t>(c)];
        }
        out[static_cast<std::size_t>(r)] = acc;
    }
    return out;
}

std::vector<double> matvec_transpose(const std::vector<double>& m, int rows, int cols,
                                     const std::vector<double>& x) {
    std::vector<double> out(static_cast<std::size_t>(cols), 0.0);
    for (int c = 0; c < cols; ++c) {
        double acc = 0.0;
        for (int r = 0; r < rows; ++r) {
            acc += m[static_cast<std::size_t>(r * cols + c)] * x[static_cast<std::size_t>(r)];
        }
        out[static_cast<std::size_t>(c)] = acc;
    }
    return out;
}

}  // namespace

CyphaLMModel::CyphaLMModel(CyphaLMConfig cfg) : cfg_(std::move(cfg)) {
    hybrid_blend_logit_ = cfg_.hybrid_blend_logit;
    init_components();
}

CyphaLMModel::~CyphaLMModel() = default;

void CyphaLMModel::init_components() {
    const auto mode = cfg_.context_mode;
    if (mode != ContextMode::CharLstm) {
        embed_ = std::make_unique<EmbedTable>(static_cast<std::uint32_t>(cfg_.vocab_size),
                                              static_cast<std::uint32_t>(cfg_.d_embed),
                                              static_cast<std::uint32_t>(cfg_.seed));
    }
    if (uses_ssm(mode)) {
        CellAISSMConfig sc;
        sc.d_input = cfg_.d_embed;
        sc.d_state = cfg_.d_state;
        sc.tau_fast = cfg_.tau_fast;
        sc.tau_slow = cfg_.tau_slow;
        sc.n_layers = cfg_.ssm_layers;
        sc.seed = static_cast<int>(cfg_.seed + 1);
        sc.use_spectral_pde = cfg_.use_spectral_pde;
        sc.use_multiscale = cfg_.use_multiscale;
        sc.use_sparse_hebbian = cfg_.use_sparse_hebbian && !cfg_.use_hebbian_stack;
        if (cfg_.use_hierarchical_ssm) {
            hierarchical_ssm_ = std::make_unique<HierarchicalSSM>(sc, cfg_.compress_interval);
        } else {
            ssm_ = std::make_unique<CellAISSM>(sc);
        }
        if (cfg_.use_hebb_graph && !cfg_.use_hebbian_stack) {
            HebbianGraphConfig gc;
            gc.n = 2 * cfg_.d_state;
            if (auto* active = active_ssm()) {
                active->enable_hebb_graph(gc);
            }
        }
        if (cfg_.use_temporal_som) {
            cypha::som::TemporalSOMConfig tc;
            tc.M = 8;
            tc.L_max = 16;
            if (auto* active = active_ssm()) {
                active->enable_temporal_som(tc);
            }
        }
        if (cfg_.use_hebbian_stack) {
            hebbian_stack_ = std::make_unique<HebbianStack>();
            HebbianStackConfig hcfg;
            hcfg.use_sparse_hebbian = cfg_.use_sparse_hebbian;
            hcfg.use_hebb_graph = cfg_.use_hebb_graph;
            hcfg.ssm_hebb_lr = cfg_.ssm_hebb_lr;
            hcfg.d_state = cfg_.d_state;
            hcfg.n_layers = cfg_.ssm_layers;
            hcfg.graph.n = 2 * cfg_.d_state;
            hebbian_stack_->configure(hcfg);
        }
        {
            std::mt19937_64 rng(cfg_.seed);
            constexpr double kScale = 0.02;
            init_proj_from_rng(proj_ssm_, cfg_.field_dim, ssm_context_dim(), rng, kScale);
            init_proj_from_rng(proj_dif_, cfg_.field_dim, cfg_.field_dim, rng, kScale);
            init_proj_from_rng(proj_embed_, cfg_.field_dim, cfg_.d_embed, rng, kScale);
            const int n_pos = 1 + std::max(0, cfg_.ngram_context);
            const int ngram_in = cfg_.field_dim + n_pos * cfg_.d_embed;
            std::normal_distribution<double> nd(0.0, kScale);
            for (int i = 0; i < cfg_.field_dim * ngram_in; ++i) {
                (void)nd(rng);
            }
        }
    }
    if (cfg_.use_gng) {
        gng_ = std::make_unique<cypha::som::GNGExpertManager>(cfg_.field_dim);
        if (cfg_.use_gria_controller) {
            gria_controller_ = std::make_unique<cypha::som::GRIAController>();
        }
    }
    if (cfg_.use_discriminative_feedback) {
        discriminative_feedback_ = std::make_unique<cypha::som::DiscriminativeFeedback>();
    }
    if (uses_gria(mode)) {
        gria_d_in_ = cfg_.field_dim;
        if (cfg_.view_id_dim > 0) {
            gria_d_in_ += cfg_.view_id_dim;
            view_emb_ = std::make_unique<ViewEmbedding>(
                std::max(1, cfg_.max_view_slots), cfg_.view_id_dim, cfg_.seed + 3,
                cfg_.view_learnable);
        }
        gria_ = std::make_unique<GRIALowRank>(
            gria_d_in_, cfg_.vocab_size, cfg_.gria_rank, cfg_.alpha_init,
            cfg_.alpha_learnable, cfg_.seed + 2);
        gria_in_.assign(static_cast<std::size_t>(gria_d_in_), 0.0);
        token_counts_.assign(static_cast<std::size_t>(cfg_.vocab_size), 1.0);
        refresh_laplace_prior();
    }
    if (mode != ContextMode::CharLstm) {
        dif_ = std::make_unique<CyphaDIF>(cfg_);
    }
    const auto ngram_path = mode == ContextMode::Hybrid || mode == ContextMode::GriaNgram ||
                            mode == ContextMode::AblationNoSsm || mode == ContextMode::Rpsm;
    const int n_pos = 1 + std::max(0, cfg_.ngram_context);
    if (ngram_path) {
        embed_history_.assign(static_cast<std::size_t>(n_pos),
                              std::vector<double>(static_cast<std::size_t>(cfg_.d_embed), 0.0));
    }
    if (cfg_.ngram_fuse_split && ngram_path) {
        const int embed_in = n_pos * cfg_.d_embed;
        const int field_in = (mode == ContextMode::AblationNoSsm) ? 0 : cfg_.field_dim;
        if (field_in > 0) {
            ngram_fusion_ = std::make_unique<NgramFusion>(
                cfg_.field_dim, field_in, embed_in, cfg_.ngram_fusion, n_pos,
                cfg_.ngram_position_weights, cfg_.seed + 4);
        } else {
            std::mt19937_64 rng(cfg_.seed + 4);
            init_proj_from_rng(proj_ngram_embed_, cfg_.field_dim, embed_in, rng, 0.02);
            const int ngram_in = embed_in;
            init_proj_from_rng(proj_ngram_, cfg_.field_dim, ngram_in, rng, 0.02);
        }
    }
    if (uses_lstm(mode)) {
        lstm_ = std::make_unique<CharLSTMHead>(cfg_.vocab_size, cfg_.lstm_hidden, cfg_.seed + 5);
        if (cfg_.use_eml_activation) {
            lstm_->set_activation_mode(LSTMActivationMode::Eml);
        }
        lstm_h_.assign(static_cast<std::size_t>(cfg_.lstm_hidden), 0.0);
        lstm_c_.assign(static_cast<std::size_t>(cfg_.lstm_hidden), 0.0);
    }
    if (mode == ContextMode::SsmGriaNoLstm || mode == ContextMode::Full) {
        selective_ = std::make_unique<SelectiveSSM>(
            static_cast<std::uint32_t>(cfg_.d_embed), static_cast<std::uint32_t>(cfg_.d_state),
            static_cast<std::uint32_t>(cfg_.field_dim), true, 0.9,
            static_cast<std::uint32_t>(cfg_.seed + 13));
    }
    if (cfg_.max_memory_slots > 0 && mode != ContextMode::CharLstm) {
        memory_ = std::make_unique<CompressiveMemory>(
            static_cast<std::uint32_t>(cfg_.field_dim),
            static_cast<std::uint32_t>(cfg_.max_memory_slots), cfg_.nig_kappa0, cfg_.nig_alpha0,
            cfg_.nig_beta0, static_cast<std::uint32_t>(cfg_.seed + 17));
        memory_->set_compress_interval(static_cast<std::uint32_t>(cfg_.compress_interval));
    }
    if (cfg_.use_context_bank) {
        context_bank_ = std::make_unique<ContextBank>(cfg_.d_embed, cfg_.context_bank_slots);
    }
    field_x_.assign(static_cast<std::size_t>(cfg_.field_dim), 0.0);
    if (gria_in_.empty()) {
        gria_in_.assign(static_cast<std::size_t>(cfg_.field_dim), 0.0);
    }
    if (!cfg_.bpe_merges_path.empty() && !cfg_.bpe_vocab_path.empty()) {
        bpe_ = std::make_unique<BpeTokenizer>(
            BpeTokenizer::load(cfg_.bpe_merges_path, cfg_.bpe_vocab_path));
    }
    if (uses_rpsm(cfg_)) {
        cypha::rpsm::RpsmSequenceConfig rc;
        rc.n_levels = cfg_.rpsm_n_levels;
        rc.state_dim = cfg_.rpsm_state_dim;
        rc.feat_dim = cfg_.rpsm_feat_dim;
        rc.n_classes = cfg_.vocab_size;
        rc.seed = cfg_.seed + 29;
        rpsm_layer_ = std::make_unique<cypha::rpsm::RpsmSequenceLayer>(rc);
        rpsm_log_probs_.assign(static_cast<std::size_t>(cfg_.vocab_size), 0.0);
    }
    if (cfg_.profile_guided_loss) {
        train_profiler_ = std::make_unique<cypha::intelligence::IntelligenceProfiler>();
    }
}

CyphaLMModel CyphaLMModel::from_json_npz(const std::string& json_path) {
    return load_cyphalm_model(json_path);
}

void CyphaLMModel::save(const std::string& base_path) const {
    save_cyphalm_model(*this, base_path);
}

void CyphaLMModel::reset_context() {
    if (hierarchical_ssm_) hierarchical_ssm_->reset();
    if (ssm_) ssm_->reset();
    if (selective_) selective_->reset();
    if (memory_) memory_->reset();
    if (context_bank_) context_bank_->reset();
    if (lstm_) {
        lstm_->reset_state();
        std::fill(lstm_h_.begin(), lstm_h_.end(), 0.0);
        std::fill(lstm_c_.begin(), lstm_c_.end(), 0.0);
    }
    hybrid_lstm_has_cache_ = false;
    last_hybrid_log_g_.clear();
    last_hybrid_log_l_.clear();
    for (auto& row : embed_history_) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    bptt_buffer_.clear();
    last_e_.clear();
    last_ctx_.clear();
    if (rpsm_layer_) rpsm_layer_->reset();
    // DIF experts persist across view/block resets (Python carry_dif=True).
    std::fill(field_x_.begin(), field_x_.end(), 0.0);
    step_count_ = 0;
}

void CyphaLMModel::record_embedding(const std::vector<double>& e) {
    if (embed_history_.empty()) return;
    for (std::size_t j = embed_history_.size() - 1; j > 0; --j) {
        embed_history_[j] = embed_history_[j - 1];
    }
    embed_history_[0] = e;
}

std::vector<double> CyphaLMModel::ngram_embedding_vector() const {
    std::vector<double> out;
    for (const auto& row : embed_history_) {
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

void CyphaLMModel::refresh_laplace_prior() {
    if (gria_ && cfg_.laplace_smoothing > 0.0 && !token_counts_.empty()) {
        gria_->set_laplace_prior(token_counts_.data(), static_cast<int>(token_counts_.size()),
                                 cfg_.laplace_smoothing);
    }
}

std::vector<double> CyphaLMModel::augment_gria_input(const std::vector<double>& v) const {
    if (cfg_.view_id_dim <= 0 || !view_emb_) return v;
    std::vector<double> out = v;
    const auto vv = view_emb_->forward(current_view_slot_);
    out.insert(out.end(), vv.begin(), vv.end());
    return out;
}

int CyphaLMModel::ssm_context_dim() const {
    if (hierarchical_ssm_) return hierarchical_ssm_->fast_tier().context_dim();
    if (ssm_) return ssm_->context_dim();
    return 0;
}

CellAISSM* CyphaLMModel::active_ssm() {
    if (hierarchical_ssm_) {
        return const_cast<CellAISSM*>(&hierarchical_ssm_->fast_tier());
    }
    return ssm_.get();
}

const CellAISSM* CyphaLMModel::active_ssm() const {
    if (hierarchical_ssm_) return &hierarchical_ssm_->fast_tier();
    return ssm_.get();
}

std::vector<double> CyphaLMModel::ssm_step(const std::vector<double>& e) {
    std::vector<double> ctx;
    if (hierarchical_ssm_) {
        ctx = hierarchical_ssm_->step(e);
    } else if (ssm_) {
        ctx = ssm_->step(e);
    }
    if (cfg_.use_differential_gate && !ctx.empty() && last_ctx_.size() == ctx.size()) {
        constexpr double kDeltaBlend = 0.25;
        for (std::size_t i = 0; i < ctx.size(); ++i) {
            ctx[i] = (1.0 - kDeltaBlend) * last_ctx_[i] + kDeltaBlend * ctx[i];
        }
    }
    return ctx;
}

void CyphaLMModel::apply_hebbian_hooks(std::vector<double>& ctx) {
    if (!hebbian_stack_) return;
    const CellAISSM* active = active_ssm();
    if (!active || active->n_layers() < 1) return;
    const auto& h = active->h_states();
    const auto& s = active->s_states();
    hebbian_stack_->on_ssm_layer_context(ctx, 0, h[0].data(), s[0].data());
}

std::vector<double> CyphaLMModel::project_field(const std::vector<double>& ctx) {
    return matvec(proj_ssm_, cfg_.field_dim, ssm_context_dim(), ctx);
}

std::vector<double> CyphaLMModel::gria_input_core(const std::vector<double>& field,
                                                  const DIFPredictOutput* dif_out) const {
    const auto mode = cfg_.context_mode;
    if (ngram_fusion_ &&
        (mode == ContextMode::Hybrid || mode == ContextMode::GriaNgram ||
         mode == ContextMode::AblationNoSsm || mode == ContextMode::Rpsm)) {
        return ngram_fusion_->forward(field, ngram_embedding_vector());
    }
    if (mode == ContextMode::AblationNoSsm && !proj_ngram_.empty()) {
        const auto embeds = ngram_embedding_vector();
        if (!proj_ngram_embed_.empty()) {
            return matvec(proj_ngram_embed_, cfg_.field_dim, static_cast<int>(embeds.size()),
                          embeds);
        }
        return matvec(proj_ngram_, cfg_.field_dim, static_cast<int>(embeds.size()), embeds);
    }
    if (mode == ContextMode::SsmGria) {
        return field;
    }
    if (dif_out != nullptr && !proj_dif_.empty()) {
        const auto mean_resized = resize_to_field(dif_out->mean, cfg_.field_dim);
        std::vector<double> mean = matvec(proj_dif_, cfg_.field_dim, cfg_.field_dim, mean_resized);
        if (mode == ContextMode::Full || mode == ContextMode::SsmGriaNoLstm) {
            const double u = dif_out->epistemic_var;
            for (std::size_t i = 0; i < mean.size() && i < field.size(); ++i) {
                mean[i] += u * field[i] * 0.01;
            }
        }
        return mean;
    }
    return field;
}

std::vector<double> CyphaLMModel::build_gria_input(const std::vector<double>& field,
                                                   const DIFPredictOutput* dif_out) {
    std::vector<double> v = gria_input_core(field, dif_out);
    if (memory_ && !field.empty()) {
        const auto bias = memory_->retrieve(field.data(), static_cast<std::uint32_t>(field.size()));
        for (std::size_t i = 0; i < bias.size() && i < v.size(); ++i) {
            v[i] += bias[i] * 0.01;
        }
    }
    if (context_bank_ && !embed_history_.empty() && !proj_embed_.empty()) {
        const auto& q = embed_history_[0];
        const auto attn = context_bank_->linear_attention(q);
        const auto attn_field = matvec(proj_embed_, cfg_.field_dim, cfg_.d_embed, attn);
        for (std::size_t i = 0; i < attn_field.size() && i < v.size(); ++i) {
            v[i] += attn_field[i] * 0.05;
        }
        context_bank_->push(q.data(), cfg_.d_embed);
    }
    return augment_gria_input(v);
}

void CyphaLMModel::fill_top_k(const std::vector<double>& log_probs, PredictNextOutput& out,
                              int k) const {
    const int n = static_cast<int>(log_probs.size());
    std::vector<int> idx(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
    std::partial_sort(idx.begin(), idx.begin() + std::min(k, n), idx.end(),
                      [&](int a, int b) { return log_probs[static_cast<std::size_t>(a)] >
                                                 log_probs[static_cast<std::size_t>(b)]; });
    const int m = std::min(k, n);
    out.top_k_tokens.resize(static_cast<std::size_t>(m));
    out.top_k_probs.resize(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
        const int id = idx[static_cast<std::size_t>(i)];
        out.top_k_tokens[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(id);
        out.top_k_probs[static_cast<std::size_t>(i)] =
            std::exp(log_probs[static_cast<std::size_t>(id)]);
    }
}

PredictNextOutput CyphaLMModel::predict_next(std::uint32_t token_id) {
    PredictNextOutput out;
    const auto mode = cfg_.context_mode;

    if (mode == ContextMode::CharLstm) {
        if (!lstm_) throw std::runtime_error("char_lstm without LSTM head");
        out.log_probs = lstm_->forward(static_cast<int>(token_id));
        fill_top_k(out.log_probs, out);
        return out;
    }

    std::vector<double> log_g(cfg_.vocab_size);
    if (embed_ && (active_ssm() || mode == ContextMode::AblationNoSsm)) {
        const auto e = embed_->embed_vec(token_id);
        record_embedding(e);
        last_e_ = e;
        if (mode == ContextMode::AblationNoSsm) {
            last_ctx_.assign(static_cast<std::size_t>(ssm_context_dim()), 0.0);
            field_x_.assign(static_cast<std::size_t>(cfg_.field_dim), 0.0);
        } else {
            auto ctx = ssm_step(e);
            apply_hebbian_hooks(ctx);
            last_ctx_ = ctx;
            field_x_ = project_field(ctx);
        }
        const bool skip_dif =
            mode == ContextMode::SsmGria || mode == ContextMode::AblationNoSsm;
        if (dif_ && !skip_dif) {
            last_dif_out_ = dif_->predict(field_x_.data(), static_cast<int>(field_x_.size()));
            out.epistemic_var = last_dif_out_.epistemic_var;
            out.aleatoric_var = last_dif_out_.aleatoric_var;
        } else {
            last_dif_out_ = {};
            last_dif_out_.mean.assign(static_cast<std::size_t>(cfg_.field_dim), 0.0);
        }
        if (selective_ && (mode == ContextMode::SsmGriaNoLstm || mode == ContextMode::Full)) {
            const auto sel = selective_->step(e.data(), static_cast<std::uint32_t>(e.size()));
            for (std::size_t i = 0; i < sel.size() && i < field_x_.size(); ++i) {
                field_x_[i] = 0.5 * field_x_[i] + 0.5 * sel[i];
            }
            if (memory_) {
                const auto pooled = selective_->pooled_state();
                memory_->maybe_store(step_count_, pooled.data(),
                                     static_cast<std::uint32_t>(pooled.size()));
            }
        } else if (memory_) {
            memory_->maybe_store(step_count_, field_x_.data(),
                                 static_cast<std::uint32_t>(field_x_.size()));
        }
        gria_in_ = build_gria_input(field_x_, skip_dif ? nullptr : &last_dif_out_);
        if (rpsm_layer_ && !field_x_.empty()) {
            (void)rpsm_layer_->step(field_x_.data(), static_cast<int>(field_x_.size()),
                                    rpsm_log_probs_.data());
            const auto& h = rpsm_layer_->hidden();
            for (std::size_t i = 0; i < gria_in_.size() && !h.empty(); ++i) {
                gria_in_[i] += 0.05 * h[i % h.size()];
            }
        }
        if (gng_ && !field_x_.empty()) {
            last_gng_bmu_ = gng_->step(field_x_);
        }
        if (gria_) {
            gria_->forward(gria_in_.data(), log_g.data());
        }
    }

    if (mode == ContextMode::Hybrid && lstm_ && gria_) {
        std::vector<double> log_l(cfg_.vocab_size);
        std::vector<double> h_new;
        std::vector<double> c_new;
        lstm_->forward_step(static_cast<int>(token_id), lstm_h_.data(), lstm_c_.data(),
                            log_l.data(), h_new, c_new, &hybrid_lstm_cache_);
        lstm_h_ = std::move(h_new);
        lstm_c_ = std::move(c_new);
        hybrid_lstm_has_cache_ = true;
        last_hybrid_log_g_ = log_g;
        last_hybrid_log_l_ = log_l;
        out.log_probs = blend_log_probs(log_g, log_l, hybrid_blend_logit_);
    } else if (uses_lstm(mode) && lstm_ && !gria_) {
        out.log_probs = lstm_->forward(static_cast<int>(token_id));
    } else {
        out.log_probs = std::move(log_g);
    }

    ++step_count_;
    fill_top_k(out.log_probs, out);
    return out;
}

TrainStepMetrics CyphaLMModel::train_step(std::uint32_t token_id, std::uint32_t next_token_id) {
    const auto pred = predict_next(token_id);
    TrainStepMetrics m;
    m.loss = -pred.log_probs[static_cast<std::size_t>(next_token_id)];
    m.epistemic_var = pred.epistemic_var;
    m.aleatoric_var = pred.aleatoric_var;
    m.active_experts = last_dif_out_.active_experts;

    const auto mode = cfg_.context_mode;
    if (mode == ContextMode::CharLstm && lstm_) {
        lstm_->backward(static_cast<int>(next_token_id), cfg_.lstm_lr);
        if (next_token_id < token_counts_.size()) {
            token_counts_[static_cast<std::size_t>(next_token_id)] += 1.0;
        }
        return m;
    }
    if (gria_ && !gria_in_.empty()) {
        const GRIALowRankGrad grad =
            gria_->cross_entropy_gradients(gria_in_.data(), static_cast<int>(next_token_id));
        gria_->update_weights(grad, cfg_.gria_lr);
        gria_->update_alpha(grad, cfg_.gria_lr);
        gria_->update_bias(grad, cfg_.gria_lr);
        if (view_emb_ && cfg_.view_learnable) {
            const auto grad_v = gria_->grad_v_cross_entropy(gria_in_.data(), static_cast<int>(next_token_id));
            view_emb_->update(current_view_slot_, grad_v.data() + cfg_.field_dim, cfg_.view_id_dim,
                              cfg_.view_lr);
        }
    }
    if (cfg_.profile_guided_loss && train_profiler_) {
        update_profiler_from_lm_token(*train_profiler_, last_e_, pred.log_probs, pred.epistemic_var,
                                      pred.aleatoric_var);
        const auto pg =
            cypha::intelligence::compute_profile_guided_loss_from_profiler(*train_profiler_);
        m.profile_guided_loss = pg.total;
        m.loss += pg.total;
    }
    if (cfg_.online && dif_ && embed_ && !proj_embed_.empty()) {
        const auto target = matvec(proj_embed_, cfg_.field_dim, cfg_.d_embed,
                                   embed_->embed_vec(next_token_id));
        dif_->train_step(field_x_.data(), static_cast<int>(field_x_.size()), target.data(),
                         static_cast<int>(target.size()));
    }
    if (hebbian_stack_ && cfg_.train_ssm && !field_x_.empty() && !last_e_.empty()) {
        const int nxt = static_cast<int>(next_token_id);
        hebbian_stack_->encoder_train_step(field_x_.data(), last_e_.data(), std::to_string(nxt),
                                           std::to_string(nxt), cfg_.ssm_lr);
    }
    bptt_ssm_update(next_token_id);
    if (mode == ContextMode::Hybrid && lstm_) {
        if (hybrid_lstm_has_cache_) {
            const CharLSTMGrad grads =
                lstm_->backward_step(hybrid_lstm_cache_, static_cast<int>(next_token_id));
            lstm_->apply_grads(grads, cfg_.lstm_lr);
            hybrid_lstm_has_cache_ = false;
        }
        if (cfg_.hybrid_blend_learnable && !last_hybrid_log_g_.empty() && !last_hybrid_log_l_.empty()) {
            hybrid_blend_logit_ -= cfg_.hybrid_blend_lr *
                                   blend_logit_grad(last_hybrid_log_g_.data(), last_hybrid_log_l_.data(),
                                                    cfg_.vocab_size, hybrid_blend_logit_,
                                                    static_cast<int>(next_token_id));
        }
    }
    if (next_token_id < token_counts_.size()) {
        token_counts_[static_cast<std::size_t>(next_token_id)] += 1.0;
    }
    if (gria_controller_ && gng_ && !field_x_.empty()) {
        std::vector<double> act;
        if (!last_dif_out_.mean.empty()) {
            act = last_dif_out_.mean;
        } else if (!pred.log_probs.empty()) {
            act = pred.log_probs;
        } else {
            act = field_x_;
        }
        gria_controller_->push(field_x_, act);
        (void)gria_controller_->act(last_gng_bmu_, *gng_);
    }
    refresh_laplace_prior();
    return m;
}

void CyphaLMModel::bptt_ssm_update(std::uint32_t next_token_id) {
    CellAISSM* active = active_ssm();
    if (cfg_.bptt_steps <= 0 || !active || !gria_ || last_e_.empty() || last_ctx_.empty()) {
        return;
    }
    const auto mode = cfg_.context_mode;
    if (mode == ContextMode::AblationNoSsm || mode == ContextMode::AblationNoDif) {
        return;
    }
    if (mode != ContextMode::Hybrid && mode != ContextMode::GriaNgram &&
        mode != ContextMode::SsmGria && mode != ContextMode::Full &&
        mode != ContextMode::SsmGriaNoLstm && mode != ContextMode::Rpsm) {
        return;
    }
    auto grad_v = gria_->grad_v_cross_entropy(gria_in_.data(), static_cast<int>(next_token_id));
    std::vector<double> grad_core(static_cast<std::size_t>(cfg_.field_dim), 0.0);
    for (int i = 0; i < cfg_.field_dim && i < static_cast<int>(grad_v.size()); ++i) {
        grad_core[static_cast<std::size_t>(i)] = grad_v[static_cast<std::size_t>(i)];
    }
    std::vector<double> grad_field;
    if (ngram_fusion_ &&
        (mode == ContextMode::Hybrid || mode == ContextMode::GriaNgram ||
         mode == ContextMode::AblationNoSsm)) {
        grad_field = ngram_fusion_->grad_field_x(grad_core);
    } else if (mode == ContextMode::AblationNoSsm && !proj_ngram_embed_.empty()) {
        grad_field = matvec_transpose(proj_ngram_embed_, cfg_.field_dim,
                                      static_cast<int>(proj_ngram_embed_.size()) / cfg_.field_dim,
                                      grad_core);
    } else if (mode == ContextMode::Full || mode == ContextMode::SsmGriaNoLstm) {
        const double u = last_dif_out_.epistemic_var;
        grad_field.resize(grad_core.size());
        for (std::size_t i = 0; i < grad_core.size(); ++i) {
            grad_field[i] = grad_core[i] * (u * 0.01);
        }
    } else {
        grad_field = std::move(grad_core);
    }
    if (discriminative_feedback_ && dif_ && !grad_field.empty()) {
        std::vector<double> delta_rows;
        std::vector<double> inv_v;
        if (dif_->discriminative_state(delta_rows, inv_v)) {
            const int K = static_cast<int>(delta_rows.size()) / cfg_.field_dim;
            const auto d = discriminative_feedback_->compute_d(delta_rows, K, cfg_.field_dim, inv_v);
            discriminative_feedback_->modulate_inplace(grad_field, d);
        }
    }
    const int ctx_dim = active->context_dim();
    std::vector<double> grad_ctx =
        matvec_transpose(proj_ssm_, cfg_.field_dim, ctx_dim, grad_field);
    if (static_cast<int>(grad_ctx.size()) < active->d_state()) return;
    std::vector<double> grad_h(grad_ctx.begin(), grad_ctx.begin() + active->d_state());
    const double lf = active->lambda_fast();
    std::vector<double> delta(static_cast<std::size_t>(active->d_state() * cfg_.d_embed), 0.0);
    for (int r = 0; r < active->d_state(); ++r) {
        for (int c = 0; c < cfg_.d_embed; ++c) {
            delta[static_cast<std::size_t>(r * cfg_.d_embed + c)] =
                (1.0 - lf) * grad_h[static_cast<std::size_t>(r)] * last_e_[static_cast<std::size_t>(c)];
        }
    }
    bptt_buffer_.push_back(std::move(delta));
    if (static_cast<int>(bptt_buffer_.size()) < cfg_.bptt_steps) return;
    std::vector<double> avg = bptt_buffer_.front();
    for (std::size_t i = 1; i < bptt_buffer_.size(); ++i) {
        for (std::size_t j = 0; j < avg.size(); ++j) {
            avg[j] += bptt_buffer_[i][j];
        }
    }
    for (double& v : avg) v /= static_cast<double>(bptt_buffer_.size());
    active->apply_bptt_delta_avg(avg, cfg_.ssm_lr);
    bptt_buffer_.clear();
}

std::vector<double> CyphaLMModel::forward_log_probs(std::uint32_t token_id) {
    return predict_next(token_id).log_probs;
}

namespace {

int view_slot_for_name(const std::string& name) {
    if (name == "forward") return 0;
    if (name == "block_shuffle") return 1;
    if (name == "rotated" || name == "rotate_start") return 2;
    if (name == "backward" || name == "reverse") return 3;
    return 0;
}

}  // namespace

void CyphaLMModel::train_sequence_views(const std::vector<int>& ids) {
    if (ids.size() < 2) return;
    const auto schedule =
        resolve_view_schedule_struct(cfg_.view_schedule, cfg_.seed, std::max(1, cfg_.train_epochs));
    const auto epochs = iter_view_epochs(ids, schedule, std::nullopt, cfg_.view_block_size);
    const double base_gria_lr = cfg_.gria_lr;
    int last_macro = -1;
    int step_i = 0;
    const char* log_env = std::getenv("CYPHALM_TRAIN_LOG_EVERY");
    const int log_every = log_env ? std::max(0, std::atoi(log_env)) : 0;
    for (const auto& item : epochs) {
        if (view_emb_) {
            set_view_slot(view_emb_->slot_for_view(item.view_spec.name));
        } else {
            set_view_slot(view_slot_for_name(item.view_spec.name));
        }
        if (item.epoch_idx != last_macro) {
            reset_context();
            last_macro = item.epoch_idx;
        } else if (item.reset_before) {
            reset_context();
        }
        cfg_.gria_lr =
            base_gria_lr * std::pow(cfg_.gria_lr_decay, static_cast<double>(item.epoch_idx));
        const int steps = static_cast<int>(item.segment_ids.size()) - 1;
        for (int i = 0; i < steps; ++i) {
            const auto m = train_step(
                static_cast<std::uint32_t>(item.segment_ids[static_cast<std::size_t>(i)]),
                static_cast<std::uint32_t>(item.segment_ids[static_cast<std::size_t>(i + 1)]));
            ++step_i;
            if (log_every > 0 && step_i % log_every == 0) {
                std::cerr << "[CyphaLM] train step " << step_i << " view=" << item.view_spec.name
                          << " loss=" << m.loss << std::endl;
            }
        }
    }
    cfg_.gria_lr = base_gria_lr;
}

std::vector<std::uint32_t> CyphaLMModel::encode_text(const std::string& text) const {
    if (!bpe_) {
        throw std::runtime_error("CyphaLMModel: BPE tokenizer not configured");
    }
    return bpe_->encode(text);
}

std::string CyphaLMModel::decode_tokens(const std::vector<std::uint32_t>& ids) const {
    if (!bpe_) {
        throw std::runtime_error("CyphaLMModel: BPE tokenizer not configured");
    }
    return bpe_->decode(ids);
}

void CyphaLMModel::train_sequence(const std::vector<int>& ids, int n_steps, int epochs) {
    if (ids.size() < 2) return;
    if (cfg_.view_schedule != "same_order") {
        const std::size_t take =
            std::min(ids.size(), static_cast<std::size_t>(std::max(1, n_steps) + 1));
        train_sequence_views(std::vector<int>(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(take)));
        return;
    }
    const int ep_count = std::max(1, epochs);
    const double base_gria_lr = cfg_.gria_lr;
    for (int ep = 0; ep < ep_count; ++ep) {
        cfg_.gria_lr = base_gria_lr * std::pow(cfg_.gria_lr_decay, static_cast<double>(ep));
        reset_context();
        const int steps = std::min(n_steps, static_cast<int>(ids.size()) - 1);
        for (int i = 0; i < steps; ++i) {
            train_step(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]),
                       static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i + 1)]));
        }
    }
    cfg_.gria_lr = base_gria_lr;
}

double CyphaLMModel::hybrid_gria_weight() const {
    return sigmoid(hybrid_blend_logit_);
}

AlphaSpectrumSnapshot CyphaLMModel::alpha_spectrum_snapshot() const {
    AlphaSpectrumSnapshot snap;
    if (gria_) snap.gria_projection_alpha = gria_->alpha;
    if (dif_) snap.expert_alpha = dif_->alpha_per_expert();
    std::vector<double> all = snap.gria_projection_alpha;
    all.insert(all.end(), snap.expert_alpha.begin(), snap.expert_alpha.end());
    if (!all.empty()) {
        double sum = 0.0;
        int near = 0;
        for (double a : all) {
            sum += a;
            if (std::abs(a - 0.5) < 0.1) ++near;
        }
        snap.mean_alpha = sum / static_cast<double>(all.size());
        snap.fraction_near_edge_of_chaos =
            static_cast<double>(near) / static_cast<double>(all.size());
    }
    return snap;
}

nlohmann::json CyphaLMModel::compression_profile() const {
    const auto snap = alpha_spectrum_snapshot();
    nlohmann::json gria_spec = nlohmann::json::object();
    if (gria_) {
        for (const auto& kv : gria_->alpha_spectrum()) {
            gria_spec[kv.first] = kv.second;
        }
    }
    const double total = last_dif_out_.epistemic_var + last_dif_out_.aleatoric_var + 1e-12;
    return {
        {"mean_epistemic_var", last_dif_out_.epistemic_var},
        {"mean_aleatoric_var", last_dif_out_.aleatoric_var},
        {"mean_alpha", snap.mean_alpha},
        {"expert_alpha_spectrum", snap.expert_alpha},
        {"gria_alpha_spectrum", gria_spec},
        {"n_experts", dif_ ? dif_->expert_count() : 0},
        {"lossless_fraction", last_dif_out_.epistemic_var / total},
        {"lossy_fraction", 1.0 - last_dif_out_.epistemic_var / total},
        {"fraction_near_edge_of_chaos", snap.fraction_near_edge_of_chaos},
    };
}

double CyphaLMModel::eval_bpc(const std::vector<int>& ids, int n_eval,
                              cypha::intelligence::IntelligenceProfiler* profiler) {
    reset_context();
    const int n = std::min(n_eval, static_cast<int>(ids.size()) - 1);
    if (n <= 0) return std::numeric_limits<double>::quiet_NaN();
    const int vocab = static_cast<int>(cfg_.vocab_size);
    double bits = 0.0;
    int scored = 0;
    for (int i = 0; i < n; ++i) {
        const std::uint32_t tok = static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]);
        std::vector<double> embed;
        if (embed_) {
            try {
                embed = embed_vector(tok);
            } catch (const std::exception&) {
                embed.clear();
            }
        }
        const auto pred = predict_next(tok);
        const int nxt = ids[static_cast<std::size_t>(i + 1)];
        // Index log_probs by vocab token id (not label/class order — legacy D04 bug).
        if (nxt < 0 || nxt >= vocab ||
            static_cast<std::size_t>(nxt) >= pred.log_probs.size()) {
            continue;
        }
        if (profiler != nullptr) {
            update_profiler_from_lm_token(*profiler, embed, pred.log_probs, pred.epistemic_var,
                                          pred.aleatoric_var);
        }
        bits += -pred.log_probs[static_cast<std::size_t>(nxt)] / kLog2;
        ++scored;
    }
    if (scored <= 0) return std::numeric_limits<double>::quiet_NaN();
    return bits / static_cast<double>(scored);
}

std::vector<double> CyphaLMModel::embed_vector(std::uint32_t token_id) const {
    if (!embed_) {
        throw std::runtime_error("CyphaLMModel: embed table unavailable");
    }
    return embed_->embed_vec(token_id);
}

double CyphaLMModel::ssm_projection_rms() const {
    if (proj_ssm_.empty()) return 0.0;
    double acc = 0.0;
    for (double v : proj_ssm_) acc += v * v;
    return std::sqrt(acc / static_cast<double>(proj_ssm_.size()));
}

nlohmann::json CyphaLMModel::ssm_diagnostic_report(const std::vector<int>& token_ids, int max_steps) {
    return diagnose_model_tokens(*this, token_ids, max_steps, "cyphalm");
}

}  // namespace cyphalm
}  // namespace cypha
