#include "cypha/cyphalm/cyphalm_model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_views.hpp"
#include "cypha/cyphalm/infinigram.hpp"
#include "hp/shard_merge.hpp"  // Predictor::reset_stream_state
#include "cypha/intelligence/intelligence_profiler.hpp"

namespace cypha {
namespace cyphalm {

namespace {

constexpr double kLog2 = 0.6931471805599453;

}  // namespace

CyphaLMModel::CyphaLMModel(CyphaLMConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.vocab_size > 256) {
        throw std::runtime_error("CyphaLMModel (hp): vocab_size must be <= 256 (byte tokens)");
    }
    init_components();
}

CyphaLMModel::~CyphaLMModel() = default;

void CyphaLMModel::init_components() {
    if (cfg_.context_mode == ContextMode::Hp || cfg_.context_mode == ContextMode::Hybrid) {
        apply_hp_lossy_env(cfg_);
        if (cfg_.hp_slot_max <= 0) {
            apply_hp_production_recipe(cfg_);
        } else {
            normalize_hp_table_bits(cfg_);
        }
    }
    hp_ = std::make_unique<HpSequenceBackend>(hp_config_from_cyphalm(cfg_));
    hp_->set_frozen_scoring(cfg_.hp_frozen_scoring);
    hp_->set_tree_prune(cfg_.hp_tree_prune);
    if (cfg_.hp_serve_compact) {
        hp_->compact_for_serve();
    }
    if (!cfg_.bpe_merges_path.empty() && !cfg_.bpe_vocab_path.empty()) {
        bpe_ = std::make_unique<BpeTokenizer>(
            BpeTokenizer::load(cfg_.bpe_merges_path, cfg_.bpe_vocab_path));
    }
    field_stub_.assign(static_cast<std::size_t>(cfg_.field_dim), 0.0);
}

CyphaLMModel CyphaLMModel::from_json_npz(const std::string& json_path) {
    return load_cyphalm_model(json_path);
}

void CyphaLMModel::compact_hp_for_serve() {
    if (hp_) {
        hp_->compact_for_serve();
    }
}

void CyphaLMModel::prune_hp_cold_slots(int min_total) {
    if (!hp_) {
        return;
    }
    const int thr = min_total > 0 ? min_total : cfg_.hp_prune_cold_min_n;
    hp_->prune_cold_slots(thr);
}

void CyphaLMModel::reset_context() {
    if (hp_) {
        hp_->reset();
    }
    last_predict_out_ = {};
    step_count_ = 0;
    last_train_loss_ = 0.0;
}

void CyphaLMModel::reset_stream(bool keep_history) {
    if (hp_) {
        hp_->reset_stream(keep_history);
    }
    last_predict_out_ = {};
}

void CyphaLMModel::set_serve_mode(bool on) {
    if (!hp_) return;
    // Mixer rates = trained x scale (in 1/16ths) while serving; trained otherwise.
    const int num = on ? std::max(1, static_cast<int>(std::lround(cfg_.hp_serve_mixer_lr_scale * 16.0))) : 16;
    hp_->set_serve_adaptation(num, 16, -1);
}

void CyphaLMModel::fold_hp_tables(int cm_bits_cap, int match_bits_cap, int pool_bits_cap,
                                  std::uint64_t cm_drop, int hebb_bits_cap,
                                  std::uint32_t match_drop) {
    if (hebb_bits_cap > 0) cfg_.hp_hebb_bits_cap = hebb_bits_cap;
    cfg_.hp_match_drop |= match_drop;
    if (!hp_) return;
    cfg_.hp_cm_drop |= cm_drop;
    if (cm_bits_cap > 0) cfg_.hp_cm_bits_cap = cm_bits_cap;
    if (match_bits_cap > 0) cfg_.hp_match_bits_cap = match_bits_cap;
    if (pool_bits_cap > 0) cfg_.hp_pool_bits_cap = pool_bits_cap;
    hp::Config target = hp_->predictor().config();
    target.cm_bits_cap = cm_bits_cap;
    target.match_bits_cap = match_bits_cap;
    target.pool_bits_cap = pool_bits_cap;
    target.cm_drop = cm_drop;
    target.hebb_bits_cap = hebb_bits_cap;
    target.match_drop = match_drop;
    hp_->fold_tables(target);
}

void CyphaLMModel::attach_infinigram(const std::string& index_path) {
    if (!hp_) throw std::runtime_error("attach_infinigram: hp backend required");
    hp_->set_infinigram(std::make_shared<const InfiniGram>(index_path));
    // Starting weights learned on held-out text, if saved next to the index
    // (cyphalm_lm_quality --ig-weights-out INDEX.weights.json).
    std::ifstream wf(index_path + ".weights.json");
    if (wf) {
        nlohmann::json j;
        wf >> j;
        hp_->set_infinigram_weights(j.at("weights").get<std::vector<double>>());
    }
}

void CyphaLMModel::attach_neural(const std::string& blm_path) {
    if (!hp_) throw std::runtime_error("attach_neural: hp backend required");
    hp_->set_neural(ByteLstmExpert::load(blm_path));
}

void CyphaLMModel::add_ensemble_member(CyphaLMModel&& other, double weight) {
    if (!hp_ || !other.hp_) throw std::runtime_error("add_ensemble_member: hp backends required");
    hp_->add_ensemble_member(std::move(other.hp_), weight);
    hp_->set_ensemble_learning_rate(cfg_.hp_ensemble_learning_rate);
}

void CyphaLMModel::reset_optim_state() {
    reset_context();
}

std::uint8_t CyphaLMModel::token_to_byte(std::uint32_t token_id) const {
    if (token_id >= static_cast<std::uint32_t>(cfg_.vocab_size)) {
        throw std::runtime_error("token id out of vocab range");
    }
    return static_cast<std::uint8_t>(token_id);
}

void CyphaLMModel::fill_top_k(const std::vector<double>& log_probs, PredictNextOutput& out,
                              int k) const {
    const int n = static_cast<int>(log_probs.size());
    const int kk = std::max(1, std::min(k, n));
    std::vector<int> idx(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        idx[static_cast<std::size_t>(i)] = i;
    }
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) {
        return log_probs[static_cast<std::size_t>(a)] > log_probs[static_cast<std::size_t>(b)];
    });
    out.top_k_tokens.clear();
    out.top_k_probs.clear();
    double mx = log_probs[static_cast<std::size_t>(idx[0])];
    double sum = 0.0;
    std::vector<double> probs(static_cast<std::size_t>(kk));
    for (int i = 0; i < kk; ++i) {
        probs[static_cast<std::size_t>(i)] =
            std::exp(log_probs[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])] - mx);
        sum += probs[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < kk; ++i) {
        out.top_k_tokens.push_back(static_cast<std::uint32_t>(idx[static_cast<std::size_t>(i)]));
        out.top_k_probs.push_back(probs[static_cast<std::size_t>(i)] / (sum + 1e-12));
    }
}

void CyphaLMModel::remember_last_predict(const PredictNextOutput& out) {
    last_predict_out_ = out;
}

PredictNextOutput CyphaLMModel::serve_predict_next(std::uint32_t token_id) {
    PredictNextOutput out;
    hp_->serve_advance_byte(token_to_byte(token_id));
    out.log_probs = hp_->serve_next_byte_log_probs(cfg_.vocab_size);
    out.epistemic_var = 0.0;
    out.aleatoric_var = 0.0;
    fill_top_k(out.log_probs, out);
    return out;
}

void CyphaLMModel::serve_advance(std::uint32_t token_id) {
    hp_->serve_advance_byte(token_to_byte(token_id));
}

std::uint32_t CyphaLMModel::serve_greedy_next(std::uint32_t token_id) {
    hp_->serve_advance_byte(token_to_byte(token_id));
    return static_cast<std::uint32_t>(hp_->serve_greedy_next_byte());
}

PredictNextOutput CyphaLMModel::predict_next(std::uint32_t token_id) {
    PredictNextOutput out = serve_predict_next(token_id);
    remember_last_predict(out);
    ++step_count_;
    return out;
}

PredictNextOutput CyphaLMModel::repredict_hybrid_blend(double) const {
    return last_predict_out_;
}

TrainStepMetrics CyphaLMModel::adapt_after_predict(std::uint32_t next_token_id, double,
                                                   cypha::intelligence::IntelligenceProfiler*,
                                                   LmIntelligenceMonitor*) {
    const PredictNextOutput& pred = last_predict_out_;
    if (pred.log_probs.empty() ||
        next_token_id >= static_cast<std::uint32_t>(pred.log_probs.size())) {
        throw std::runtime_error("adapt_after_predict: require predict_next first / token OOB");
    }
    set_serve_mode(false);
    TrainStepMetrics m;
    m.loss = -pred.log_probs[static_cast<std::size_t>(next_token_id)];
    m.epistemic_var = pred.epistemic_var;
    m.aleatoric_var = pred.aleatoric_var;
    m.alpha_gria = pred.epistemic_var;
    last_train_loss_ = m.loss;
    hp_->observe_next_byte(token_to_byte(next_token_id));
    return m;
}

TrainStepMetrics CyphaLMModel::train_step(std::uint32_t token_id, std::uint32_t next_token_id,
                                          cypha::intelligence::IntelligenceProfiler* profiler,
                                          LmIntelligenceMonitor* monitor) {
    (void)predict_next(token_id);
    return adapt_after_predict(next_token_id, 1.0, profiler, monitor);
}

std::vector<double> CyphaLMModel::forward_log_probs(std::uint32_t token_id) {
    return predict_next(token_id).log_probs;
}

void CyphaLMModel::train_sequence(const std::vector<int>& ids, int n_steps, int epochs,
                                  cypha::intelligence::IntelligenceProfiler*) {
    if (ids.size() < 2) {
        return;
    }
    const int ep_count = std::max(1, epochs);
    set_serve_mode(false);
    for (int ep = 0; ep < ep_count; ++ep) {
        // First epoch trains from scratch (historic behaviour); later epochs keep
        // what was learned and only restart the stream.
        if (ep == 0) {
            reset_context();
        } else {
            reset_stream();
        }
        const int steps = std::min(n_steps, static_cast<int>(ids.size()) - 1);
        for (int i = 0; i < steps; ++i) {
            if (i == 0) {
                hp_->consume_byte(token_to_byte(static_cast<std::uint32_t>(ids[0])));
            }
            const int nxt = ids[static_cast<std::size_t>(i + 1)];
            last_train_loss_ = -hp_->log_prob_byte(token_to_byte(static_cast<std::uint32_t>(nxt)));
            hp_->observe_next_byte(token_to_byte(static_cast<std::uint32_t>(nxt)));
            ++step_count_;
        }
    }
}

void CyphaLMModel::train_sequence_views(const std::vector<int>& ids,
                                        cypha::intelligence::IntelligenceProfiler* profiler) {
    const auto schedule = resolve_view_schedule_struct(cfg_.view_schedule, cfg_.seed, 1);
    const auto epochs = iter_view_epochs(ids, schedule);
    for (const auto& ep : epochs) {
        (void)ep;
        train_sequence(ids, static_cast<int>(ids.size()) - 1, 1, profiler);
    }
}

double CyphaLMModel::eval_bpc(const std::vector<int>& ids, int n_eval,
                              cypha::intelligence::IntelligenceProfiler*) {
    /// Bit-serial compress-equivalent BPC (``observe_next_byte`` on ``n_eval`` bytes).
    /// Does not use the 256-way ``next_byte_log_probs`` fan-out.
    return eval_bpc_compress_equivalent(ids, n_eval);
}

double CyphaLMModel::eval_bpc_compress_equivalent(const std::vector<int>& ids, int n_eval) {
    reset_context();
    const int n = std::min(n_eval, static_cast<int>(ids.size()));
    if (n <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        bytes[static_cast<std::size_t>(i)] = token_to_byte(static_cast<std::uint32_t>(ids[i]));
    }
    const double bits = hp_->observe_stream_bits(bytes.data(), bytes.size());
    return bits / static_cast<double>(n);
}

void CyphaLMModel::accumulate_intelligence_profile(const std::vector<int>& ids, int n_steps,
                                                   cypha::intelligence::IntelligenceProfiler&) {
    train_sequence(ids, n_steps, 1, nullptr);
}

std::vector<std::uint32_t> CyphaLMModel::encode_text(const std::string& text) const {
    if (bpe_) {
        return bpe_->encode(text);
    }
    std::vector<std::uint32_t> out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        if (static_cast<int>(c) < cfg_.vocab_size) {
            out.push_back(static_cast<std::uint32_t>(c));
        }
    }
    return out;
}

std::string CyphaLMModel::decode_tokens(const std::vector<std::uint32_t>& ids) const {
    if (bpe_) {
        return bpe_->decode(ids);
    }
    std::string out;
    out.reserve(ids.size());
    for (std::uint32_t id : ids) {
        if (id < static_cast<std::uint32_t>(cfg_.vocab_size)) {
            out.push_back(static_cast<char>(id));
        }
    }
    return out;
}

AlphaSpectrumSnapshot CyphaLMModel::alpha_spectrum_snapshot() const {
    AlphaSpectrumSnapshot snap;
    const auto& gria = hp_->predictor().gria();
    const double alpha = static_cast<double>(gria.bucket()) / 255.0;
    snap.mean_alpha = alpha;
    snap.gria_projection_alpha.push_back(alpha);
    snap.fraction_near_edge_of_chaos = (alpha > 0.45 && alpha < 0.55) ? 1.0 : 0.0;
    return snap;
}

nlohmann::json CyphaLMModel::compression_profile() const {
    return {
        {"algorithm", "hp"},
        {"hp_table_bits", cfg_.hp_table_bits},
        {"hp_effective_table_bits", hp_effective_table_bits(cfg_)},
        {"hp_lossy_mem", cfg_.hp_lossy_mem},
        {"hp_serve_compact", cfg_.hp_serve_compact},
        {"hp_prune_cold_min_n", cfg_.hp_prune_cold_min_n},
        {"hp_lossy_tier", cfg_.hp_lossy_tier},
        {"hp_cm_drop", cfg_.hp_cm_drop},
        {"hp_cm_bits_cap", cfg_.hp_cm_bits_cap},
        {"hp_gate_drop", cfg_.hp_gate_drop},
        {"hp_mixer_skip", cfg_.hp_mixer_skip},
        {"hp_lr1_scale", cfg_.hp_lr1_scale},
        {"hp_mixer_scale", cfg_.hp_mixer_scale},
        {"hp_mixer_skip_l1", cfg_.hp_mixer_skip_l1},
        {"hp_match_bits_cap", cfg_.hp_match_bits_cap},
        {"hp_pool_slots", cfg_.hp_pool_slots},
        {"hp_pool_bits_cap", cfg_.hp_pool_bits_cap},
        {"hp_hebb_bits_cap", cfg_.hp_hebb_bits_cap},
        {"hp_match_drop", cfg_.hp_match_drop},
        {"hp_frozen_scoring", cfg_.hp_frozen_scoring},
        {"hp_serve_mixer_lr_scale", cfg_.hp_serve_mixer_lr_scale},
        {"hp_ensemble_learning_rate", cfg_.hp_ensemble_learning_rate},
        {"hp_tree_prune", cfg_.hp_tree_prune},
        {"hp_slot_max", cfg_.hp_slot_max},
        {"hp_slot_compile_max", hp_compile_slot_max()},
        {"hp_mixer_lr", cfg_.hp_mixer_lr},
        {"hp_gria", cfg_.hp_gria},
        {"vocab_size", cfg_.vocab_size},
        {"context_mode", context_mode_name(cfg_.context_mode)},
        {"hp_profile", "gate24"},
        {"rss_lab_reference",
         nlohmann::json{{"slot_max_24_mem22_kb", 1600000},
                        {"source", "hp/tools/hp_harness.sh RECORD H34"}}},
        {"note",
         "Cypha BPC (eval_bpc / compress_equivalent_bpc) uses bit-serial observe NLL; "
         "hp archive bytes are a separate metric (see CYPHALM_BPC_GAP_REPORT.md)."},
    };
}

nlohmann::json CyphaLMModel::ssm_diagnostic_report(const std::vector<int>& token_ids,
                                                   int max_steps) {
    nlohmann::json report;
    report["algorithm"] = "hp";
    report["steps"] = std::min(max_steps, static_cast<int>(token_ids.size()));
    report["hp_table_bits"] = cfg_.hp_table_bits;
    return report;
}

std::vector<double> CyphaLMModel::embed_vector(std::uint32_t token_id) const {
    std::vector<double> out(static_cast<std::size_t>(cfg_.d_embed), 0.0);
    if (!out.empty()) {
        out[0] = static_cast<double>(token_id) / static_cast<double>(std::max(1, cfg_.vocab_size));
    }
    return out;
}

void CyphaLMModel::save(const std::string& base_path) const {
    save_cyphalm_model(*this, base_path);
}

}  // namespace cyphalm
}  // namespace cypha
