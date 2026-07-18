#include "cypha/cyphalm/cyphalm_config.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "cypha/env.hpp"

namespace cypha::cyphalm {

namespace {

std::string lower_copy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

}  // namespace

ContextMode parse_context_mode(const std::string& s) {
    const std::string k = lower_copy(s);
    if (k == "full") return ContextMode::Full;
    if (k == "gria_ngram") return ContextMode::GriaNgram;
    if (k == "hybrid" || k == "hybrid_gria_lstm") return ContextMode::Hybrid;
    if (k == "char_lstm") return ContextMode::CharLstm;
    if (k == "ssm_gria" || k == "ssm_only" || k == "ssm-only") return ContextMode::SsmGria;
    if (k == "ssm_gria_no_lstm") return ContextMode::SsmGriaNoLstm;
    if (k == "ablation_no_dif") return ContextMode::AblationNoDif;
    if (k == "ablation_no_ssm") return ContextMode::AblationNoSsm;
    if (k == "rpsm") return ContextMode::Rpsm;
    throw std::runtime_error("unknown context mode: " + s);
}

std::string context_mode_name(ContextMode mode) {
    switch (mode) {
        case ContextMode::Full: return "full";
        case ContextMode::GriaNgram: return "gria_ngram";
        case ContextMode::Hybrid: return "hybrid";
        case ContextMode::CharLstm: return "char_lstm";
        case ContextMode::SsmGria: return "ssm_gria";
        case ContextMode::SsmGriaNoLstm: return "ssm_gria_no_lstm";
        case ContextMode::AblationNoDif: return "ablation_no_dif";
        case ContextMode::AblationNoSsm: return "ablation_no_ssm";
        case ContextMode::Rpsm: return "rpsm";
    }
    return "unknown";
}

std::string context_mode_string(ContextMode mode) {
    switch (mode) {
        case ContextMode::Hybrid: return "hybrid_gria_lstm";
        case ContextMode::SsmGria: return "ssm_only";
        default: return context_mode_name(mode);
    }
}

BenchMode parse_bench_mode(const std::string& s) {
    if (s == "char_lstm") return BenchMode::CharLstm;
    if (s == "ssm") return BenchMode::Ssm;
    if (s == "hybrid") return BenchMode::Hybrid;
    if (s == "ssm_gria") return BenchMode::SsmGria;
    if (s == "context_bank") return BenchMode::ContextBank;
    if (s == "spectral") return BenchMode::Spectral;
    if (s == "rpsm") return BenchMode::Rpsm;
    throw std::runtime_error("unknown bench mode: " + s);
}

void apply_bench_mode(BenchMode mode, CyphaLMConfig& cfg) {
    cfg.use_context_bank = false;
    cfg.use_spectral_pde = false;
    switch (mode) {
        case BenchMode::CharLstm:
            cfg.context_mode = ContextMode::CharLstm;
            break;
        case BenchMode::Ssm:
            cfg.context_mode = ContextMode::SsmGria;
            break;
        case BenchMode::Hybrid:
            cfg.context_mode = ContextMode::Hybrid;
            break;
        case BenchMode::SsmGria:
            cfg.context_mode = ContextMode::GriaNgram;
            break;
        case BenchMode::ContextBank:
            cfg.context_mode = ContextMode::GriaNgram;
            cfg.use_context_bank = true;
            break;
        case BenchMode::Spectral:
            cfg.context_mode = ContextMode::SsmGria;
            cfg.use_spectral_pde = true;
            break;
        case BenchMode::Rpsm:
            // Do not hardcode Tiny (L=4,D=128,feat=64) here — that overwrote profile JSON after
            // apply_bench_profile and blocked Small-tier capacity gates (BACKLOG Phase C1).
            // Dims/memory knobs come from the profile (d21 Tiny lock vs d21_small).
            cfg.context_mode = ContextMode::Rpsm;
            cfg.use_rpsm_layer = true;
            break;
    }
}

std::string bench_mode_name(BenchMode mode) {
    switch (mode) {
        case BenchMode::CharLstm: return "char_lstm";
        case BenchMode::Ssm: return "ssm";
        case BenchMode::Hybrid: return "hybrid";
        case BenchMode::SsmGria: return "ssm_gria";
        case BenchMode::ContextBank: return "context_bank";
        case BenchMode::Spectral: return "spectral";
        case BenchMode::Rpsm: return "rpsm";
    }
    return "unknown";
}

namespace {

std::string repo_root_from_config() {
    namespace fs = std::filesystem;
    const fs::path native = fs::path(__FILE__).parent_path().parent_path().parent_path();
    return native.parent_path().string();
}

void merge_json_config(const nlohmann::json& j, CyphaLMConfig& cfg) {
    auto set_i = [&](const char* k, int& v) {
        if (j.contains(k)) v = j.at(k).get<int>();
    };
    auto set_u64 = [&](const char* k, std::uint64_t& v) {
        if (j.contains(k)) v = j.at(k).get<std::uint64_t>();
    };
    auto set_d = [&](const char* k, double& v) {
        if (j.contains(k)) v = j.at(k).get<double>();
    };
    auto set_b = [&](const char* k, bool& v) {
        if (j.contains(k)) v = j.at(k).get<bool>();
    };
    auto set_s = [&](const char* k, std::string& v) {
        if (j.contains(k)) v = j.at(k).get<std::string>();
    };

    set_i("vocab_size", cfg.vocab_size);
    set_i("d_embed", cfg.d_embed);
    set_i("d_state", cfg.d_state);
    set_d("tau_fast", cfg.tau_fast);
    set_d("tau_slow", cfg.tau_slow);
    set_i("ssm_layers", cfg.ssm_layers);
    set_i("field_dim", cfg.field_dim);
    set_i("max_experts", cfg.max_experts);
    set_i("n_experts", cfg.n_experts);
    set_b("use_soft_expert_updates", cfg.use_soft_expert_updates);
    set_b("use_routing_entropy_floor", cfg.use_routing_entropy_floor);
    set_d("routing_entropy_lambda", cfg.routing_entropy_lambda);
    set_d("routing_entropy_floor_frac", cfg.routing_entropy_floor_frac);
    set_d("alpha_init", cfg.alpha_init);
    set_b("alpha_learnable", cfg.alpha_learnable);
    set_i("gria_rank", cfg.gria_rank);
    set_i("context_length", cfg.context_length);
    if (j.contains("context_mode")) {
        cfg.context_mode = parse_context_mode(j.at("context_mode").get<std::string>());
    }
    set_i("ngram_context", cfg.ngram_context);
    set_i("train_epochs", cfg.train_epochs);
    set_s("view_schedule", cfg.view_schedule);
    set_i("view_block_size", cfg.view_block_size);
    set_i("view_id_dim", cfg.view_id_dim);
    set_b("view_learnable", cfg.view_learnable);
    set_i("bptt_steps", cfg.bptt_steps);
    set_d("laplace_smoothing", cfg.laplace_smoothing);
    set_d("gria_lr_decay", cfg.gria_lr_decay);
    set_i("lstm_hidden", cfg.lstm_hidden);
    set_d("lstm_lr", cfg.lstm_lr);
    set_i("lstm_bptt_steps", cfg.lstm_bptt_steps);
    set_s("lstm_optim", cfg.lstm_optim);
    set_d("lstm_grad_clip", cfg.lstm_grad_clip);
    set_s("lstm_init", cfg.lstm_init);
    set_d("hybrid_blend_logit", cfg.hybrid_blend_logit);
    set_b("hybrid_blend_learnable", cfg.hybrid_blend_learnable);
    set_d("hybrid_blend_lr", cfg.hybrid_blend_lr);
    set_u64("seed", cfg.seed);
    set_d("gria_lr", cfg.gria_lr);
    set_b("online", cfg.online);
    set_b("use_spectral_pde", cfg.use_spectral_pde);
    set_b("use_sparse_hebbian", cfg.use_sparse_hebbian);
    set_b("use_multiscale", cfg.use_multiscale);
    set_b("train_ssm", cfg.train_ssm);
    set_d("ssm_lr", cfg.ssm_lr);
    set_s("ngram_fusion", cfg.ngram_fusion);
    set_b("ngram_position_weights", cfg.ngram_position_weights);
    set_b("ngram_bilinear_fusion", cfg.ngram_bilinear_fusion);
    set_b("ngram_fuse_split", cfg.ngram_fuse_split);
    set_s("bpe_merges_path", cfg.bpe_merges_path);
    set_s("bpe_vocab_path", cfg.bpe_vocab_path);
    set_b("use_rpsm_layer", cfg.use_rpsm_layer);
    set_i("rpsm_n_levels", cfg.rpsm_n_levels);
    set_i("rpsm_state_dim", cfg.rpsm_state_dim);
    set_i("rpsm_feat_dim", cfg.rpsm_feat_dim);
    set_d("rpsm_lr", cfg.rpsm_lr);
    set_i("rpsm_n_memory_slots", cfg.rpsm_n_memory_slots);
    set_d("rpsm_beta_memory", cfg.rpsm_beta_memory);
    set_d("rpsm_surprise_threshold", cfg.rpsm_surprise_threshold);
    set_d("rpsm_hierarchy_loss_weight", cfg.rpsm_hierarchy_loss_weight);
    set_i("rpsm_bptt_window", cfg.rpsm_bptt_window);
    set_b("profile_guided_loss", cfg.profile_guided_loss);
    set_b("use_full_navigation_loss", cfg.use_full_navigation_loss);
    set_b("use_profile_curriculum", cfg.use_profile_curriculum);
    set_i("navigation_loss_warmup_steps", cfg.navigation_loss_warmup_steps);
    set_b("use_adaptive_navigation_lambdas", cfg.use_adaptive_navigation_lambdas);
    set_d("kappa_lambda_target", cfg.kappa_lambda_target);
    set_b("use_nig_state_cell", cfg.use_nig_state_cell);
    set_b("use_alpha_forget_gate", cfg.use_alpha_forget_gate);
    set_b("use_tau_forget_gate", cfg.use_tau_forget_gate);
    set_b("use_kappa_trajectory_lambdas", cfg.use_kappa_trajectory_lambdas);
    set_i("kappa_trajectory_window", cfg.kappa_trajectory_window);
    set_b("use_per_stat_deviation_lambdas", cfg.use_per_stat_deviation_lambdas);
    set_d("per_stat_deviation_span", cfg.per_stat_deviation_span);
    set_b("use_kappa_ceiling_lambdas", cfg.use_kappa_ceiling_lambdas);
    set_b("use_lstm_d_eff_hidden_nudge", cfg.use_lstm_d_eff_hidden_nudge);
    set_b("use_eigenvalue_d_eff", cfg.use_eigenvalue_d_eff);
    set_b("use_reu_forget_gate", cfg.use_reu_forget_gate);
    set_d("kappa_ceiling_strength", cfg.kappa_ceiling_strength);
    set_d("kappa_ceiling_min_scale", cfg.kappa_ceiling_min_scale);
    set_b("use_kappa_trajectory_ceiling", cfg.use_kappa_trajectory_ceiling);
    set_b("use_kappa_excess_grad_nudge", cfg.use_kappa_excess_grad_nudge);
    set_d("kappa_excess_grad_scale", cfg.kappa_excess_grad_scale);
    set_d("kappa_excess_grad_margin", cfg.kappa_excess_grad_margin);
    set_b("use_kappa_kernel_blend_scale", cfg.use_kappa_kernel_blend_scale);
    set_d("kappa_kernel_blend_floor", cfg.kappa_kernel_blend_floor);
    set_b("use_kappa_navigation_warmup_scale", cfg.use_kappa_navigation_warmup_scale);
    set_d("kappa_navigation_warmup_strength", cfg.kappa_navigation_warmup_strength);
    set_d("kappa_navigation_warmup_floor", cfg.kappa_navigation_warmup_floor);
    set_d("reu_forget_gate_blend", cfg.reu_forget_gate_blend);
    set_b("use_kernel_llr", cfg.use_kernel_llr);
    set_d("kernel_blend", cfg.kernel_blend);
    set_i("kernel_m", cfg.kernel_m);
    set_d("kernel_gamma_scale", cfg.kernel_gamma_scale);
    set_d("kernel_lr_scale", cfg.kernel_lr_scale);
    set_b("use_gria_gated_mixture", cfg.use_gria_gated_mixture);
    set_b("use_ood_branching", cfg.use_ood_branching);
    set_b("use_free_energy_loss", cfg.use_free_energy_loss);
    set_d("free_energy_beta", cfg.free_energy_beta);
    set_b("use_mdl_forget", cfg.use_mdl_forget);
    set_d("mdl_forget_max_norm", cfg.mdl_forget_max_norm);
    set_b("use_priority_replay", cfg.use_priority_replay);
    set_b("use_hebbian_stack", cfg.use_hebbian_stack);
    set_d("ewc_lambda", cfg.ewc_lambda);
}

}  // namespace

void apply_bench_profile(const std::string& profile, CyphaLMConfig& cfg) {
    namespace fs = std::filesystem;
    const fs::path root = fs::path(repo_root_from_config()) / "bench" / "config" / "profiles";
    fs::path path;
    if (profile == "d17") {
        path = root / "cyphalm_d17_wikitext.json";
    } else if (profile == "d17_bpe") {
        path = root / "cyphalm_d17_wikitext_bpe.json";
    } else if (profile == "d21") {
        path = root / "cyphalm_d21_rpsm.json";
    } else if (profile == "d21_small") {
        path = root / "cyphalm_d21_rpsm_small.json";
    } else if (profile == "d04") {
        path = root / "cyphalm_d04_gutenberg.json";
        if (!fs::is_regular_file(path)) {
            path = root / "cyphalm_d04.json";
        }
    } else {
        path = root / ("cyphalm_" + profile + ".json");
    }
    if (!fs::is_regular_file(path)) {
        return;
    }
    std::ifstream in(path);
    if (!in) return;
    nlohmann::json j;
    in >> j;
    merge_json_config(j, cfg);
    const fs::path repo = fs::path(repo_root_from_config());
    auto resolve = [&](std::string& p) {
        if (p.empty()) return;
        const fs::path cand = fs::path(p);
        if (cand.is_absolute() && fs::is_regular_file(cand)) return;
        const fs::path under = repo / cand;
        if (fs::is_regular_file(under)) p = under.string();
    };
    resolve(cfg.bpe_merges_path);
    resolve(cfg.bpe_vocab_path);
    apply_lstm_recipe_env(cfg);
}

void apply_lstm_recipe_env(CyphaLMConfig& cfg) {
    if (const auto v = cypha::env_get("CYPHA_LSTM_BPTT"); v.has_value() && !v->empty()) {
        cfg.lstm_bptt_steps = std::max(1, std::stoi(*v));
    }
    if (const auto v = cypha::env_get("CYPHA_LSTM_OPTIM"); v.has_value() && !v->empty()) {
        cfg.lstm_optim = lower_copy(*v);
    }
    if (const auto v = cypha::env_get("CYPHA_LSTM_GRAD_CLIP"); v.has_value() && !v->empty()) {
        cfg.lstm_grad_clip = std::stod(*v);
    }
    if (const auto v = cypha::env_get("CYPHA_LSTM_INIT"); v.has_value() && !v->empty()) {
        cfg.lstm_init = lower_copy(*v);
    }
}

}  // namespace cypha::cyphalm
