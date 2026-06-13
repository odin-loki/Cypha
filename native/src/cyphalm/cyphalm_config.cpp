#include "cypha/cyphalm/cyphalm_config.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

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
    set_b("ngram_fuse_split", cfg.ngram_fuse_split);
    set_s("bpe_merges_path", cfg.bpe_merges_path);
    set_s("bpe_vocab_path", cfg.bpe_vocab_path);
}

}  // namespace

void apply_bench_profile(const std::string& profile, CyphaLMConfig& cfg) {
    namespace fs = std::filesystem;
    const fs::path root = fs::path(repo_root_from_config()) / "bench" / "config" / "profiles";
    fs::path path;
    if (profile == "d17") {
        path = root / "cyphalm_d17_wikitext.json";
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
}

}  // namespace cypha::cyphalm
