#include "cypha/cyphalm/cyphalm_checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace cypha::cyphalm {

namespace {

namespace fs = std::filesystem;

nlohmann::json config_to_json(const CyphaLMConfig& cfg) {
    return {
        {"algorithm", "hp"},
        {"vocab_size", cfg.vocab_size},
        {"d_embed", cfg.d_embed},
        {"d_state", cfg.d_state},
        {"field_dim", cfg.field_dim},
        {"context_mode", context_mode_string(cfg.context_mode)},
        {"hp_table_bits", cfg.hp_table_bits},
        {"hp_mixer_lr", cfg.hp_mixer_lr},
        {"hp_gria", cfg.hp_gria},
        {"train_epochs", cfg.train_epochs},
        {"view_schedule", cfg.view_schedule},
        {"view_block_size", cfg.view_block_size},
        {"view_id_dim", cfg.view_id_dim},
        {"view_learnable", cfg.view_learnable},
        {"seed", cfg.seed},
        {"bpe_merges_path", cfg.bpe_merges_path},
        {"bpe_vocab_path", cfg.bpe_vocab_path},
    };
}

CyphaLMConfig config_from_json(const nlohmann::json& c) {
    CyphaLMConfig cfg;
    cfg.context_mode = ContextMode::Hp;
    auto get_i = [&](const char* k, int& v) {
        if (c.contains(k)) v = c.at(k).get<int>();
    };
    auto get_b = [&](const char* k, bool& v) {
        if (c.contains(k)) v = c.at(k).get<bool>();
    };
    auto get_u64 = [&](const char* k, std::uint64_t& v) {
        if (c.contains(k)) v = c.at(k).get<std::uint64_t>();
    };
    get_i("vocab_size", cfg.vocab_size);
    get_i("d_embed", cfg.d_embed);
    get_i("d_state", cfg.d_state);
    get_i("field_dim", cfg.field_dim);
    get_i("hp_table_bits", cfg.hp_table_bits);
    get_i("hp_mixer_lr", cfg.hp_mixer_lr);
    get_b("hp_gria", cfg.hp_gria);
    if (c.contains("context_mode")) {
        cfg.context_mode = parse_context_mode(c.at("context_mode").get<std::string>());
    }
    get_i("train_epochs", cfg.train_epochs);
    if (c.contains("view_schedule")) cfg.view_schedule = c.at("view_schedule").get<std::string>();
    get_i("view_block_size", cfg.view_block_size);
    get_i("view_id_dim", cfg.view_id_dim);
    get_b("view_learnable", cfg.view_learnable);
    get_u64("seed", cfg.seed);
    if (c.contains("bpe_merges_path")) cfg.bpe_merges_path = c.at("bpe_merges_path").get<std::string>();
    if (c.contains("bpe_vocab_path")) cfg.bpe_vocab_path = c.at("bpe_vocab_path").get<std::string>();
    return cfg;
}

fs::path resolve_json_path(const std::string& path) {
    fs::path p(path);
    if (p.extension() == ".json") return p;
    p.replace_extension(".json");
    return p;
}

}  // namespace

void save_cyphalm_model(const CyphaLMModel& model, const std::string& base_path) {
    fs::path base(base_path);
    if (base.extension() == ".json") base.replace_extension("");
    fs::path json_file = base;
    json_file.replace_extension(".json");
    fs::create_directories(base.parent_path());

    nlohmann::json meta;
    meta["algorithm"] = "hp";
    meta["config"] = config_to_json(model.config());
    meta["train_step_count"] = model.train_step_count();
    meta["note"] =
        "hp predictor tables are session-local (online adaptation). Config is persisted; "
        "re-run train_sequence to warm tables for a corpus.";

    std::ofstream out(json_file);
    if (!out) throw std::runtime_error("cannot write checkpoint json: " + json_file.string());
    out << meta.dump(2) << "\n";
}

CyphaLMModel load_cyphalm_model(const std::string& json_path) {
    const fs::path jp = resolve_json_path(json_path);
    std::ifstream in(jp);
    if (!in) throw std::runtime_error("cannot open checkpoint json: " + jp.string());
    nlohmann::json meta;
    in >> meta;
    if (!meta.contains("config")) throw std::runtime_error("checkpoint missing config");
    return CyphaLMModel(config_from_json(meta.at("config")));
}

}  // namespace cypha::cyphalm
