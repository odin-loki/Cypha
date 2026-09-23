#include "cypha/cyphalm/cyphalm_checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

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
        {"hp_slot_max", cfg.hp_slot_max},
        {"hp_mixer_lr", cfg.hp_mixer_lr},
        {"hp_gria", cfg.hp_gria},
        {"hp_lossy_mem", cfg.hp_lossy_mem},
        {"hp_lossy_tier", cfg.hp_lossy_tier},
        {"hp_cm_drop", cfg.hp_cm_drop},
        {"hp_cm_bits_cap", cfg.hp_cm_bits_cap},
        {"hp_gate_drop", cfg.hp_gate_drop},
        {"hp_mixer_skip", cfg.hp_mixer_skip},
        {"hp_match_bits_cap", cfg.hp_match_bits_cap},
        {"hp_pool_slots", cfg.hp_pool_slots},
        {"hp_pool_bits_cap", cfg.hp_pool_bits_cap},
        {"hp_frozen_scoring", cfg.hp_frozen_scoring},
        {"hp_serve_mixer_lr_scale", cfg.hp_serve_mixer_lr_scale},
        {"hp_ensemble_learning_rate", cfg.hp_ensemble_learning_rate},
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
    get_i("hp_slot_max", cfg.hp_slot_max);
    get_i("hp_mixer_lr", cfg.hp_mixer_lr);
    get_b("hp_gria", cfg.hp_gria);
    get_i("hp_lossy_mem", cfg.hp_lossy_mem);
    if (c.contains("hp_lossy_tier")) cfg.hp_lossy_tier = c.at("hp_lossy_tier").get<std::string>();
    get_u64("hp_cm_drop", cfg.hp_cm_drop);
    get_i("hp_cm_bits_cap", cfg.hp_cm_bits_cap);
    if (c.contains("hp_gate_drop")) cfg.hp_gate_drop = c.at("hp_gate_drop").get<std::uint32_t>();
    get_i("hp_mixer_skip", cfg.hp_mixer_skip);
    get_i("hp_match_bits_cap", cfg.hp_match_bits_cap);
    get_i("hp_pool_slots", cfg.hp_pool_slots);
    get_i("hp_pool_bits_cap", cfg.hp_pool_bits_cap);
    get_b("hp_frozen_scoring", cfg.hp_frozen_scoring);
    if (c.contains("hp_ensemble_learning_rate"))
        cfg.hp_ensemble_learning_rate = c.at("hp_ensemble_learning_rate").get<double>();
    if (c.contains("hp_serve_mixer_lr_scale"))
        cfg.hp_serve_mixer_lr_scale = c.at("hp_serve_mixer_lr_scale").get<double>();
    normalize_hp_table_bits(cfg);
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

fs::path hpbin_path(const fs::path& base) {
    fs::path p = base;
    p.replace_extension(".hpbin");
    return p;
}

void write_hpbin(const CyphaLMModel& model, const fs::path& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write hp checkpoint: " + path.string());
    }
    model.hp_backend().predictor().write_checkpoint(out);
    if (!out) {
        throw std::runtime_error("hp checkpoint write failed: " + path.string());
    }
}

void read_hpbin(CyphaLMModel& model, const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open hp checkpoint: " + path.string());
    }
    model.hp_backend().predictor().read_checkpoint(in);
    if (!in) {
        throw std::runtime_error("hp checkpoint read failed: " + path.string());
    }
}

}  // namespace

void save_cyphalm_model(const CyphaLMModel& model, const std::string& base_path) {
    fs::path base(base_path);
    if (base.extension() == ".json") base.replace_extension("");
    fs::path json_file = base;
    json_file.replace_extension(".json");
    const fs::path bin_file = hpbin_path(base);
    fs::create_directories(base.parent_path());

    write_hpbin(model, bin_file);

    nlohmann::json meta;
    meta["algorithm"] = "hp";
    meta["config"] = config_to_json(model.config());
    meta["train_step_count"] = model.train_step_count();
    meta["hp_checkpoint"] = bin_file.filename().string();
    meta["note"] =
        "hp predictor state in sibling .hpbin (HPCP v2). JSON carries config metadata only.";

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
    CyphaLMModel model(config_from_json(meta.at("config")));

    const fs::path bin_file = hpbin_path(jp);
    if (fs::exists(bin_file)) {
        read_hpbin(model, bin_file);
    }
    return model;
}

}  // namespace cypha::cyphalm
