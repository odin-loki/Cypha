#include "cypha/cyphalm/cyphalm_checkpoint.hpp"

#include <cstdlib>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace cypha::cyphalm {

namespace {

namespace fs = std::filesystem;

nlohmann::json config_to_json(const CyphaLMConfig& cfg) {
    nlohmann::json j = {
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
        {"hp_lr1_scale", cfg.hp_lr1_scale},
        {"hp_mixer_scale", cfg.hp_mixer_scale},
        {"hp_mixer_skip_l1", cfg.hp_mixer_skip_l1},
        {"hp_match_bits_cap", cfg.hp_match_bits_cap},
        {"hp_pool_slots", cfg.hp_pool_slots},
        {"hp_pool_bits_cap", cfg.hp_pool_bits_cap},
        {"hp_hebb_bits_cap", cfg.hp_hebb_bits_cap},
        {"hp_match_drop", cfg.hp_match_drop},
        {"hp_frozen_scoring", cfg.hp_frozen_scoring},
        {"hp_serve_mixer_lr_scale", cfg.hp_serve_mixer_lr_scale},
        {"hp_ensemble_learning_rate", cfg.hp_ensemble_learning_rate},
        {"hp_tree_prune", cfg.hp_tree_prune},
        {"train_epochs", cfg.train_epochs},
        {"view_schedule", cfg.view_schedule},
        {"view_block_size", cfg.view_block_size},
        {"view_id_dim", cfg.view_id_dim},
        {"view_learnable", cfg.view_learnable},
        {"seed", cfg.seed},
        {"bpe_merges_path", cfg.bpe_merges_path},
        {"bpe_vocab_path", cfg.bpe_vocab_path},
    };
    // Written only when set, so gate24 checkpoints stay byte-identical.
    if (cfg.hp_extra_cms != 0) j["hp_extra_cms"] = cfg.hp_extra_cms;
    return j;
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
    get_i("hp_lr1_scale", cfg.hp_lr1_scale);
    get_i("hp_mixer_scale", cfg.hp_mixer_scale);
    get_i("hp_mixer_skip_l1", cfg.hp_mixer_skip_l1);
    if (c.contains("hp_extra_cms")) cfg.hp_extra_cms = c.at("hp_extra_cms").get<std::uint32_t>();
    get_i("hp_match_bits_cap", cfg.hp_match_bits_cap);
    get_i("hp_pool_slots", cfg.hp_pool_slots);
    get_i("hp_pool_bits_cap", cfg.hp_pool_bits_cap);
    get_i("hp_hebb_bits_cap", cfg.hp_hebb_bits_cap);
    if (c.contains("hp_match_drop")) cfg.hp_match_drop = c.at("hp_match_drop").get<std::uint32_t>();
    get_b("hp_frozen_scoring", cfg.hp_frozen_scoring);
    if (c.contains("hp_tree_prune")) cfg.hp_tree_prune = c.at("hp_tree_prune").get<double>();
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

/// Large tables are mapped copy-on-write from the checkpoint file instead of
/// copied into anonymous memory (hp::MapScope); CYPHA_HP_MMAP=0 turns it off.
bool hp_mmap_enabled() {
    const char* v = std::getenv("CYPHA_HP_MMAP");
    return v == nullptr || v[0] != '0';
}

/// Why ``path`` did not load, from its 8-byte header (magic ``HPCP`` + version).
std::string hpbin_failure(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    char head[8] = {};
    in.read(head, 8);
    if (in.gcount() < 8 || std::string(head, 4) != "HPCP") {
        return "not an hp checkpoint (no HPCP header): " + path.string();
    }
    std::uint32_t ver = 0;
    std::memcpy(&ver, head + 4, 4);
    if (ver < 1 || ver > 5) {
        return "unsupported hp checkpoint version " + std::to_string(ver) + ": " + path.string();
    }
    return "hp checkpoint read failed (truncated, or its tables or mixer do not match the "
           "JSON config): " + path.string();
}

void read_hpbin(CyphaLMModel& model, const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open hp checkpoint: " + path.string());
    }
    hp::MapSource src;
#if !defined(_WIN32)
    if (hp_mmap_enabled()) src.fd = ::open(path.c_str(), O_RDONLY);
    // Mappings outlive the descriptor; close it on every exit, throws included.
    struct FdClose {
        int fd;
        ~FdClose() {
            if (fd >= 0) ::close(fd);
        }
    } fd_close{src.fd};
#endif
    {
        hp::MapScope scope(&src);
        model.hp_backend().predictor().read_checkpoint(in);
    }
    if (!in) {
        throw std::runtime_error(hpbin_failure(path));
    }
}

}  // namespace

void save_cyphalm_model(const CyphaLMModel& model, const std::string& base_path) {
    fs::path base(base_path);
    if (base.extension() == ".json") base.replace_extension("");
    fs::path json_file = base;
    json_file.replace_extension(".json");
    const fs::path bin_file = hpbin_path(base);
    if (base.has_parent_path()) fs::create_directories(base.parent_path());  // "--save name" in the cwd

    write_hpbin(model, bin_file);

    nlohmann::json meta;
    meta["algorithm"] = "hp";
    meta["config"] = config_to_json(model.config());
    meta["train_step_count"] = model.train_step_count();
    meta["hp_checkpoint"] = bin_file.filename().string();
    meta["note"] = model.config().hp_extra_cms != 0
                       ? "hp predictor state in sibling .hpbin (HPCP v5). JSON carries config metadata only."
                       : "hp predictor state in sibling .hpbin (HPCP v4). JSON carries config metadata only.";

    std::ofstream out(json_file);
    if (!out) throw std::runtime_error("cannot write checkpoint json: " + json_file.string());
    out << meta.dump(2) << "\n";
}

namespace {

/// Ensemble manifest: {"cyphalm_ensemble": 1, "members": [{"checkpoint": path,
/// "weight": w?}, ...], "learning_rate": r?}. The first member is the primary;
/// paths are relative to the manifest; weights default to equal shares.
CyphaLMModel load_ensemble_manifest(const fs::path& jp, const nlohmann::json& meta) {
    const auto& ms = meta.at("members");
    if (!ms.is_array() || ms.empty()) throw std::runtime_error("ensemble manifest has no members");
    auto resolve = [&](const std::string& p) {
        const fs::path q(p);
        return (q.is_absolute() ? q : jp.parent_path() / q).string();
    };
    const double equal = 1.0 / static_cast<double>(ms.size());
    CyphaLMModel model = load_cyphalm_model(resolve(ms.at(0).at("checkpoint").get<std::string>()));
    for (std::size_t i = 1; i < ms.size(); ++i) {
        const auto& m = ms.at(i);
        model.add_ensemble_member(load_cyphalm_model(resolve(m.at("checkpoint").get<std::string>())),
                                  m.value("weight", equal));
    }
    if (meta.contains("learning_rate")) {
        model.hp_backend().set_ensemble_learning_rate(meta.at("learning_rate").get<double>());
    }
    if (meta.contains("infinigram")) {
        // A stored index, or the corpus itself (indexed at load, first
        // "infinigram_bytes" bytes).
        model.attach_infinigram(resolve(meta.at("infinigram").get<std::string>()),
                                meta.value("infinigram_bytes", std::size_t{0}));
    }
    if (meta.contains("neural")) {  // one path or a list
        const auto& nj = meta.at("neural");
        const double eta = meta.value("neural_learning_rate", 0.1);
        if (nj.is_array())
            for (const auto& p : nj) model.attach_neural(resolve(p.get<std::string>()), eta);
        else
            model.attach_neural(resolve(nj.get<std::string>()), eta);
        if (meta.contains("neural_adapt")) model.hp_backend().set_neural_adaptation(meta.at("neural_adapt").get<double>());
    }
    if (meta.value("session_cache", false)) model.hp_backend().set_session_cache(true);
    return model;
}

}  // namespace

void save_cyphalm_ensemble_manifest(const std::string& manifest_path,
                                    const std::vector<std::string>& member_checkpoints,
                                    double learning_rate) {
    nlohmann::json meta;
    meta["cyphalm_ensemble"] = 1;
    meta["note"] = "Serve-time ensemble: the first member is the primary; distributions are "
                   "mixed geometrically (CyphaLMModel::add_ensemble_member).";
    meta["members"] = nlohmann::json::array();
    for (const auto& c : member_checkpoints) meta["members"].push_back({{"checkpoint", c}});
    meta["learning_rate"] = learning_rate;
    std::ofstream out(manifest_path);
    if (!out) throw std::runtime_error("cannot write ensemble manifest: " + manifest_path);
    out << meta.dump(2) << "\n";
}

CyphaLMModel load_cyphalm_model(const std::string& json_path) {
    const fs::path jp = resolve_json_path(json_path);
    std::ifstream in(jp);
    if (!in) throw std::runtime_error("cannot open checkpoint json: " + jp.string());
    nlohmann::json meta;
    in >> meta;
    if (meta.contains("cyphalm_ensemble")) return load_ensemble_manifest(jp, meta);
    if (!meta.contains("config")) throw std::runtime_error("checkpoint missing config");
    CyphaLMModel model(config_from_json(meta.at("config")));

    const fs::path bin_file = hpbin_path(jp);
    if (fs::exists(bin_file)) {
        read_hpbin(model, bin_file);
    } else if (meta.contains("hp_checkpoint") || meta.value("algorithm", std::string()) == "hp") {
        // save_cyphalm_model wrote the state next to the JSON: without it the
        // model would load untrained. (Legacy JSON-only checkpoints have neither key.)
        throw std::runtime_error("hp checkpoint missing: " + bin_file.string() + " (for " + jp.string() + ")");
    }
    return model;
}

}  // namespace cypha::cyphalm
