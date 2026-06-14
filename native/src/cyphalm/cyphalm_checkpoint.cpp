#include "cypha/cyphalm/cyphalm_checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hierarchical_ssm.hpp"
#include "cypha/cyphalm/npz_util.hpp"

namespace cypha::cyphalm {

namespace {

namespace fs = std::filesystem;

nlohmann::json vec_to_json(const std::vector<double>& v) {
    return nlohmann::json::array_t(v.begin(), v.end());
}

std::vector<double> json_to_vec(const nlohmann::json& j) {
    std::vector<double> out;
    if (!j.is_array()) return out;
    if (!j.empty() && j[0].is_array()) {
        for (const auto& row : j) {
            for (const auto& x : row) out.push_back(x.get<double>());
        }
        return out;
    }
    out.reserve(j.size());
    for (const auto& x : j) out.push_back(x.get<double>());
    return out;
}

nlohmann::json config_to_json(const CyphaLMConfig& cfg) {
    return {
        {"vocab_size", cfg.vocab_size},
        {"d_embed", cfg.d_embed},
        {"d_state", cfg.d_state},
        {"tau_fast", cfg.tau_fast},
        {"tau_slow", cfg.tau_slow},
        {"ssm_layers", cfg.ssm_layers},
        {"use_spectral_pde", cfg.use_spectral_pde},
        {"use_multiscale", cfg.use_multiscale},
        {"use_sparse_hebbian", cfg.use_sparse_hebbian},
        {"field_dim", cfg.field_dim},
        {"alpha_init", cfg.alpha_init},
        {"alpha_learnable", cfg.alpha_learnable},
        {"gria_rank", cfg.gria_rank},
        {"context_mode", context_mode_string(cfg.context_mode)},
        {"ngram_context", cfg.ngram_context},
        {"train_epochs", cfg.train_epochs},
        {"view_schedule", cfg.view_schedule},
        {"view_block_size", cfg.view_block_size},
        {"view_id_dim", cfg.view_id_dim},
        {"view_learnable", cfg.view_learnable},
        {"ngram_fusion", cfg.ngram_fusion},
        {"ngram_position_weights", cfg.ngram_position_weights},
        {"ngram_fuse_split", cfg.ngram_fuse_split},
        {"gria_lr_decay", cfg.gria_lr_decay},
        {"bptt_steps", cfg.bptt_steps},
        {"laplace_smoothing", cfg.laplace_smoothing},
        {"online", cfg.online},
        {"gria_lr", cfg.gria_lr},
        {"ssm_lr", cfg.ssm_lr},
        {"train_ssm", cfg.train_ssm},
        {"lstm_hidden", cfg.lstm_hidden},
        {"lstm_lr", cfg.lstm_lr},
        {"hybrid_blend_logit", cfg.hybrid_blend_logit},
        {"hybrid_blend_learnable", cfg.hybrid_blend_learnable},
        {"hybrid_blend_lr", cfg.hybrid_blend_lr},
        {"use_context_bank", cfg.use_context_bank},
        {"context_bank_slots", cfg.context_bank_slots},
        {"use_hierarchical_ssm", cfg.use_hierarchical_ssm},
        {"use_hebb_graph", cfg.use_hebb_graph},
        {"use_hebbian_stack", cfg.use_hebbian_stack},
        {"use_temporal_som", cfg.use_temporal_som},
        {"use_gng", cfg.use_gng},
        {"use_gria_controller", cfg.use_gria_controller},
        {"use_discriminative_feedback", cfg.use_discriminative_feedback},
        {"ssm_hebb_lr", cfg.ssm_hebb_lr},
        {"compress_interval", cfg.compress_interval},
        {"max_memory_slots", cfg.max_memory_slots},
        {"seed", cfg.seed},
        {"bpe_merges_path", cfg.bpe_merges_path},
        {"bpe_vocab_path", cfg.bpe_vocab_path},
    };
}

CyphaLMConfig config_from_json(const nlohmann::json& c) {
    CyphaLMConfig cfg;
    auto get_i = [&](const char* k, int& v) {
        if (c.contains(k)) v = c.at(k).get<int>();
    };
    auto get_d = [&](const char* k, double& v) {
        if (c.contains(k)) v = c.at(k).get<double>();
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
    get_d("tau_fast", cfg.tau_fast);
    get_d("tau_slow", cfg.tau_slow);
    get_i("ssm_layers", cfg.ssm_layers);
    get_b("use_spectral_pde", cfg.use_spectral_pde);
    get_b("use_multiscale", cfg.use_multiscale);
    get_b("use_sparse_hebbian", cfg.use_sparse_hebbian);
    get_i("field_dim", cfg.field_dim);
    get_d("alpha_init", cfg.alpha_init);
    get_b("alpha_learnable", cfg.alpha_learnable);
    get_i("gria_rank", cfg.gria_rank);
    if (c.contains("context_mode")) {
        cfg.context_mode = parse_context_mode(c.at("context_mode").get<std::string>());
    }
    get_i("ngram_context", cfg.ngram_context);
    get_i("train_epochs", cfg.train_epochs);
    if (c.contains("view_schedule")) cfg.view_schedule = c.at("view_schedule").get<std::string>();
    get_i("view_block_size", cfg.view_block_size);
    get_i("view_id_dim", cfg.view_id_dim);
    get_b("view_learnable", cfg.view_learnable);
    if (c.contains("ngram_fusion")) cfg.ngram_fusion = c.at("ngram_fusion").get<std::string>();
    get_b("ngram_position_weights", cfg.ngram_position_weights);
    get_b("ngram_fuse_split", cfg.ngram_fuse_split);
    get_d("gria_lr_decay", cfg.gria_lr_decay);
    get_i("bptt_steps", cfg.bptt_steps);
    get_d("laplace_smoothing", cfg.laplace_smoothing);
    get_b("online", cfg.online);
    get_d("gria_lr", cfg.gria_lr);
    get_d("ssm_lr", cfg.ssm_lr);
    get_b("train_ssm", cfg.train_ssm);
    get_i("lstm_hidden", cfg.lstm_hidden);
    get_d("lstm_lr", cfg.lstm_lr);
    get_d("hybrid_blend_logit", cfg.hybrid_blend_logit);
    get_b("hybrid_blend_learnable", cfg.hybrid_blend_learnable);
    get_d("hybrid_blend_lr", cfg.hybrid_blend_lr);
    get_b("use_context_bank", cfg.use_context_bank);
    get_i("context_bank_slots", cfg.context_bank_slots);
    get_b("use_hierarchical_ssm", cfg.use_hierarchical_ssm);
    get_b("use_hebb_graph", cfg.use_hebb_graph);
    get_b("use_hebbian_stack", cfg.use_hebbian_stack);
    get_b("use_temporal_som", cfg.use_temporal_som);
    get_b("use_gng", cfg.use_gng);
    get_b("use_gria_controller", cfg.use_gria_controller);
    get_b("use_discriminative_feedback", cfg.use_discriminative_feedback);
    get_d("ssm_hebb_lr", cfg.ssm_hebb_lr);
    get_i("compress_interval", cfg.compress_interval);
    get_i("max_memory_slots", cfg.max_memory_slots);
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

void add_matrix(NpzWriter& w, const std::string& name, const std::vector<double>& data, int rows,
                int cols) {
    if (data.empty()) return;
    w.add_f64(name, data, {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)});
}

}  // namespace

void save_cyphalm_model(const CyphaLMModel& model, const std::string& base_path) {
    fs::path base(base_path);
    if (base.extension() == ".json") base.replace_extension("");
    const fs::path json_path = base;
    fs::path json_file = json_path;
    json_file.replace_extension(".json");
    fs::path npz_file = json_path;
    npz_file.replace_extension(".npz");
    fs::create_directories(base.parent_path());

    const CyphaLMConfig& cfg = model.config();
    nlohmann::json meta;
    meta["config"] = config_to_json(cfg);
    meta["token_counts"] = vec_to_json(model.token_counts_);
    if (!model.ngram_count_table_.empty()) {
        nlohmann::json ngram_counts = nlohmann::json::object();
        for (const auto& [key, counts] : model.ngram_count_table_) {
            ngram_counts[std::to_string(key)] = vec_to_json(counts);
        }
        meta["ngram_count_table"] = std::move(ngram_counts);
    }
    meta["hybrid_blend_logit"] = model.hybrid_blend_logit_;

    if (model.gria_) {
        meta["gria"] = {
            {"format", "lowrank"},
            {"field_dim", model.gria_->field_dim},
            {"vocab_size", model.gria_->vocab_size},
            {"rank", model.gria_->rank},
            {"U", vec_to_json(model.gria_->U)},
            {"V", vec_to_json(model.gria_->V)},
            {"alpha", vec_to_json(model.gria_->alpha)},
            {"bias", vec_to_json(model.gria_->bias)},
            {"alpha_learnable", model.gria_->alpha_learnable},
        };
    }
    if (model.lstm_) {
        meta["lstm"] = {
            {"E", vec_to_json(model.lstm_->E)},
            {"Wx", vec_to_json(model.lstm_->Wx)},
            {"Wh", vec_to_json(model.lstm_->Wh)},
            {"b", vec_to_json(model.lstm_->b)},
            {"Wy", vec_to_json(model.lstm_->Wy)},
            {"by", vec_to_json(model.lstm_->by)},
        };
    }
    if (model.hierarchical_ssm_) {
        meta["hierarchical_ssm"] = model.hierarchical_ssm_->get_state();
    } else if (model.ssm_) {
        meta["ssm"] = model.ssm_->get_state();
    }
    if (model.dif_) {
        meta["dif"] = model.dif_->get_state();
    }

    std::ofstream out(json_file);
    if (!out) throw std::runtime_error("cannot write checkpoint json: " + json_file.string());
    out << meta.dump(2) << "\n";

    NpzWriter npz;
    if (!model.proj_ssm_.empty()) {
        const int ctx_dim = model.hierarchical_ssm_
                                ? model.hierarchical_ssm_->fast_tier().context_dim()
                                : (model.ssm_ ? model.ssm_->context_dim() : cfg.field_dim);
        add_matrix(npz, "proj_ssm", model.proj_ssm_, cfg.field_dim, ctx_dim);
    }
    if (!model.proj_dif_.empty()) {
        add_matrix(npz, "proj_dif", model.proj_dif_, cfg.field_dim, cfg.field_dim);
    }
    if (!model.proj_embed_.empty()) {
        add_matrix(npz, "proj_embed", model.proj_embed_, cfg.field_dim, cfg.d_embed);
    }
    if (model.ngram_fusion_) {
        add_matrix(npz, "ngram_W_field", model.ngram_fusion_->w_field(), cfg.field_dim, cfg.field_dim);
        add_matrix(npz, "ngram_W_embed", model.ngram_fusion_->w_embed(), cfg.field_dim,
                   model.ngram_fusion_->embed_in());
        add_matrix(npz, "ngram_W_gate", model.ngram_fusion_->w_gate(), cfg.field_dim,
                   model.ngram_fusion_->embed_in());
        if (!model.ngram_fusion_->pos_weights().empty()) {
            npz.add_f64("ngram_pos_weights", model.ngram_fusion_->pos_weights(),
                        {static_cast<std::uint32_t>(model.ngram_fusion_->pos_weights().size())});
        }
    }
    if (model.view_emb_ && !model.view_emb_->table().empty()) {
        add_matrix(npz, "view_embed", model.view_emb_->table(), cfg.max_view_slots, cfg.view_id_dim);
    }
    npz.write(npz_file.string());
}

CyphaLMModel load_cyphalm_model(const std::string& json_path) {
    const fs::path jp = resolve_json_path(json_path);
    std::ifstream in(jp);
    if (!in) throw std::runtime_error("cannot open checkpoint json: " + jp.string());
    nlohmann::json meta;
    in >> meta;
    if (!meta.contains("config")) throw std::runtime_error("checkpoint missing config");
    CyphaLMConfig cfg = config_from_json(meta.at("config"));
    CyphaLMModel model(cfg);

    if (meta.contains("token_counts")) {
        model.token_counts_ = json_to_vec(meta.at("token_counts"));
    }
    if (meta.contains("ngram_count_table")) {
        model.ngram_count_table_.clear();
        for (const auto& [key_str, counts_json] : meta.at("ngram_count_table").items()) {
            model.ngram_count_table_[std::stoull(key_str)] = json_to_vec(counts_json);
        }
    }
    if (meta.contains("hybrid_blend_logit")) {
        model.hybrid_blend_logit_ = meta.at("hybrid_blend_logit").get<double>();
    }

    if (meta.contains("gria") && model.gria_) {
        const auto& g = meta.at("gria");
        if (g.contains("format") && g.at("format").get<std::string>() == "lowrank") {
            model.gria_->load_state(json_to_vec(g.at("U")), json_to_vec(g.at("V")),
                                    json_to_vec(g.at("alpha")), json_to_vec(g.at("bias")),
                                    g.value("alpha_learnable", true));
        } else if (g.contains("W")) {
            const auto w = json_to_vec(g.at("W"));
            const int d_in = g.value("d_input", model.gria_->field_dim);
            model.gria_->load_from_full_w(w, d_in, model.cfg_.gria_rank);
            if (g.contains("alpha")) {
                model.gria_->alpha = json_to_vec(g.at("alpha"));
            }
            if (g.contains("bias")) {
                model.gria_->bias = json_to_vec(g.at("bias"));
            }
            if (g.contains("alpha_learnable")) {
                model.gria_->alpha_learnable = g.at("alpha_learnable").get<bool>();
            }
        }
    }
    if (meta.contains("lstm") && model.lstm_) {
        const auto& l = meta.at("lstm");
        model.lstm_->load_state(json_to_vec(l.at("E")), json_to_vec(l.at("Wx")), json_to_vec(l.at("Wh")),
                                json_to_vec(l.at("b")), json_to_vec(l.at("Wy")), json_to_vec(l.at("by")));
    }
    if (meta.contains("hierarchical_ssm") && model.hierarchical_ssm_) {
        model.hierarchical_ssm_->set_state(meta.at("hierarchical_ssm"));
    } else if (meta.contains("ssm") && model.ssm_) {
        model.ssm_->set_state(meta.at("ssm"));
    }
    if (meta.contains("dif") && model.dif_) {
        model.dif_->set_state(meta.at("dif"));
    }

    fs::path npz_path = jp;
    npz_path.replace_extension(".npz");
    if (fs::is_regular_file(npz_path)) {
        const NpzReader npz = NpzReader::open(npz_path.string());
        if (npz.has("proj_ssm")) model.proj_ssm_ = npz.read_f64("proj_ssm");
        if (npz.has("proj_dif")) model.proj_dif_ = npz.read_f64("proj_dif");
        if (npz.has("proj_embed")) model.proj_embed_ = npz.read_f64("proj_embed");
        if (model.ngram_fusion_) {
            std::vector<double> pw;
            if (npz.has("ngram_pos_weights")) pw = npz.read_f64("ngram_pos_weights");
            model.ngram_fusion_->load_weights(
                npz.has("ngram_W_field") ? npz.read_f64("ngram_W_field") : std::vector<double>{},
                npz.has("ngram_W_embed") ? npz.read_f64("ngram_W_embed") : std::vector<double>{},
                npz.has("ngram_W_gate") ? npz.read_f64("ngram_W_gate") : std::vector<double>{}, pw);
        }
        if (model.view_emb_ && npz.has("view_embed")) {
            model.view_emb_->set_table(npz.read_f64("view_embed"));
        }
    }

    model.refresh_laplace_prior();
    return model;
}

}  // namespace cypha::cyphalm
