#include "cypha/branch_a_router.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "cypha/create_model.hpp"
#include "cypha/load_cypha.hpp"

namespace cypha {
namespace {

namespace fs = std::filesystem;

constexpr double kEps = 1e-8;

// MurmurHash3 32-bit (x86) — sklearn HashingVectorizer compatible seed handling.
std::uint32_t murmurhash3_32(const std::uint8_t* data, std::size_t len, std::uint32_t seed) {
    const std::uint32_t c1 = 0xcc9e2d51;
    const std::uint32_t c2 = 0x1b873593;
    std::uint32_t h1 = seed;
    const int nblocks = static_cast<int>(len / 4);
    const auto* blocks = reinterpret_cast<const std::uint32_t*>(data);

    for (int i = 0; i < nblocks; ++i) {
        std::uint32_t k1 = blocks[i];
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;
        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> 19);
        h1 = h1 * 5 + 0xe6546b64;
    }

    const std::uint8_t* tail = data + nblocks * 4;
    std::uint32_t k1 = 0;
    switch (len & 3) {
        case 3:
            k1 ^= static_cast<std::uint32_t>(tail[2]) << 16;
            [[fallthrough]];
        case 2:
            k1 ^= static_cast<std::uint32_t>(tail[1]) << 8;
            [[fallthrough]];
        case 1:
            k1 ^= static_cast<std::uint32_t>(tail[0]);
            k1 *= c1;
            k1 = (k1 << 15) | (k1 >> 17);
            k1 *= c2;
            h1 ^= k1;
    }

    h1 ^= static_cast<std::uint32_t>(len);
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;
    return h1;
}

std::uint32_t hash_token(const std::string& token, std::uint32_t seed) {
    return murmurhash3_32(reinterpret_cast<const std::uint8_t*>(token.data()), token.size(), seed);
}

std::vector<std::string> tokenize_ngrams(const std::string& text) {
    std::string lower;
    lower.reserve(text.size());
    for (unsigned char c : text) {
        lower.push_back(static_cast<char>(std::tolower(c)));
    }
    std::istringstream iss(lower);
    std::vector<std::string> words;
    std::string w;
    while (iss >> w) {
        if (!w.empty()) {
            words.push_back(w);
        }
    }
    std::vector<std::string> out;
    out.reserve(words.size() * 2);
    for (const auto& word : words) {
        out.push_back(word);
    }
    for (std::size_t i = 0; i + 1 < words.size(); ++i) {
        out.push_back(words[i] + " " + words[i + 1]);
    }
    return out;
}

fs::path resolve_path(const fs::path& base, const std::string& rel) {
    fs::path p(rel);
    if (p.is_absolute()) {
        return p;
    }
    return (base.parent_path() / p).lexically_normal();
}

int infer_field_dim(const CNode& root) {
    if (const CNode* w = map_get(root, "world")) {
        if (const CNode* ff = map_get(*w, "F_field")) {
            if (ff->shape.size() == 2) {
                return static_cast<int>(ff->shape[1]);
            }
        }
    }
    if (const CNode* fd = map_get(root, "field_dim")) {
        if (fd->kind == CNode::Int) {
            return static_cast<int>(fd->i);
        }
        if (fd->kind == CNode::Float) {
            return static_cast<int>(fd->f);
        }
    }
    return 24;
}

}  // namespace

BranchARouter::BranchARouter() = default;

bool BranchARouter::is_trained() const {
    std::lock_guard<std::mutex> lk(mu_);
    return trained_ && model_ && !mean_.empty() && !std_.empty();
}

std::vector<double> BranchARouter::hashing_features(const std::string& text, int n_features) {
    std::vector<double> acc(static_cast<std::size_t>(n_features), 0.0);
    for (const auto& tok : tokenize_ngrams(text)) {
        const std::uint32_t h = hash_token(tok, 0);
        const int idx = static_cast<int>(h % static_cast<std::uint32_t>(n_features));
        acc[static_cast<std::size_t>(idx)] += 1.0;
    }
    double norm = 0.0;
    for (double v : acc) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > kEps) {
        for (double& v : acc) {
            v /= norm;
        }
    }
    return acc;
}

std::vector<double> BranchARouter::project_features(const std::vector<double>& raw,
                                                    const HashingEmbedConfig& cfg) {
    const int d_out = cfg.n_components;
    std::vector<double> out(static_cast<std::size_t>(d_out), 0.0);
    if (!cfg.projection.empty()) {
        const int cols = cfg.n_features;
        const int rows = static_cast<int>(cfg.projection.size()) / cols;
        const int use_rows = std::min(rows, d_out);
        for (int r = 0; r < use_rows; ++r) {
            double s = 0.0;
            for (int c = 0; c < cols && c < static_cast<int>(raw.size()); ++c) {
                s += cfg.projection[static_cast<std::size_t>(r * cols + c)] * raw[static_cast<std::size_t>(c)];
            }
            out[static_cast<std::size_t>(r)] = s;
        }
        return out;
    }
    for (int i = 0; i < d_out && i < static_cast<int>(raw.size()); ++i) {
        out[static_cast<std::size_t>(i)] = raw[static_cast<std::size_t>(i)];
    }
    return out;
}

double BranchARouter::shannon_entropy(const std::vector<double>& probs) {
    double h = 0.0;
    for (double p : probs) {
        const double q = std::max(p, kEps);
        h -= q * std::log(q);
    }
    return h;
}

std::vector<double> BranchARouter::embed_text_unlocked(const std::string& text,
                                                       EmbedMeta* meta_out) const {
    const auto raw = hashing_features(text, embed_cfg_.n_features);
    auto dense = project_features(raw, embed_cfg_);
    if (meta_out) {
        meta_out->backend = embedding_backend_;
        meta_out->n_features = embed_cfg_.n_features;
        meta_out->n_components = embed_cfg_.n_components;
    }
    return dense;
}

void BranchARouter::ensure_trained() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (trained_) {
        return;
    }
    if (!checkpoint_base_.empty()) {
        const_cast<BranchARouter*>(this)->try_load_checkpoint(checkpoint_base_);
    }
    if (!trained_) {
        throw std::runtime_error(
            "Branch A router not trained — provide --branch-a-json or CYPHA_BRANCH_A_CHECKPOINT");
    }
}

bool BranchARouter::try_load_checkpoint(const std::string& json_path) {
    const std::string path = json_path.empty() ? checkpoint_base_ : json_path;
    if (path.empty()) {
        return false;
    }
    try {
        load_checkpoint(path);
        return true;
    } catch (...) {
        return false;
    }
}

void BranchARouter::load_checkpoint(const std::string& json_path) {
    const std::string path = json_path.empty() ? checkpoint_base_ : json_path;
    if (path.empty()) {
        throw std::invalid_argument("branch A checkpoint path empty");
    }

    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Branch A checkpoint JSON missing: " + path);
    }
    nlohmann::json meta;
    f >> meta;

    const fs::path json_file = fs::path(path);
    const std::string cypha_rel = meta.at("model_cypha").get<std::string>();
    const fs::path cypha_path = resolve_path(json_file, cypha_rel);
    if (!fs::is_regular_file(cypha_path)) {
        throw std::runtime_error("Branch A model.cypha missing: " + cypha_path.string());
    }

    CNode root = load_cypha_file(cypha_path.string().c_str());
    std::vector<double> f_field;
    const double* ff_ptr = nullptr;
    if (meta.contains("f_field_json") && meta["f_field_json"].is_string()) {
        const fs::path ff_path = resolve_path(json_file, meta["f_field_json"].get<std::string>());
        std::ifstream ff(ff_path);
        if (!ff) {
            throw std::runtime_error("f_field_json missing: " + ff_path.string());
        }
        nlohmann::json ffj;
        ff >> ffj;
        f_field = ffj.at("F_field").get<std::vector<double>>();
        ff_ptr = f_field.data();
    }

    const int field_dim = infer_field_dim(root);
    auto infer = std::make_unique<CyphaInferModel>(CyphaInferModel::from_root(root, ff_ptr, field_dim));
    auto mem = std::make_unique<CyphaDifMemoryState>(
        CyphaDifMemoryState::from_cypha_root(root, ff_ptr, field_dim));

    mean_ = meta.at("mean").get<std::vector<double>>();
    std_ = meta.at("std").get<std::vector<double>>();
    for (double& s : std_) {
        if (std::abs(s) < kEps) {
            s = 1.0;
        }
    }

    epistemic_threshold_ = meta.value("epistemic_threshold", 0.5);
    n_train_samples_ = meta.value("n_train_samples", 0);
    embedding_backend_ = meta.value("embedding_backend", std::string("hashing_svd"));
    if (meta.contains("train_seconds") && !meta["train_seconds"].is_null()) {
        train_seconds_ = meta["train_seconds"].get<double>();
    }

    embed_cfg_.n_features = meta.value("hash_n_features", 512);
    embed_cfg_.n_components = meta.value("hash_n_components", static_cast<int>(mean_.size()));
    embed_cfg_.seed = meta.value("hash_seed", 42);
    if (meta.contains("projection") && meta["projection"].is_array()) {
        embed_cfg_.projection = meta["projection"].get<std::vector<double>>();
    }

    model_ = std::move(infer);
    mem_ = std::move(mem);
    model_root_ = std::move(root);
    checkpoint_base_ = path;
    trained_ = true;
}

std::string BranchARouter::save_checkpoint(const std::string& base) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!trained_ || !model_) {
        throw std::runtime_error("router is not trained");
    }

    const fs::path root = base.empty() ? fs::path(checkpoint_base_).parent_path() : fs::path(base);
    const fs::path json_path = root / "branch_a_router.json";
    const fs::path cypha_path = root / "router_model.cypha";

    fs::create_directories(root);

    FreshModelParams fmp;
    fmp.input_dim = model_->d_latent;
    fmp.field_dim = model_->field_dim;
    fmp.temperature = model_->temperature;
    CNode save_root =
        model_root_.kind == CNode::Map ? clone_cnode(model_root_) : create_fresh_model_root(fmp);
    if (mem_) {
        save_root = CyphaDifMemoryState::merge_state_into_root_for_save(save_root, *mem_);
    }
    save_cypha_file(cypha_path.string().c_str(), save_root);

    nlohmann::json meta;
    meta["version"] = 1;
    meta["format"] = "native";
    meta["model_cypha"] = cypha_path.filename().string();
    meta["mean"] = mean_;
    meta["std"] = std_;
    meta["epistemic_threshold"] = epistemic_threshold_;
    meta["embedding_backend"] = embedding_backend_;
    meta["hash_n_features"] = embed_cfg_.n_features;
    meta["hash_n_components"] = embed_cfg_.n_components;
    meta["hash_seed"] = embed_cfg_.seed;
    meta["n_train_samples"] = n_train_samples_;
    meta["n_classes"] = model_->labels.size();
    if (train_seconds_) {
        meta["train_seconds"] = *train_seconds_;
    }
    if (!embed_cfg_.projection.empty()) {
        meta["projection"] = embed_cfg_.projection;
    }

    std::ofstream out(json_path);
    out << meta.dump(2) << '\n';
    checkpoint_base_ = json_path.string();
    return json_path.string();
}

BranchARouteResult BranchARouter::route(const std::string& text,
                                        std::optional<double> epistemic_threshold) const {
    ensure_trained();
    std::lock_guard<std::mutex> lk(mu_);
    if (!model_) {
        throw std::runtime_error("Branch A model not loaded");
    }

    EmbedMeta em;
    auto dense = embed_text_unlocked(text, &em);
    if (dense.size() != mean_.size()) {
        throw std::runtime_error("embedding dim mismatch after projection");
    }

    std::vector<double> x(dense.size());
    for (std::size_t i = 0; i < dense.size(); ++i) {
        x[i] = (dense[i] - mean_[i]) / std_[i];
    }

    CyphaInferOptions opt;
    opt.use_field = true;
    const auto pred = infer_at_h(*model_, x.data(), opt);

    const int k = static_cast<int>(model_->labels.size());
    std::vector<double> probs(static_cast<std::size_t>(k), 0.0);
    if (k > 0) {
        std::vector<double> llr(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) {
            llr[static_cast<std::size_t>(i)] =
                i < static_cast<int>(pred.llrs.size()) ? pred.llrs[static_cast<std::size_t>(i)] : 0.0;
        }
        softmax_batch_reference(llr.data(), 1, k, kEps, probs);
    }

    const double threshold =
        epistemic_threshold.has_value() ? *epistemic_threshold : epistemic_threshold_;
    const double epistemic = shannon_entropy(probs);
    const bool abstain = epistemic > threshold;

    BranchARouteResult out;
    out.label = pred.label;
    out.confidence = pred.confidence;
    out.epistemic_var = epistemic;
    out.abstain = abstain;
    out.embedding_backend = em.backend;
    out.action = abstain ? "fallback_llm" : "cypha_route";
    return out;
}

nlohmann::json BranchARouter::summary() const {
    std::lock_guard<std::mutex> lk(mu_);
    nlohmann::json j;
    j["trained"] = trained_;
    j["n_train_samples"] = n_train_samples_;
    j["epistemic_threshold"] = epistemic_threshold_;
    j["embedding_backend"] = embedding_backend_;
    if (train_seconds_) {
        j["train_seconds"] = *train_seconds_;
    } else {
        j["train_seconds"] = nullptr;
    }
    j["n_classes"] = model_ ? model_->labels.size() : 0;
    j["checkpoint_base"] = checkpoint_base_;
    j["checkpoint_exists"] = !checkpoint_base_.empty() && fs::is_regular_file(fs::path(checkpoint_base_));
    return j;
}

std::vector<int> encode_prompt_chars(const std::string& text, int vocab_size) {
    std::unordered_map<char, int> char2id;
    std::vector<char> uniq;
    for (char c : text) {
        if (std::find(uniq.begin(), uniq.end(), c) == uniq.end()) {
            uniq.push_back(c);
        }
    }
    std::sort(uniq.begin(), uniq.end());
    const int limit = std::max(vocab_size - 1, 1);
    if (static_cast<int>(uniq.size()) > limit) {
        uniq.resize(static_cast<std::size_t>(limit));
    }
    for (std::size_t i = 0; i < uniq.size(); ++i) {
        char2id[uniq[i]] = static_cast<int>(i) + 1;
    }
    std::vector<int> ids;
    ids.reserve(text.size());
    for (char c : text) {
        auto it = char2id.find(c);
        ids.push_back(it == char2id.end() ? 0 : it->second);
    }
    return ids;
}

std::string decode_generated_ids(const std::vector<int>& ids, const std::string& prompt, int vocab_size) {
    std::vector<char> uniq;
    for (char c : prompt) {
        if (std::find(uniq.begin(), uniq.end(), c) == uniq.end()) {
            uniq.push_back(c);
        }
    }
    std::sort(uniq.begin(), uniq.end());
    const int limit = std::max(vocab_size - 1, 1);
    if (static_cast<int>(uniq.size()) > limit) {
        uniq.resize(static_cast<std::size_t>(limit));
    }
    std::unordered_map<int, char> id2char;
    id2char[0] = '?';
    for (std::size_t i = 0; i < uniq.size(); ++i) {
        id2char[static_cast<int>(i) + 1] = uniq[i];
    }
    std::string out;
    out.reserve(ids.size());
    for (int t : ids) {
        auto it = id2char.find(t);
        out.push_back(it == id2char.end() ? '?' : it->second);
    }
    return out;
}

}  // namespace cypha
