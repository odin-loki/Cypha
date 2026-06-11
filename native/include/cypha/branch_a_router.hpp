#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"

namespace cypha {

/// Frozen hashing text embedder for Branch A (no sentence-transformers).
struct HashingEmbedConfig {
    int n_features{512};
    int n_components{128};
    int seed{42};
    /// Row-major ``(n_components × n_features)``; empty → hash-only truncate/pad to ``n_components``.
    std::vector<double> projection;
};

struct EmbedMeta {
    std::string backend{"hashing_svd"};
    int n_features{512};
    int n_components{128};
};

/// Route decision aligned with ``cypha_studio.core.branch_a_router.BranchARouter.route``.
struct BranchARouteResult {
    std::string label;
    double confidence{};
    double epistemic_var{};
    bool abstain{};
    std::string embedding_backend;
    std::string action;  // ``cypha_route`` | ``fallback_llm``
};

struct BranchAGenerationResult {
    std::string provider;  // ``cypha_lm`` | ``ollama`` | ``none``
    std::string text;
    std::string reason;
    std::string error;
    std::vector<int> generated_ids;
    double latency_ms{};
    int n_tokens{};
};

/// Hashing embedder + CyphaDIF router (load native JSON checkpoint or train stub).
class BranchARouter {
 public:
    BranchARouter();

    void set_epistemic_threshold(double t) { epistemic_threshold_ = t; }
    double epistemic_threshold() const { return epistemic_threshold_; }

    void set_checkpoint_base(std::string base) { checkpoint_base_ = std::move(base); }
    const std::string& checkpoint_base() const { return checkpoint_base_; }

    bool is_trained() const;
    bool try_load_checkpoint(const std::string& json_path = "");
    void load_checkpoint(const std::string& json_path = "");
    std::string save_checkpoint(const std::string& base = "");

    BranchARouteResult route(const std::string& text,
                             std::optional<double> epistemic_threshold = std::nullopt) const;

    struct DispatchGenerateOptions {
        std::optional<double> epistemic_threshold;
        int max_tokens{128};
        std::optional<std::string> ollama_model;
        std::optional<std::string> ollama_system;
        std::string cypha_lm_strategy{"top_p"};
        double cypha_lm_temperature{0.9};
    };

    nlohmann::json summary() const;

 private:
    void ensure_trained() const;
    std::vector<double> embed_text_unlocked(const std::string& text, EmbedMeta* meta_out = nullptr) const;
    static std::vector<double> hashing_features(const std::string& text, int n_features);
    static std::vector<double> project_features(const std::vector<double>& raw,
                                                const HashingEmbedConfig& cfg);
    static double shannon_entropy(const std::vector<double>& probs);

    mutable std::mutex mu_;
    mutable bool trained_{false};

    double epistemic_threshold_{0.5};
    int n_train_samples_{0};
    std::string checkpoint_base_;
    HashingEmbedConfig embed_cfg_;
    std::string embedding_backend_{"hashing_svd"};
    std::optional<double> train_seconds_;

    std::vector<double> mean_;
    std::vector<double> std_;
    std::unique_ptr<CyphaInferModel> model_;
    std::unique_ptr<CyphaDifMemoryState> mem_;
    CNode model_root_;
};

std::vector<int> encode_prompt_chars(const std::string& text, int vocab_size = 128);
std::string decode_generated_ids(const std::vector<int>& ids, const std::string& prompt,
                                 int vocab_size = 128);

}  // namespace cypha
