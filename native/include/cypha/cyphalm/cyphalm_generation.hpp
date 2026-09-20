#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/intelligence/epistemic_threshold.hpp"

namespace cypha::intelligence {
class IntelligenceProfiler;
}  // namespace cypha::intelligence

namespace cypha::cyphalm {
class LmIntelligenceMonitor;
}  // namespace cypha::cyphalm

namespace cypha::cyphalm {

enum class DecodeStrategy { Greedy, Temperature, TopK, TopP, Beam, UncertaintyGated };

struct DecodeParams {
    DecodeStrategy strategy = DecodeStrategy::Temperature;
    double temperature = 0.9;
    int top_k = 40;
    double top_p = 0.9;
    /// Byte-level beam width (1 = off). ``strategy=beam`` or ``beam_width>1`` selects beam search.
    int beam_width = 1;
    std::optional<double> uncertainty_threshold;
    /// Halt when epistemic ratio r_eu exceeds ``EpistemicThreshold`` (Paper IV).
    bool epistemic_halt = false;
    /// On high r_eu, emit one greedy token before halting (LM self-correct stub).
    bool self_correct = false;
    std::uint64_t seed = 42;
    /// Bytes fed via ``serve_advance`` before the prompt (context priming; serve-time only).
    std::vector<int> warmup_ids;
    /// Hard-ban bytes appearing in the last ``ban_last_k`` context bytes (0 = off).
    int ban_last_k = 0;
    /// Divide repeat-byte mass by this factor within ``repetition_window`` (1.0 = off).
    double repetition_penalty = 1.0;
    /// Lookback window for ``repetition_penalty`` (bytes).
    int repetition_window = 32;
    /// Soft log-prob bonus for printable ASCII / common whitespace (serve-time only; 0 = off).
    double text_like_prior = 0.0;
};

DecodeStrategy decode_strategy_from_string(const std::string& name);

struct GenerateStep {
    int token_id = 0;
    double loss = 0.0;
    double epistemic_var = 0.0;
    double aleatoric_var = 0.0;
    int active_experts = 0;
    bool halted = false;
};

struct GenerateOutput {
    std::vector<int> generated_ids;
    std::vector<GenerateStep> per_step;
    bool halted_on_uncertainty = false;
    bool halted_on_epistemic = false;
    double r_eu_proxy = 0.0;
    bool self_corrected = false;
    int self_correct_passes = 0;
    DecodeStrategy strategy = DecodeStrategy::Temperature;
};

GenerateOutput generate_decode(CyphaLMModel& model, const std::vector<int>& prompt_ids, int max_tokens,
                               const DecodeParams& params,
                               cypha::intelligence::EpistemicThreshold* epistemic_threshold = nullptr,
                               cypha::intelligence::IntelligenceProfiler* profiler = nullptr,
                               LmIntelligenceMonitor* monitor = nullptr);

/// Byte-level beam search on bit-tree log probs (``beam_width`` hypotheses, predictor snapshots).
GenerateOutput generate_beam(CyphaLMModel& model, const std::vector<int>& prompt_ids, int max_bytes,
                             const DecodeParams& params);

/// Greedy decode (legacy wrapper).
GenerateOutput generate_greedy(CyphaLMModel& model, const std::vector<int>& prompt_ids, int max_tokens);

/// Temperature + top_k sampling (legacy wrapper).
GenerateOutput generate_sample(CyphaLMModel& model, const std::vector<int>& prompt_ids, int max_tokens,
                               double temperature, int top_k, std::uint64_t seed = 42);

/// Invoke ``cb`` once per SSE chunk; stop early if ``cb`` returns false.
void stream_generate(CyphaLMModel& model, const std::vector<int>& prompt_ids, int max_tokens,
                     const DecodeParams& params, const std::function<bool(const nlohmann::json&)>& cb,
                     cypha::intelligence::EpistemicThreshold* epistemic_threshold = nullptr,
                     cypha::intelligence::IntelligenceProfiler* profiler = nullptr,
                     LmIntelligenceMonitor* monitor = nullptr);

nlohmann::json predict_next_json(CyphaLMModel& model, int token_id);

/// Load up to ``max_bytes`` raw bytes from ``path`` as token ids (vocab-clamped).
std::vector<int> load_warmup_bytes(const std::string& path, int max_bytes, int vocab_size);

/// Advance serve context on ``warmup_ids`` without resetting first (caller may ``reset_context``).
void warmup_serve_context(CyphaLMModel& model, const std::vector<int>& warmup_ids);

nlohmann::json lm_summary_json(const CyphaLMModel& model, const std::string& source_path, int n_generations);

}  // namespace cypha::cyphalm
