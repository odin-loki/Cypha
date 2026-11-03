#pragma once

#include "cypha/lm/cyphalm_model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cypha::cyphalm {

struct PredictiveCodecOptions {
    bool use_mixer = true;
    int ngram_max_order = 3;
    double mixer_lr = 0.15;
    double laplace = 0.5;
    bool use_match_expert = true;
    bool use_knn_expert = true;
    /// Bias/SSE is opt-in: on warm WikiText it diluted the neural before enough steps.
    bool use_bias_expert = false;
    bool online_adapt = true;
    double online_adapt_lr_scale = 0.25;
    /// Blend model ``hidden_knn_log_probs()`` into neural before the mixer (0 = off).
    bool use_hidden_knn = true;
    double hidden_knn_mix = 0.15;
    /// Softmax score prior for the neural expert (higher = less dilution).
    double neural_prior = 6.0;
    /// Floor on neural mix weight (warm-model safety). Raise to trust neural more.
    double neural_weight_floor = 0.55;
    /// N-gram experts stay cold until this many context counts.
    int min_context_count = 16;
};

struct PredictiveCodecResult {
    std::vector<std::uint8_t> bytes;
    double model_bpc = 0.0;
    double coded_bpc = 0.0;
    double neural_bpc = 0.0;
    std::size_t n_coded = 0;
    std::size_t n_tokens = 0;
    std::string detail;
};

PredictiveCodecResult compress_tokens(CyphaLMModel& model, const std::vector<std::uint32_t>& tokens,
                                      const PredictiveCodecOptions& opt = {});

std::vector<std::uint32_t> decompress_tokens(CyphaLMModel& model, const std::vector<std::uint8_t>& bytes,
                                             std::uint32_t seed, std::size_t n_tokens,
                                             std::string* detail = nullptr,
                                             const PredictiveCodecOptions& opt = {});

std::vector<std::uint32_t> generate_via_bits(CyphaLMModel& model, const std::vector<std::uint32_t>& prefix,
                                            std::size_t n_new, std::uint64_t rng_seed,
                                            std::string* detail = nullptr,
                                            const PredictiveCodecOptions& opt = {});

}  // namespace cypha::cyphalm
