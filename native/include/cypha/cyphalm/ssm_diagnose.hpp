#pragma once

#include <cstddef>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cellai_ssm.hpp"

namespace cypha::cyphalm {

class CyphaLMModel;

/// L2 norm of a state vector (0 for empty).
double vector_l2_norm(const std::vector<double>& v);

/// Mean L2 norm across SSM layers for fast (h) or slow (s) states.
double layer_stack_mean_norm(const std::vector<std::vector<double>>& states);

/// Run CellAI SSM on a sequence of input embeddings; return Phase-5 diagnostic JSON.
nlohmann::json diagnose_cellai_sequence(CellAISSM& ssm,
                                        const std::vector<std::vector<double>>& inputs,
                                        int sample_stride, const char* domain_tag);

/// CyphaLM token stream probe (embed → SSM → field projection connectivity).
nlohmann::json diagnose_model_tokens(CyphaLMModel& model, const std::vector<int>& token_ids,
                                     int max_steps, const char* domain_tag);

/// Pad or truncate a feature row to ``d_input`` (zero fill).
std::vector<double> fit_input_dim(const std::vector<double>& row, int d_input);

/// Actionable tuning hints from Phase-5 SSM diagnostic JSON fragments.
nlohmann::json build_ssm_recommendations(const nlohmann::json& summary, const nlohmann::json& decay_rates,
                                           const nlohmann::json* projection = nullptr);

}  // namespace cypha::cyphalm
