#pragma once

#include <string>
#include <vector>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_generation.hpp"

namespace cypha::cyphalm {

/// Register ``/lm/load``, ``/lm/metrics``, ``/lm/predict_next``, ``/generate``, ``/generate/stream``.
void register_cyphalm_rest_routes(httplib::Server& svr);

bool cyphalm_rest_lm_loaded();

/// Load CyphaLM checkpoint (``.json`` + ``.npz``). Thread-safe; throws on failure.
void cyphalm_rest_lm_load(const std::string& checkpoint_path);

/// Run autoregressive decode; returns JSON with ``generated_ids``, ``n_tokens``, etc.
nlohmann::json cyphalm_rest_generate_json(const std::vector<int>& prompt_ids, int max_tokens,
                                          const DecodeParams& params);

}  // namespace cypha::cyphalm
