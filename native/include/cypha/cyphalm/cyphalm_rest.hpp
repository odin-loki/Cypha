#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_generation.hpp"

namespace cypha {
class Cypha;
}

namespace cypha::cyphalm {

/// Bind process ``Cypha`` instance owned by ``cypha_rest`` main (single sequence model).
void cyphalm_rest_configure(std::mutex* mu, cypha::Cypha* cypha);

/// Register ``/sequence/load``, ``/sequence/metrics``, ``/predict_next``, ``/generate``, ``/generate/stream``.
void register_cyphalm_rest_routes(httplib::Server& svr);

bool cyphalm_rest_lm_loaded();

/// Load Cypha sequence checkpoint (``.json`` + ``.npz``). Thread-safe; throws on failure.
void cyphalm_rest_lm_load(const std::string& checkpoint_path);

/// Run autoregressive decode; returns JSON with ``generated_ids``, ``n_tokens``, etc.
nlohmann::json cyphalm_rest_generate_json(const std::vector<int>& prompt_ids, int max_tokens,
                                          const DecodeParams& params);

}  // namespace cypha::cyphalm
