#pragma once

#include <string>

#include "httplib.h"

namespace cypha::cyphalm {

/// Register ``/lm/load``, ``/lm/metrics``, ``/lm/predict_next``, ``/generate``, ``/generate/stream``.
void register_cyphalm_rest_routes(httplib::Server& svr);

bool cyphalm_rest_lm_loaded();

/// Load CyphaLM checkpoint (``.json`` + ``.npz``). Thread-safe; throws on failure.
void cyphalm_rest_lm_load(const std::string& checkpoint_path);

}  // namespace cypha::cyphalm
