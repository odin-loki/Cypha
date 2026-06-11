#pragma once

#include <string>

#include "httplib.h"
#include <nlohmann/json.hpp>

namespace cypha {

/// Configure Branch A router checkpoint JSON (``--branch-a-json`` or ``CYPHA_BRANCH_A_CHECKPOINT``).
void branch_a_rest_configure(const std::string& checkpoint_json_path);

/// Register ``/route/health``, ``/route/text``, ``/route/generate``, ``/route/save``.
void register_branch_a_rest_routes(httplib::Server& svr);

bool branch_a_rest_router_trained();

nlohmann::json branch_a_rest_summary_json();

}  // namespace cypha
