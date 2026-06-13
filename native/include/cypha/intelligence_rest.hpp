#pragma once

#include <mutex>

#include "httplib.h"

namespace cypha::intelligence {

class IntelligenceProfiler;

}  // namespace cypha::intelligence

namespace cypha {

/// Bind session mutex + profiler owned by ``cypha_rest`` main.
void intelligence_rest_configure(std::mutex* mu, cypha::intelligence::IntelligenceProfiler* profiler);

/// Register ``GET /intelligence/profile``.
void register_intelligence_rest_routes(httplib::Server& svr);

}  // namespace cypha
