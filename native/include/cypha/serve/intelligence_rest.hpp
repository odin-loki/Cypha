#pragma once

#include <mutex>

#include "httplib.h"

namespace cypha::intelligence {

class CausalGraphMonitor;
class IntelligenceProfiler;

}  // namespace cypha::intelligence

namespace cypha {

/// Bind session mutex + profiler + causal graph owned by ``cypha_rest`` main.
void intelligence_rest_configure(std::mutex* mu, cypha::intelligence::IntelligenceProfiler* profiler,
                                 cypha::intelligence::CausalGraphMonitor* causal_graph);

/// Register ``GET /intelligence/profile``, ``GET /intelligence/report``, ``GET /intelligence/simulation``,
/// ``GET /intelligence/criticality``.
void register_intelligence_rest_routes(httplib::Server& svr);

}  // namespace cypha
