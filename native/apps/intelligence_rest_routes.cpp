#include "cypha/intelligence_rest.hpp"

#include <mutex>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "cypha/intelligence/intelligence_profile_json.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"

namespace cypha {
namespace {

std::mutex* g_mu{nullptr};
cypha::intelligence::IntelligenceProfiler* g_profiler{nullptr};

}  // namespace

void intelligence_rest_configure(std::mutex* mu, cypha::intelligence::IntelligenceProfiler* profiler) {
  g_mu = mu;
  g_profiler = profiler;
}

void register_intelligence_rest_routes(httplib::Server& svr) {
  svr.Get("/intelligence/profile", [](const httplib::Request&, httplib::Response& res) {
    if (g_mu == nullptr || g_profiler == nullptr) {
      res.status = 503;
      res.set_content(R"({"detail":"intelligence profiler not configured"})", "application/json");
      return;
    }
    std::lock_guard<std::mutex> lk(*g_mu);
    nlohmann::json payload = cypha::intelligence::intelligence_profile_to_json(*g_profiler);
    payload["source"] = "intelligence_profiler";
    res.set_content(payload.dump(), "application/json");
  });
}

}  // namespace cypha
