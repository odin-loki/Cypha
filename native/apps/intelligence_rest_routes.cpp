#include "cypha/intelligence_rest.hpp"

#include <mutex>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/intelligence/causal_graph.hpp"
#include "cypha/intelligence/criticality_vector.hpp"
#include "cypha/intelligence/intelligence_profile_json.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_completeness.hpp"
#include "cypha/intelligence/profile_from_model.hpp"

namespace cypha {

namespace {

std::mutex* g_mu{nullptr};
cypha::intelligence::IntelligenceProfiler* g_profiler{nullptr};
cypha::intelligence::CausalGraphMonitor* g_causal_graph{nullptr};

}  // namespace

void intelligence_rest_configure(std::mutex* mu, cypha::intelligence::IntelligenceProfiler* profiler,
                                 cypha::intelligence::CausalGraphMonitor* causal_graph) {
  g_mu = mu;
  g_profiler = profiler;
  g_causal_graph = causal_graph;
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
    const auto completeness =
        cypha::intelligence::validate_profile_completeness(*g_profiler);
    payload["profile_completeness"] =
        cypha::intelligence::profile_completeness_to_json(completeness);
    payload["source"] = "intelligence_profiler";
    res.set_content(payload.dump(), "application/json");
  });

  svr.Get("/intelligence/report", [](const httplib::Request& req, httplib::Response& res) {
    const std::string source = req.get_param_value("source");
    if (source == "live") {
      if (g_mu == nullptr || g_profiler == nullptr) {
        res.status = 503;
        res.set_content(R"({"detail":"intelligence profiler not configured"})", "application/json");
        return;
      }
      try {
        std::lock_guard<std::mutex> lk(*g_mu);
        nlohmann::json payload =
            cypha::intelligence::intelligence_profile_report_json(*g_profiler);
        const auto completeness =
            cypha::intelligence::validate_profile_completeness(*g_profiler);
        payload["profile_completeness"] =
            cypha::intelligence::profile_completeness_to_json(completeness);
        payload["source"] = "live_profiler";
        res.set_content(payload.dump(), "application/json");
      } catch (const std::exception& ex) {
        res.status = 500;
        nlohmann::json err{{"detail", ex.what()}};
        res.set_content(err.dump(), "application/json");
      }
      return;
    }

    try {
      const auto profiler =
          cypha::intelligence::profile_from_reference_fixture(cypha::bench::repo_root());
      nlohmann::json payload = cypha::intelligence::intelligence_profile_report_json(profiler);
      payload["source"] = "reference_fixture";
      res.set_content(payload.dump(), "application/json");
    } catch (const std::exception& ex) {
      res.status = 500;
      nlohmann::json err{{"detail", ex.what()}};
      res.set_content(err.dump(), "application/json");
    }
  });

  svr.Get("/intelligence/simulation", [](const httplib::Request&, httplib::Response& res) {
    if (g_mu == nullptr || g_causal_graph == nullptr) {
      res.status = 503;
      res.set_content(R"({"detail":"causal graph monitor not configured"})", "application/json");
      return;
    }
    std::lock_guard<std::mutex> lk(*g_mu);
    nlohmann::json payload = g_causal_graph->to_json();
    payload["source"] = "causal_graph_monitor";
    res.set_content(payload.dump(), "application/json");
  });

  svr.Get("/intelligence/criticality", [](const httplib::Request& req, httplib::Response& res) {
    if (g_mu == nullptr || g_profiler == nullptr) {
      res.status = 503;
      res.set_content(R"({"detail":"intelligence profiler not configured"})", "application/json");
      return;
    }
    cypha::intelligence::CriticalityBuildOptions opts;
    if (req.has_param("mid")) {
      opts.enable_mid = req.get_param_value("mid") == "1" || req.get_param_value("mid") == "true";
    }
    if (req.has_param("step")) {
      opts.step = std::stoll(req.get_param_value("step"));
    }
    std::lock_guard<std::mutex> lk(*g_mu);
    const auto vec = g_profiler->criticality_vector({}, {}, opts);
    nlohmann::json payload = cypha::intelligence::criticality_vector_to_json(vec);
    payload["source"] = "intelligence_profiler";
    res.set_content(payload.dump(), "application/json");
  });
}

}  // namespace cypha
