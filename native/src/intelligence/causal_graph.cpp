#include "cypha/intelligence/causal_graph.hpp"

#include <algorithm>
#include <cmath>

namespace cypha::intelligence {

namespace {

double clamp01(double x) { return std::clamp(x, 0.0, 1.0); }

}  // namespace

CausalGraphMonitor::CausalGraphMonitor() = default;

void CausalGraphMonitor::record_edge(std::string from, std::string to, double weight) {
  edges_.push_back(CausalEdge{std::move(from), std::move(to), clamp01(weight)});
}

void CausalGraphMonitor::observe_profile(const ProfileObservation& obs) {
  if (has_last_obs_) {
    const double delta_reu = obs.r_eu - last_obs_.r_eu;
    record_edge("r_eu", "tau", clamp01(0.5 + 0.5 * delta_reu));
    record_edge("alpha", "sigma_branch", clamp01(std::abs(obs.alpha - last_obs_.alpha)));
    if (delta_reu > 0.0) {
      soft_world_.record_acquisition(last_obs_.r_eu, obs.r_eu);
    }
  }
  last_obs_ = obs;
  has_last_obs_ = true;
  record_edge("alpha", "calibration", clamp01(1.0 - std::abs(obs.alpha - 0.5)));
  record_edge("tau", "r_eu", clamp01(obs.tau * obs.r_eu));
}

void CausalGraphMonitor::record_acquisition(double r_eu_before, double r_eu_after) {
  soft_world_.record_acquisition(r_eu_before, r_eu_after);
  record_edge("query", "r_eu", clamp01(r_eu_before - r_eu_after));
}

void CausalGraphMonitor::record_simulation(double resolution) {
  soft_world_.record_simulation(resolution);
  record_edge("simulation", "world_model", clamp01(resolution));
}

nlohmann::json CausalGraphMonitor::to_json() const {
  nlohmann::json edges = nlohmann::json::array();
  for (const auto& e : edges_) {
    edges.push_back({{"from", e.from}, {"to", e.to}, {"weight", e.weight}});
  }
  return {
      {"edges", edges},
      {"soft_world",
       {{"maturation_level", soft_world_.maturation_level()},
        {"query_quality", soft_world_.query_quality()}}},
      {"last_observation",
       has_last_obs_
           ? nlohmann::json{{"alpha", last_obs_.alpha},
                            {"tau", last_obs_.tau},
                            {"r_eu", last_obs_.r_eu}}
           : nullptr},
  };
}

}  // namespace cypha::intelligence
