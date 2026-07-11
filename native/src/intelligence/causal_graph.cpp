#include "cypha/intelligence/causal_graph.hpp"

#include <algorithm>
#include <cmath>

namespace cypha::intelligence {

namespace {

double clamp01(double x) { return std::clamp(x, 0.0, 1.0); }

}  // namespace

void OnlineCorrelation::update(double x, double y) {
  ++n_;
  const double dx = x - mean_x_;
  mean_x_ += dx / static_cast<double>(n_);
  const double dy = y - mean_y_;
  mean_y_ += dy / static_cast<double>(n_);
  cov_ += dx * (y - mean_y_);
  var_x_ += dx * (x - mean_x_);
  var_y_ += dy * (y - mean_y_);
}

double OnlineCorrelation::correlation() const {
  constexpr double kMinVariance = 1e-12;
  if (n_ < 2 || var_x_ <= kMinVariance || var_y_ <= kMinVariance) {
    return 0.0;
  }
  return std::clamp(cov_ / std::sqrt(var_x_ * var_y_), -1.0, 1.0);
}

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

  // Edge weight estimated from the real (alpha, calibration) history via Pearson
  // correlation, rather than a fixed formula of alpha alone (which never read the
  // observed calibration value at all). Weight is 0 until >=2 observations exist.
  alpha_calibration_corr_.update(obs.alpha, obs.calibration);
  record_edge("alpha", "calibration", clamp01(std::abs(alpha_calibration_corr_.correlation())));

  // Edge weight estimated from the real (tau, r_eu) history via Pearson correlation,
  // rather than the instantaneous product ``tau * r_eu`` of a single observation.
  tau_r_eu_corr_.update(obs.tau, obs.r_eu);
  record_edge("tau", "r_eu", clamp01(std::abs(tau_r_eu_corr_.correlation())));
}

void CausalGraphMonitor::record_acquisition(double r_eu_before, double r_eu_after) {
  soft_world_.record_acquisition(r_eu_before, r_eu_after);
  record_edge("query", "r_eu", clamp01(r_eu_before - r_eu_after));
}

void CausalGraphMonitor::record_simulation(double resolution) {
  soft_world_.record_simulation(resolution);
  record_edge("simulation", "world_model", clamp01(resolution));
}

void CausalGraphMonitor::simulation_step(double r_eu_before, double r_eu_after, double resolution) {
  soft_world_.simulation_step(r_eu_before, r_eu_after, resolution);
  record_edge("query", "r_eu", clamp01(r_eu_before - r_eu_after));
  record_edge("simulation", "world_model", clamp01(resolution));
  record_edge("world_model", "maturation", clamp01(soft_world_.maturation_level()));
  trajectory_.push_back(SimulationStepEvent{
      r_eu_before, r_eu_after, resolution, soft_world_.maturation_level()});
}

void CausalGraphMonitor::run_simulation_trajectory(int n_steps, const ProfileObservation& obs,
                                                   double resolution_scale) {
  if (n_steps <= 0) {
    return;
  }
  observe_profile(obs);
  double r_eu = std::clamp(obs.r_eu, 0.1, 1.0);
  const double scale = std::max(0.01, resolution_scale);
  for (int i = 0; i < n_steps; ++i) {
    const double decay = scale * (0.75 + 0.05 * static_cast<double>(i));
    const double r_after = std::max(0.05, r_eu - decay);
    const double resolution = std::max(0.0, r_eu - r_after);
    simulation_step(r_eu, r_after, resolution);
    record_edge("maturation", "tau", clamp01(obs.tau * soft_world_.maturation_level()));
    r_eu = r_after;
  }
}

nlohmann::json CausalGraphMonitor::trajectory_json() const {
  nlohmann::json steps = nlohmann::json::array();
  for (const auto& s : trajectory_) {
    steps.push_back({{"r_eu_before", s.r_eu_before},
                     {"r_eu_after", s.r_eu_after},
                     {"resolution", s.resolution},
                     {"maturation_level", s.maturation_level}});
  }
  return {
      {"trajectory", steps},
      {"step_count", trajectory_.size()},
      {"soft_world",
       {{"maturation_level", soft_world_.maturation_level()},
        {"query_quality", soft_world_.query_quality()}}},
  };
}

nlohmann::json CausalGraphMonitor::to_json() const {
  nlohmann::json edges = nlohmann::json::array();
  for (const auto& e : edges_) {
    edges.push_back({{"from", e.from}, {"to", e.to}, {"weight", e.weight}});
  }
  nlohmann::json traj = trajectory_json();
  return {
      {"edges", edges},
      {"trajectory", traj.at("trajectory")},
      {"step_count", traj.at("step_count")},
      {"soft_world",
       {{"maturation_level", soft_world_.maturation_level()},
        {"query_quality", soft_world_.query_quality()}}},
      {"last_observation",
       has_last_obs_
           ? nlohmann::json{{"alpha", last_obs_.alpha},
                            {"tau", last_obs_.tau},
                            {"r_eu", last_obs_.r_eu}}
           : nullptr},
      {"edge_estimation",
       {{"alpha_calibration_correlation", alpha_calibration_corr_.correlation()},
        {"alpha_calibration_n", alpha_calibration_corr_.n()},
        {"tau_r_eu_correlation", tau_r_eu_corr_.correlation()},
        {"tau_r_eu_n", tau_r_eu_corr_.n()}}},
  };
}

}  // namespace cypha::intelligence
