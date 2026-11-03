#include "cypha/intelligence/profile_completeness.hpp"

#include <cmath>

namespace cypha::intelligence {

namespace {

bool value_in_unit_interval(double value) {
  return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

}  // namespace

ProfileCompletenessResult validate_profile_completeness(const IntelligenceProfiler& profiler,
                                                        int min_updates_per_stat) {
  ProfileCompletenessResult result;
  result.update_counts = profiler.get_statistic_update_counts();
  result.kappa = profiler.criticality_score();

  bool all_complete = true;
  static constexpr const char* kStatNames[] = {"alpha", "d_eff", "sigma_branch", "tau",
                                               "r_eu", "lipschitz", "calibration"};
  for (std::size_t i = 0; i < kProfileStatisticCount; ++i) {
    result.stat_observed[i] = result.update_counts[i] >= min_updates_per_stat;
    if (!result.stat_observed[i]) {
      all_complete = false;
      result.missing_stats.push_back(kStatNames[i]);
    }
  }
  result.all_complete = all_complete && std::isfinite(result.kappa);
  return result;
}

bool profile_observation_complete(const ProfileObservation& obs) {
  return value_in_unit_interval(obs.alpha) && value_in_unit_interval(obs.d_eff) &&
         value_in_unit_interval(obs.sigma_branch) && value_in_unit_interval(obs.tau) &&
         value_in_unit_interval(obs.r_eu) && value_in_unit_interval(obs.lipschitz) &&
         value_in_unit_interval(obs.calibration);
}

nlohmann::json profile_completeness_to_json(const ProfileCompletenessResult& result) {
  nlohmann::json stat_observed = nlohmann::json::array();
  nlohmann::json update_counts = nlohmann::json::array();
  for (std::size_t i = 0; i < kProfileStatisticCount; ++i) {
    stat_observed.push_back(result.stat_observed[i]);
    update_counts.push_back(result.update_counts[i]);
  }
  return nlohmann::json{
      {"all_complete", result.all_complete},
      {"stat_observed", stat_observed},
      {"update_counts", update_counts},
      {"kappa", result.kappa},
      {"missing_stats", result.missing_stats},
  };
}

}  // namespace cypha::intelligence
