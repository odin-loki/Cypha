#pragma once

#include <array>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_statistic.hpp"

namespace cypha::intelligence {

struct ProfileCompletenessResult {
  bool all_complete = false;
  std::array<bool, kProfileStatisticCount> stat_observed{};
  std::array<int, kProfileStatisticCount> update_counts{};
  std::vector<std::string> missing_stats;
  double kappa = 0.0;
};

ProfileCompletenessResult validate_profile_completeness(const IntelligenceProfiler& profiler,
                                                        int min_updates_per_stat = 1);

bool profile_observation_complete(const ProfileObservation& obs);

nlohmann::json profile_completeness_to_json(const ProfileCompletenessResult& result);

}  // namespace cypha::intelligence
