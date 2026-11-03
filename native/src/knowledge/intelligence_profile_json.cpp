#include "cypha/intelligence/intelligence_profile_json.hpp"

#include "cypha/intelligence/criticality_vector.hpp"

namespace cypha::intelligence {

namespace {

const char* statistic_name(ProfileStatistic stat) {
  switch (stat) {
    case ProfileStatistic::Alpha:
      return "alpha";
    case ProfileStatistic::DEff:
      return "d_eff";
    case ProfileStatistic::SigmaBranch:
      return "sigma_branch";
    case ProfileStatistic::Tau:
      return "tau";
    case ProfileStatistic::REu:
      return "r_eu";
    case ProfileStatistic::Lipschitz:
      return "lipschitz";
    case ProfileStatistic::Calibration:
      return "calibration";
  }
  return "unknown";
}

}  // namespace

nlohmann::json intelligence_profile_to_json(const IntelligenceProfiler& profiler) {
  const auto matrix = profiler.get_profile_matrix();
  const auto targets = IntelligenceProfiler::critical_targets();
  nlohmann::json stats = nlohmann::json::array();
  for (std::size_t i = 0; i < kProfileStatisticCount; ++i) {
    stats.push_back(nlohmann::json{
        {"name", statistic_name(static_cast<ProfileStatistic>(i))},
        {"point", matrix[i][0]},
        {"epistemic", matrix[i][1]},
        {"aleatoric", matrix[i][2]},
        {"critical_target", targets[i]},
    });
  }
  nlohmann::json out{
      {"statistics", stats},
      {"criticality_score", profiler.criticality_score()},
      {"health_signal", profiler.health_signal()},
      {"critical_targets",
       nlohmann::json{{"alpha", targets[0]},
                      {"d_eff", targets[1]},
                      {"sigma_branch", targets[2]},
                      {"tau", targets[3]},
                      {"r_eu", targets[4]},
                      {"lipschitz", targets[5]},
                      {"calibration", targets[6]}}},
  };
  if (criticality_vector_enabled()) {
    out["criticality_vector"] = criticality_vector_to_json(profiler.criticality_vector());
  } else {
    out["criticality_vector"] = nullptr;
  }
  return out;
}

}  // namespace cypha::intelligence
