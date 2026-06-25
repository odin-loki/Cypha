#include "cypha/cyphalm/cyphalm_intelligence_hook.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "cypha/cyphalm/lm_intelligence_monitor.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_completeness.hpp"
#include "cypha/intelligence/profile_from_model.hpp"

namespace cypha::cyphalm {

namespace {

constexpr double kEps = 1e-12;

}  // namespace

double log_probs_entropy(const double* log_probs, int vocab_size) {
  if (log_probs == nullptr || vocab_size <= 0) {
    return 0.0;
  }
  double max_lp = log_probs[0];
  for (int i = 1; i < vocab_size; ++i) {
    max_lp = std::max(max_lp, log_probs[i]);
  }
  double sum = 0.0;
  for (int i = 0; i < vocab_size; ++i) {
    sum += std::exp(log_probs[i] - max_lp);
  }
  const double log_z = max_lp + std::log(sum + kEps);
  double h = 0.0;
  for (int i = 0; i < vocab_size; ++i) {
    const double p = std::exp(log_probs[i] - log_z);
    if (p > kEps) {
      h -= p * std::log(p + kEps);
    }
  }
  return h;
}

void update_profiler_from_lm_step(cypha::intelligence::IntelligenceProfiler& profiler,
                                 LmIntelligenceMonitor& monitor,
                                 const std::vector<double>& input_embed,
                                 const std::vector<double>& field_hidden,
                                 const std::vector<double>& log_probs, double epistemic_var,
                                 double aleatoric_var, std::int64_t next_token_id, int vocab_size) {
  monitor.observe_token(input_embed, field_hidden, log_probs, epistemic_var, aleatoric_var,
                        next_token_id, vocab_size);
  monitor.flush_to_profiler(profiler);
}

void update_profiler_from_lm_token(cypha::intelligence::IntelligenceProfiler& profiler,
                                   const std::vector<double>& input_embed,
                                   const std::vector<double>& log_probs, double epistemic_var,
                                   double aleatoric_var) {
  LmIntelligenceMonitor monitor;
  const int vocab_size = static_cast<int>(log_probs.size());
  monitor.observe_token(input_embed, {}, log_probs, epistemic_var, aleatoric_var, -1, vocab_size);
  monitor.flush_to_profiler(profiler);
}

nlohmann::json export_intelligence_monitor_report(
    const cypha::intelligence::IntelligenceProfiler& profiler) {
  nlohmann::json report = cypha::intelligence::intelligence_profile_report_json(profiler);
  const auto completeness = cypha::intelligence::validate_profile_completeness(profiler);
  report["profile_completeness"] = cypha::intelligence::profile_completeness_to_json(completeness);
  return report;
}

}  // namespace cypha::cyphalm
