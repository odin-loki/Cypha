#pragma once

#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

namespace cypha::intelligence {
class IntelligenceProfiler;
}  // namespace cypha::intelligence

namespace cypha::cyphalm {

class LmIntelligenceMonitor;

/// Full seven-stat LM step hook (embed, field hidden state, logits, DIF vars, target token).
void update_profiler_from_lm_step(cypha::intelligence::IntelligenceProfiler& profiler,
                                  LmIntelligenceMonitor& monitor,
                                  const std::vector<double>& input_embed,
                                  const std::vector<double>& field_hidden,
                                  const std::vector<double>& log_probs, double epistemic_var,
                                  double aleatoric_var, std::int64_t next_token_id, int vocab_size);

/// Backward-compatible per-token hook (α and r_eu only when field/target omitted).
void update_profiler_from_lm_token(cypha::intelligence::IntelligenceProfiler& profiler,
                                   const std::vector<double>& input_embed,
                                   const std::vector<double>& log_probs, double epistemic_var,
                                   double aleatoric_var);

/// Merged intelligence report JSON (profile report + completeness).
nlohmann::json export_intelligence_monitor_report(
    const cypha::intelligence::IntelligenceProfiler& profiler);

/// Shannon entropy (nats) of a log-probability row.
double log_probs_entropy(const double* log_probs, int vocab_size);

}  // namespace cypha::cyphalm
