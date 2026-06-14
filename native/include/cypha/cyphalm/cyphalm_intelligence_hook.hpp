#pragma once

#include <cstdint>
#include <vector>

namespace cypha::intelligence {
class IntelligenceProfiler;
}  // namespace cypha::intelligence

namespace cypha::cyphalm {

/// Update profiler with entropy→α from one LM eval step (embed vs log-prob vector).
void update_profiler_from_lm_token(cypha::intelligence::IntelligenceProfiler& profiler,
                                   const std::vector<double>& input_embed,
                                   const std::vector<double>& log_probs, double epistemic_var,
                                   double aleatoric_var);

/// Shannon entropy (nats) of a log-probability row.
double log_probs_entropy(const double* log_probs, int vocab_size);

}  // namespace cypha::cyphalm
