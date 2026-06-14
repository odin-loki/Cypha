#include "cypha/cyphalm/cyphalm_intelligence_hook.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/measurers.hpp"

namespace cypha::cyphalm {

namespace {

constexpr double kEps = 1e-12;

double histogram_entropy(const double* values, int n, int n_bins = 16) {
  if (n <= 0) {
    return 0.0;
  }
  const int bins = std::max(1, n_bins);
  double lo = values[0];
  double hi = values[0];
  for (int i = 1; i < n; ++i) {
    lo = std::min(lo, values[i]);
    hi = std::max(hi, values[i]);
  }
  if (hi <= lo) {
    hi = lo + 1.0;
  }
  std::vector<double> hist(static_cast<std::size_t>(bins), 0.0);
  const double width = (hi - lo) / static_cast<double>(bins);
  for (int i = 0; i < n; ++i) {
    int idx = static_cast<int>((values[i] - lo) / width);
    idx = std::clamp(idx, 0, bins - 1);
    hist[static_cast<std::size_t>(idx)] += 1.0;
  }
  double ent = 0.0;
  const double norm = static_cast<double>(n) + kEps * static_cast<double>(bins);
  for (double c : hist) {
    const double p = (c + kEps) / norm;
    ent -= p * std::log(p);
  }
  return ent;
}

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

void update_profiler_from_lm_token(cypha::intelligence::IntelligenceProfiler& profiler,
                                   const std::vector<double>& input_embed,
                                   const std::vector<double>& log_probs, double epistemic_var,
                                   double aleatoric_var) {
  cypha::intelligence::ProfileObservation obs;
  if (!log_probs.empty()) {
    const double h_out =
        log_probs_entropy(log_probs.data(), static_cast<int>(log_probs.size()));
    double h_in = h_out + kEps;
    if (!input_embed.empty()) {
      h_in = histogram_entropy(input_embed.data(), static_cast<int>(input_embed.size()));
    }
    if (h_in > kEps) {
      obs.alpha = std::clamp(1.0 - h_out / h_in, 0.0, 1.0);
    }
  }
  if (epistemic_var > 0.0 || aleatoric_var > 0.0) {
    obs.r_eu = cypha::intelligence::compute_epistemic_ratio(epistemic_var, aleatoric_var);
  }
  profiler.update(obs);
}

}  // namespace cypha::cyphalm
