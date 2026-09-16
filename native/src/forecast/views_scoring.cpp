#include "cypha/forecast/views_scoring.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cypha::forecast {

double crps(const std::vector<double>& bin_probs, const std::vector<double>& bin_centers,
            double observed) {
  if (bin_probs.empty() || bin_probs.size() != bin_centers.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double score = 0.0;
  double prev_cdf = 0.0;
  double prev_x = 0.0;
  for (std::size_t i = 0; i < bin_probs.size(); ++i) {
    const double cdf = prev_cdf + bin_probs[i];
    const double x = bin_centers[i];
    const double h = (observed <= x) ? 1.0 : 0.0;
    score += (cdf - h) * (cdf - h) * (x - prev_x);
    prev_cdf = cdf;
    prev_x = x;
  }
  return score;
}

double ignorance_score(const std::vector<double>& bin_probs, const std::vector<double>& bin_centers,
                       double observed) {
  if (bin_probs.empty() || bin_probs.size() != bin_centers.size()) {
    return std::numeric_limits<double>::infinity();
  }
  std::size_t best = 0;
  for (std::size_t i = 1; i < bin_centers.size(); ++i) {
    if (std::abs(bin_centers[i] - observed) < std::abs(bin_centers[best] - observed)) {
      best = i;
    }
  }
  const double p = std::max(bin_probs[best], 1e-12);
  return -std::log(p);
}

ViewsValidationResult validate_against_views(const std::vector<FatalityForecast>& forecasts,
                                             const std::vector<ViewsObservation>& held_out) {
  ViewsValidationResult out;
  if (forecasts.empty() || held_out.empty()) {
    out.detail = "empty forecasts or held-out set";
    return out;
  }

  double crps_sum = 0.0;
  double ign_sum = 0.0;
  int n = 0;
  for (const auto& obs : held_out) {
    for (const auto& fc : forecasts) {
      if (fc.country != obs.country || fc.year != obs.year || fc.month != obs.month) {
        continue;
      }
      crps_sum += crps(fc.bin_probs, fc.bin_centers, obs.fatalities);
      ign_sum += ignorance_score(fc.bin_probs, fc.bin_centers, obs.fatalities);
      ++n;
      break;
    }
  }
  out.n_scored = n;
  if (n > 0) {
    out.mean_crps = crps_sum / static_cast<double>(n);
    out.mean_ignorance = ign_sum / static_cast<double>(n);
  } else {
    out.detail = "no matching forecast/observation pairs";
  }
  return out;
}

}  // namespace cypha::forecast
