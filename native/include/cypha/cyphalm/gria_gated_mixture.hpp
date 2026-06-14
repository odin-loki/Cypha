#pragma once

#include <cmath>

namespace cypha::cyphalm {

/// H09: shift hybrid blend logit from GRIA α trajectory (ordered vs chaotic routing).
/// High mean α → more GRIA (ordered); low α → more LSTM (chaotic). ``delta_alpha`` is step change.
inline double gria_gated_blend_logit(double base_logit, double mean_alpha, double delta_alpha) {
    constexpr double kAlphaCenter = 0.5;
    constexpr double kMeanScale = 1.2;
    constexpr double kDeltaScale = 0.6;
    const double ordered_bias = (mean_alpha - kAlphaCenter) * kMeanScale;
    const double trajectory_bias = delta_alpha * kDeltaScale;
    return base_logit + ordered_bias + trajectory_bias;
}

}  // namespace cypha::cyphalm
