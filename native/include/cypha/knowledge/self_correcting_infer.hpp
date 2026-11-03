#pragma once

#include "cypha/infer_cpu.hpp"
#include "cypha/intelligence/epistemic_threshold.hpp"

namespace cypha::intelligence {

/// Paper IV self-correcting loop: re-infer with deepened deliberation when r_eu exceeds θ_eu.
struct SelfCorrectingResult {
  cypha::InferAtHResult infer{};
  double r_eu_proxy{0.5};
  int correction_passes{0};
  bool corrected{false};
};

/// Encode ``x`` (length ``d``), run up to ``max_passes`` infer passes; deepen deliberation when
/// epistemic ratio (1 − confidence) exceeds learned threshold.
SelfCorrectingResult self_correcting_infer(cypha::CyphaInferModel& model, const double* x, int d,
                                           cypha::CyphaInferOptions opt, EpistemicThreshold& threshold,
                                           int max_passes = 3);

/// Same loop on pre-encoded latent ``h`` (length ``model.d_latent``).
SelfCorrectingResult self_correcting_infer_at_h(cypha::CyphaInferModel& model, const double* h,
                                                cypha::CyphaInferOptions opt,
                                                EpistemicThreshold& threshold, int max_passes = 3);

}  // namespace cypha::intelligence
