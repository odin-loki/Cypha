#include "cypha/intelligence/self_correcting_infer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cypha::intelligence {

namespace {

double r_eu_from_confidence(double confidence) {
  const double c = std::clamp(confidence, 0.0, 1.0);
  return std::clamp(1.0 - c, 0.0, 1.0);
}

SelfCorrectingResult self_correcting_infer_at_h_impl(cypha::CyphaInferModel& model, const double* h,
                                                     cypha::CyphaInferOptions opt,
                                                     EpistemicThreshold& threshold, int max_passes) {
  SelfCorrectingResult out;
  max_passes = std::max(1, max_passes);

  cypha::CyphaInferOptions pass_opt = opt;
  const cypha::InferAtHResult first = cypha::infer_at_h(model, h, pass_opt);
  const double first_conf = first.confidence;

  cypha::InferAtHResult best = first;
  double best_conf = first.confidence;
  double r_eu = r_eu_from_confidence(first.confidence);
  out.correction_passes = 1;

  for (int pass = 1; pass < max_passes; ++pass) {
    if (!threshold.should_correct(r_eu)) {
      break;
    }
    out.corrected = true;
    pass_opt.deliberation_lo = std::max(0.05, pass_opt.deliberation_lo * 0.82);
    pass_opt.deliberation_hi = std::min(1.0, pass_opt.deliberation_hi + 0.05);
    const cypha::InferAtHResult inf = cypha::infer_at_h(model, h, pass_opt);
    r_eu = r_eu_from_confidence(inf.confidence);
    if (inf.confidence > best_conf) {
      best = inf;
      best_conf = inf.confidence;
    }
    out.correction_passes = pass + 1;
  }

  threshold.update(r_eu, best_conf > first_conf + 1e-6);

  out.infer = best;
  out.r_eu_proxy = r_eu;
  return out;
}

}  // namespace

SelfCorrectingResult self_correcting_infer(cypha::CyphaInferModel& model, const double* x, int /*d*/,
                                           cypha::CyphaInferOptions opt, EpistemicThreshold& threshold,
                                           int max_passes) {
  std::vector<double> h;
  cypha::batch_encode(model, x, 1, h);
  return self_correcting_infer_at_h_impl(model, h.data(), opt, threshold, max_passes);
}

SelfCorrectingResult self_correcting_infer_at_h(cypha::CyphaInferModel& model, const double* h,
                                                cypha::CyphaInferOptions opt,
                                                EpistemicThreshold& threshold, int max_passes) {
  return self_correcting_infer_at_h_impl(model, h, opt, threshold, max_passes);
}

}  // namespace cypha::intelligence
