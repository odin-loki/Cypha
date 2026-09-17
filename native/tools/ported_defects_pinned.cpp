// Guards two defects that were inherited from the Python reference and have now been FIXED.
//
// Both were faithful ports of Python behaviour. Python was decommissioned at P7
// (CHANGELOG.md:55) — native C++ is the sole runtime, and nothing can execute the reference any
// more — so bit-compatibility with it stopped being a reason to keep a defect. See
// docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md, findings R3 and R4.
//
// This file previously pinned the DEFECTIVE behaviour, to stop anyone changing it by accident
// while the parity question was open. That question is settled; it now asserts the CORRECT
// behaviour instead.
//
// R3 - compute_ece_bins (native/src/infer_cpu.cpp)
//      was: bins are [lo, hi) throughout, so confidence exactly 1.0 fell in no bin and was
//           dropped from the numerator while n still counted it; NaN was dropped the same way,
//           so an all-NaN evaluation scored a perfect 0.0 and WON the temperature search.
//      now: the top bin is closed at 1.0; non-finite confidences are excluded from both
//           numerator and denominator; an evaluation with no finite sample returns +infinity.
//
// R4 - adapt_temperature_ece (native/src/infer_cpu.cpp)
//      was: best_ece seeded at +infinity, so the incumbent temperature was never scored and the
//           first grid point beat it unconditionally - the routine could install a worse
//           temperature and report success.
//      now: the incumbent is scored first, so the search can only improve on it.
//
// Exit 0 on success, 1 on failure. Registered as CTest `native_ported_defects_pinned`.

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "cypha/infer_cpu.hpp"

namespace {

// Calls the REAL compute_ece_bins from native/src/infer_cpu.cpp - not a copy of it.
double ece_bins(const std::vector<double>& confs, const std::vector<double>& correct, int n_bins) {
  return cypha::compute_ece_bins(confs.data(), correct.data(), static_cast<int>(confs.size()), n_bins);
}

bool expect(const char* what, double got, double want, double tol) {
  if (std::isinf(want) ? (std::isinf(got) && got > 0) : (std::fabs(got - want) <= tol)) {
    std::printf("  ok    %-54s %.9f\n", what, got);
    return true;
  }
  std::printf("  FAIL  %-54s got %.9f, want %.9f\n", what, got, want);
  return false;
}

}  // namespace

int main() {
  bool ok = true;
  std::printf("R2/R3/R4 - previously-ported defects, now fixed\n");

  // --- R3 -----------------------------------------------------------------------------------
  // Ten perfectly confident samples, every one wrong. True ECE is 1.0. This reported 0.0 before
  // the fix, because confidence == 1.0 fell outside the top bin.
  {
    const std::vector<double> confs(10, 1.0);
    const std::vector<double> correct(10, 0.0);
    ok &= expect("conf==1.0, all wrong -> ECE is 1.0 (was 0.0)", ece_bins(confs, correct, 10), 1.0, 1e-12);
  }
  // The cliff is gone: one part in a million lower must give essentially the same answer.
  {
    const std::vector<double> confs(10, 0.999999);
    const std::vector<double> correct(10, 0.0);
    ok &= expect("conf==0.999999, all wrong -> ECE", ece_bins(confs, correct, 10), 0.999999, 1e-9);
  }
  // Perfectly confident and perfectly RIGHT is genuinely well calibrated: ECE 0.
  {
    const std::vector<double> confs(10, 1.0);
    const std::vector<double> correct(10, 1.0);
    ok &= expect("conf==1.0, all correct -> ECE is 0.0", ece_bins(confs, correct, 10), 0.0, 1e-12);
  }
  // An all-NaN evaluation must never win a minimisation.
  {
    const std::vector<double> confs(10, std::numeric_limits<double>::quiet_NaN());
    const std::vector<double> correct(10, 0.0);
    ok &= expect("all-NaN evaluation -> +inf (was a perfect 0.0)", ece_bins(confs, correct, 10),
                 std::numeric_limits<double>::infinity(), 0.0);
  }
  // Partial NaN must not dilute the score: the finite rows are scored on their own count.
  {
    std::vector<double> confs(10, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> correct(10, 0.0);
    for (int i = 0; i < 5; ++i) {
      confs[static_cast<std::size_t>(i)] = 1.0;  // confident and wrong
    }
    ok &= expect("5 NaN + 5 confident-wrong -> ECE is 1.0, not 0.5", ece_bins(confs, correct, 10), 1.0, 1e-12);
  }

  // --- R4, against the real shipped function -------------------------------------------------
  // Build a calibration set, find the grid's own best temperature, then hand that back as the
  // incumbent with a single-point grid pinned somewhere worse. Before the fix the incumbent was
  // never scored, so the lone grid point won unconditionally and the temperature got worse.
  {
    const int d = 4, K = 3, n_cal = 400;
    cypha::CyphaInferModel m;
    m.d_latent = d;
    m.field_dim = 0;
    for (int k = 0; k < K; ++k) {
      m.labels.push_back("c" + std::to_string(k));
    }
    m.n_obs.assign(static_cast<std::size_t>(K), 100.0);
    m.class_pi.assign(static_cast<std::size_t>(K), 1.0 / static_cast<double>(K));
    m.class_n_comp.assign(static_cast<std::size_t>(K), 1);
    m.D.assign(static_cast<std::size_t>(K * d), 0.0);
    for (int k = 0; k < K; ++k) {
      m.D[static_cast<std::size_t>(k * d + (k % d))] = 6.0;
    }
    m.mu_world.assign(static_cast<std::size_t>(d), 0.0);
    m.inv_v.assign(static_cast<std::size_t>(d), 1.0);
    m.v_mean = 1.0;
    m.temperature = 1.0;
    m.has_mahal_ema = true;
    m.mahal_ema = 1.0;
    m.mahal_std_ema = 0.5;

    std::mt19937 rng(20260917);
    std::normal_distribution<double> noise(0.0, 3.0);
    std::vector<double> h(static_cast<std::size_t>(n_cal * d), 0.0);
    std::vector<int> y(static_cast<std::size_t>(n_cal), 0);
    for (int i = 0; i < n_cal; ++i) {
      const int k = i % K;
      y[static_cast<std::size_t>(i)] = k;
      for (int j = 0; j < d; ++j) {
        h[static_cast<std::size_t>(i * d + j)] = (j == (k % d) ? 6.0 : 0.0) + noise(rng);
      }
    }

    // Let the full grid pick its own optimum, and adopt it as the incumbent.
    cypha::CyphaInferModel a = m;
    const double t_opt = cypha::adapt_temperature_ece(a, h.data(), n_cal, y.data(), 20, 0.3, 8.0, 10);

    // Now re-run with that optimum as the incumbent and a single grid point elsewhere.
    cypha::CyphaInferModel b = m;
    b.temperature = t_opt;
    const double t_after = cypha::adapt_temperature_ece(b, h.data(), n_cal, y.data(), 1, 0.3, 8.0, 10);

    std::printf("  ....  grid optimum T = %.6f; single-point grid pinned at T_min = 0.3\n", t_opt);
    if (std::fabs(t_after - t_opt) < 1e-12) {
      std::printf("  ok    %-54s %.6f\n", "real adapt_temperature_ece keeps the better incumbent", t_after);
    } else {
      std::printf("  FAIL  %-54s %.6f (incumbent was %.6f)\n",
                  "real adapt_temperature_ece keeps the better incumbent", t_after, t_opt);
      ok = false;
    }
    if (std::fabs(b.temperature - t_after) > 1e-12) {
      std::printf("  FAIL  model temperature (%.6f) disagrees with the return value\n", b.temperature);
      ok = false;
    }
  }

  // --- R2, the anomaly score's units ---------------------------------------------------------
  // r_eff is in h^2 units, formed against r_base = 1/mean(inv_v). Dividing it by mahal_ema (a
  // dimensionless per-dim Mahalanobis EMA) mixed units, and on any model whose latent variance
  // is below mahal_ema the score was pinned at 0 for EVERY input - the OOD flag never fired.
  // It is now max(0, r_eff/r_base - 1): dimensionless, 0 in distribution, growing with anomaly.
  {
    // Scale invariance: multiplying both r_eff and its baseline by the same factor must not
    // change the score. This fails under the old formula, which has a fixed denominator.
    const double a1 = cypha::gh_infer_anomaly_score(4.0, 1.0);
    const double a2 = cypha::gh_infer_anomaly_score(0.4, 0.1);
    const double a3 = cypha::gh_infer_anomaly_score(400.0, 100.0);
    ok &= expect("anomaly is scale-invariant (r_eff=4, base=1)", a1, 3.0, 1e-12);
    ok &= expect("  ... same ratio at 1/10 scale", a2, 3.0, 1e-12);
    ok &= expect("  ... same ratio at 100x scale", a3, 3.0, 1e-12);

    // No inflation at all is exactly in-distribution.
    ok &= expect("no inflation (r_eff == r_base) -> 0", cypha::gh_infer_anomaly_score(0.0875, 0.0875), 0.0, 1e-12);

    // A small-variance model must not be permanently pinned at 0. Under the old formula, a model
    // with r_base = 0.0875 against mahal_ema = 1.0 scored 0 for every r_eff below 1.0.
    const double small_var = cypha::gh_infer_anomaly_score(0.35, 0.0875);
    ok &= expect("small-variance model still scores (was pinned at 0)", small_var, 3.0, 1e-12);
  }

  std::printf("%s\n", ok ? "PASS" : "FAILED");
  return ok ? 0 : 1;
}
