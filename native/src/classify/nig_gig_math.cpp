#include "cypha/nig_gig_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "cypha/bessel_table.hpp"
#include "cypha/nig_gig_score_match.hpp"

namespace cypha {

namespace {

constexpr double kEps = 1e-8;

double np_interp_k0k1(double x) {
  using cypha::detail::kBesselK0K1;
  using cypha::detail::kBesselN;
  using cypha::detail::kBesselX0;
  using cypha::detail::kBesselX1;
  if (x <= kBesselX0) {
    return kBesselK0K1[0];
  }
  if (x >= kBesselX1) {
    return kBesselK0K1[kBesselN - 1];
  }
  const double step = (kBesselX1 - kBesselX0) / static_cast<double>(kBesselN - 1);
  double pos = (x - kBesselX0) / step;
  std::size_t i = static_cast<std::size_t>(pos);
  if (i >= kBesselN - 1) {
    i = kBesselN - 2;
  }
  double t = pos - static_cast<double>(i);
  return kBesselK0K1[i] * (1.0 - t) + kBesselK0K1[i + 1] * t;
}

// K_0/K_1 for the LUT backend, with no mode dispatch: closed-form series below the blend window,
// the shipped table through the middle, closed-form asymptotic past the table's last node.
//
// The table's first cell spans [1e-6, 7.3e-3] — four decades — and K_0/K_1 varies by three orders
// of magnitude across it, so interpolating there was 46% wrong; below x = 1e-6 the old code
// clamped to the first node and was 7415% wrong. See the audit report, finding L9.
//
// Comparing worst-in-cell error (linear interpolation is exact at the nodes and worst mid-cell,
// so a per-point comparison flatters it), the series beats the table up to x ~ 0.44 and the table
// wins above. The blend window [0.35, 0.65] straddles that crossover.
double k0k1_lut(double x) {
  if (x <= 0.0) {
    return 0.0;
  }
  if (x >= cypha::detail::kBesselX1) {
    return cypha::detail::k0k1_large_x(x);
  }
  if (x <= cypha::detail::kK0K1BlendLo) {
    return cypha::detail::k0k1_small_x(x);
  }
  return cypha::detail::k0k1_blend(x, np_interp_k0k1(x));
}

}  // namespace

double gig_k2k1_lut(double x) {
  // Exact Bessel recurrence K_2 = K_0 + (2/x)·K_1, hence K_2/K_1 = 2/x + K_0/K_1.
  //
  // This used to interpolate a separate K_2/K_1 table. K_2/K_1 carries a 2/x pole, and linear
  // interpolation of a pole across the first (four-decade-wide) table cell was 183,084% wrong at
  // its worst — the single largest numerical error in the kernel. K_0/K_1 is bounded in [0,1) and
  // smooth, so interpolating that and adding the pole back analytically is exact in the pole and
  // accurate to ~1.9e-6 over the whole domain — that figure is for K_2/K_1; the K_0/K_1 it is
  // built from is good to ~2.5e-5, which is what gig_e_v_lam_neg1 inherits. Audit finding L9.
  const double xs = std::max(x, kEps);
  return 2.0 / xs + k0k1_lut(xs);
}

namespace {

double active_k2k1(double x) {
  if (gig_normalisation_mode() == GigNormalisationMode::ScoreMatch) {
    return gig_k2k1_score_match(x);
  }
  return gig_k2k1_lut(x);
}

double active_k0k1(double x) {
  if (gig_normalisation_mode() == GigNormalisationMode::ScoreMatch) {
    return gig_k0k1_score_match(x);
  }
  return k0k1_lut(x);
}

}  // namespace

// Moments of V ~ GIG(lambda = -1, chi, psi), x = sqrt(chi*psi):
//   E[1/V] = sqrt(psi/chi) * K_2(x)/K_1(x)
//   E[V]   = sqrt(chi/psi) * K_0(x)/K_1(x)
//
// Both used to carry hand-written special cases for small and large x, and every one of them
// returned the wrong limit (audit findings N1, N2, N3, L7):
//   - small x: they returned psi/chi and chi/psi; the true limits are 2/chi and 0.
//   - large x: E[1/V] returned psi/chi instead of sqrt(psi/chi), a 41% error at chi=1e4, psi=2e4.
// Now that gig_k2k1_lut reproduces the 2/x pole analytically rather than interpolating it, no
// special case is needed: clamping the parameters away from zero and evaluating the closed form
// gives every limit correctly and continuously.
//
//   psi -> 0   E[1/V] -> 2/chi  (the GIG degenerates to InvGamma(1, chi/2)). E[V] is genuinely
//              INFINITE there -- InvGamma(1, .) has no first moment -- so what comes back is the
//              series evaluated at the clamped psi = kEps. Its RATE is right (E[V] ~ (chi/2)*ln(1/psi),
//              so it grows the way the true divergence does), but its MAGNITUDE is set by kEps,
//              which exists to guard a division. Treat it as a bounded stand-in for infinity, not
//              as an estimate. Shipped code never reaches it: gh_psi is initialised to 1.0 at every
//              construction site and infer_cpu.cpp rejects gh_psi <= 0 outright.
//   chi -> 0   E[1/V] -> 2/chi (diverges, clamped), E[V] -> 0.
//   x -> inf   E[1/V] -> sqrt(psi/chi), E[V] -> sqrt(chi/psi).
double gig_e_inv_v_lam_neg1(double chi0, double psi) {
  const double chi_g = std::max(chi0, kEps);
  const double psi_g = std::max(psi, kEps);
  const double x = std::sqrt(chi_g * psi_g);
  return std::sqrt(psi_g / chi_g) * active_k2k1(x);
}

double gig_e_v_lam_neg1(double chi0, double psi) {
  const double chi_g = std::max(chi0, kEps);
  const double psi_g = std::max(psi, kEps);
  const double x = std::sqrt(chi_g * psi_g);
  return std::sqrt(chi_g / psi_g) * active_k0k1(x);
}

double nig_adapt_chi_impl(double chi, double psi, double innovation_sq, double R, double alpha) {
  double chi_post = chi + innovation_sq / std::max(R, kEps);
  double ev = gig_e_v_lam_neg1(chi_post, psi);
  return std::clamp(alpha * ev, 1e-4, 1e3);
}

double nig_r_eff_scalar(double mp, double r, double chi, double psi) {
  mp = std::max(mp, 0.0);
  double chi_post = chi + mp / std::max(r, kEps);
  double e_inv = gig_e_inv_v_lam_neg1(chi_post, psi);
  return r / std::max(e_inv, kEps);
}

double nig_delta_posterior_scale(double n_obs, double v_mean) {
  return v_mean / (std::max(n_obs, 0.0) + 1.0);
}

double nig_delta_posterior_var_j(double n_obs, double v_mean, double inv_v_j) {
  const double tau = nig_delta_posterior_scale(n_obs, v_mean);
  return tau / std::max(inv_v_j, kEps);
}

double nig_delta_bma_llr_correction(int d, double n_obs, double v_mean, const double* inv_v,
                                    const double* r) {
  if (d <= 0 || inv_v == nullptr || r == nullptr) {
    return 0.0;
  }
  const double tau = nig_delta_posterior_scale(n_obs, v_mean);
  double r_sq_inv = 0.0;
  for (int j = 0; j < d; ++j) {
    const double inv_j = inv_v[static_cast<std::size_t>(j)];
    const double rj = r[static_cast<std::size_t>(j)];
    r_sq_inv += (rj * rj) / std::max(inv_j, kEps);
  }
  return 0.5 * tau * (static_cast<double>(d) + r_sq_inv);
}

double nig_delta_bma_epistemic_var(double n_obs, double v_mean, const double* inv_v, const double* r,
                                   int d) {
  if (d <= 0 || inv_v == nullptr || r == nullptr) {
    return 0.0;
  }
  const double tau = nig_delta_posterior_scale(n_obs, v_mean);
  double r_sq_inv = 0.0;
  for (int j = 0; j < d; ++j) {
    const double inv_j = inv_v[static_cast<std::size_t>(j)];
    const double rj = r[static_cast<std::size_t>(j)];
    r_sq_inv += (rj * rj) / std::max(inv_j, kEps);
  }
  return tau * r_sq_inv;
}

double nig_delta_credible_lower(double prob, double epistemic_std, double z, double temperature) {
  const double T = std::max(temperature, kEps);
  const double se = epistemic_std / T;
  return std::clamp(prob - z * se, 0.0, 1.0);
}

}  // namespace cypha
