#pragma once

namespace cypha {

/// GH/NIG GIG normalisation backend for ``gig_e_*`` helpers (Phase 7).
enum class GigNormalisationMode {
  /// Interpolated ``K_n/K_{n-1}`` Bessel-ratio LUT (default; unchanged numerics).
  Lut = 0,
  /// Hyvärinen score-matching rational fit — no partition function / Bessel LUT on hot path.
  ScoreMatch = 1,
};

/// Active mode: default ``Lut``; set ``CYPHA_GIG_SCORE_MATCH=1`` for ``ScoreMatch``.
GigNormalisationMode gig_normalisation_mode();

/// Override mode (tests); pass ``nullopt`` to revert to env/default.
void set_gig_normalisation_mode_override(GigNormalisationMode mode, bool active);

/// ``K_2(x)/K_1(x)`` via score-matching rational approximation (``x·ratio`` Padé fit).
double gig_k2k1_score_match(double x);

/// ``K_0(x)/K_1(x)`` derived from ``K_2/K_1`` and the exact Bessel recurrence ``K_0 = K_2 - 2K_1/x``.
double gig_k0k1_score_match(double x);

/// Held-out NIG gate predictive log-likelihood proxy (Tier-1 acceptance metric for Phase 7).
/// Uses ``k2k1_fn`` for the Bessel-ratio increment between ``chi`` and ``chi + mp/r``.
double nig_gate_predictive_loglik(double mp, double r_base, double chi, double psi,
                                  double (*k2k1_fn)(double));

}  // namespace cypha
