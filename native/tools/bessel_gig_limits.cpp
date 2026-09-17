// Guards the Bessel-ratio and GIG-moment numerics against the defects recorded in
// docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md (findings N1-N4, L5, L6, L7, L8, L9).
//
// Every reference value below comes from scipy.special.kv at double precision. The suite checks
// three separate things, because the defects fell into three groups:
//
//   1. ACCURACY.    K_2/K_1 was interpolated from a 16384-point uniform table. K_2/K_1 has a 2/x
//                   pole, and the table's first cell spans [1e-6, 7.3e-3] -- four decades -- so
//                   linear interpolation across it was 183,084% wrong (L9). It is now built from
//                   the exact recurrence K_2/K_1 = 2/x + K_0/K_1.
//
//   2. RANGE/SHAPE. K_0/K_1 is confined to [0,1) and increases; K_2/K_1 exceeds 1 and decreases.
//                   The score-match backend computed K_0/K_1 as a difference that cancelled and
//                   went NEGATIVE on 44.1% of its domain (L5).
//
//   3. LIMITS.      The GIG moments carried hand-written small-x and large-x branches, and every
//                   one returned the wrong limit (N1, N2, N3, L6, L7). They are now evaluated in
//                   closed form, so the limits fall out instead of being asserted by hand.
//
// Both normalisation backends are driven, since the defects differed between them.
//
// Exit 0 on success, 1 on failure. Registered as CTest `native_bessel_gig_limits`.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cypha/nig_gig_math.hpp"
#include "cypha/nig_gig_score_match.hpp"

namespace {

int g_failures = 0;

void check_rel(const char* what, double got, double want, double tol) {
  const double rel = std::fabs(got - want) / std::fabs(want);
  if (std::isfinite(got) && rel <= tol) {
    std::printf("  ok    %-46s %.12g  (rel %.2e)\n", what, got, rel);
    return;
  }
  std::printf("  FAIL  %-46s got %.12g, want %.12g (rel %.3e > %.1e)\n", what, got, want, rel, tol);
  ++g_failures;
}

void check_true(const char* what, bool cond, const char* detail) {
  if (cond) {
    std::printf("  ok    %-46s %s\n", what, detail);
    return;
  }
  std::printf("  FAIL  %-46s %s\n", what, detail);
  ++g_failures;
}

struct Ref {
  double x;
  double k0k1;
  double k2k1;
};

// scipy.special.kv reference, spanning both blend seams (0.35, 0.65) and the table edge (120).
const std::vector<Ref>& refs() {
  static const std::vector<Ref> r = {
      {1e-6, 0.000013931442, 2000000.000013931189},  {1e-5, 0.000116288570, 200000.000116288573},
      {1e-4, 0.000932627237, 20000.000932627237},    {0.01, 0.047224775746, 200.047224775746},
      {0.1, 0.246306804976, 20.246306804976},        {0.35, 0.481691178512, 6.195976892798},
      {0.5, 0.558075418477, 4.558075418477},         {0.65, 0.613553952004, 3.690477028927},
      {1.0, 0.699483935594, 2.699483935594},         {2.0, 0.814307758764, 1.814307758764},
      {5.0, 0.912596069766, 1.312596069766},         {20.0, 0.975893463067, 1.075893463067},
      {119.0, 0.995824580397, 1.012631303086},       {120.0, 0.995859160326, 1.012525826993},
      {200.0, 0.997509328430, 1.007509328430},       {10000.0, 0.999950003750, 1.000150003750},
  };
  return r;
}

// K_0/K_1 for the active backend, reached through the public GIG moment: E[V] at psi = chi is
// exactly K_0/K_1, since sqrt(chi/psi) == 1 there. This probe only reaches x >= kEps = 1e-8,
// because both parameters are clamped there; smaller x is covered by the direct entry points.
double k0k1_via_moment(double x) { return cypha::gig_e_v_lam_neg1(x, x); }
// Likewise E[1/V] at psi = chi is exactly K_2/K_1.
double k2k1_via_moment(double x) { return cypha::gig_e_inv_v_lam_neg1(x, x); }

void run_backend(const char* name, cypha::GigNormalisationMode mode) {
  cypha::set_gig_normalisation_mode_override(mode, true);
  std::printf("\n=== %s backend ===\n", name);

  // --- 1. accuracy against scipy, across every seam --------------------------------------------
  // 3e-5 is the worst relative error of the small-x series, which is where the bound binds.
  constexpr double kTol = 3e-5;
  double worst_k0 = 0.0, worst_k2 = 0.0;
  for (const Ref& r : refs()) {
    const double g0 = k0k1_via_moment(r.x);
    const double g2 = k2k1_via_moment(r.x);
    worst_k0 = std::max(worst_k0, std::fabs(g0 - r.k0k1) / r.k0k1);
    worst_k2 = std::max(worst_k2, std::fabs(g2 - r.k2k1) / r.k2k1);
  }
  std::printf("  worst relative error over %zu reference points: K0/K1 %.3e, K2/K1 %.3e\n", refs().size(),
              worst_k0, worst_k2);
  check_true("K0/K1 accurate to 3e-5 everywhere", worst_k0 <= kTol, "");
  check_true("K2/K1 accurate to 3e-5 everywhere", worst_k2 <= kTol, "");

  // The specific regression: before the recurrence rewrite, K2/K1 mid-way through the LUT's first
  // cell was 183,084% wrong. x = 0.00365 is the worst point.
  check_rel("K2/K1 at x=0.00365 (was 183,084% wrong)", k2k1_via_moment(0.00365), 547.9498632, 1e-4);

  // --- 2. range and shape ----------------------------------------------------------------------
  bool in_range = true, k0_increasing = true, k2_above_one = true, k2_decreasing = true;
  double prev0 = -1.0, prev2 = 1e308;
  double min_k0 = 1e308;
  for (int i = 0; i <= 4000; ++i) {
    const double x = std::pow(10.0, -8.0 + 12.0 * static_cast<double>(i) / 4000.0);  // 1e-8 .. 1e4
    const double v0 = k0k1_via_moment(x);
    const double v2 = k2k1_via_moment(x);
    min_k0 = std::min(min_k0, v0);
    if (!(v0 >= 0.0 && v0 < 1.0)) in_range = false;
    if (v0 < prev0 - 1e-12) k0_increasing = false;
    if (!(v2 > 1.0)) k2_above_one = false;
    if (v2 > prev2 + 1e-12) k2_decreasing = false;
    prev0 = v0;
    prev2 = v2;
  }
  // This is the L5 guard: the score-match backend returned a negative K0/K1 on 44.1% of the domain.
  check_true("K0/K1 stays in [0,1)", in_range, min_k0 < 0.0 ? "went NEGATIVE" : "never left the interval");
  check_true("K0/K1 is non-decreasing", k0_increasing, "no backward step across the blend seams");
  check_true("K2/K1 stays above 1", k2_above_one, "");
  check_true("K2/K1 is non-increasing", k2_decreasing, "");

  // --- 3. limits -------------------------------------------------------------------------------
  // N2/N3: as psi -> 0 the GIG degenerates to InvGamma(1, chi/2), whose first inverse moment is
  // 2/chi. The old code returned psi/chi, i.e. ~0.
  check_rel("E[1/V] -> 2/chi as psi -> 0  (chi=1)", cypha::gig_e_inv_v_lam_neg1(1.0, 0.0), 2.0, 1e-6);
  check_rel("E[1/V] -> 2/chi as psi -> 0  (chi=0.5)", cypha::gig_e_inv_v_lam_neg1(0.5, 0.0), 4.0, 1e-6);
  check_rel("E[1/V] -> 2/chi as psi -> 0  (chi=0.25)", cypha::gig_e_inv_v_lam_neg1(0.25, 1e-30), 8.0, 1e-6);

  // N1: as x -> infinity, E[1/V] -> sqrt(psi/chi), NOT psi/chi. At chi=1e4, psi=2e4 the old
  // large-x branch returned 2.0 against a true 1.41436 -- a 41% overstatement.
  check_rel("E[1/V] -> sqrt(psi/chi) at large x", cypha::gig_e_inv_v_lam_neg1(1e4, 2e4), 1.41436356502, 1e-5);
  check_true("  ... and is nowhere near the old psi/chi = 2",
             std::fabs(cypha::gig_e_inv_v_lam_neg1(1e4, 2e4) - 2.0) > 0.5, "");

  // L8: the LUT used to clamp to its last node past x = 120, leaving a 1.25% floor that never
  // decayed. K2/K1 must keep falling towards 1.
  check_rel("K2/K1 at x=200  (was pinned at 1.0125)", k2k1_via_moment(200.0), 1.007509328430, 1e-6);
  check_rel("K2/K1 at x=1e4  (was pinned at 1.0125)", k2k1_via_moment(1e4), 1.000150003750, 1e-6);
  check_true("  ... continuous across the table edge",
             std::fabs(k2k1_via_moment(120.1) - k2k1_via_moment(119.9)) < 1e-4, "no step at x=120");

  // L6/L7: E[V] -> 0 as chi -> 0, and diverges (does not vanish) as psi -> 0.
  check_true("E[V] -> 0 as chi -> 0", cypha::gig_e_v_lam_neg1(0.0, 1.0) < 1e-6, "");
  check_true("E[V] diverges, not vanishes, as psi -> 0",
             cypha::gig_e_v_lam_neg1(1.0, 0.0) > 1.0, "InvGamma(1, chi/2) has no mean");

  // General agreement with the closed form on ordinary inputs.
  check_rel("E[1/V](chi=1, psi=1)", cypha::gig_e_inv_v_lam_neg1(1.0, 1.0), 2.69948393559, 1e-5);
  check_rel("E[V]  (chi=1, psi=1)", cypha::gig_e_v_lam_neg1(1.0, 1.0), 0.699483935594, 1e-5);
  check_rel("E[1/V](chi=2, psi=0.5)", cypha::gig_e_inv_v_lam_neg1(2.0, 0.5), 1.3497419678, 1e-5);
  check_rel("E[V]  (chi=0.25, psi=4)", cypha::gig_e_v_lam_neg1(0.25, 4.0), 0.174870983898, 1e-5);

  cypha::set_gig_normalisation_mode_override(mode, false);
}

}  // namespace

// The CUDA device path in accel_cuda.cu re-implements the same approximation, and CPU/GPU parity
// tests compare the two paths against EACH OTHER -- so if a constant is retuned on the host and
// not mirrored on the device, parity still passes and both are simply wrong together. This checks
// the shared constants appear verbatim in both files. It needs no GPU and no CUDA toolchain.
void check_device_mirror(const char* src_dir) {
  const char* kShared[] = {
      "0.5772156649015329",     // Euler-Mascheroni
      "0.23513844126021524",    // x^5 L^3
      "0.37929893852576335",    // x^5 L^2
      "0.50510404194744363",    // x^5 L^1
      "0.20476904886486919",    // x^5 L^0
      "0.35",                   // blend window lower edge
      "0.65",                   // blend window upper edge
  };
  const std::string host_path = std::string(src_dir) + "/nig_gig_score_match.cpp";
  const std::string dev_path = std::string(src_dir) + "/accel_cuda.cu";
  std::ifstream hf(host_path), df(dev_path);
  if (!hf || !df) {
    std::printf("  SKIP  device-mirror check (sources not readable at %s)\n", src_dir);
    return;
  }
  std::stringstream hb, db;
  hb << hf.rdbuf();
  db << df.rdbuf();
  const std::string host = hb.str(), dev = db.str();
  for (const char* c : kShared) {
    const bool in_host = host.find(c) != std::string::npos;
    const bool in_dev = dev.find(c) != std::string::npos;
    if (in_host && in_dev) {
      std::printf("  ok    %-46s present in both host and device\n", c);
    } else {
      std::printf("  FAIL  %-46s host=%d device=%d -- the CUDA path has drifted\n", c, in_host, in_dev);
      ++g_failures;
    }
  }
  // The device must consume the K_0/K_1 column and rebuild K_2/K_1, exactly as the host does.
  check_true("device uploads the K0/K1 table, not K2/K1",
             dev.find("kBesselK0K1") != std::string::npos && dev.find("kBesselK2K1") == std::string::npos,
             "");
  check_true("device rebuilds K2/K1 from the recurrence", dev.find("2.0 / xv + d_k0k1") != std::string::npos, "");
}

int main(int argc, char** argv) {
  std::printf("Bessel-ratio and GIG-moment limits (audit N1-N4, L5, L6, L7, L8, L9)\n");

  run_backend("LUT", cypha::GigNormalisationMode::Lut);
  run_backend("ScoreMatch", cypha::GigNormalisationMode::ScoreMatch);

  // N4: the score-match large-x series had 6.75/x^2 where the standard asymptotic expansion for
  // K_nu gives 3/8 = 0.375 -- an 18x overstatement of the second-order term.
  std::printf("\n=== direct backend entry points ===\n");
  check_rel("gig_k2k1_score_match(1e4)", cypha::gig_k2k1_score_match(1e4), 1.000150003750, 1e-6);
  check_rel("gig_k2k1_lut(1e4)", cypha::gig_k2k1_lut(1e4), 1.000150003750, 1e-6);
  check_rel("gig_k0k1_score_match(1e-4)", cypha::gig_k0k1_score_match(1e-4), 0.000932627237, 1e-5);
  check_rel("gig_k2k1_lut(1e-6)", cypha::gig_k2k1_lut(1e-6), 2000000.000013931189, 1e-9);
  // Below the table's first node the old code clamped K_2/K_1 to that node and was 7415% wrong.
  // It is now analytic there. Both K_2/K_1 entry points still clamp the ARGUMENT at kEps = 1e-8 to
  // keep the 2/x pole finite, so they agree with each other and saturate rather than diverge --
  // that clamp is deliberate and no caller reaches below it (gig_e_* clamps chi and psi first).
  check_rel("gig_k2k1_lut(1e-9) saturates at the kEps clamp", cypha::gig_k2k1_lut(1e-9), 2.0e8, 1e-6);
  check_true("  ... and both backends clamp identically",
             cypha::gig_k2k1_lut(1e-9) == cypha::gig_k2k1_score_match(1e-9), "");
  // K_0/K_1 has no pole, so its entry point needs no clamp and stays accurate into the denormals.
  check_rel("gig_k0k1_score_match(1e-9)", cypha::gig_k0k1_score_match(1e-9), 2.0839e-08, 1e-4);
  // The two backends must agree: they approximate the same function.
  double worst_gap = 0.0;
  for (const Ref& r : refs()) {
    const double a = cypha::gig_k2k1_lut(r.x);
    const double b = cypha::gig_k2k1_score_match(r.x);
    worst_gap = std::max(worst_gap, std::fabs(a - b) / a);
  }
  check_true("LUT and ScoreMatch agree to 5e-6", worst_gap <= 5e-6, "both approximate the same function");
  std::printf("        worst cross-backend gap: %.3e\n", worst_gap);

  if (argc > 1) {
    std::printf("\n=== CUDA device mirror (accel_cuda.cu) ===\n");
    check_device_mirror(argv[1]);
  }

  std::printf("\n%s\n", g_failures == 0 ? "PASS" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
