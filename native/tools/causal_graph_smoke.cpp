/// Smoke test: CausalGraphMonitor ingests a short deterministic sequence of correlated
/// synthetic profiler observations and reports a finite, well-defined causal_fidelity().
///
/// CausalGraphMonitor's edge *topology* is fixed (see causal_graph.hpp), not something the
/// caller constructs; the alpha->calibration and tau->r_eu edges are what get estimated
/// online from observation history via OnlineCorrelation. This feeds a fixed-seed sequence
/// where calibration is a noisy linear function of alpha and r_eu is a noisy linear
/// function of tau, then checks: causal_fidelity() is finite and in the documented [0, 1)
/// contract; the two OnlineCorrelation edges report |correlation| <= 1 with n matching the
/// number of observations; and every recorded edge weight stays clamped to [0, 1].
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

#include "cypha/intelligence/causal_graph.hpp"
#include "cypha/intelligence/profile_observation.hpp"

int main() {
    using cypha::intelligence::CausalGraphMonitor;
    using cypha::intelligence::ProfileObservation;

    constexpr int kSteps = 12;
    std::mt19937 rng(42);  // fixed seed for reproducibility
    std::uniform_real_distribution<double> noise(-0.02, 0.02);

    CausalGraphMonitor graph;
    for (int i = 0; i < kSteps; ++i) {
        ProfileObservation obs;
        obs.alpha = 0.1 + 0.05 * static_cast<double>(i);
        obs.calibration = 0.8 * obs.alpha + noise(rng);  // noisy linear fn of alpha
        obs.tau = 0.2 + 0.03 * static_cast<double>(i);
        obs.r_eu = 0.9 * obs.tau + noise(rng);            // noisy linear fn of tau
        obs.d_eff = 0.5;
        obs.sigma_branch = 0.5;
        obs.lipschitz = 0.5;
        graph.observe_profile(obs);
    }

    int rc = 0;

    const double alpha_calib_corr = graph.alpha_calibration_correlation();
    const double tau_r_eu_corr = graph.tau_r_eu_correlation();
    const int alpha_calib_n = graph.alpha_calibration_n();
    const int tau_r_eu_n = graph.tau_r_eu_n();

    if (!std::isfinite(alpha_calib_corr) || std::abs(alpha_calib_corr) > 1.0) {
        std::fprintf(stderr,
                     "causal_graph_smoke: alpha_calibration_correlation out of range: %.6f\n",
                     alpha_calib_corr);
        rc = 1;
    }
    if (!std::isfinite(tau_r_eu_corr) || std::abs(tau_r_eu_corr) > 1.0) {
        std::fprintf(stderr, "causal_graph_smoke: tau_r_eu_correlation out of range: %.6f\n",
                     tau_r_eu_corr);
        rc = 1;
    }
    if (alpha_calib_n != kSteps || tau_r_eu_n != kSteps) {
        std::fprintf(stderr,
                     "causal_graph_smoke: unexpected sample counts alpha_calib_n=%d "
                     "tau_r_eu_n=%d (expected %d)\n",
                     alpha_calib_n, tau_r_eu_n, kSteps);
        rc = 1;
    }
    // Strongly correlated synthetic inputs (small noise on a linear relationship) should
    // yield a strong |correlation| estimate, not a degenerate near-zero one.
    if (std::abs(alpha_calib_corr) < 0.9 || std::abs(tau_r_eu_corr) < 0.9) {
        std::fprintf(stderr,
                     "causal_graph_smoke: expected strong correlation from linear synthetic "
                     "data, got alpha_calib=%.6f tau_r_eu=%.6f\n",
                     alpha_calib_corr, tau_r_eu_corr);
        rc = 1;
    }

    const double fidelity = graph.causal_fidelity();
    if (!std::isfinite(fidelity) || fidelity < 0.0 || fidelity >= 1.0) {
        std::fprintf(stderr,
                     "causal_graph_smoke: causal_fidelity() out of documented [0,1) range: %.6f\n",
                     fidelity);
        rc = 1;
    }

    if (graph.edges().empty()) {
        std::fprintf(stderr, "causal_graph_smoke: no edges recorded\n");
        rc = 1;
    }
    for (const auto& edge : graph.edges()) {
        if (!std::isfinite(edge.weight) || edge.weight < 0.0 || edge.weight > 1.0) {
            std::fprintf(stderr,
                         "causal_graph_smoke: edge %s->%s weight out of [0,1]: %.6f\n",
                         edge.from.c_str(), edge.to.c_str(), edge.weight);
            rc = 1;
            break;
        }
    }

    if (rc == 0) {
        std::printf(
            "causal_graph_smoke: alpha_calib_corr=%.4f tau_r_eu_corr=%.4f fidelity=%.4f "
            "edges=%zu PASS\n",
            alpha_calib_corr, tau_r_eu_corr, fidelity, graph.edges().size());
        std::puts("causal_graph_smoke: PASS");
    }
    return rc;
}
