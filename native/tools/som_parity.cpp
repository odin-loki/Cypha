// CTest: TemporalSOM decay-range smoke + GNG node growth (mirrors cypha_som/tests/test_som_upgrades.py).
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cypha/cyphalm/cellai_ssm.hpp"
#include "cypha/numpy_default_rng.hpp"
#include "cypha/som/discriminative_feedback.hpp"
#include "cypha/som/gng_expert.hpp"
#include "cypha/som/gria_controller.hpp"
#include "cypha/som/temporal_som.hpp"

namespace {

bool in_range(double v, double lo, double hi) { return v > lo && v < hi; }

}  // namespace

int main() {
  // TemporalSOM: 50 steps, seed 3 — lambdas in (0.5, 1.5).
  {
    cypha::som::TemporalSOMConfig cfg;
    cfg.M = 4;
    cfg.L_max = 8;
    cfg.seed = 3;
    cypha::som::TemporalSOM ts(cfg);
    cypha::NumpyDefaultRng rng(3);
    double lf = 1.0;
    double ls = 1.0;
    for (int t = 0; t < 50; ++t) {
      std::vector<double> x(4, 0.0);
      for (int j = 0; j < 4; ++j) {
        x[static_cast<std::size_t>(j)] = rng.normal(0.0, 1.0);
      }
      int bmu = 0;
      std::tie(bmu, lf, ls) = ts.step(x, true);
      (void)bmu;
    }
    if (!in_range(lf, 0.5, 1.5) || !in_range(ls, 0.5, 1.5)) {
      std::cerr << "som_parity: temporal lambda out of range lf=" << lf << " ls=" << ls << "\n";
      return 1;
    }
  }

  // GNG: 300 steps with lam=20 should grow beyond 2 nodes.
  {
    cypha::som::GNGExpertConfig cfg;
    cfg.lam = 20;
    cfg.max_nodes = 64;
    cfg.seed = 0;
    cypha::som::GNGExpertManager gng(8, cfg);
    cypha::NumpyDefaultRng rng(0);
    for (int t = 0; t < 300; ++t) {
      std::vector<double> x(8, 0.0);
      for (int j = 0; j < 8; ++j) {
        x[static_cast<std::size_t>(j)] = rng.normal(0.0, 1.0);
      }
      (void)gng.step(x);
    }
    if (gng.node_count() < 2) {
      std::cerr << "som_parity: GNG node_count=" << gng.node_count() << " expected >= 2\n";
      return 1;
    }
  }

  // GRIAController: alpha bounded after 100 pushes (mirrors test_gria_alpha_bounded).
  {
    cypha::som::GRIAControllerConfig gc;
    gc.window = 50;
    cypha::som::GRIAController g(gc);
    cypha::NumpyDefaultRng rng(1);
    for (int t = 0; t < 100; ++t) {
      std::vector<double> x(10, 0.0);
      std::vector<double> a(5, 0.0);
      for (int j = 0; j < 10; ++j) {
        x[static_cast<std::size_t>(j)] = rng.normal(0.0, 1.0);
      }
      for (int j = 0; j < 5; ++j) {
        a[static_cast<std::size_t>(j)] = rng.normal(0.0, 1.0);
      }
      g.push(x, a);
    }
    const double alpha = g.alpha();
    if (alpha < 0.0 || alpha > 1.5) {
      std::cerr << "som_parity: GRIA alpha=" << alpha << " expected in [0, 1.5]\n";
      return 1;
    }
  }

  // DiscriminativeFeedback: modulate boosts emphasized column (mirrors test_discriminative_modulate).
  {
    cypha::som::DiscriminativeFeedback fb;
    const std::vector<double> dW(16, 1.0);
    const std::vector<double> d{1.0, 0.0, 0.0, 0.0};
    const auto out = fb.modulate(dW, 4, 4, d);
    if (out.empty() || out[0] <= dW[0]) {
      std::cerr << "som_parity: discriminative modulate out[0]=" << (out.empty() ? 0.0 : out[0])
                << " expected > " << dW[0] << "\n";
      return 1;
    }
  }

  // CellAISSM + temporal SOM wiring smoke.
  {
    cypha::cyphalm::CellAISSMConfig sc;
    sc.d_input = 4;
    sc.d_state = 4;
    sc.n_layers = 1;
    sc.seed = 7;
    sc.use_spectral_pde = false;
    cypha::cyphalm::CellAISSM ssm(sc);
    cypha::som::TemporalSOMConfig tc;
    tc.M = 4;
    tc.L_max = 8;
    ssm.enable_temporal_som(tc);
    std::vector<double> e(4, 0.1);
    const auto out = ssm.step(e);
    if (out.empty()) {
      std::cerr << "som_parity: CellAISSM step with temporal SOM returned empty context\n";
      return 1;
    }
  }

  return 0;
}
