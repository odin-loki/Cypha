/// Tiny smoke for native ``PGMCell`` (H23): a few steps, print context_dim + finite norms.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cypha/cyphalm/pgm_cell.hpp"

namespace {

double l2_norm(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) {
    s += x * x;
  }
  return std::sqrt(s);
}

bool all_finite(const std::vector<double>& v) {
  for (double x : v) {
    if (!std::isfinite(x)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  using cypha::cyphalm::PGMCell;
  using cypha::cyphalm::PGMCellConfig;

  PGMCellConfig cfg;
  cfg.d_input = 16;
  cfg.hidden = 16;
  cfg.n_sub = 4;
  cfg.levels = 3;  // N = 64
  cfg.chunk_len = 4;
  cfg.topk = 4;
  cfg.beam = 2;
  cfg.rehash_t = 8;
  cfg.hops = 2;
  cfg.seed = 7;

  PGMCell cell(cfg);
  std::printf("pgm_cell_smoke: context_dim=%d n_slots=%d levels=%d n_sub=%d\n", cell.context_dim(),
              cell.n_slots(), cell.levels(), cell.n_sub());

  if (cell.context_dim() != 16) {
    std::printf("pgm_cell_smoke: FAIL context_dim\n");
    return 1;
  }
  if (cell.n_slots() != 64) {
    std::printf("pgm_cell_smoke: FAIL n_slots=%d (expected 64)\n", cell.n_slots());
    return 1;
  }

  std::vector<double> x(16, 0.0);
  for (int t = 0; t < 12; ++t) {
    for (int i = 0; i < 16; ++i) {
      x[static_cast<std::size_t>(i)] = std::sin(0.17 * (t + 1) * (i + 1));
    }
    const auto h = cell.step(x);
    if (static_cast<int>(h.size()) != cell.context_dim()) {
      std::printf("pgm_cell_smoke: FAIL step size at t=%d\n", t);
      return 1;
    }
    if (!all_finite(h)) {
      std::printf("pgm_cell_smoke: FAIL non-finite hidden at t=%d\n", t);
      return 1;
    }
    const double n = l2_norm(h);
    std::printf("  step %2d  ||h||=%.6f  edges=%zu  occupied=%zu\n", t, n, cell.edge_count(),
                cell.occupied_count());
    if (!std::isfinite(n)) {
      std::printf("pgm_cell_smoke: FAIL non-finite norm at t=%d\n", t);
      return 1;
    }
  }

  cell.reset();
  if (cell.edge_count() != 0 || cell.occupied_count() != 0) {
    std::printf("pgm_cell_smoke: FAIL reset did not clear graph\n");
    return 1;
  }

  std::puts("pgm_cell_smoke: PASS");
  return 0;
}
