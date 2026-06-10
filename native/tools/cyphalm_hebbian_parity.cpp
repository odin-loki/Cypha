// CTest: one biochemical encoder update + sparse SSM + graph diffuse vs Python golden.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cypha/cyphalm/hebbian_encoder.hpp"
#include "cypha/cyphalm/hebbian_graph.hpp"
#include "cypha/cyphalm/hebbian_ssm.hpp"

namespace {

constexpr int kD = 4;
constexpr double kTol = 1e-14;

bool near_eq(double a, double b) { return std::abs(a - b) <= kTol; }

bool vec_near(const std::vector<double>& got, const std::vector<double>& want) {
  if (got.size() != want.size()) {
    return false;
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (!near_eq(got[i], want[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  // Golden from Cypha.py EncoderProjection.hebbian_update (d=4 fixture).
  const double f[kD] = {0.1, 0.2, -0.3, 0.4};
  const double h[kD] = {0.8, -0.2, 0.4, 0.3};
  const double mu_k[kD] = {0.1, 0.0, 0.0, 0.0};
  const double v_k[kD] = {0.5, 0.5, 0.5, 0.5};
  const double mu_j[kD] = {0.5, 0.2, 0.1, 0.0};
  const double v_j[kD] = {0.25, 0.25, 0.25, 0.25};
  const double w_golden[kD * kD] = {
      0.50003158005123594, 6.3160102471969276e-05,  -9.4740153707953894e-05, 0.00012632020494393855,
      -7.8950128089961595e-06, 0.49998420997438203,  2.3685038426988474e-05,  -3.1580051235984638e-05,
      1.5790025617992319e-05,  3.1580051235984638e-05, 0.49995262992314604,   6.3160102471969276e-05,
      1.1842519213494237e-05,  2.3685038426988474e-05, -3.5527557640482714e-05, 0.50004737007685396};

  cypha::cyphalm::HebbianEncoder enc;
  enc.d = kD;
  enc.w.assign(kD * kD, 0.0);
  for (int i = 0; i < kD; ++i) {
    enc.w[static_cast<std::size_t>(i * kD + i)] = 0.5;
  }
  enc.update_with_stats(f, h, mu_k, v_k, mu_j, v_j, 0.002, 1.0);
  if (!vec_near(enc.w, std::vector<double>(w_golden, w_golden + kD * kD))) {
    std::cerr << "cyphalm_hebbian_parity: encoder W mismatch\n";
    return 1;
  }

  // Golden from cellai_ssm.sparse_hebbian_update.
  const double pre[kD] = {0.5, -0.1, 0.3, 0.2};
  const double post[kD] = {0.2, 0.4, -0.2, 0.1};
  const double w_ssm_golden[kD * kD] = {1e-5, 0.0, 0.0, 0.0, 2e-5, 0.0, 0.0, 0.0,
                                        -1e-5, 0.0, 0.0, 0.0, 5e-6, 0.0, 0.0, 0.0};
  cypha::cyphalm::HebbianSSMState ssm;
  ssm.resize(kD, 1);
  cypha::cyphalm::sparse_hebbian_update(ssm, pre, post, 1e-4, 0);
  if (!vec_near(ssm.w[0], std::vector<double>(w_ssm_golden, w_ssm_golden + kD * kD))) {
    std::cerr << "cyphalm_hebbian_parity: sparse SSM W mismatch\n";
    return 1;
  }

  // Golden from DynamicHebbianGraph.diffuse (ring init, gamma=0.1).
  const double ctx[kD] = {1.0, 0.5, -0.3, 0.2};
  const double diffuse_golden[kD] = {1.034999999999965, 0.534999999999965, -0.265000000000035,
                                     0.23499999999996501};
  cypha::cyphalm::HebbianGraphConfig gc;
  gc.n = kD;
  gc.gamma = 0.1;
  cypha::cyphalm::HebbianGraph graph(gc);
  std::vector<double> out(static_cast<std::size_t>(kD), 0.0);
  graph.diffuse(ctx, out.data());
  if (!vec_near(out, std::vector<double>(diffuse_golden, diffuse_golden + kD))) {
    std::cerr << "cyphalm_hebbian_parity: graph diffuse mismatch\n";
    return 1;
  }

  return 0;
}
