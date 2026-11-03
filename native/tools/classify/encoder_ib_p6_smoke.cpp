// Phase 6: opt-in variational IB encoder vs Fisher–Rao contrastive on 2-class blobs.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cypha/create_model.hpp"
#include "cypha/encoder_contrastive.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/mt19937_rng.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/sync_infer.hpp"
#include "cypha/train_step_vector.hpp"

namespace {

struct Split {
  std::vector<std::vector<double>> x_tr;
  std::vector<std::vector<double>> x_te;
  std::vector<std::string> y_tr;
  std::vector<std::string> y_te;
};

Split make_blobs(int n, int d, int seed, double test_frac) {
  cypha::NumpyDefaultRng rng(seed);
  std::vector<std::vector<double>> x(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(d)));
  std::vector<int> y(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    const int cls = i % 2;
    y[static_cast<std::size_t>(i)] = cls;
    const double shift = cls == 0 ? -1.5 : 1.5;
    for (int j = 0; j < d; ++j) {
      x[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = shift + rng.normal(0.0, 0.35);
    }
  }
  const std::vector<int> idx = rng.permutation(n);
  const int n_test = std::max(1, static_cast<int>(n * test_frac));
  Split out;
  for (int t = 0; t < n_test; ++t) {
    const int ix = idx[static_cast<std::size_t>(t)];
    out.x_te.push_back(x[static_cast<std::size_t>(ix)]);
    out.y_te.push_back(std::to_string(y[static_cast<std::size_t>(ix)]));
  }
  for (int t = n_test; t < n; ++t) {
    const int ix = idx[static_cast<std::size_t>(t)];
    out.x_tr.push_back(x[static_cast<std::size_t>(ix)]);
    out.y_tr.push_back(std::to_string(y[static_cast<std::size_t>(ix)]));
  }
  return out;
}

double eval_acc(cypha::CyphaInferModel& infer, const Split& sp) {
  int correct = 0;
  cypha::CyphaInferOptions opt;
  opt.use_field = true;
  opt.use_kernel_llr = false;
  for (std::size_t i = 0; i < sp.x_te.size(); ++i) {
    std::vector<double> h;
    cypha::batch_encode(infer, sp.x_te[i].data(), 1, h);
    const cypha::InferAtHResult res = cypha::infer_at_h(infer, h.data(), opt);
    if (res.label == sp.y_te[i]) {
      correct += 1;
    }
  }
  return static_cast<double>(correct) / static_cast<double>(sp.x_te.size());
}

double collect_mi_proxy(cypha::CyphaInferModel& infer, const Split& sp) {
  std::vector<std::vector<double>> h_samples;
  h_samples.reserve(sp.x_te.size());
  for (const auto& row : sp.x_te) {
    std::vector<double> h;
    cypha::batch_encode(infer, row.data(), 1, h);
    h_samples.push_back(std::move(h));
  }
  return cypha::latent_class_mi_proxy(h_samples, sp.y_te);
}

struct RunResult {
  double acc{0.0};
  double mi_proxy{0.0};
  double fro_norm{0.0};
};

RunResult run_blobs(int seed, bool use_ib, double beta) {
  constexpr int n = 2400;
  constexpr int d = 12;
  constexpr int passes = 6;
  Split sp = make_blobs(n, d, seed, 0.25);

  cypha::FreshModelParams fp;
  fp.input_dim = d;
  fp.field_dim = 64;
  fp.temperature = 1.0;
  fp.world_lr = 0.008;
  fp.delta_lr = 0.05;
  cypha::CNode root = cypha::create_fresh_model_root(fp);
  cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root, nullptr, fp.field_dim);
  cypha::init_encoder_projection_w(d, static_cast<std::uint64_t>(seed), infer.enc_w);
  cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim);
  cypha::ReplayBuffer replay(10000);
  cypha::TrainStepParams tsp;
  tsp.replay_ratio = 0.0;
  cypha::TrainStepExtras extras;
  extras.use_variational_ib_encoder = use_ib;
  extras.ib_beta = beta;
  int total_steps = 0;
  extras.total_steps = &total_steps;

  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  int enc_updates = 0;
  std::vector<std::size_t> order(sp.x_tr.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  for (int p = 0; p < passes; ++p) {
    std::shuffle(order.begin(), order.end(), rng);
    for (const std::size_t i : order) {
      cypha::dif_train_step_vector(infer, mem, replay, sp.x_tr[i].data(), d, sp.y_tr[i], 0.008, 0.05, 0.008, 0.05,
                                   12.0, tsp, rng, enc_updates, nullptr, &extras);
    }
  }
  cypha::sync_infer_model_from_memory(infer, mem);

  RunResult out;
  out.acc = eval_acc(infer, sp);
  out.mi_proxy = collect_mi_proxy(infer, sp);
  double fro = 0.0;
  for (double w : infer.enc_w) {
    fro += w * w;
  }
  out.fro_norm = std::sqrt(fro);
  return out;
}

bool direct_ib_update_smoke() {
  constexpr int d = 4;
  std::vector<double> w(d * d, 0.01);
  const double f[d] = {0.5, -0.2, 0.3, 0.1};
  const double h[d] = {0.4, -0.1, 0.2, 0.0};
  const double mu_k[d] = {0.1, 0.0, 0.0, 0.0};
  const double v_k[d] = {1.0, 1.0, 1.0, 1.0};
  const double mu_j[d] = {-0.2, 0.3, 0.0, 0.0};
  const double v_j[d] = {1.0, 1.0, 1.0, 1.0};
  int cnt = 0;
  cypha::variational_ib_update_encoder_w(w, d, f, h, mu_k, v_k, mu_j, v_j, 1.0, 0.002, 1.0, cnt);
  for (double x : w) {
    if (!std::isfinite(x)) {
      return false;
    }
  }
  return cnt == 1;
}

}  // namespace

int main() {
  if (!direct_ib_update_smoke()) {
    std::cout << "encoder_ib_p6_smoke: FAIL direct update\n";
    return 1;
  }

  constexpr int kSeeds = 3;
  constexpr double kBeta = 1.0;
  double fr_acc = 0.0;
  double ib_acc = 0.0;
  double fr_mi = 0.0;
  double ib_mi = 0.0;
  double max_fro = 0.0;

  for (int s = 0; s < kSeeds; ++s) {
    const RunResult fr = run_blobs(s + 11, false, kBeta);
    const RunResult ib = run_blobs(s + 11, true, kBeta);
    fr_acc += fr.acc;
    ib_acc += ib.acc;
    fr_mi += fr.mi_proxy;
    ib_mi += ib.mi_proxy;
    max_fro = std::max(max_fro, std::max(fr.fro_norm, ib.fro_norm));
    std::cerr << "  seed=" << s << " fr_acc=" << fr.acc << " ib_acc=" << ib.acc << " fr_mi=" << fr.mi_proxy
              << " ib_mi=" << ib.mi_proxy << " fro=" << ib.fro_norm << "\n";
  }

  fr_acc /= static_cast<double>(kSeeds);
  ib_acc /= static_cast<double>(kSeeds);
  fr_mi /= static_cast<double>(kSeeds);
  ib_mi /= static_cast<double>(kSeeds);
  const double acc_delta = ib_acc - fr_acc;
  const double mi_delta = ib_mi - fr_mi;

  std::cout << "encoder_ib_p6_smoke:\n"
            << "  fisher_rao_mean_acc=" << fr_acc << "\n"
            << "  variational_ib_mean_acc=" << ib_acc << "\n"
            << "  acc_delta_ib_minus_fr=" << acc_delta << "\n"
            << "  fisher_rao_mi_proxy=" << fr_mi << "\n"
            << "  variational_ib_mi_proxy=" << ib_mi << "\n"
            << "  mi_proxy_delta=" << mi_delta << "\n"
            << "  max_enc_fro_norm=" << max_fro << "\n"
            << "  ib_beta=" << kBeta << "\n";

  if (max_fro > 8.01) {
    std::cout << "encoder_ib_p6_smoke: FAIL Frobenius cap\n";
    return 1;
  }
  if (!(fr_acc > 0.85 && ib_acc > 0.80)) {
    std::cout << "encoder_ib_p6_smoke: FAIL blob accuracy sanity\n";
    return 1;
  }
  if (ib_mi <= 0.0 || fr_mi <= 0.0) {
    std::cout << "encoder_ib_p6_smoke: FAIL MI proxy zero\n";
    return 1;
  }

  std::cout << "encoder_ib_p6_smoke: PASS\n";
  return 0;
}
