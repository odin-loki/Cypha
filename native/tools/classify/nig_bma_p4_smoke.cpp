// Phase 4: opt-in NIG BMA over class deltas — ECE / credible-interval coverage smoke.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cypha/bench/bench_metrics.hpp"
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

void make_blobs(int n_per_class, int d, int n_classes, cypha::NumpyDefaultRng& rng,
                std::vector<std::vector<double>>& x, std::vector<int>& y) {
  x.clear();
  y.clear();
  for (int c = 0; c < n_classes; ++c) {
    for (int i = 0; i < n_per_class; ++i) {
      std::vector<double> row(static_cast<std::size_t>(d));
      for (int j = 0; j < d; ++j) {
        const double center = (j == 0) ? static_cast<double>(c) * 2.5 : 0.0;
        row[static_cast<std::size_t>(j)] = center + rng.normal(0.0, 0.55);
      }
      x.push_back(row);
      y.push_back(c);
    }
  }
}

Split scale_split(const std::vector<std::vector<double>>& x, const std::vector<int>& y, int seed,
                  double test_frac) {
  const int n = static_cast<int>(x.size());
  const int d = static_cast<int>(x[0].size());
  cypha::NumpyDefaultRng perm_rng(seed);
  const std::vector<int> idx = perm_rng.permutation(n);
  const int n_test = std::max(1, static_cast<int>(n * test_frac));
  std::vector<double> mu(static_cast<std::size_t>(d), 0.0);
  std::vector<double> stdv(static_cast<std::size_t>(d), 1.0);
  for (int j = 0; j < d; ++j) {
    double s = 0.0;
    for (int t = n_test; t < n; ++t) {
      s += x[static_cast<std::size_t>(idx[static_cast<std::size_t>(t)])][static_cast<std::size_t>(j)];
    }
    mu[static_cast<std::size_t>(j)] = s / static_cast<double>(n - n_test);
    double v = 0.0;
    for (int t = n_test; t < n; ++t) {
      const double diff =
          x[static_cast<std::size_t>(idx[static_cast<std::size_t>(t)])][static_cast<std::size_t>(j)] -
          mu[static_cast<std::size_t>(j)];
      v += diff * diff;
    }
    stdv[static_cast<std::size_t>(j)] =
        std::sqrt(v / static_cast<double>(n - n_test)) + 1e-8;
  }
  auto norm_row = [&](int ix) {
    std::vector<double> row(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
      row[static_cast<std::size_t>(j)] =
          (x[static_cast<std::size_t>(ix)][static_cast<std::size_t>(j)] - mu[static_cast<std::size_t>(j)]) /
          stdv[static_cast<std::size_t>(j)];
    }
    return row;
  };
  Split out;
  for (int t = 0; t < n_test; ++t) {
    out.x_te.push_back(norm_row(idx[static_cast<std::size_t>(t)]));
    out.y_te.push_back(std::to_string(y[static_cast<std::size_t>(idx[static_cast<std::size_t>(t)])]));
  }
  for (int t = n_test; t < n; ++t) {
    out.x_tr.push_back(norm_row(idx[static_cast<std::size_t>(t)]));
    out.y_tr.push_back(std::to_string(y[static_cast<std::size_t>(idx[static_cast<std::size_t>(t)])]));
  }
  return out;
}

struct EvalMetrics {
  double ece{0.0};
  double coverage{0.0};
  double accuracy{0.0};
};

EvalMetrics eval_split(cypha::CyphaInferModel& infer, const Split& sp, bool use_bma) {
  EvalMetrics m;
  cypha::CyphaInferOptions opt;
  opt.use_field = true;
  opt.use_kernel_llr = false;
  infer.use_nig_bma = use_bma;

  std::vector<double> confidences;
  std::vector<double> correct;
  std::vector<std::string> preds;
  confidences.reserve(sp.x_te.size());
  correct.reserve(sp.x_te.size());
  preds.reserve(sp.x_te.size());

  for (std::size_t i = 0; i < sp.x_te.size(); ++i) {
    std::vector<double> h;
    cypha::batch_encode(infer, sp.x_te[i].data(), 1, h);
    const cypha::InferAtHResult res = cypha::infer_at_h(infer, h.data(), opt);
    confidences.push_back(res.confidence);
    preds.push_back(res.label);
    correct.push_back(res.label == sp.y_te[i] ? 1.0 : 0.0);
  }

  m.ece = cypha::bench::expected_calibration_error(confidences, correct, 10);
  m.accuracy = cypha::bench::accuracy(sp.y_te, preds);

  int corr_n = 0;
  int cov_n = 0;
  for (std::size_t i = 0; i < confidences.size(); ++i) {
    if (correct[i] > 0.5) {
      corr_n += 1;
      if (confidences[i] >= 0.5) {
        cov_n += 1;
      }
    }
  }
  m.coverage = corr_n > 0 ? static_cast<double>(cov_n) / static_cast<double>(corr_n) : 1.0;
  return m;
}

}  // namespace

int main() {
  constexpr int kSeed = 7;
  cypha::NumpyDefaultRng data_rng(4242);
  std::vector<std::vector<double>> x;
  std::vector<int> y;
  make_blobs(200, 12, 4, data_rng, x, y);
  Split sp = scale_split(x, y, kSeed, 0.25);

  cypha::FreshModelParams fp;
  fp.input_dim = 12;
  fp.field_dim = 64;
  fp.temperature = 1.0;
  fp.world_lr = 0.01;
  fp.delta_lr = 0.06;
  cypha::CNode root = cypha::create_fresh_model_root(fp);
  cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root, nullptr, fp.field_dim);
  cypha::init_encoder_projection_w(12, static_cast<std::uint64_t>(kSeed), infer.enc_w);
  cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim);
  cypha::ReplayBuffer replay(10000);
  cypha::TrainStepParams tsp;
  tsp.replay_ratio = 0.0;
  int total_steps = 0;
  cypha::TrainStepExtras extras;
  extras.total_steps = &total_steps;
  std::mt19937 rng(static_cast<std::uint32_t>(kSeed));
  int enc_updates = 0;
  std::vector<std::size_t> order(sp.x_tr.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  for (int p = 0; p < 10; ++p) {
    std::shuffle(order.begin(), order.end(), rng);
    for (const std::size_t i : order) {
      cypha::dif_train_step_vector(infer, mem, replay, sp.x_tr[i].data(), 12, sp.y_tr[i], fp.world_lr,
                                   fp.delta_lr, fp.world_lr, fp.delta_lr, 12.0, tsp, rng, enc_updates,
                                   nullptr, &extras);
    }
  }
  cypha::sync_infer_model_from_memory(infer, mem);

  const EvalMetrics off = eval_split(infer, sp, false);
  const EvalMetrics on = eval_split(infer, sp, true);

  std::cout << "nig_bma_p4_smoke:\n"
            << "  ece_map=" << off.ece << "\n"
            << "  ece_bma=" << on.ece << "\n"
            << "  acc_map=" << off.accuracy << "\n"
            << "  acc_bma=" << on.accuracy << "\n"
            << "  coverage_bma=" << on.coverage << "\n";

  constexpr double kEceTol = 0.05;
  if (!(std::isfinite(off.ece) && std::isfinite(on.ece))) {
    std::cout << "nig_bma_p4_smoke: FAIL non-finite ECE\n";
    return 1;
  }
  if (on.ece > off.ece + kEceTol) {
    std::cout << "nig_bma_p4_smoke: FAIL ECE worsened (on=" << on.ece << " off=" << off.ece << ")\n";
    return 1;
  }
  // Correct-row fraction with conf ≥ 0.5 (proxy for CI lower bound usefulness).
  if (on.coverage < 0.50) {
    std::cout << "nig_bma_p4_smoke: FAIL coverage below tolerance\n";
    return 1;
  }
  if (off.accuracy < 0.70) {
    std::cout << "nig_bma_p4_smoke: FAIL baseline accuracy too low for calibration test\n";
    return 1;
  }

  std::printf("nig_bma_p4_smoke: ece_delta=%.4f PASS\n", on.ece - off.ece);
  return 0;
}
