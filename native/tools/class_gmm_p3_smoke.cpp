// Phase 3: opt-in per-class GMM vs linear LLR on S3 XOR (no kernels).
// Measures XOR lift; format migration must succeed. ≥75% is not required for
// CTest PASS — see docs/reports/OPTIMALITY_PHASE3_*.md for acceptance status.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
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

void make_xor(int n, int d, cypha::NumpyDefaultRng& rng, std::vector<std::vector<double>>& x,
              std::vector<int>& y) {
  x.assign(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(d)));
  y.assign(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      x[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = rng.normal(0.0, 1.0);
    }
    y[static_cast<std::size_t>(i)] =
        ((x[static_cast<std::size_t>(i)][0] > 0.0) ^ (x[static_cast<std::size_t>(i)][1] > 0.0)) ? 1 : 0;
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

double run_xor_linear(int seed, bool use_gmm, int gmm_m) {
  constexpr int n = 4000;
  constexpr int d = 20;
  constexpr int passes = 8;
  cypha::NumpyDefaultRng data_rng(42);
  std::vector<std::vector<double>> x;
  std::vector<int> y;
  make_xor(n, d, data_rng, x, y);
  Split sp = scale_split(x, y, seed, 0.25);

  cypha::FreshModelParams fp;
  fp.input_dim = d;
  fp.field_dim = 128;
  fp.temperature = 1.15;
  fp.world_lr = 0.008;
  fp.delta_lr = 0.05;
  cypha::CNode root = cypha::create_fresh_model_root(fp);
  cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root, nullptr, fp.field_dim);
  cypha::init_encoder_projection_w(d, static_cast<std::uint64_t>(seed), infer.enc_w);
  cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim);
  mem.use_class_gmm = use_gmm;
  mem.class_gmm_m = gmm_m;
  cypha::ReplayBuffer replay(10000);
  cypha::TrainStepParams tsp;
  tsp.replay_ratio = 0.0;
  cypha::TrainStepExtras extras;
  extras.use_class_gmm = use_gmm;
  extras.class_gmm_m = gmm_m;
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
      cypha::dif_train_step_vector(infer, mem, replay, sp.x_tr[i].data(), d, sp.y_tr[i], 0.008, 0.05,
                                   0.008, 0.05, 15.0, tsp, rng, enc_updates, nullptr, &extras);
    }
  }
  cypha::sync_infer_model_from_memory(infer, mem);
  return eval_acc(infer, sp);
}

bool format_roundtrip_ok() {
  cypha::FreshModelParams fp;
  fp.input_dim = 4;
  fp.field_dim = 8;
  cypha::CNode root = cypha::create_fresh_model_root(fp);
  cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim);
  mem.use_class_gmm = true;
  mem.class_gmm_m = 2;
  std::unordered_map<std::string, double> ctx;
  std::vector<double> h(4, 0.1);
  mem.memory_train(h.data(), "0", nullptr, ctx, 1.0, 12.0, 0.008, 0.05, nullptr);
  cypha::CNode saved = cypha::CyphaDifMemoryState::merge_state_into_root_for_save(root, mem);
  cypha::CyphaDifMemoryState reload =
      cypha::CyphaDifMemoryState::from_cypha_root(saved, nullptr, fp.field_dim);
  if (!reload.use_class_gmm || reload.class_n_comp.empty() || reload.class_n_comp[0] < 1) {
    return false;
  }
  // Legacy M=1 class entry still loads when GMM enabled.
  cypha::CyphaDifMemoryState legacy = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim);
  legacy.use_class_gmm = false;
  legacy.memory_train(h.data(), "1", nullptr, ctx, 1.0, 12.0, 0.008, 0.05, nullptr);
  cypha::CNode legacy_saved = cypha::CyphaDifMemoryState::merge_state_into_root_for_save(root, legacy);
  cypha::CyphaDifMemoryState legacy_reload =
      cypha::CyphaDifMemoryState::from_cypha_root(legacy_saved, nullptr, fp.field_dim);
  return !legacy_reload.use_class_gmm && !legacy_reload.D.empty();
}

}  // namespace

int main() {
  constexpr int kSeeds = 3;
  constexpr int kGmmM = 2;
  double off_sum = 0.0;
  double on_sum = 0.0;
  for (int s = 0; s < kSeeds; ++s) {
    const double off = run_xor_linear(s, false, kGmmM);
    const double on = run_xor_linear(s, true, kGmmM);
    off_sum += off;
    on_sum += on;
    std::cerr << "  seed=" << s << " gmm_off=" << off << " gmm_on=" << on << "\n";
  }
  const double off_mean = off_sum / static_cast<double>(kSeeds);
  const double on_mean = on_sum / static_cast<double>(kSeeds);
  const double lift = on_mean - off_mean;
  const bool xor_target = on_mean >= 0.75;
  const bool xor_lift = lift >= 0.05;

  std::cout << "class_gmm_p3_smoke:\n"
            << "  xor_linear_mean_off=" << off_mean << "\n"
            << "  xor_linear_mean_on=" << on_mean << "\n"
            << "  lift=" << lift << "\n"
            << "  xor_target_75=" << (xor_target ? "true" : "false") << "\n"
            << "  xor_lift_5pp=" << (xor_lift ? "true" : "false") << "\n";

  if (!format_roundtrip_ok()) {
    std::cout << "class_gmm_p3_smoke: FAIL format roundtrip\n";
    return 1;
  }
  // Linear wall sanity: OFF stays near chance on XOR (no accidental kernel).
  if (!(off_mean > 0.40 && off_mean < 0.65)) {
    std::cout << "class_gmm_p3_smoke: FAIL unexpected OFF accuracy\n";
    return 1;
  }

  std::cout << "class_gmm_p3_smoke: PASS\n";
  return 0;
}
