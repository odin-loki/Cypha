// XOR linear vs Nyström kernel LLR benchmark (native CyphaDIF train path).
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cypha/create_model.hpp"
#include "cypha/encoder_contrastive.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/kernel_memory.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/numpy_default_rng.hpp"
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

double eval_acc(cypha::CyphaInferModel& infer, const Split& sp, cypha::KernelMemory* km, bool use_kernel,
                double kernel_blend) {
  int correct = 0;
  cypha::CyphaInferOptions opt;
  opt.use_field = true;
  opt.kernel_mem = km;
  opt.use_kernel_llr = use_kernel;
  opt.kernel_blend = kernel_blend;
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

double run_seed(int seed, bool use_kernel, double kernel_blend, int passes, int kernel_m) {
  constexpr int n = 4000;
  constexpr int d = 20;
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
  cypha::ReplayBuffer replay(10000);
  cypha::TrainStepParams tsp;
  tsp.replay_ratio = 0.0;
  cypha::TrainStepExtras extras;
  int total_steps = 0;
  extras.total_steps = &total_steps;
  cypha::KernelMemory km(d, kernel_m, static_cast<std::uint64_t>(seed));
  extras.kernel_mem = use_kernel ? &km : nullptr;
  extras.use_kernel_llr = use_kernel;
  extras.kernel_blend = kernel_blend;

  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  int enc_updates = 0;
  constexpr double kWorldLr = 0.008;
  constexpr double kDeltaLr = 0.05;
  constexpr double kOodSigma = 15.0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int p = 0; p < passes; ++p) {
    for (std::size_t i = 0; i < sp.x_tr.size(); ++i) {
      cypha::dif_train_step_vector(infer, mem, replay, sp.x_tr[i].data(), d, sp.y_tr[i], kWorldLr, kDeltaLr, kWorldLr,
                                   kDeltaLr, kOodSigma, tsp, rng, enc_updates, nullptr, &extras);
    }
  }
  cypha::sync_infer_model_from_memory(infer, mem);
  const double sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const double acc = eval_acc(infer, sp, use_kernel ? &km : nullptr, use_kernel, kernel_blend);
  std::cerr << "  seed=" << seed << " kernel=" << (use_kernel ? "1" : "0") << " acc=" << acc
            << " train_sec=" << sec << " n_basis=" << km.n_basis() << "\n";
  return acc;
}

}  // namespace

int main(int argc, char** argv) {
  int seeds = 3;
  int passes = 8;
  double kernel_blend = 1.0;
  int kernel_m = 256;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--seeds" && i + 1 < argc) {
      seeds = std::atoi(argv[++i]);
    } else if (arg == "--passes" && i + 1 < argc) {
      passes = std::atoi(argv[++i]);
    } else if (arg == "--kernel-blend" && i + 1 < argc) {
      kernel_blend = std::atof(argv[++i]);
    } else if (arg == "--kernel-m" && i + 1 < argc) {
      kernel_m = std::atoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "usage: xor_kernel_bench [--seeds N] [--passes N] [--kernel-blend B] [--kernel-m M]\n";
      return 0;
    }
  }

  double lin_sum = 0.0;
  double ker_sum = 0.0;
  std::cout << "{\n  \"dataset\": \"S3_nonlinear_xor_native\",\n  \"seeds\": " << seeds
            << ",\n  \"passes\": " << passes << ",\n  \"kernel_blend\": " << kernel_blend
            << ",\n  \"kernel_m\": " << kernel_m << ",\n  \"linear\": [\n";
  for (int s = 0; s < seeds; ++s) {
    if (s) {
      std::cout << ",\n";
    }
    const double acc = run_seed(s, false, kernel_blend, passes, kernel_m);
    std::cout << "    " << acc;
    lin_sum += acc;
  }
  std::cout << "\n  ],\n  \"kernel\": [\n";
  for (int s = 0; s < seeds; ++s) {
    if (s) {
      std::cout << ",\n";
    }
    const double acc = run_seed(s, true, kernel_blend, passes, kernel_m);
    std::cout << "    " << acc;
    ker_sum += acc;
  }
  const double lin_mean = lin_sum / static_cast<double>(seeds);
  const double ker_mean = ker_sum / static_cast<double>(seeds);
  std::cout << "\n  ],\n  \"linear_mean_acc\": " << lin_mean << ",\n  \"kernel_mean_acc\": " << ker_mean
            << ",\n  \"delta_pp\": " << 100.0 * (ker_mean - lin_mean) << "\n}\n";
  return 0;
}
