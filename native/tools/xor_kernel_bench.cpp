// XOR linear vs Nyström kernel LLR benchmark (native CyphaDIF train path).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cypha/create_model.hpp"
#include "cypha/encoder_contrastive.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/kernel_memory.hpp"
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

struct BenchConfig {
  int seeds = 3;
  int passes = 8;
  double kernel_blend = 1.0;
  int kernel_m = 512;
  double gamma_scale = 2.0;
  double kernel_lr_scale = 2.0;
  bool shuffle_train = true;
  /// ``latent``, ``raw_x``, or ``xor_pair`` (default) kernel features.
  std::string kernel_feature_mode = "xor_pair";
  /// ``nystrom`` (default, online landmark sketch) or ``rff`` (fixed Random Fourier Features basis).
  std::string kernel_basis = "nystrom";
  /// RFF projection dimension (``M``-equivalent) when ``kernel_basis == "rff"``.
  int rff_dim = 512;
  /// Multiplier applied to the auto (median-heuristic) gamma when calibrating the RFF basis.
  double rff_gamma_scale = 1.0;
  /// Fixed RBF bandwidth for RFF, bypassing the auto-gamma median heuristic (comparison baseline).
  /// Sentinel <= 0 means "use auto-gamma" (the default).
  double rff_fixed_gamma = -1.0;
  /// ``uniform`` (default) or ``leverage`` Nyström landmark reservoir (Phase 5 opt-in).
  std::string nystrom_landmark_sampling = "uniform";
  /// ``iid`` (default) or ``sorf`` RFF weight initialization (Phase 5 opt-in).
  std::string rff_projection = "iid";
};

struct SeedResult {
  double linear_acc = 0.0;
  double kernel_acc = 0.0;
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
                double kernel_blend, const BenchConfig& cfg, int raw_d) {
  int correct = 0;
  cypha::CyphaInferOptions opt;
  opt.use_field = true;
  opt.kernel_mem = km;
  opt.use_kernel_llr = use_kernel;
  opt.kernel_blend = kernel_blend;
  std::vector<double> kfeat_buf;
  for (std::size_t i = 0; i < sp.x_te.size(); ++i) {
    std::vector<double> h;
    cypha::batch_encode(infer, sp.x_te[i].data(), 1, h);
    if (use_kernel && cfg.kernel_feature_mode == "raw_x") {
      opt.kernel_x = sp.x_te[i].data();
    } else if (use_kernel && cfg.kernel_feature_mode == "xor_pair") {
      kfeat_buf = cypha::build_xor_pair_features(sp.x_te[i].data(), raw_d);
      opt.kernel_x = kfeat_buf.data();
    } else {
      opt.kernel_x = nullptr;
    }
    const cypha::InferAtHResult res = cypha::infer_at_h(infer, h.data(), opt);
    if (res.label == sp.y_te[i]) {
      correct += 1;
    }
  }
  return static_cast<double>(correct) / static_cast<double>(sp.x_te.size());
}

int kernel_feature_dim(const BenchConfig& cfg, int input_dim, int latent_dim) {
  if (cfg.kernel_feature_mode == "raw_x") {
    return input_dim;
  }
  if (cfg.kernel_feature_mode == "xor_pair") {
    return 5;
  }
  return latent_dim;
}

const double* kernel_features_for_x(const BenchConfig& cfg, const double* x, int d,
                                    std::vector<double>& buf) {
  if (cfg.kernel_feature_mode == "raw_x") {
    return x;
  }
  if (cfg.kernel_feature_mode == "xor_pair") {
    buf = cypha::build_xor_pair_features(x, d);
    return buf.data();
  }
  return nullptr;
}

double run_seed_mode(int seed, bool use_kernel, const BenchConfig& cfg) {
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
  const int kdim = use_kernel ? kernel_feature_dim(cfg, d, infer.d_latent) : infer.d_latent;

  // Calibration batch for RFF auto-gamma and/or leverage Nyström init (median pairwise-distance
  // heuristic over the same feature representation used at train/eval time — latent h, raw x, or
  // xor_pair — computed before any weight updates so the projection is frozen and unbiased by
  // training order).
  std::vector<double> calib_rowmajor;
  const bool need_calib =
      use_kernel &&
      (cfg.kernel_basis == "rff" ||
       (cfg.kernel_basis == "nystrom" && cfg.nystrom_landmark_sampling == "leverage"));
  if (need_calib) {
    std::vector<double> feat_buf;
    const std::size_t n_calib = std::min<std::size_t>(sp.x_tr.size(), 256);
    calib_rowmajor.reserve(n_calib * static_cast<std::size_t>(kdim));
    for (std::size_t i = 0; i < n_calib; ++i) {
      std::vector<double> h;
      cypha::batch_encode(infer, sp.x_tr[i].data(), 1, h);
      const double* feat = (cfg.kernel_feature_mode == "latent")
                               ? h.data()
                               : kernel_features_for_x(cfg, sp.x_tr[i].data(), d, feat_buf);
      for (int j = 0; j < kdim; ++j) {
        calib_rowmajor.push_back(feat[j]);
      }
    }
  }

  cypha::KernelMemory km = [&]() {
    if (use_kernel && cfg.kernel_basis == "rff") {
      const double gamma = (cfg.rff_fixed_gamma > 0.0)
                                ? cfg.rff_fixed_gamma
                                : cypha::KernelMemory::auto_gamma_median_heuristic(
                                      calib_rowmajor.data(),
                                      static_cast<int>(calib_rowmajor.size() / static_cast<std::size_t>(kdim)),
                                      kdim, cfg.rff_gamma_scale, 256, static_cast<std::uint64_t>(seed));
      if (cfg.rff_projection == "sorf") {
        return cypha::KernelMemory::make_orthogonal_rff(kdim, cfg.rff_dim, gamma,
                                                         static_cast<std::uint64_t>(seed));
      }
      return cypha::KernelMemory::make_rff(kdim, cfg.rff_dim, gamma, static_cast<std::uint64_t>(seed));
    }
    cypha::KernelMemory nystrom_km(kdim, cfg.kernel_m, static_cast<std::uint64_t>(seed));
    nystrom_km.set_gamma_scale(cfg.gamma_scale);
    if (cfg.nystrom_landmark_sampling == "leverage") {
      nystrom_km.set_landmark_sampling(cypha::KernelMemory::LandmarkSamplingKind::LeverageScore);
      if (!calib_rowmajor.empty()) {
        nystrom_km.init_leverage_landmarks_from_samples(
            calib_rowmajor.data(),
            static_cast<int>(calib_rowmajor.size() / static_cast<std::size_t>(kdim)), kdim);
      }
    }
    return nystrom_km;
  }();
  extras.kernel_mem = use_kernel ? &km : nullptr;
  extras.use_kernel_llr = use_kernel;
  extras.kernel_blend = cfg.kernel_blend;
  extras.kernel_lr_scale = cfg.kernel_lr_scale;
  std::vector<double> kernel_feat_buf;

  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  int enc_updates = 0;
  constexpr double kWorldLr = 0.008;
  constexpr double kDeltaLr = 0.05;
  constexpr double kOodSigma = 15.0;
  const auto t0 = std::chrono::steady_clock::now();

  std::vector<std::size_t> order(sp.x_tr.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }

  for (int p = 0; p < cfg.passes; ++p) {
    if (cfg.shuffle_train) {
      std::shuffle(order.begin(), order.end(), rng);
    }
    for (const std::size_t i : order) {
      if (use_kernel && cfg.kernel_feature_mode != "latent") {
        extras.kernel_features =
            kernel_features_for_x(cfg, sp.x_tr[i].data(), d, kernel_feat_buf);
      } else {
        extras.kernel_features = nullptr;
      }
      cypha::dif_train_step_vector(infer, mem, replay, sp.x_tr[i].data(), d, sp.y_tr[i], kWorldLr, kDeltaLr,
                                   kWorldLr, kDeltaLr, kOodSigma, tsp, rng, enc_updates, nullptr, &extras);
    }
  }
  cypha::sync_infer_model_from_memory(infer, mem);
  const double sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const double acc = eval_acc(infer, sp, use_kernel ? &km : nullptr, use_kernel, cfg.kernel_blend, cfg, d);
  std::cerr << "  seed=" << seed << " kernel=" << (use_kernel ? "1" : "0") << " basis="
            << (use_kernel ? cfg.kernel_basis : "-") << " mode=" << cfg.kernel_feature_mode << " acc=" << acc
            << " train_sec=" << sec << " n_basis=" << km.n_basis() << " gamma=" << km.gamma() << "\n";
  return acc;
}

SeedResult run_seed_pair(int seed, const BenchConfig& cfg) {
  SeedResult out;
  out.linear_acc = run_seed_mode(seed, false, cfg);
  out.kernel_acc = run_seed_mode(seed, true, cfg);
  return out;
}

BenchConfig parse_bench_config(int argc, char** argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--seeds" && i + 1 < argc) {
      cfg.seeds = std::atoi(argv[++i]);
    } else if (arg == "--passes" && i + 1 < argc) {
      cfg.passes = std::atoi(argv[++i]);
    } else if (arg == "--kernel-blend" && i + 1 < argc) {
      cfg.kernel_blend = std::atof(argv[++i]);
    } else if (arg == "--kernel-m" && i + 1 < argc) {
      cfg.kernel_m = std::atoi(argv[++i]);
    } else if (arg == "--gamma-scale" && i + 1 < argc) {
      cfg.gamma_scale = std::atof(argv[++i]);
    } else if (arg == "--kernel-lr-scale" && i + 1 < argc) {
      cfg.kernel_lr_scale = std::atof(argv[++i]);
    } else if (arg == "--no-shuffle") {
      cfg.shuffle_train = false;
    } else if (arg == "--kernel-raw-x") {
      cfg.kernel_feature_mode = "raw_x";
    } else if (arg == "--kernel-xor-features") {
      cfg.kernel_feature_mode = "xor_pair";
    } else if (arg == "--kernel-feature-mode" && i + 1 < argc) {
      cfg.kernel_feature_mode = argv[++i];
    } else if (arg == "--kernel-basis" && i + 1 < argc) {
      cfg.kernel_basis = argv[++i];
    } else if (arg == "--rff-dim" && i + 1 < argc) {
      cfg.rff_dim = std::atoi(argv[++i]);
    } else if (arg == "--rff-gamma-scale" && i + 1 < argc) {
      cfg.rff_gamma_scale = std::atof(argv[++i]);
    } else if (arg == "--rff-fixed-gamma" && i + 1 < argc) {
      cfg.rff_fixed_gamma = std::atof(argv[++i]);
    } else if (arg == "--nystrom-landmark-sampling" && i + 1 < argc) {
      cfg.nystrom_landmark_sampling = argv[++i];
    } else if (arg == "--rff-projection" && i + 1 < argc) {
      cfg.rff_projection = argv[++i];
    }
  }
  return cfg;
}

void usage() {
  std::cout << "usage: xor_kernel_bench [options]\n"
            << "  --seeds N            number of seeds (default 3)\n"
            << "  --passes N           training passes (default 8)\n"
            << "  --kernel-blend B     kernel LLR blend in [0,1] (default 1.0)\n"
            << "  --kernel-m M         Nyström landmarks (default 512)\n"
            << "  --gamma-scale G      RBF bandwidth multiplier (default 2.0)\n"
            << "  --kernel-lr-scale S  kernel weight lr scale (default 2.0)\n"
            << "  --kernel-raw-x       Nyström kernel on standardized raw x (not latent h)\n"
            << "  --kernel-xor-features  kernel on [x0,x1,x0*x1,x0^2,x1^2] (5-d)\n"
            << "  --kernel-feature-mode {latent,raw_x,xor_pair}\n"
            << "  --kernel-basis {nystrom,rff}  landmark sketch (default) or fixed RFF projection\n"
            << "  --rff-dim N          RFF projection dimension (default 512, kernel_basis=rff)\n"
            << "  --rff-gamma-scale G  multiplier on RFF auto (median-heuristic) gamma (default 1.0)\n"
            << "  --rff-fixed-gamma G  fixed RBF bandwidth for RFF, bypassing auto-gamma (comparison)\n"
            << "  --nystrom-landmark-sampling {uniform,leverage}  Nyström reservoir (default uniform)\n"
            << "  --rff-projection {iid,sorf}  RFF weight init (default iid; sorf = SORF orthogonal)\n"
            << "  --no-shuffle         disable per-pass train shuffle\n"
            << "  --tune               grid search M x gamma_scale x blend\n"
            << "  --tune-seeds N       seeds per tune cell (default 2)\n";
}

struct TuneCell {
  int kernel_m;
  double gamma_scale;
  double kernel_blend;
  double kernel_lr_scale;
  double kernel_mean = 0.0;
  double linear_mean = 0.0;
  double delta_pp = 0.0;
};

std::vector<int> parse_int_list(const char* csv) {
  std::vector<int> out;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) {
      out.push_back(std::atoi(tok.c_str()));
    }
  }
  return out;
}

std::vector<double> parse_double_list(const char* csv) {
  std::vector<double> out;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) {
      out.push_back(std::atof(tok.c_str()));
    }
  }
  return out;
}

int run_tune(int argc, char** argv) {
  BenchConfig base = parse_bench_config(argc, argv);
  int tune_seeds = 2;
  std::vector<int> m_grid = {128, 256, 512};
  std::vector<double> gamma_grid = {0.25, 0.5, 1.0, 2.0, 4.0};
  std::vector<double> blend_grid = {0.75, 1.0};
  std::vector<double> lr_grid = {1.0, 2.0};

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tune-seeds" && i + 1 < argc) {
      tune_seeds = std::atoi(argv[++i]);
    } else if (arg == "--tune-m" && i + 1 < argc) {
      m_grid = parse_int_list(argv[++i]);
    } else if (arg == "--tune-gamma" && i + 1 < argc) {
      gamma_grid = parse_double_list(argv[++i]);
    } else if (arg == "--tune-blend" && i + 1 < argc) {
      blend_grid = parse_double_list(argv[++i]);
    } else if (arg == "--tune-lr" && i + 1 < argc) {
      lr_grid = parse_double_list(argv[++i]);
    }
  }

  std::vector<TuneCell> cells;
  for (int m : m_grid) {
    for (double g : gamma_grid) {
      for (double b : blend_grid) {
        for (double lr : lr_grid) {
          TuneCell cell{m, g, b, lr};
          BenchConfig cfg = base;
          cfg.seeds = tune_seeds;
          cfg.kernel_m = m;
          cfg.gamma_scale = g;
          cfg.kernel_blend = b;
          cfg.kernel_lr_scale = lr;
          double lin_sum = 0.0;
          double ker_sum = 0.0;
          for (int s = 0; s < tune_seeds; ++s) {
            const SeedResult r = run_seed_pair(s, cfg);
            lin_sum += r.linear_acc;
            ker_sum += r.kernel_acc;
          }
          cell.linear_mean = lin_sum / static_cast<double>(tune_seeds);
          cell.kernel_mean = ker_sum / static_cast<double>(tune_seeds);
          cell.delta_pp = 100.0 * (cell.kernel_mean - cell.linear_mean);
          cells.push_back(cell);
        }
      }
    }
  }

  std::sort(cells.begin(), cells.end(), [](const TuneCell& a, const TuneCell& b) {
    if (a.kernel_mean != b.kernel_mean) {
      return a.kernel_mean > b.kernel_mean;
    }
    return a.delta_pp > b.delta_pp;
  });

  std::cout << "{\n  \"mode\": \"tune\",\n  \"tune_seeds\": " << tune_seeds << ",\n  \"passes\": " << base.passes
            << ",\n  \"cells\": [\n";
  for (std::size_t i = 0; i < cells.size(); ++i) {
    const TuneCell& c = cells[i];
    if (i) {
      std::cout << ",\n";
    }
    std::cout << "    {\"kernel_m\": " << c.kernel_m << ", \"gamma_scale\": " << c.gamma_scale
              << ", \"kernel_blend\": " << c.kernel_blend << ", \"kernel_lr_scale\": " << c.kernel_lr_scale
              << ", \"linear_mean_acc\": " << c.linear_mean << ", \"kernel_mean_acc\": " << c.kernel_mean
              << ", \"delta_pp\": " << c.delta_pp << "}";
  }
  if (!cells.empty()) {
    const TuneCell& best = cells.front();
    std::cout << "\n  ],\n  \"best\": {\"kernel_m\": " << best.kernel_m << ", \"gamma_scale\": "
              << best.gamma_scale << ", \"kernel_blend\": " << best.kernel_blend << ", \"kernel_lr_scale\": "
              << best.kernel_lr_scale << ", \"kernel_mean_acc\": " << best.kernel_mean
              << ", \"delta_pp\": " << best.delta_pp << "}\n}\n";
  } else {
    std::cout << "\n  ],\n  \"best\": null\n}\n";
  }
  return 0;
}

}  // namespace

int xor_kernel_bench_main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
      usage();
      return 0;
    }
    if (std::string(argv[i]) == "--tune") {
      return run_tune(argc, argv);
    }
  }

  const BenchConfig cfg = parse_bench_config(argc, argv);
  double lin_sum = 0.0;
  double ker_sum = 0.0;
  std::cout << "{\n  \"dataset\": \"S3_nonlinear_xor_native\",\n  \"seeds\": " << cfg.seeds
            << ",\n  \"passes\": " << cfg.passes << ",\n  \"kernel_blend\": " << cfg.kernel_blend
            << ",\n  \"kernel_basis\": \"" << cfg.kernel_basis << "\",\n  \"kernel_m\": " << cfg.kernel_m
            << ",\n  \"gamma_scale\": " << cfg.gamma_scale << ",\n  \"kernel_lr_scale\": " << cfg.kernel_lr_scale
            << ",\n  \"rff_dim\": " << cfg.rff_dim << ",\n  \"rff_gamma_scale\": " << cfg.rff_gamma_scale
            << ",\n  \"rff_fixed_gamma\": " << cfg.rff_fixed_gamma << ",\n  \"nystrom_landmark_sampling\": \""
            << cfg.nystrom_landmark_sampling << "\",\n  \"rff_projection\": \"" << cfg.rff_projection
            << "\",\n  \"kernel_feature_mode\": \"" << cfg.kernel_feature_mode << "\",\n  \"shuffle_train\": "
            << (cfg.shuffle_train ? "true" : "false") << ",\n  \"linear\": [\n";
  for (int s = 0; s < cfg.seeds; ++s) {
    if (s) {
      std::cout << ",\n";
    }
    const double acc = run_seed_mode(s, false, cfg);
    std::cout << "    " << acc;
    lin_sum += acc;
  }
  std::cout << "\n  ],\n  \"kernel\": [\n";
  for (int s = 0; s < cfg.seeds; ++s) {
    if (s) {
      std::cout << ",\n";
    }
    const double acc = run_seed_mode(s, true, cfg);
    std::cout << "    " << acc;
    ker_sum += acc;
  }
  const double lin_mean = lin_sum / static_cast<double>(cfg.seeds);
  const double ker_mean = ker_sum / static_cast<double>(cfg.seeds);
  std::cout << "\n  ],\n  \"linear_mean_acc\": " << lin_mean << ",\n  \"kernel_mean_acc\": " << ker_mean
            << ",\n  \"delta_pp\": " << 100.0 * (ker_mean - lin_mean) << "\n}\n";
  return 0;
}

#ifndef CYPHA_KERNEL_TUNE_STANDALONE
int main(int argc, char** argv) { return xor_kernel_bench_main(argc, argv); }
#endif
