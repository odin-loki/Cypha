/// Phase 1 promotion gate: XOR + D01 golden linear-sep (same split/training as cypha_bench_run D01).
#include "cypha/create_model.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/portable_shuffle.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/sync_infer.hpp"
#include "cypha/train_step_vector.hpp"

#include "../src/bench/d01_synthetic_golden.inc"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {

struct ClfSplit {
  std::vector<std::vector<double>> tr;
  std::vector<std::vector<double>> te;
  std::vector<std::string> y_tr;
  std::vector<std::string> y_te;
};

std::mt19937 make_rng(std::uint64_t seed) {
  return std::mt19937(static_cast<std::uint32_t>(seed));
}

ClfSplit make_xor(int n, int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  ClfSplit s;
  const int n_te = std::max(1, n / 5);
  const int n_tr = n - n_te;
  for (int i = 0; i < n; ++i) {
    std::vector<double> x(2);
    x[0] = nd(rng);
    x[1] = nd(rng);
    const bool label = (x[0] > 0.0) ^ (x[1] > 0.0);
    const std::string y = label ? "1" : "0";
    if (i < n_tr) {
      s.tr.push_back(x);
      s.y_tr.push_back(y);
    } else {
      s.te.push_back(x);
      s.y_te.push_back(y);
    }
  }
  return s;
}

/// D01 linearly_separable_2class: golden draws, sequential 80/20 split (matches train_eval_classifier).
ClfSplit make_d01_golden() {
  std::vector<std::vector<double>> xs;
  std::vector<std::string> ys;
  if (!cypha::bench::d01_golden::try_load(400, 10, 42, xs, ys)) {
    std::cerr << "orf_encoder_bench: D01 golden load failed\n";
    std::exit(2);
  }
  const int train_n = static_cast<int>(static_cast<double>(xs.size()) * 0.8);
  ClfSplit s;
  for (int i = 0; i < train_n; ++i) {
    s.tr.push_back(xs[static_cast<std::size_t>(i)]);
    s.y_tr.push_back(ys[static_cast<std::size_t>(i)]);
  }
  for (int i = train_n; i < static_cast<int>(xs.size()); ++i) {
    s.te.push_back(xs[static_cast<std::size_t>(i)]);
    s.y_te.push_back(ys[static_cast<std::size_t>(i)]);
  }
  return s;
}

double train_eval(const ClfSplit& s, cypha::RffProjectionKind kind, int rff_dim, int passes) {
  const int n_feat = static_cast<int>(s.tr.front().size());
  cypha::PreprocessorState pre;
  pre.scale = true;
  pre.rff_dim = rff_dim;
  pre.rff_orf = (kind == cypha::RffProjectionKind::Orf);
  pre.rff_sorf = (kind == cypha::RffProjectionKind::Sorf);
  if (kind == cypha::RffProjectionKind::IidGaussian) {
    pre.rff_sorf = false;
    pre.rff_orf = false;
  }
  pre.seed = 42;
  std::vector<double> flat;
  for (const auto& row : s.tr) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  pre.fit_from_design_matrix(flat, static_cast<int>(s.tr.size()), n_feat);
  auto tr = s.tr;
  auto te = s.te;
  for (auto& row : tr) {
    row = pre.transform_one(row);
  }
  for (auto& row : te) {
    row = pre.transform_one(row);
  }
  const int d = pre.output_dim;

  cypha::FreshModelParams fp;
  fp.input_dim = d;
  fp.field_dim = 128;
  fp.world_lr = 0.008;
  fp.delta_lr = 0.03;
  fp.temperature = 0.9975;
  const cypha::CNode root = cypha::create_fresh_model_root(fp);
  auto infer = cypha::CyphaInferModel::from_root(root, nullptr, fp.field_dim);
  auto mem = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim);
  cypha::ReplayBuffer replay(10000);
  cypha::TrainStepParams tsp;
  tsp.enc_lr = 0.002;
  tsp.replay_ratio = 0.22;
  tsp.replay_cap = 10000;
  std::mt19937 rng = make_rng(42);
  int enc = 0;
  const int train_n = static_cast<int>(tr.size());
  for (int p = 0; p < passes; ++p) {
    std::vector<int> order(static_cast<std::size_t>(train_n));
    for (int i = 0; i < train_n; ++i) {
      order[static_cast<std::size_t>(i)] = i;
    }
    cypha::portable_shuffle(order.begin(), order.end(), make_rng(42ULL + static_cast<std::uint64_t>(p)));
    for (int idx : order) {
      cypha::dif_train_step_vector(infer, mem, replay, tr[static_cast<std::size_t>(idx)].data(), d,
                                   s.y_tr[static_cast<std::size_t>(idx)], fp.world_lr, fp.delta_lr,
                                   fp.world_lr, fp.delta_lr, 12.0, tsp, rng, enc, nullptr, nullptr);
    }
  }
  cypha::sync_infer_model_from_memory(infer, mem);
  int correct = 0;
  for (std::size_t i = 0; i < te.size(); ++i) {
    const auto out = cypha::infer_at_h(infer, te[i].data(), {});
    if (out.label == s.y_te[i]) {
      ++correct;
    }
  }
  return static_cast<double>(correct) / static_cast<double>(te.size());
}

}  // namespace

int main() {
  constexpr int kRffDim = 256;
  constexpr int kPasses = 4;
  constexpr double kD01Pin = 0.9875;
  constexpr double kXorPin = 0.482;
  constexpr double kD01Margin = 0.02;

  const auto xor_split = make_xor(400, 42);
  const auto d01_split = make_d01_golden();

  const double xor_iid = train_eval(xor_split, cypha::RffProjectionKind::IidGaussian, kRffDim, kPasses);
  const double xor_sorf = train_eval(xor_split, cypha::RffProjectionKind::Sorf, kRffDim, kPasses);
  const double xor_orf = train_eval(xor_split, cypha::RffProjectionKind::Orf, kRffDim, kPasses);

  const double d01_iid = train_eval(d01_split, cypha::RffProjectionKind::IidGaussian, kRffDim, kPasses);
  const double d01_sorf = train_eval(d01_split, cypha::RffProjectionKind::Sorf, kRffDim, kPasses);
  const double d01_orf = train_eval(d01_split, cypha::RffProjectionKind::Orf, kRffDim, kPasses);

  std::cout << "xor_linear_iid=" << xor_iid << " (pin~" << kXorPin << ")\n";
  std::cout << "xor_linear_sorf=" << xor_sorf << "\n";
  std::cout << "xor_linear_orf=" << xor_orf << "\n";
  std::cout << "d01_golden_iid=" << d01_iid << " (pin~" << kD01Pin << ")\n";
  std::cout << "d01_golden_sorf=" << d01_sorf << "\n";
  std::cout << "d01_golden_orf=" << d01_orf << "\n";

  const bool sorf_xor_win = xor_sorf > xor_iid + 0.05;
  const bool sorf_d01_pin_ok = d01_sorf >= kD01Pin - kD01Margin;
  const bool sorf_d01_lift_ok = d01_sorf >= d01_iid + 0.30;
  const bool orf_xor_win = xor_orf > xor_iid + 0.05;
  const bool orf_d01_pin_ok = d01_orf >= kD01Pin - kD01Margin;
  const bool orf_d01_lift_ok = d01_orf >= d01_iid + 0.30;

  if (sorf_xor_win && (sorf_d01_pin_ok || sorf_d01_lift_ok)) {
    std::cout << "promote=sorf (XOR win + D01 pin or strong lift)\n";
  } else if (orf_xor_win && (orf_d01_pin_ok || orf_d01_lift_ok)) {
    std::cout << "promote=orf (XOR win + D01 pin or strong lift)\n";
  } else {
    std::cout << "promote=none (keep iid default for explicit overrides)\n";
    if (sorf_xor_win && sorf_d01_lift_ok) {
      std::cout << "note=SORF passes lift gate; full D01 pin: CYPHA_BENCH_USE_PROFILE=1\n";
    }
  }

  std::cout << "orf_encoder_bench OK\n";
  return 0;
}
