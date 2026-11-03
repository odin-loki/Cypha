#include "cypha/forecast/node_estimator.hpp"

#include "cypha/classify/create_model.hpp"
#include "cypha/classify/infer_cpu.hpp"
#include "cypha/classify/memory_train.hpp"
#include "cypha/online/replay_buffer.hpp"
#include "cypha/classify/sync_infer.hpp"
#include "cypha/classify/train_step_vector.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace cypha::forecast {

NodeEstimator::NodeEstimator(NodeEstimatorConfig cfg) : cfg_(cfg) {}

NodeEstimatorResult NodeEstimator::train_on_mid(const std::vector<DisputeRecord>& records,
                                                double eval_frac) {
  NodeEstimatorResult out;
  if (records.size() < 4) {
    out.detail = "need at least 4 MID records";
    return out;
  }

  std::vector<double> x_all;
  std::vector<std::string> y_all;
  int n_feat = 0;
  mid_to_features(records, x_all, y_all, n_feat);

  const int n = static_cast<int>(y_all.size());
  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
  std::mt19937 rng(42);
  std::shuffle(idx.begin(), idx.end(), rng);

  const int n_eval = std::max(1, static_cast<int>(std::round(eval_frac * n)));
  const int n_train = n - n_eval;

  std::vector<std::vector<double>> tr;
  std::vector<std::vector<double>> te;
  std::vector<std::string> y_tr;
  std::vector<std::string> y_te;
  tr.reserve(static_cast<std::size_t>(n_train));
  te.reserve(static_cast<std::size_t>(n_eval));
  for (int i = 0; i < n_train; ++i) {
    const int r = idx[static_cast<std::size_t>(i)];
    std::vector<double> row(n_feat);
    for (int f = 0; f < n_feat; ++f) {
      row[static_cast<std::size_t>(f)] = x_all[static_cast<std::size_t>(r * n_feat + f)];
    }
    tr.push_back(std::move(row));
    y_tr.push_back(y_all[static_cast<std::size_t>(r)]);
  }
  for (int i = n_train; i < n; ++i) {
    const int r = idx[static_cast<std::size_t>(i)];
    std::vector<double> row(n_feat);
    for (int f = 0; f < n_feat; ++f) {
      row[static_cast<std::size_t>(f)] = x_all[static_cast<std::size_t>(r * n_feat + f)];
    }
    te.push_back(std::move(row));
    y_te.push_back(y_all[static_cast<std::size_t>(r)]);
  }

  pre_.scale = true;
  pre_.rff_dim = cfg_.rff_dim;
  pre_.rff_orf = cfg_.use_orf;
  pre_.rff_sorf = !cfg_.use_orf;
  pre_.auto_rff_gamma_cv = true;
  std::vector<double> flat_tr;
  flat_tr.reserve(tr.size() * static_cast<std::size_t>(n_feat));
  for (const auto& row : tr) {
    flat_tr.insert(flat_tr.end(), row.begin(), row.end());
  }
  pre_.fit_from_design_matrix(flat_tr, n_train, n_feat);
  for (auto& row : tr) {
    row = pre_.transform_one(row);
  }
  for (auto& row : te) {
    row = pre_.transform_one(row);
  }
  const int d = pre_.output_dim;

  FreshModelParams fp;
  fp.input_dim = d;
  fp.field_dim = 64;
  fp.world_lr = 0.008;
  fp.delta_lr = 0.03;
  const CNode root = create_fresh_model_root(fp);
  auto infer = CyphaInferModel::from_root(root, nullptr, fp.field_dim);
  auto mem = CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim);
  ReplayBuffer replay(2000);
  TrainStepParams tsp;
  tsp.replay_ratio = 0.1;
  std::mt19937 trng(42);
  int enc_updates = 0;

  for (int ep = 0; ep < cfg_.epochs; ++ep) {
    for (int i = 0; i < n_train; ++i) {
      dif_train_step_vector(infer, mem, replay, tr[static_cast<std::size_t>(i)].data(), d,
                            y_tr[static_cast<std::size_t>(i)], fp.world_lr, fp.delta_lr, fp.world_lr,
                            fp.delta_lr, 12.0, tsp, trng, enc_updates, nullptr, nullptr);
    }
  }
  sync_infer_model_from_memory(infer, mem);

  int correct_train = 0;
  for (int i = 0; i < n_train; ++i) {
    const auto inf = infer_at_h(infer, tr[static_cast<std::size_t>(i)].data(), {});
    if (inf.label == y_tr[static_cast<std::size_t>(i)]) {
      ++correct_train;
    }
  }
  int correct_eval = 0;
  for (int i = 0; i < n_eval; ++i) {
    const auto inf = infer_at_h(infer, te[static_cast<std::size_t>(i)].data(), {});
    if (inf.label == y_te[static_cast<std::size_t>(i)]) {
      ++correct_eval;
    }
  }

  cypha_.infer_owned() = std::make_unique<CyphaInferModel>(std::move(infer));
  cypha_.mem_owned() = std::make_unique<CyphaDifMemoryState>(std::move(mem));
  cypha_.preprocessor_owned() = std::make_unique<PreprocessorState>(pre_);
  cypha_.ewc_lambda() = cfg_.ewc_lambda;

  out.n_train = n_train;
  out.n_eval = n_eval;
  out.train_accuracy = static_cast<double>(correct_train) / static_cast<double>(n_train);
  out.eval_accuracy = static_cast<double>(correct_eval) / static_cast<double>(n_eval);
  return out;
}

}  // namespace cypha::forecast
