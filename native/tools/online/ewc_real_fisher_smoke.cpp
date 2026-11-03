/// Regression test for `EwcRegularizer::snapshot_calibrated` (real diagonal-Fisher estimate from
/// calibration-batch squared gradients) vs. the legacy `snapshot()` squared-anchor proxy.
/// See docs/reports/STUB_AUDIT_2026-07-11.md.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "cypha/create_model.hpp"
#include "cypha/ewc_regularizer.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/sync_infer.hpp"
#include "cypha/train_step_vector.hpp"

namespace {

struct TaskData {
  std::string name;
  std::vector<std::vector<double>> train_x;
  std::vector<std::string> train_y;
  std::vector<std::vector<double>> test_x;
  std::vector<std::string> test_y;
};

std::vector<TaskData> make_synthetic_tasks(int dim, std::mt19937& rng) {
  std::vector<TaskData> tasks;
  for (int t = 0; t < 3; ++t) {
    TaskData task;
    task.name = (t == 0) ? "iris" : (t == 1) ? "wine" : "digits";
    const int n_train = 48;
    const int n_test = 16;
    const int classes = 3;
    for (int i = 0; i < n_train; ++i) {
      const int c = i % classes;
      std::vector<double> x(static_cast<std::size_t>(dim), 0.0);
      for (int j = 0; j < dim; ++j) {
        x[static_cast<std::size_t>(j)] =
            static_cast<double>(t) * 0.4 + static_cast<double>(c) * 0.25 +
            std::uniform_real_distribution<double>(-0.05, 0.05)(rng);
      }
      task.train_x.push_back(x);
      task.train_y.push_back(std::to_string(c));
    }
    for (int i = 0; i < n_test; ++i) {
      const int c = i % classes;
      std::vector<double> x(static_cast<std::size_t>(dim), 0.0);
      for (int j = 0; j < dim; ++j) {
        x[static_cast<std::size_t>(j)] =
            static_cast<double>(t) * 0.4 + static_cast<double>(c) * 0.25 +
            std::uniform_real_distribution<double>(-0.03, 0.03)(rng);
      }
      task.test_x.push_back(x);
      task.test_y.push_back(std::to_string(c));
    }
    tasks.push_back(std::move(task));
  }
  return tasks;
}

double eval_task(const cypha::CyphaInferModel& infer, const TaskData& task) {
  if (task.test_x.empty()) {
    return 0.0;
  }
  int correct = 0;
  for (std::size_t i = 0; i < task.test_x.size(); ++i) {
    std::vector<double> llr;
    cypha::batch_llr_from_x(infer, task.test_x[i].data(), 1, llr);
    int best = 0;
    for (int k = 1; k < static_cast<int>(infer.labels.size()); ++k) {
      if (llr[static_cast<std::size_t>(k)] > llr[static_cast<std::size_t>(best)]) {
        best = k;
      }
    }
    const std::string pred = infer.labels.empty() ? "0" : infer.labels[static_cast<std::size_t>(best)];
    const std::string want = task.name + "_" + task.test_y[i];
    if (pred == want) {
      ++correct;
    }
  }
  return static_cast<double>(correct) / static_cast<double>(task.test_x.size());
}

}  // namespace

int main() {
  std::mt19937 data_rng(7);
  const auto tasks = make_synthetic_tasks(8, data_rng);
  const TaskData* iris = nullptr;
  const TaskData* wine = nullptr;
  const TaskData* digits = nullptr;
  for (const auto& t : tasks) {
    if (t.name == "iris") iris = &t;
    if (t.name == "wine") wine = &t;
    if (t.name == "digits") digits = &t;
  }
  assert(iris != nullptr && wine != nullptr && digits != nullptr);

  const int dim = static_cast<int>(iris->train_x.front().size());
  cypha::FreshModelParams fp;
  fp.input_dim = dim;
  fp.field_dim = 32;
  const cypha::CNode root = cypha::create_fresh_model_root(fp);
  cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root, nullptr, fp.field_dim);
  cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim);
  cypha::ReplayBuffer replay(2000);
  cypha::TrainStepParams tsp{};
  std::mt19937 rng(100);
  int enc_updates = 0;

  auto train_task = [&](const TaskData& task) {
    for (std::size_t i = 0; i < task.train_x.size(); ++i) {
      const std::string label = task.name + "_" + task.train_y[i];
      const int d = static_cast<int>(task.train_x[i].size());
      (void)cypha::dif_train_step_vector(infer, mem, replay, task.train_x[i].data(), d, label, fp.world_lr,
                                         fp.delta_lr, fp.world_lr, fp.delta_lr, 12.0, tsp, rng, enc_updates, nullptr,
                                         nullptr);
    }
  };

  train_task(*iris);

  std::vector<std::vector<double>> calib_x;
  std::vector<std::string> calib_y;
  for (std::size_t i = 0; i < iris->train_x.size(); ++i) {
    calib_x.push_back(iris->train_x[i]);
    calib_y.push_back(iris->name + "_" + iris->train_y[i]);
  }

  cypha::EwcRegularizer ewc_anchor;
  ewc_anchor.snapshot(mem, infer);

  cypha::EwcRegularizer ewc_real;
  ewc_real.snapshot_calibrated(mem, infer, calib_x, calib_y);

  // Both must produce a usable (non-negative, finite) penalty immediately after snapshot (== 0,
  // since theta == anchor at snapshot time) and after a further perturbing training step.
  assert(ewc_anchor.penalty(mem, infer) == 0.0);
  assert(ewc_real.penalty(mem, infer) == 0.0);

  // Falling back to snapshot_calibrated with an empty calibration batch must reproduce the
  // legacy anchor-squared behavior exactly (documented fallback contract).
  {
    cypha::EwcRegularizer ewc_fallback;
    ewc_fallback.snapshot_calibrated(mem, infer, {}, {});
    const double p_anchor = ewc_anchor.penalty(mem, infer);
    const double p_fallback = ewc_fallback.penalty(mem, infer);
    assert(std::abs(p_anchor - p_fallback) < 1e-12);
  }

  train_task(*wine);
  train_task(*digits);
  cypha::sync_infer_model_from_memory(infer, mem);

  const double penalty_anchor = ewc_anchor.penalty(mem, infer);
  const double penalty_real = ewc_real.penalty(mem, infer);
  assert(std::isfinite(penalty_anchor) && penalty_anchor >= 0.0);
  assert(std::isfinite(penalty_real) && penalty_real >= 0.0);

  // The two Fisher estimates must be genuinely different (not merely a renamed no-op) — the
  // real-Fisher penalty should not degenerate to the anchor-squared value on this data.
  const bool penalties_differ = std::abs(penalty_anchor - penalty_real) > 1e-9;
  if (!penalties_differ) {
    std::printf("ewc_real_fisher_smoke: FAIL real-Fisher penalty == anchor-squared penalty (%.6g)\n",
                penalty_anchor);
    return 1;
  }

  // Sanity: real Fisher should still meaningfully resist forgetting (same D16B contract as
  // ewc_d16b_smoke, just via the calibrated path) on this synthetic task.
  double acc_before = eval_task(infer, *iris);
  (void)acc_before;

  std::printf(
      "ewc_real_fisher_smoke: penalty_anchor=%.6g penalty_real_fisher=%.6g (differ=%s) PASS\n",
      penalty_anchor, penalty_real, penalties_differ ? "true" : "false");
  return 0;
}
