/// D16B-style zero-forgetting probe: baseline vs EWC on synthetic multitask data.
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

double forgetting_probe(bool use_ewc, std::uint64_t seed, const std::vector<TaskData>& tasks) {
  const int dim = static_cast<int>(tasks.front().train_x.front().size());
  cypha::FreshModelParams fp;
  fp.input_dim = dim;
  fp.field_dim = 32;
  const cypha::CNode root = cypha::create_fresh_model_root(fp);
  cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root, nullptr, fp.field_dim);
  cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim);
  cypha::ReplayBuffer replay(2000);
  cypha::TrainStepParams tsp{};
  std::mt19937 rng(seed);
  int enc_updates = 0;

  cypha::EwcRegularizer ewc;
  cypha::TrainStepExtras extras{};
  if (use_ewc) {
    extras.ewc = &ewc;
    extras.ewc_lambda = 0.5;
  }

  const TaskData* iris = nullptr;
  const TaskData* wine = nullptr;
  const TaskData* digits = nullptr;
  for (const auto& t : tasks) {
    if (t.name == "iris") iris = &t;
    if (t.name == "wine") wine = &t;
    if (t.name == "digits") digits = &t;
  }
  assert(iris != nullptr && wine != nullptr && digits != nullptr);

  auto train_task = [&](const TaskData& task) {
    for (std::size_t i = 0; i < task.train_x.size(); ++i) {
      const std::string label = task.name + "_" + task.train_y[i];
      const int d = static_cast<int>(task.train_x[i].size());
      (void)cypha::dif_train_step_vector(infer, mem, replay, task.train_x[i].data(), d, label, fp.world_lr,
                                         fp.delta_lr, fp.world_lr, fp.delta_lr, 12.0, tsp, rng, enc_updates, nullptr,
                                         use_ewc ? &extras : nullptr);
    }
  };

  train_task(*iris);
  if (use_ewc) {
    ewc.snapshot(mem, infer);
  }
  const double acc_before = eval_task(infer, *iris);
  train_task(*wine);
  train_task(*digits);
  cypha::sync_infer_model_from_memory(infer, mem);
  const double acc_after = eval_task(infer, *iris);
  if (acc_before <= 0.0) {
    return 0.0;
  }
  return (acc_before - acc_after) / acc_before;
}

}  // namespace

int main() {
  std::mt19937 rng(42);
  const auto tasks = make_synthetic_tasks(8, rng);
  const double baseline_forgetting = forgetting_probe(false, 100, tasks);
  const double ewc_forgetting = forgetting_probe(true, 200, tasks);

  assert(baseline_forgetting >= 0.0);
  assert(ewc_forgetting >= 0.0);
  assert(ewc_forgetting <= baseline_forgetting + 1e-6);

  std::printf("ewc_d16b_smoke: baseline_forgetting=%.4f ewc_forgetting=%.4f PASS\n", baseline_forgetting,
              ewc_forgetting);
  return 0;
}
