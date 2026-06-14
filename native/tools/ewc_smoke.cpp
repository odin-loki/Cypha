/// Smoke test for elastic weight consolidation overlay on D / enc_w.
#include <cassert>
#include <cstdio>
#include <vector>

#include "cypha/create_model.hpp"
#include "cypha/ewc_regularizer.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/train_step_vector.hpp"

int main() {
  cypha::FreshModelParams fp;
  fp.input_dim = 6;
  fp.field_dim = 16;
  const cypha::CNode root = cypha::create_fresh_model_root(fp);
  cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root, nullptr, fp.field_dim);
  cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim);
  cypha::ReplayBuffer replay(100);
  cypha::TrainStepParams tsp{};
  cypha::EwcRegularizer ewc;
  ewc.snapshot(mem, infer);
  assert(ewc.penalty(mem, infer) == 0.0);

  cypha::TrainStepExtras extras{};
  extras.ewc = &ewc;
  extras.ewc_lambda = 0.5;
  std::mt19937 rng(42);
  int enc_updates = 0;
  const std::vector<double> x = {0.2, 0.4, 0.1, 0.3, 0.5, 0.2};
  (void)cypha::dif_train_step_vector(infer, mem, replay, x.data(), static_cast<int>(x.size()), "class_a", 0.008, 0.03,
                                     0.008, 0.03, 12.0, tsp, rng, enc_updates, nullptr, &extras);
  const double penalty_after = ewc.penalty(mem, infer);
  assert(penalty_after >= 0.0);

  (void)cypha::dif_train_step_vector(infer, mem, replay, x.data(), static_cast<int>(x.size()), "class_b", 0.008, 0.03,
                                     0.008, 0.03, 12.0, tsp, rng, enc_updates, nullptr, &extras);
  assert(ewc.penalty(mem, infer) >= penalty_after);

  std::puts("ewc_smoke: PASS");
  return 0;
}
