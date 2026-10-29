/// End-to-end forecasting framework smoke (Phases 1-9).
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "cypha/forecast/forecast_pipeline.hpp"
#include "cypha/forecast/rollout_tree.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

namespace {

// Rollouts read the trained model (SLOW_TEST_TRIAGE, "Fix"): a model trained on the cycle 2 3 4 5 continues it in
// rollouts and a cold one does not; rollouts learn nothing (the learned digest is unchanged) and carry nothing from one
// rollout to the next (the same seed gives the same tree twice).
int rollout_check() {
  namespace cy = cypha::cyphalm;
  cy::CyphaLMConfig cfg;
  cfg.vocab_size = 256;
  cfg.hp_table_bits = 16;
  cy::apply_hp_production_recipe(cfg);
  std::vector<int> ids;
  for (int i = 0; i < 400; ++i) ids.push_back(2 + i % 4);
  cy::CyphaLMModel trained(cfg);
  trained.train_sequence(ids, static_cast<int>(ids.size()) - 1, 2);
  cy::CyphaLMModel cold(cfg);

  auto share_on_cycle = [](cy::CyphaLMModel& m) {
    std::mt19937 rng(7);
    int on = 0, n = 0;
    for (int r = 0; r < 20; ++r) {
      const auto path = cypha::forecast::ancestral_rollout(m, {2, 3}, 8, 0.5, rng);
      for (std::size_t i = 2; i < path.size(); ++i) {
        const std::uint32_t prev = path[i - 1];
        ++n;
        on += prev >= 2 && prev <= 5 && path[i] == 2 + (prev - 1) % 4;
      }
    }
    return n > 0 ? static_cast<double>(on) / n : 0.0;
  };
  const std::uint64_t digest = trained.hp_backend().predictor().learned_digest();
  const double trained_share = share_on_cycle(trained), cold_share = share_on_cycle(cold);
  std::cout << "rollout_cycle trained=" << trained_share << " cold=" << cold_share << "\n";
  if (!(trained_share >= 0.9)) {
    std::cerr << "forecast_smoke: rollouts should follow the trained cycle (share " << trained_share << ")\n";
    return 3;
  }
  if (!(cold_share <= 0.2)) {
    std::cerr << "forecast_smoke: a cold model should not know the cycle (share " << cold_share << ")\n";
    return 3;
  }
  if (trained.hp_backend().predictor().learned_digest() != digest) {
    std::cerr << "forecast_smoke: rollouts changed what the model learned\n";
    return 3;
  }
  const cypha::forecast::EventVocabulary vocab;
  cypha::forecast::RolloutTreeConfig tcfg;
  tcfg.n_rollouts = 30;
  std::mt19937 r1(11), r2(11);
  const auto t1 = cypha::forecast::generate_rollout_tree(trained, vocab, {2, 3, 4}, tcfg, r1);
  const auto t2 = cypha::forecast::generate_rollout_tree(trained, vocab, {2, 3, 4}, tcfg, r2);
  bool same = t1.nodes.size() == t2.nodes.size();
  for (std::size_t i = 0; same && i < t1.nodes.size(); ++i) {
    same = t1.nodes[i].token_id == t2.nodes[i].token_id && t1.nodes[i].visit_count == t2.nodes[i].visit_count &&
           t1.nodes[i].active_query_score == t2.nodes[i].active_query_score;
  }
  if (!same) {
    std::cerr << "forecast_smoke: the same seed gave two different trees (state carried between rollouts)\n";
    return 3;
  }
  std::cout << "rollout_trained OK nodes=" << t1.nodes.size() << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (const int rc = rollout_check()) return rc;

  std::filesystem::path data_dir = "bench/data/forecast";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--data" && i + 1 < argc) {
      data_dir = argv[++i];
    }
  }

  const char* candidates[] = {
      "bench/data/forecast",
      "../bench/data/forecast",
      "../../bench/data/forecast",
  };
  if (!std::filesystem::exists(data_dir)) {
    for (const char* c : candidates) {
      if (std::filesystem::exists(c)) {
        data_dir = c;
        break;
      }
    }
  }

  cypha::forecast::ForecastPipeline pipe;
  const auto result = pipe.run(data_dir);

  if (!result.detail.empty()) {
    std::cerr << "forecast_smoke: " << result.detail << "\n";
    return 2;
  }

  std::cout << "node_eval_acc=" << result.node_result.eval_accuracy << "\n";
  std::cout << "tree_nodes=" << result.scenario_tree.nodes.size() << "\n";
  std::cout << "forecast_smoke OK\n";
  return 0;
}
