/// End-to-end forecasting framework smoke (Phases 1-9).
#include "cypha/forecast/event_model.hpp"
#include "cypha/forecast/forecast_pipeline.hpp"
#include "cypha/forecast/rollout_tree.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

namespace {

// Rollouts read the trained model (SLOW_TEST_TRIAGE, "Fix"), for every event model and for ids of 256 and above
// (FORECAST_EVENT_VOCAB.md: before FCAST such an id threw in hp's training). A model trained on a cycle of four
// events in four theaters continues it in rollouts and a cold one does not; rollouts learn nothing (the learned
// digest is unchanged) and carry nothing from one rollout to the next (the same seed gives the same tree twice).
int rollout_check() {
  using namespace cypha::forecast;
  const EventVocabulary vocab;
  const std::vector<std::uint32_t> cycle{300, 450, 600, 750};  // Ukraine, Middle East, Korea, Africa
  std::vector<std::uint32_t> ids;
  for (int i = 0; i < 400; ++i) ids.push_back(cycle[static_cast<std::size_t>(i % 4)]);
  auto next_of = [&](std::uint32_t prev) -> std::int64_t {
    for (std::size_t i = 0; i < cycle.size(); ++i) {
      if (cycle[i] == prev) return cycle[(i + 1) % cycle.size()];
    }
    return -1;
  };
  for (const char* kind : {"hp_pair", "hp_base256", "hp_theater", "ngram"}) {
    EventModelConfig cfg;
    cfg.kind = kind;
    cfg.hp_table_bits = 16;
    cfg.theater_table_bits = 16;
    auto trained = make_event_model(vocab, cfg);
    trained->train(ids, 2);
    auto cold = make_event_model(vocab, cfg);
    auto share_on_cycle = [&](EventModel& m) {
      std::mt19937 rng(7);
      int on = 0, n = 0;
      for (int r = 0; r < 20; ++r) {
        const auto path = ancestral_rollout(m, {300, 450}, 8, 0.5, rng);
        for (std::size_t i = 2; i < path.size(); ++i) {
          ++n;
          on += next_of(path[i - 1]) == static_cast<std::int64_t>(path[i]);
        }
      }
      return n > 0 ? static_cast<double>(on) / n : 0.0;
    };
    const std::uint64_t digest = trained->learned_digest();
    const double trained_share = share_on_cycle(*trained), cold_share = share_on_cycle(*cold);
    std::cout << "rollout_cycle " << kind << " trained=" << trained_share << " cold=" << cold_share << "\n";
    if (!(trained_share >= 0.9)) {
      std::cerr << "forecast_smoke: " << kind << " rollouts should follow the trained cycle (share " << trained_share
                << ")\n";
      return 3;
    }
    if (!(cold_share <= 0.2)) {
      std::cerr << "forecast_smoke: a cold " << kind << " model should not know the cycle (share " << cold_share
                << ")\n";
      return 3;
    }
    if (trained->learned_digest() != digest) {
      std::cerr << "forecast_smoke: " << kind << " rollouts changed what the model learned\n";
      return 3;
    }
    RolloutTreeConfig tcfg;
    tcfg.n_rollouts = 30;
    std::mt19937 r1(11), r2(11);
    const auto t1 = generate_rollout_tree(*trained, {300, 450, 600}, tcfg, r1);
    const auto t2 = generate_rollout_tree(*trained, {300, 450, 600}, tcfg, r2);
    bool same = t1.nodes.size() == t2.nodes.size();
    for (std::size_t i = 0; same && i < t1.nodes.size(); ++i) {
      same = t1.nodes[i].token_id == t2.nodes[i].token_id && t1.nodes[i].visit_count == t2.nodes[i].visit_count &&
             t1.nodes[i].active_query_score == t2.nodes[i].active_query_score;
    }
    if (!same) {
      std::cerr << "forecast_smoke: " << kind << ": the same seed gave two different trees\n";
      return 3;
    }
    std::cout << "rollout_trained " << kind << " OK nodes=" << t1.nodes.size() << "\n";
  }
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
  std::cout << "sequence_bits_per_event=" << result.sequence_eval_bpc << "\n";
  std::cout << "tree_nodes=" << result.scenario_tree.nodes.size() << "\n";
  // The repo's events include ids of 256 and above (Ukraine, Middle East, Korea, South Asia); they used to throw.
  std::uint32_t max_id = 0;
  for (const auto& node : result.scenario_tree.nodes) max_id = std::max(max_id, node.token_id);
  for (const auto id : result.scenario_tree.root_context) max_id = std::max(max_id, id);
  std::cout << "max_event_id=" << max_id << "\n";
  if (!(result.sequence_eval_bpc > 0.0) || result.scenario_tree.nodes.size() < 2) {
    std::cerr << "forecast_smoke: the pipeline did not train or roll out\n";
    return 3;
  }
  std::cout << "forecast_smoke OK\n";
  return 0;
}
