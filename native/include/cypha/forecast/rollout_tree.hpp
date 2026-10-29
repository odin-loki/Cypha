#pragma once

#include "cypha/forecast/event_model.hpp"
#include "cypha/forecast/event_vocabulary.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha::forecast {

/// One node in the generated scenario tree.
struct TreeNode {
  std::uint32_t token_id{0};
  int depth{0};
  int visit_count{0};
  double transition_prob{0.0};
  double active_query_score{0.0};
  double epistemic_var{0.0};
  double leaf_value{0.0};
  bool drift_flag{false};
  int parent{-1};
  std::vector<int> children;
};

/// MCTS-style guided rollout tree over an ``EventModel`` (Phases 5 + 3.3).
struct RolloutTreeConfig {
  int max_depth{8};
  int n_rollouts{200};
  int expand_top_k{3};
  double exploration_weight{1.5};
  double temperature{1.0};
  double drift_threshold{0.05};
};

struct RolloutTree {
  std::vector<TreeNode> nodes;
  std::vector<std::uint32_t> root_context;
};

/// Build a guided scenario tree from seed context using ancestral sampling + active-query expansion. Rollouts read
/// the trained model with learning off, each from the same fresh stream (FORECAST_EVENT_VOCAB.md; SLOW_TEST_TRIAGE).
RolloutTree generate_rollout_tree(EventModel& model, const std::vector<std::uint32_t>& seed_tokens,
                                  const RolloutTreeConfig& cfg, std::mt19937& rng);

/// Sample one ancestral path (temperature) for quick forecasts.
std::vector<std::uint32_t> ancestral_rollout(EventModel& model, const std::vector<std::uint32_t>& seed, int horizon,
                                             double temperature, std::mt19937& rng);

}  // namespace cypha::forecast
