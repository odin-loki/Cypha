#include "cypha/forecast/rollout_tree.hpp"

#include "cypha/infer_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cypha::forecast {
namespace {

double active_query_from_log_probs(const std::vector<double>& log_probs) {
  if (log_probs.empty()) {
    return 0.0;
  }
  const int k = static_cast<int>(log_probs.size());
  std::vector<double> z(static_cast<std::size_t>(k));
  double max_z = log_probs[0];
  for (int i = 1; i < k; ++i) {
    max_z = std::max(max_z, log_probs[static_cast<std::size_t>(i)]);
  }
  for (int i = 0; i < k; ++i) {
    z[static_cast<std::size_t>(i)] = log_probs[static_cast<std::size_t>(i)] - max_z;
  }
  std::vector<double> probs(static_cast<std::size_t>(k));
  softmax_batch_reference(z.data(), 1, k, 1e-8, probs);
  return active_query_score_from_probs(probs.data(), k, 1e-8);
}

// Rollouts read the trained model, never a cold one (until 2026-10-29 each rollout and tree node called
// reset_context(), which in the hp wrapper builds a new, untrained predictor). Learning is off while rollouts run, so
// sampled events teach the model nothing, and every rollout and node starts from the same state: one fresh stream on
// the trained model, marked and restored before each, so nothing carries from one rollout into the next. The model is
// left on that fresh stream, with learning as it was. Since FCAST (FORECAST_EVENT_VOCAB.md) the tree reads an
// EventModel over all 842 event ids; a hp model sees each event as bytes behind that interface.
class RolloutBase {
 public:
  explicit RolloutBase(EventModel& model) : model_(model), prev_learning_(model.learning()) {
    model_.set_learning(false);
    model_.begin_stream();
    model_.mark();
  }
  ~RolloutBase() {
    model_.rewind();
    model_.unmark();
    model_.set_learning(prev_learning_);
  }
  RolloutBase(const RolloutBase&) = delete;
  RolloutBase& operator=(const RolloutBase&) = delete;

  // The base state, then the model reads every event of ``context``.
  void sync(const std::vector<std::uint32_t>& context) {
    model_.rewind();
    for (const auto id : context) {
      (void)model_.observe(id);
    }
  }

 private:
  EventModel& model_;
  bool prev_learning_;
};

std::vector<std::uint32_t> rollout_from(RolloutBase& base, EventModel& model, const std::vector<std::uint32_t>& seed,
                                        int horizon, double temperature, std::mt19937& rng) {
  std::vector<std::uint32_t> path = seed;
  if (path.empty()) {
    return path;
  }
  base.sync(seed);
  for (int step = 0; step < horizon; ++step) {
    const std::uint32_t next = model.sample(temperature, rng);
    (void)model.observe(next);
    path.push_back(next);
  }
  return path;
}

std::vector<double> log_of(const std::vector<double>& p) {
  std::vector<double> out(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) {
    out[i] = std::log(std::max(p[i], 1e-300));
  }
  return out;
}

}  // namespace

std::vector<std::uint32_t> ancestral_rollout(EventModel& model, const std::vector<std::uint32_t>& seed, int horizon,
                                             double temperature, std::mt19937& rng) {
  RolloutBase base(model);
  return rollout_from(base, model, seed, horizon, temperature, rng);
}

RolloutTree generate_rollout_tree(EventModel& model, const std::vector<std::uint32_t>& seed_tokens,
                                  const RolloutTreeConfig& cfg, std::mt19937& rng) {
  RolloutTree tree;
  tree.root_context = seed_tokens;
  if (seed_tokens.empty()) {
    return tree;
  }

  TreeNode root;
  root.token_id = seed_tokens.back();
  root.depth = 0;
  root.visit_count = 1;
  root.parent = -1;
  tree.nodes.push_back(root);

  std::unordered_map<std::string, int> edge_index;
  RolloutBase base(model);

  for (int rollout = 0; rollout < cfg.n_rollouts; ++rollout) {
    auto path = rollout_from(base, model, seed_tokens, cfg.max_depth, cfg.temperature, rng);
    int parent_idx = 0;
    for (std::size_t d = seed_tokens.size(); d < path.size(); ++d) {
      const std::uint32_t tok = path[d];
      const std::string key = std::to_string(parent_idx) + "->" + std::to_string(tok);
      auto it = edge_index.find(key);
      int child_idx;
      if (it == edge_index.end()) {
        TreeNode child;
        child.token_id = tok;
        child.depth = tree.nodes[static_cast<std::size_t>(parent_idx)].depth + 1;
        child.parent = parent_idx;
        child.visit_count = 1;
        // The distribution after this node's whole context, path[0..d].
        base.sync(std::vector<std::uint32_t>(path.begin(), path.begin() + static_cast<std::ptrdiff_t>(d) + 1));
        child.active_query_score = active_query_from_log_probs(log_of(model.distribution()));
        // No model behind this interface reports an epistemic variance (hp's was always 0).
        child.epistemic_var = 0.0;
        child.leaf_value = 0.0;
        child.drift_flag = child.epistemic_var > cfg.drift_threshold;
        child_idx = static_cast<int>(tree.nodes.size());
        tree.nodes.push_back(child);
        tree.nodes[static_cast<std::size_t>(parent_idx)].children.push_back(child_idx);
        edge_index[key] = child_idx;
      } else {
        child_idx = it->second;
        tree.nodes[static_cast<std::size_t>(child_idx)].visit_count += 1;
      }
      parent_idx = child_idx;
    }
    tree.nodes[static_cast<std::size_t>(parent_idx)].visit_count += 1;
  }

  for (auto& node : tree.nodes) {
    if (node.parent >= 0 && node.visit_count > 0) {
      const int parent_visits = tree.nodes[static_cast<std::size_t>(node.parent)].visit_count;
      node.transition_prob =
          parent_visits > 0 ? static_cast<double>(node.visit_count) / static_cast<double>(parent_visits) : 0.0;
      node.transition_prob += cfg.exploration_weight * node.active_query_score /
                             static_cast<double>(std::max(node.visit_count, 1));
    }
  }

  return tree;
}

}  // namespace cypha::forecast
