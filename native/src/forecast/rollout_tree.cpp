#include "cypha/forecast/rollout_tree.hpp"

#include "cypha/cyphalm/hp_backend.hpp"
#include "cypha/infer_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

namespace cypha::forecast {
namespace {

double softmax_sample(const std::vector<double>& log_probs, double temperature, std::mt19937& rng) {
  if (log_probs.empty()) {
    return 0.0;
  }
  const int V = static_cast<int>(log_probs.size());
  double max_lp = log_probs[0];
  for (int i = 1; i < V; ++i) {
    max_lp = std::max(max_lp, log_probs[static_cast<std::size_t>(i)]);
  }
  std::vector<double> p(static_cast<std::size_t>(V));
  double sum = 0.0;
  for (int i = 0; i < V; ++i) {
    p[static_cast<std::size_t>(i)] =
        std::exp((log_probs[static_cast<std::size_t>(i)] - max_lp) / std::max(temperature, 1e-6));
    sum += p[static_cast<std::size_t>(i)];
  }
  std::uniform_real_distribution<double> u(0.0, sum);
  double draw = u(rng);
  double acc = 0.0;
  for (int i = 0; i < V; ++i) {
    acc += p[static_cast<std::size_t>(i)];
    if (draw <= acc) {
      return static_cast<double>(i);
    }
  }
  return static_cast<double>(V - 1);
}

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
// reset_context(), which in the hp wrapper builds a new, untrained predictor: the tree sampled from an untrained
// model, about 1,200 production-size rebuilds a run). Learning is off while rollouts run, so sampled bytes teach the
// model nothing, and every rollout and node starts from the same state: one fresh stream on the trained model,
// restored exactly before each (hp::StreamRewind, as beam decoding does), so nothing carries from one rollout into
// the next. The model is left in that state, with learning as it was.
class RolloutBase {
 public:
  explicit RolloutBase(cyphalm::CyphaLMModel& model)
      : model_(model), hp_(model.hp_backend()), prev_learning_(hp_.learning()) {
    hp_.set_learning(false);
    model_.reset_stream(false);
    rewind_.emplace(hp_.all_predictors());
    if (hp_.has_neural()) {
      nn_saved_ = hp_.neural_states();
    }
    session_saved_ = hp_.session_size();
  }
  ~RolloutBase() {
    restore();
    rewind_.reset();
    hp_.set_learning(prev_learning_);
  }
  RolloutBase(const RolloutBase&) = delete;
  RolloutBase& operator=(const RolloutBase&) = delete;

  // The base state, then the model reads every token of context but the last; predict_next(last) then gives the
  // distribution after the whole context.
  void sync_all_but_last(const std::vector<std::uint32_t>& context) {
    restore();
    for (std::size_t i = 0; i + 1 < context.size(); ++i) {
      (void)model_.predict_next(context[i]);
    }
  }

 private:
  void restore() {
    rewind_->rewind();
    if (hp_.has_neural()) {
      hp_.set_neural_states(nn_saved_);
    }
    hp_.truncate_session(session_saved_);
    hp_.invalidate_scoring_cache();
  }

  cyphalm::CyphaLMModel& model_;
  cyphalm::HpSequenceBackend& hp_;
  bool prev_learning_;
  std::optional<hp::StreamRewind> rewind_;
  std::vector<cyphalm::ByteNeuralExpert::State> nn_saved_;
  std::size_t session_saved_ = 0;
};

std::vector<std::uint32_t> rollout_from(RolloutBase& base, cyphalm::CyphaLMModel& model,
                                        const std::vector<std::uint32_t>& seed, int horizon, double temperature,
                                        std::mt19937& rng) {
  std::vector<std::uint32_t> path = seed;
  if (path.empty()) {
    return path;
  }
  base.sync_all_but_last(seed);
  for (int step = 0; step < horizon; ++step) {
    const auto pred = model.predict_next(path.back());
    if (pred.log_probs.empty()) {
      break;
    }
    const int next = static_cast<int>(softmax_sample(pred.log_probs, temperature, rng));
    path.push_back(static_cast<std::uint32_t>(next));
  }
  return path;
}

}  // namespace

std::vector<std::uint32_t> ancestral_rollout(cyphalm::CyphaLMModel& model,
                                             const std::vector<std::uint32_t>& seed, int horizon,
                                             double temperature, std::mt19937& rng) {
  RolloutBase base(model);
  return rollout_from(base, model, seed, horizon, temperature, rng);
}

RolloutTree generate_rollout_tree(cyphalm::CyphaLMModel& model, const EventVocabulary& vocab,
                                  const std::vector<std::uint32_t>& seed_tokens,
                                  const RolloutTreeConfig& cfg, std::mt19937& rng) {
  (void)vocab;
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
        // The distribution after this node's whole context, path[0..d] (it used to read path[d - 1] twice).
        base.sync_all_but_last(std::vector<std::uint32_t>(path.begin(), path.begin() + static_cast<std::ptrdiff_t>(d) + 1));
        const auto pred = model.predict_next(path[d]);
        child.active_query_score = active_query_from_log_probs(pred.log_probs);
        child.epistemic_var = pred.epistemic_var;
        child.leaf_value = -pred.epistemic_var;
        child.drift_flag = pred.epistemic_var > cfg.drift_threshold;
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
