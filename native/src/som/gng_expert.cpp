#include "cypha/som/gng_expert.hpp"

#include "cypha/numpy_default_rng.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cypha::som {

namespace {

double sq_dist(const std::vector<double>& x, const std::vector<double>& w) {
  double s = 0.0;
  const int n = static_cast<int>(std::min(x.size(), w.size()));
  for (int i = 0; i < n; ++i) {
    const double d = x[static_cast<std::size_t>(i)] - w[static_cast<std::size_t>(i)];
    s += d * d;
  }
  for (std::size_t i = static_cast<std::size_t>(n); i < x.size(); ++i) {
    s += x[i] * x[i];
  }
  for (std::size_t i = static_cast<std::size_t>(n); i < w.size(); ++i) {
    s += w[i] * w[i];
  }
  return s;
}

}  // namespace

GNGExpertManager::GNGExpertManager(int d, GNGExpertConfig cfg)
    : d_(d),
      eps_b_(cfg.eps_b),
      eps_n_(cfg.eps_n),
      lam_(std::max(1, cfg.lam)),
      age_max_(cfg.age_max),
      alpha_gng_(cfg.alpha_gng),
      max_nodes_(std::max(2, cfg.max_nodes)),
      seed_(cfg.seed) {
  if (d_ <= 0) {
    throw std::invalid_argument("GNGExpertManager: d must be positive");
  }
  init_two_nodes();
}

void GNGExpertManager::init_two_nodes() {
  cypha::NumpyDefaultRng rng(static_cast<int>(seed_));
  nodes_.clear();
  errors_.clear();
  edges_.clear();
  next_id_ = 0;
  step_count_ = 0;

  for (int i = 0; i < 2; ++i) {
    const int nid = next_id_++;
    nodes_[nid] = std::vector<double>(static_cast<std::size_t>(d_), 0.0);
    for (int j = 0; j < d_; ++j) {
      nodes_[nid][static_cast<std::size_t>(j)] = rng.normal(0.0, 1.0) * 0.1;
    }
    errors_[nid] = 0.0;
  }
  auto it = nodes_.begin();
  const int a = it->first;
  ++it;
  const int b = it->first;
  add_edge(a, b);
}

void GNGExpertManager::add_edge(int i, int j) {
  if (i == j) {
    return;
  }
  const int lo = std::min(i, j);
  const int hi = std::max(i, j);
  edges_[{lo, hi}] = 0;
}

std::vector<int> GNGExpertManager::neighbors(int i) const {
  std::vector<int> out;
  for (const auto& kv : edges_) {
    if (kv.first.first == i) {
      out.push_back(kv.first.second);
    } else if (kv.first.second == i) {
      out.push_back(kv.first.first);
    }
  }
  return out;
}

std::pair<int, int> GNGExpertManager::two_closest(const std::vector<double>& x) const {
  int best_i = -1;
  double best_d = std::numeric_limits<double>::infinity();
  int second_i = -1;
  double second_d = std::numeric_limits<double>::infinity();
  for (const auto& kv : nodes_) {
    const double d = sq_dist(x, kv.second);
    if (d < best_d) {
      second_i = best_i;
      second_d = best_d;
      best_i = kv.first;
      best_d = d;
    } else if (d < second_d) {
      second_i = kv.first;
      second_d = d;
    }
  }
  if (second_i < 0) {
    second_i = best_i;
  }
  return {best_i, second_i};
}

void GNGExpertManager::prune_old_edges() {
  std::vector<std::pair<int, int>> dead;
  for (const auto& kv : edges_) {
    if (kv.second > age_max_) {
      dead.push_back(kv.first);
    }
  }
  for (const auto& k : dead) {
    edges_.erase(k);
  }

  std::vector<int> isolated;
  for (const auto& kv : nodes_) {
    if (neighbors(kv.first).empty() && static_cast<int>(nodes_.size()) > 2) {
      isolated.push_back(kv.first);
    }
  }
  for (int nid : isolated) {
    remove_node(nid);
  }
}

void GNGExpertManager::remove_node(int nid) {
  nodes_.erase(nid);
  errors_.erase(nid);
  std::vector<std::pair<int, int>> dead;
  for (const auto& kv : edges_) {
    if (kv.first.first == nid || kv.first.second == nid) {
      dead.push_back(kv.first);
    }
  }
  for (const auto& k : dead) {
    edges_.erase(k);
  }
}

void GNGExpertManager::insert_node() {
  if (errors_.empty()) {
    return;
  }
  int q = errors_.begin()->first;
  double q_err = errors_.begin()->second;
  for (const auto& kv : errors_) {
    if (kv.second > q_err) {
      q = kv.first;
      q_err = kv.second;
    }
  }
  const std::vector<int> nbrs = neighbors(q);
  if (nbrs.empty()) {
    return;
  }
  int f = nbrs[0];
  double f_err = errors_.count(f) ? errors_.at(f) : 0.0;
  for (int j : nbrs) {
    const double e = errors_.count(j) ? errors_.at(j) : 0.0;
    if (e > f_err) {
      f = j;
      f_err = e;
    }
  }

  const int r = next_id_++;
  nodes_[r] = std::vector<double>(static_cast<std::size_t>(d_), 0.0);
  for (int j = 0; j < d_; ++j) {
    nodes_[r][static_cast<std::size_t>(j)] =
        0.5 * (nodes_[q][static_cast<std::size_t>(j)] + nodes_[f][static_cast<std::size_t>(j)]);
  }
  errors_[r] = errors_[q];
  errors_[q] *= alpha_gng_;
  errors_[f] *= alpha_gng_;
  add_edge(q, r);
  add_edge(r, f);
  const int lo = std::min(q, f);
  const int hi = std::max(q, f);
  edges_.erase({lo, hi});
}

void GNGExpertManager::decay_errors() {
  for (auto& kv : errors_) {
    kv.second *= 0.995;
  }
}

int GNGExpertManager::step(const std::vector<double>& x) {
  if (static_cast<int>(x.size()) != d_) {
    throw std::invalid_argument("GNGExpertManager::step: dimension mismatch");
  }

  const auto [bmu, bmu2] = two_closest(x);

  for (int j = 0; j < d_; ++j) {
    nodes_[bmu][static_cast<std::size_t>(j)] +=
        eps_b_ * (x[static_cast<std::size_t>(j)] - nodes_[bmu][static_cast<std::size_t>(j)]);
  }
  for (int nb : neighbors(bmu)) {
    for (int j = 0; j < d_; ++j) {
      nodes_[nb][static_cast<std::size_t>(j)] +=
          eps_n_ * (x[static_cast<std::size_t>(j)] - nodes_[nb][static_cast<std::size_t>(j)]);
    }
  }

  const double err = sq_dist(x, nodes_[bmu]);
  errors_[bmu] = (errors_.count(bmu) ? errors_[bmu] : 0.0) + err;

  add_edge(bmu, bmu2);
  const int lo = std::min(bmu, bmu2);
  const int hi = std::max(bmu, bmu2);
  edges_[{lo, hi}] = 0;

  std::vector<std::pair<int, int>> touch;
  for (const auto& kv : edges_) {
    if (kv.first.first == bmu || kv.first.second == bmu) {
      touch.push_back(kv.first);
    }
  }
  for (const auto& k : touch) {
    edges_[k] += 1;
  }

  prune_old_edges();
  step_count_ += 1;
  if (step_count_ % lam_ == 0 && static_cast<int>(nodes_.size()) < max_nodes_) {
    insert_node();
  }
  decay_errors();
  return bmu;
}

std::vector<std::vector<double>> GNGExpertManager::get_prototypes() const {
  std::vector<std::vector<double>> out;
  out.reserve(nodes_.size());
  for (const auto& kv : nodes_) {
    out.push_back(kv.second);
  }
  return out;
}

void GNGExpertManager::force_insert(int node_id) {
  const std::vector<int> nbrs = neighbors(node_id);
  if (nbrs.empty() || nodes_.count(node_id) == 0) {
    insert_node();
    return;
  }
  const int f = nbrs[0];
  const int r = next_id_++;
  nodes_[r] = std::vector<double>(static_cast<std::size_t>(d_), 0.0);
  for (int j = 0; j < d_; ++j) {
    nodes_[r][static_cast<std::size_t>(j)] =
        0.5 * (nodes_[node_id][static_cast<std::size_t>(j)] + nodes_[f][static_cast<std::size_t>(j)]);
  }
  errors_[r] = errors_.count(node_id) ? errors_.at(node_id) : 1.0;
  add_edge(node_id, r);
}

void GNGExpertManager::merge_with_nearest(int node_id) {
  if (nodes_.count(node_id) == 0) {
    return;
  }
  const std::vector<int> nbrs = neighbors(node_id);
  if (nbrs.empty()) {
    return;
  }
  const int other = nbrs[0];
  for (int j = 0; j < d_; ++j) {
    nodes_[other][static_cast<std::size_t>(j)] =
        0.5 * (nodes_[other][static_cast<std::size_t>(j)] +
               nodes_[node_id][static_cast<std::size_t>(j)]);
  }
  remove_node(node_id);
}

}  // namespace cypha::som
