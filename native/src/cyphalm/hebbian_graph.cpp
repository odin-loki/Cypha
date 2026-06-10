#include "cypha/cyphalm/hebbian_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cypha::cyphalm {

HebbianGraph::HebbianGraph(const HebbianGraphConfig& cfg) : cfg_(cfg) { ring_init(); }

HebbianGraph::EdgeKey HebbianGraph::edge_key(int i, int j) {
  if (i > j) {
    std::swap(i, j);
  }
  return EdgeKey{i, j};
}

void HebbianGraph::ring_init() {
  edges_.clear();
  for (int i = 0; i < cfg_.n; ++i) {
    const int j = (i + 1) % cfg_.n;
    edges_[edge_key(i, j)] = 0.5;
  }
}

void HebbianGraph::update(const double* activations) {
  if (activations == nullptr || cfg_.n <= 0) {
    return;
  }
  const int n = cfg_.n;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double co = activations[static_cast<std::size_t>(i)] * activations[static_cast<std::size_t>(j)];
      const EdgeKey key = edge_key(i, j);
      auto it = edges_.find(key);
      if (it != edges_.end()) {
        double e = it->second + cfg_.eta_edge * co;
        e *= 1.0 - cfg_.lambda_decay;
        if (e < cfg_.theta_prune) {
          edges_.erase(it);
        } else {
          it->second = e;
        }
      } else if (co > cfg_.theta_form) {
        edges_[key] = cfg_.theta_form;
      }
    }
  }
}

void HebbianGraph::build_normalized_adjacency(std::vector<double>& adj_row_major) const {
  const int n = cfg_.n;
  adj_row_major.assign(static_cast<std::size_t>(n * n), 0.0);
  for (const auto& kv : edges_) {
    const int i = kv.first.first;
    const int j = kv.first.second;
    const double w = kv.second;
    adj_row_major[static_cast<std::size_t>(i * n + j)] = w;
    adj_row_major[static_cast<std::size_t>(j * n + i)] = w;
  }

  if (cfg_.k_neighbors > 0 && cfg_.k_neighbors < n) {
    for (int i = 0; i < n; ++i) {
      std::vector<std::pair<double, int>> neighbors;
      neighbors.reserve(static_cast<std::size_t>(n));
      for (int j = 0; j < n; ++j) {
        if (i == j) {
          continue;
        }
        neighbors.emplace_back(adj_row_major[static_cast<std::size_t>(i * n + j)], j);
      }
      std::sort(neighbors.begin(), neighbors.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          adj_row_major[static_cast<std::size_t>(i * n + j)] = 0.0;
        }
      }
      const int keep = std::min(cfg_.k_neighbors, static_cast<int>(neighbors.size()));
      for (int k = 0; k < keep; ++k) {
        const int j = neighbors[static_cast<std::size_t>(k)].second;
        const double w = neighbors[static_cast<std::size_t>(k)].first;
        adj_row_major[static_cast<std::size_t>(i * n + j)] = w;
        adj_row_major[static_cast<std::size_t>(j * n + i)] = w;
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    double deg = 0.0;
    for (int j = 0; j < n; ++j) {
      deg += adj_row_major[static_cast<std::size_t>(i * n + j)];
    }
    deg += 1e-12;
    for (int j = 0; j < n; ++j) {
      adj_row_major[static_cast<std::size_t>(i * n + j)] /= deg;
    }
  }
}

void HebbianGraph::diffuse(const double* ctx, double* ctx_out) const {
  if (ctx == nullptr || ctx_out == nullptr || cfg_.n <= 0) {
    return;
  }
  const int n = cfg_.n;
  std::vector<double> adj;
  build_normalized_adjacency(adj);

  std::vector<double> ax(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j < n; ++j) {
      s += adj[static_cast<std::size_t>(i * n + j)] * ctx[static_cast<std::size_t>(j)];
    }
    ax[static_cast<std::size_t>(i)] = s;
  }

  for (int i = 0; i < n; ++i) {
    const double v = ctx[static_cast<std::size_t>(i)] + cfg_.gamma * ax[static_cast<std::size_t>(i)];
    if (!std::isfinite(v)) {
      ctx_out[static_cast<std::size_t>(i)] = ctx[static_cast<std::size_t>(i)];
    } else {
      ctx_out[static_cast<std::size_t>(i)] = v;
    }
  }
}

std::vector<double> HebbianGraph::diffuse(const std::vector<double>& ctx) const {
  std::vector<double> out(static_cast<std::size_t>(cfg_.n), 0.0);
  const int n = cfg_.n;
  std::vector<double> padded(static_cast<std::size_t>(n), 0.0);
  const int copy_n = std::min(n, static_cast<int>(ctx.size()));
  for (int i = 0; i < copy_n; ++i) {
    padded[static_cast<std::size_t>(i)] = ctx[static_cast<std::size_t>(i)];
  }
  diffuse(padded.data(), out.data());
  return out;
}

}  // namespace cypha::cyphalm
