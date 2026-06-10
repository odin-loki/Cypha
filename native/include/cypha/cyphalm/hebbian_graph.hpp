#pragma once

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

namespace cypha::cyphalm {

/// Sparse lateral Hebbian graph (Cypha Tests 2D / `DynamicHebbianGraph`).
struct HebbianGraphConfig {
  int n{0};
  double eta_edge{0.01};
  double lambda_decay{0.001};
  double theta_prune{0.01};
  double theta_form{0.3};
  double gamma{0.1};
  /// Limit diffusion to top-k neighbors per node by edge weight (0 = use all edges).
  int k_neighbors{0};
};

class HebbianGraph {
 public:
  explicit HebbianGraph(const HebbianGraphConfig& cfg);

  int size() const { return cfg_.n; }
  const HebbianGraphConfig& config() const { return cfg_; }

  /// `ctx' = ctx + gamma * (D^{-1} A) ctx` with optional per-node k-sparsity.
  void diffuse(const double* ctx, double* ctx_out) const;
  std::vector<double> diffuse(const std::vector<double>& ctx) const;

  /// Hebbian edge update from co-activation `a[i]*a[j]`.
  void update(const double* activations);

  std::size_t edge_count() const { return edges_.size(); }

 private:
  using EdgeKey = std::pair<int, int>;

  HebbianGraphConfig cfg_;
  std::map<EdgeKey, double> edges_;

  void ring_init();
  static EdgeKey edge_key(int i, int j);
  void build_normalized_adjacency(std::vector<double>& adj_row_major) const;
};

}  // namespace cypha::cyphalm
