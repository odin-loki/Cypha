#pragma once

/// Growing Neural Gas auxiliary latent prototypes (mirrors cypha_som.gng_expert.GNGExpertManager).

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace cypha::som {

struct GNGExpertConfig {
  double eps_b{0.05};
  double eps_n{0.006};
  int lam{100};
  int age_max{50};
  double alpha_gng{0.5};
  int max_nodes{256};
  std::uint64_t seed{42};
};

class GNGExpertManager {
 public:
  explicit GNGExpertManager(int d, GNGExpertConfig cfg = {});

  int dim() const { return d_; }
  int step_count() const { return step_count_; }

  /// BMU node id after GNG update.
  int step(const std::vector<double>& x);

  int node_count() const { return static_cast<int>(nodes_.size()); }

  /// Stack of prototype weight vectors (row-major).
  std::vector<std::vector<double>> get_prototypes() const;

  /// GRIA controller: force a split near ``node_id`` (mirrors Python ``force_insert``).
  void force_insert(int node_id);

  /// GRIA controller: merge ``node_id`` into a neighbor (mirrors Python ``merge_with_nearest``).
  void merge_with_nearest(int node_id);

 private:
  void init_two_nodes();
  void add_edge(int i, int j);
  std::vector<int> neighbors(int i) const;
  std::pair<int, int> two_closest(const std::vector<double>& x) const;
  void prune_old_edges();
  void remove_node(int nid);
  void insert_node();
  void decay_errors();

  int d_;
  double eps_b_;
  double eps_n_;
  int lam_;
  int age_max_;
  double alpha_gng_;
  int max_nodes_;
  std::uint64_t seed_;
  int step_count_{0};
  int next_id_{0};
  std::map<int, std::vector<double>> nodes_;
  std::map<int, double> errors_;
  std::map<std::pair<int, int>, int> edges_;
};

}  // namespace cypha::som
