#pragma once

/// Plastic Graph Machine (PGM) native cell — hierarchical log-N addressing,
/// sparse Hebbian slot graph, T1 chunk buffer, beam-2 max-plus retrieval.
/// Complexity: O(m log n) via branching factor ``n_sub`` and ``levels`` (no dense O(n²) attention).

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace cypha::cyphalm {

struct PGMCellConfig {
  int d_input = 64;
  /// Branching factor b for hierarchical addressing (N = n_sub^levels).
  int n_sub = 16;
  /// Hierarchy depth L; total slots N = n_sub^levels (capped internally).
  int levels = 3;
  /// Optional hard slot count; 0 = use n_sub^levels.
  int n_slots = 0;
  int hidden = 64;
  int chunk_len = 16;
  int topk = 4;
  int beam = 2;
  int rehash_t = 16;
  int hops = 2;
  std::uint64_t seed = 42;
};

/// Nested T1 (chunk buffer) + T2 (sparse edges) + T3 (slot content) memory cell.
class PGMCell {
 public:
  explicit PGMCell(PGMCellConfig cfg = {});

  void reset();

  /// Ingest token embedding ``x`` (length ``d_input``); return context / hidden state.
  std::vector<double> step(const std::vector<double>& x);

  int context_dim() const { return cfg_.hidden; }
  int d_input() const { return cfg_.d_input; }
  int n_slots() const { return n_slots_; }
  int levels() const { return cfg_.levels; }
  int n_sub() const { return cfg_.n_sub; }
  std::size_t edge_count() const;
  std::size_t occupied_count() const { return occupied_.size(); }
  const PGMCellConfig& config() const { return cfg_; }

  /// Checkpoint serialization (codebooks, sparse V/edges, T1, h).
  nlohmann::json get_state() const;
  void set_state(const nlohmann::json& state);

 private:
  struct EdgeNbr {
    int dst = 0;
    double w = 0.0;
  };

  PGMCellConfig cfg_;
  int n_slots_ = 0;

  /// Per-level codebooks: levels × n_sub × d_input (unit rows).
  std::vector<std::vector<std::vector<double>>> codebooks_;
  /// Sparse content bank (T3): slot -> vector (d_input).
  std::unordered_map<int, std::vector<double>> V_;
  std::unordered_map<int, bool> occupied_;
  /// Sparse directed edges (T2): src -> top-k neighbors.
  std::unordered_map<int, std::vector<EdgeNbr>> edges_;
  /// T1 recency buffer.
  std::vector<std::vector<double>> t1_;
  std::vector<double> h_;
  double eta_ = 1.0;
  double sim_thresh_ = 0.9;

  void init_codebooks();
  static double dot(const std::vector<double>& a, const std::vector<double>& b);
  static void l2_normalize(std::vector<double>& v);
  std::vector<double> project_input(const std::vector<double>& x) const;

  /// Hierarchical primary address: O(levels · n_sub · d).
  int primary_slot(const std::vector<double>& k) const;
  /// Beam-expand hierarchical candidates for rehash; returns up to ``t`` slots.
  std::vector<int> candidate_slots(const std::vector<double>& k, int t) const;
  int assign_rehash(const std::vector<double>& k);
  int locate_rehash(const std::vector<double>& q) const;
  void store_content(int slot, const std::vector<double>& x);
  void write_link(const std::vector<double>& a, const std::vector<double>& b);
  void sparsify_row(int src);
  void consolidate_adjacent();
  /// Beam-``beam`` max-plus hops; returns best tip content (or zeros).
  std::vector<double> retrieve(const std::vector<double>& q) const;
};

}  // namespace cypha::cyphalm
