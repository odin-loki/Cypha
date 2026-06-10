#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/hebbian_graph.hpp"

namespace cypha::cyphalm {

/// Real FFT circulant state transition: O(D log D) when use_spectral_pde is enabled.
std::vector<double> spectral_step(const std::vector<double>& state, const std::vector<double>& kernel);

struct CellAISSMConfig {
  int d_input = 64;
  int d_state = 128;
  double tau_fast = 10.0;
  double tau_slow = 100.0;
  int n_layers = 2;
  int seed = 0;
  bool use_spectral_pde = true;
  bool use_multiscale = true;
  bool use_sparse_hebbian = true;
};

/// Multi-scale rank-2 SSM (CellAI): fast/slow tracks per layer, optional spectral PDE.
class CellAISSM {
 public:
  explicit CellAISSM(CellAISSMConfig cfg = {});

  void reset();
  void reset_fast_only();
  void reset_slow_only();

  /// Single timestep: embedding e_t -> concatenated layer context vectors.
  std::vector<double> step(const std::vector<double>& e_t);

  /// Run step() for each row of embeddings (T x d_input).
  std::vector<std::vector<double>> process_sequence(
      const std::vector<std::vector<double>>& embeddings);

  int d_input() const { return cfg_.d_input; }
  int d_state() const { return cfg_.d_state; }
  int n_layers() const { return cfg_.n_layers; }
  int context_dim() const { return 2 * cfg_.d_state * cfg_.n_layers; }

  const std::vector<std::vector<double>>& h_states() const { return h_; }
  const std::vector<std::vector<double>>& s_states() const { return s_; }

  /// Mean-pool fast states across layers (for hierarchical slow tier).
  std::vector<double> mean_fast_state() const;

  void sparse_hebbian_update(const std::vector<double>& pre, const std::vector<double>& post,
                             double lr, int layer);

  /// Tier-4: attach lateral Hebbian graph (matches `wire_cellai` / `_hebb_graph`).
  void enable_hebb_graph(const HebbianGraphConfig& cfg);
  bool has_hebb_graph() const { return hebb_graph_ != nullptr; }

  /// Parity / checkpoint load: replace generated projection weights.
  void set_projection_weights(int layer, const std::vector<double>& w_fast,
                              const std::vector<double>& w_slow);

  double lambda_fast() const { return lambda_fast_; }
  /// Truncated BPTT nudge on layer-0 fast weights (matches Python ``_bptt_ssm_update``).
  void apply_bptt_fast_layer0(const std::vector<double>& grad_h, const std::vector<double>& e,
                              double ssm_lr, double scale = 0.001);
  /// Apply pre-averaged outer-product delta to layer-0 fast weights (``W_fast[0] -= lr * scale * avg``).
  void apply_bptt_delta_avg(const std::vector<double>& avg_delta, double ssm_lr, double scale = 0.001);

  nlohmann::json get_state() const;
  void set_state(const nlohmann::json& state);

 private:
  CellAISSMConfig cfg_;
  double lambda_fast_{};
  double lambda_slow_{};

  std::vector<int> layer_input_dims_;
  std::vector<std::vector<double>> W_fast_;
  std::vector<std::vector<double>> W_slow_;
  std::vector<std::vector<double>> a_kernel_fast_;
  std::vector<std::vector<double>> a_kernel_slow_;
  std::vector<double> alpha_;
  std::vector<std::vector<double>> W_hebb_;

  std::vector<std::vector<double>> h_;
  std::vector<std::vector<double>> s_;
  std::unique_ptr<HebbianGraph> hebb_graph_;

  static std::vector<double> matvec(const std::vector<double>& mat, int rows, int cols,
                                    const std::vector<double>& x);
  static double clip(double v, double lo, double hi);
};

}  // namespace cypha::cyphalm
