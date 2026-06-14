#pragma once



#include <cstdint>

#include <vector>



#include "cypha/rpsm/psi_matrices.hpp"



namespace cypha::rpsm {



/// Izaac grammar-search stub: activation mix variants for hierarchy levels (seed-selected).

enum class IzaacActivationMix : std::uint8_t {

  TanhOnly = 0,

  TanhRelu = 1,

  TanhSigmoid = 2,

  ReluTanh = 3,

  LinearTanh = 4,

  Count = 5,

};



constexpr int kIzaacActivationMixCount = static_cast<int>(IzaacActivationMix::Count);



IzaacActivationMix select_izaac_activation_mix(std::uint64_t seed);



/// Fixed-size working-memory slot ring (M_slots scaffold).

class RpsmGlobalMemory {

 public:

  RpsmGlobalMemory(int n_slots, int dim);



  void reset();



  /// Soft-attention read into ``out`` (length ``dim``).

  void soft_read(const double* query, int query_dim, double* out) const;



  /// Ring write of one state vector; gated by ``surprise`` (prediction-error magnitude).

  void ring_write(const double* vec, int vec_dim, double surprise = 1.0);



  int n_slots() const { return n_slots_; }

  int dim() const { return dim_; }

  const std::vector<double>& slots() const { return slots_; }



 private:

  int n_slots_;

  int dim_;

  int write_head_{0};

  std::vector<double> slots_;

};



/// Tiny/Small RPSM sequence layer config (Option B scaffold).

struct RpsmSequenceConfig {

  int n_levels = 4;

  int state_dim = 128;

  int feat_dim = 64;

  int n_classes = 128;

  int n_memory_slots = 32;

  double alpha_carry = 0.5;

  double beta_memory = 0.1;

  double hierarchy_loss_weight = 0.1;

  double surprise_threshold = 0.05;

  std::uint64_t seed = 42;

  /// H19-style seed-offset init for projection matrices (``seed + 991 + matrix_id``).

  bool use_izaac_init = false;

  /// Grammar-search stub override; when ``Count`` (255), pick from ``seed``.

  IzaacActivationMix activation_mix = IzaacActivationMix::TanhOnly;

};



struct RpsmTrainStepMetrics {

  double loss = 0.0;

  double nll = 0.0;

  double hierarchy_loss = 0.0;

  double surprise = 0.0;

};



/// RPSM sequence layer: level-0 batched LLR, multi-level carry, W_up/W_down hierarchy, M_slots.

class RpsmSequenceLayer {

 public:

  explicit RpsmSequenceLayer(RpsmSequenceConfig cfg);



  void reset();



  /// One timestep. ``input`` length ``input_dim``; ``log_probs_out`` length ``n_classes`` (log space).

  /// Returns Frobenius norm of level-0 hidden state (diagnostics).

  double step(const double* input, int input_dim, double* log_probs_out);



  /// Forward + SGD on NLL + hierarchy reconstruction error; updates ``w_enc``, ``w_carry``, ``w_up``.

  RpsmTrainStepMetrics train_step(const double* input, int input_dim, int target_class, double lr);

  /// Gradient w.r.t. last ``train_step`` input (length ``state_dim``; stub for embed backprop).
  const std::vector<double>& input_grad() const { return input_grad_; }

  /// Level-0 carry (CyphaLM GRIA injection).

  const std::vector<double>& hidden() const { return h_levels_[0]; }

  const std::vector<double>& level_hidden(int level) const;

  const std::vector<std::vector<double>>& all_levels() const { return h_levels_; }

  const PsiMatrices& psi() const { return psi_; }

  const std::vector<double>& w_up() const { return w_up_; }

  const RpsmGlobalMemory& global_memory() const { return global_memory_; }

  IzaacActivationMix activation_mix() const { return activation_mix_; }

  int n_classes() const { return cfg_.n_classes; }

  int n_levels() const { return cfg_.n_levels; }



 private:

  RpsmSequenceConfig cfg_;

  IzaacActivationMix activation_mix_{IzaacActivationMix::TanhOnly};

  PsiMatrices psi_;

  std::vector<std::vector<double>> h_levels_;

  std::vector<double> psi_rows_;

  std::vector<double> w_up_;

  std::vector<double> w_enc_;

  std::vector<double> w_carry_;

  RpsmGlobalMemory global_memory_;

  std::vector<double> feat_buf_;

  std::vector<double> llr_buf_;

  std::vector<double> work_up_;

  std::vector<double> work_down_;

  std::vector<double> work_err_;

  std::vector<double> mem_read_;

  std::vector<double> enc_pre_;

  std::vector<double> enc_grad_;

  std::vector<double> input_grad_;

  double last_surprise_{0.0};



  double apply_activation(double x) const;

  void inject_input_multilevel(const double* input, int input_dim);

  void encode_level0_features(const double* input, int input_dim);

  double hierarchy_update();

  static void log_softmax_row(const double* logits, int k, double* log_out);

};



}  // namespace cypha::rpsm

