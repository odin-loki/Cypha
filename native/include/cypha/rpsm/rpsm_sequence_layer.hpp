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

  /// Phase -1 (RPSM_UPGRADE_PLAN.md Fix 1, RESEARCH_STATUS.md:393): replace the fixed
  /// ``alpha_carry`` blend weight in ``hierarchy_update()`` with ``gria_alpha_spectral(Psi)``,
  /// computed each step from the top singular value of the previous step's hierarchy state
  /// ``psi_rows_`` (L x D). Opt-in: default-off preserves the exact Phase 0/0b behaviour.

  bool use_spectral_alpha = false;

  /// Phase -1 Fix 2: normalise the SGD learning rate used inside ``train_step`` by the
  /// Frobenius norm of the current step's multi-level prediction error (``eta = eta_base /
  /// (||E||_F + eps)``), per RPSM_IMPLEMENTATION.md:46-53. Opt-in: default-off preserves the
  /// exact Phase 0/0b behaviour.

  bool use_normalized_eta = false;

  /// Safety clamp on the normalised-eta scale factor (``eta / lr``) to avoid a divide-by-near-zero
  /// blowup once the hierarchy error converges toward 0. Not part of the original spec formula;
  /// added because the spec's own training notes assume a grad-clip-equivalent safeguard exists.

  double eta_norm_max_scale = 10.0;

  /// §14 (RPSM_UPGRADE_PLAN.md §13.6/§13.7(a)): number of steps between real
  /// backprop-through-time passes through this layer's own hierarchy recurrence
  /// (`h_levels_`, the SSM-like carried state `hierarchy_update()` transforms every step).
  /// Before §14, `train_step` updated `w_up_`/`w_enc_`/`w_carry_` from *only* the current
  /// step's local activations -- zero credit assignment for how a hidden-state decision several
  /// steps ago affected the current loss, *and* from a couple of latent timing bugs (stale
  /// `work_err_`, post-overwrite `h_levels_`) that `bptt_backward_and_apply()` also fixes by
  /// construction regardless of window size. Every `bptt_window` calls, `train_step` now runs a
  /// proper reverse-time pass that accumulates the *recursive* gradient of the whole window's
  /// loss through the state carry, then applies one *averaged* update (mean gradient x mean
  /// per-step effective_lr over the window) -- i.e. this is non-overlapping batched gradient
  /// accumulation, not a sliding/streaming truncated-BPTT that updates every step from a
  /// K-step lookback (that variant would need every step to keep an O(window) forward-state
  /// cache alive simultaneously and was out of scope here; see RPSM_UPGRADE_PLAN.md §14).
  ///
  /// Default is intentionally **1** (i.e. every step *is* its own window: apply the correctly-
  /// derived local gradient immediately, no batching, no multi-step credit assignment), *not*
  /// the hybrid GRIA+LSTM path's `CyphaLMConfig::bptt_steps` production default of 64
  /// (cyphalm_d17_wikitext.json). RPSM_UPGRADE_PLAN.md §14 measured `bptt_window` in
  /// {1,2,4,8,16,32,64} at n_train=5000 and found eval BPC degrades *monotonically* with window
  /// size (4.75 at 1 -> 6.02 at 64, i.e. batching hurts, it does not help, for this layer/lr
  /// regime) -- the opposite of the hybrid path's own measured effect of its 64-step window.
  /// `window=1` is the only setting that does not risk regressing the D21 profile's shipped BPC
  /// relative to pre-§14 behaviour; larger windows are kept fully implemented, wired, and
  /// finite-difference-verified (`native_rpsm_bptt_grad_finite_diff`, and via the
  /// `CYPHALM_RPSM_BPTT_WINDOW` env var override in `CyphaLMModel::init_components()`) for
  /// further research, but are a deliberate *opt-in*, not the default.
  int bptt_window = 1;

};

/// Fix 1 helper: top singular value of the L x D hierarchy state ``psi`` (row-major), normalised
/// by ``sqrt(state_dim)`` and squashed into ``[0.3, 0.6]`` (edge-of-chaos target band, matching
/// RPSM_IMPLEMENTATION.md:44 and the existing ``alpha_init=0.485`` convention in
/// ``cypha_cell_hypothesis.cpp``). Computed via power iteration on the L x L Gram matrix
/// ``Psi @ Psi^T`` (cheap: L is the small hierarchy depth, e.g. 4-32, not D).
double gria_alpha_spectral(const double* psi_row_major, int n_levels, int state_dim);



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
  const std::vector<double>& w_enc() const { return w_enc_; }
  const std::vector<double>& w_carry() const { return w_carry_; }

  /// Test-only mutators (mirror CellAISSM's `w_fast_layer0_mut()` convention): let a
  /// finite-difference/regression test inject a specific weight snapshot so numeric and
  /// analytic gradients can be compared against an identical, controlled forward trajectory.
  std::vector<double>& w_up_mut() { return w_up_; }
  std::vector<double>& w_enc_mut() { return w_enc_; }
  std::vector<double>& w_carry_mut() { return w_carry_; }

  const RpsmGlobalMemory& global_memory() const { return global_memory_; }

  /// §14 BPTT: true only immediately after the `train_step` call that just completed a window
  /// flush (i.e. exactly every `cfg_.bptt_window` calls). When true, `bptt_window_input_grads()`
  /// holds one gradient vector per step of the just-flushed window (length `state_dim` each), in
  /// original chronological order (index 0 = oldest step of the window, index N-1 = newest).
  /// This is a *deeper* signal than `input_grad()` (which stays the pre-existing single-step
  /// local gradient, unchanged, for backward compatibility with the Finding #2 embed-backprop
  /// path): it includes the hierarchy/injection paths' contribution to d(loss)/d(input), which
  /// `input_grad()` has never included, plus genuine multi-step recursive credit.
  bool bptt_window_flushed() const { return bptt_flushed_this_call_; }

  const std::vector<std::vector<double>>& bptt_window_input_grads() const {
    return bptt_flush_input_grads_;
  }

  IzaacActivationMix activation_mix() const { return activation_mix_; }

  int n_classes() const { return cfg_.n_classes; }

  int n_levels() const { return cfg_.n_levels; }



 private:

  /// §14 BPTT: one snapshot of everything a reverse-time pass needs to reconstruct the exact
  /// local Jacobians used at this step's forward pass, without re-running any RNG-dependent or
  /// weight-dependent computation (weights are held fixed for the whole window; see
  /// `bptt_backward_and_apply()`). `up_pre`/`down_pre`/`blended_pre` are *pre-activation* values
  /// (activation_derivative() takes the pre-activation argument, so these must be the raw
  /// matvec/blend outputs, not the post-activation `work_up_`/`work_down_`/`h` the forward loop
  /// already overwrites in place).
  struct BpttStepCache {
    int in_n = 0;                                  // input_dim actually used this step, clamped >=0.
    std::vector<double> input;                     // length state_dim, zero-padded/truncated.
    std::vector<std::vector<double>> h_inj;        // [n_levels][state_dim], state *entering*
                                                     // hierarchy_update (post-injection/encode).
    std::vector<std::vector<double>> up_pre;        // [n_levels][state_dim].
    std::vector<std::vector<double>> down_pre;      // [n_levels][state_dim].
    std::vector<std::vector<double>> blended_pre;   // [n_levels][state_dim].
    std::vector<double> inj_acc;                    // length state_dim; level-independent
                                                     // pre-activation injection accumulator.
    std::vector<double> enc_pre;                    // length feat_dim.
    std::vector<double> enc_grad;                   // length feat_dim; classifier->feat gradient
                                                     // *as it stood at this step* (computed from
                                                     // psi_.mu before that step's own SGD update).
    double alpha = 0.0;                             // hierarchy_update's alpha_carry/spectral-a
                                                     // used this step.
    double effective_lr = 0.0;                      // this step's (possibly normalised-eta) lr.
  };

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

  // §14 BPTT window state.
  std::vector<BpttStepCache> bptt_cache_;
  int bptt_fill_{0};
  bool bptt_flushed_this_call_{false};
  std::vector<std::vector<double>> bptt_flush_input_grads_;

  double apply_activation(double x) const;

  void inject_input_multilevel(const double* input, int input_dim, BpttStepCache* cache = nullptr);

  void encode_level0_features(const double* input, int input_dim);

  double hierarchy_update(BpttStepCache* cache = nullptr);

  /// §14 BPTT: reverse-time pass over the just-filled `bptt_cache_` window. Accumulates the
  /// recursive gradient of the *whole window's* loss w.r.t. `w_up_`/`w_enc_`/`w_carry_` (and,
  /// per-step, w.r.t. that step's `input`, exposed via `bptt_flush_input_grads_`), then applies
  /// one averaged SGD update. See rpsm_sequence_layer.cpp for the full derivation.
  void bptt_backward_and_apply();

  static void log_softmax_row(const double* logits, int k, double* log_out);

};



}  // namespace cypha::rpsm

