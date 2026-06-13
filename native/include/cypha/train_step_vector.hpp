#pragma once

#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/replay_buffer.hpp"

#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha {

struct TrainStepParams {
  double enc_lr{0.002};
  /// Priority replay trigger; if <= 0, skip sampling (matches Python ``CyphaDIF`` when ``replay_ratio <= 0``).
  double replay_ratio{0.30};
  int replay_cap{10000};
  /// Python `_ALIGN_EVERY` (encoder `align_to_offsets`).
  int align_every{500};
  /// If > 0, call `auto_recalibrate_temperature` when `total_steps % this == 0` (Python has no per-step hook; default off).
  int temp_recalib_every{0};
};

/// Optional session state for REST / full `train_step` parity (may be nullptr for harnesses).
struct TrainStepExtras {
  int* total_steps{nullptr};
  double* ood_sigma{nullptr};
  double* llr_ema{nullptr};
  /// Optional recorded U(0,1) stream: replay gate then ``len_`` draws per replay ``sample`` (parity vs Python).
  /// When non-null, replay does not use ``rng`` for these draws.
  const double* replay_u01{nullptr};
  std::size_t replay_u01_len{0};
  std::size_t* replay_u01_pos{nullptr};
  /// Optional Nyström kernel LLR (Python ``CyphaDIF._kernel_mem`` + ``use_kernel_llr``).
  KernelMemory* kernel_mem{nullptr};
  bool use_kernel_llr{false};
  double kernel_blend{0.5};
  /// Scales ``delta_lr`` for kernel weight updates only (default 1.0).
  double kernel_lr_scale{1.0};
};

/// `CyphaDIF.train_step` for `VectorEncoder`: memory → sync → replay.push → contrastive (if misclassified) → replay `memory.train` (unscaled lr).
double dif_train_step_vector(CyphaInferModel& infer, CyphaDifMemoryState& mem, ReplayBuffer& replay,
                             const double* x_preprocessed, int d, const std::string& y_label,
                             double world_lr_step, double delta_lr_step, double world_lr_replay,
                             double delta_lr_replay, double ood_sigma, const TrainStepParams& tsp,
                             std::mt19937& rng, int& enc_update_count, MemoryTrainMeta* meta_out,
                             TrainStepExtras* extras);

/// Python `CyphaDIF.gh_train_step`: GH LR scale on world/delta/encoder + replay uses the same scaled LRs; returns NIG diagnostics.
struct GhTrainStepResult {
  double loss{std::numeric_limits<double>::quiet_NaN()};
  double r_eff{0};
  double chi_new{0};
  double psi_new{0};
};

GhTrainStepResult dif_gh_train_step_vector(
    CyphaInferModel& infer, CyphaDifMemoryState& mem, ReplayBuffer& replay, const double* x_preprocessed, int d,
    const std::string& y_label, const std::vector<double>& gh_inv_v_clean, double gh_r_base, double chi, double psi,
    double nig_alpha, double world_lr_nominal, double delta_lr_nominal, double ood_sigma, const TrainStepParams& tsp,
    std::mt19937& rng, int& enc_update_count, MemoryTrainMeta* meta_out, TrainStepExtras* extras);

/// Run ``xs.size()`` consecutive ``dif_train_step_vector`` calls (Python ``Trainer.fit`` inner loop without Python).
/// Each ``xs[i]`` must have length ``d`` (same as model latent input dim).
[[nodiscard]] std::vector<double> dif_train_classify_sequence(
    CyphaInferModel& infer, CyphaDifMemoryState& mem, ReplayBuffer& replay,
    const std::vector<std::vector<double>>& xs, const std::vector<std::string>& labels, double world_lr_step,
    double delta_lr_step, double world_lr_replay, double delta_lr_replay, double ood_sigma, const TrainStepParams& tsp,
    std::mt19937& rng, int& enc_update_count, TrainStepExtras* extras);

/// Same as ``dif_train_classify_sequence`` but each step is ``dif_gh_train_step_vector``; ``chi`` / ``psi`` advance.
[[nodiscard]] std::vector<double> dif_gh_train_classify_sequence(
    CyphaInferModel& infer, CyphaDifMemoryState& mem, ReplayBuffer& replay,
    const std::vector<std::vector<double>>& xs, const std::vector<std::string>& labels,
    const std::vector<double>& gh_inv_v_clean, double gh_r_base, double& chi, double& psi, double nig_alpha,
    double world_lr_nominal, double delta_lr_nominal, double ood_sigma, const TrainStepParams& tsp,
    std::mt19937& rng, int& enc_update_count, TrainStepExtras* extras);

}  // namespace cypha
