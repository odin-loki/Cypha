#include "cypha/cyphalm/lm_intelligence_monitor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "cypha/intelligence/measurers.hpp"

namespace cypha::cyphalm {

namespace {

constexpr double kEps = 1e-12;
constexpr int kTauMaxSteps = 512;

// See `LmIntelligenceMonitor::feed_causal_checkpoints` (docs/reports/
// SOFT_WORLD_CAUSAL_GRAPH_PLAN.md §9.7): number of growing-prefix checkpoints reconstructed
// from this monitor's per-token history per flush, and the minimum sample count a given
// per-token vector must have reached before a checkpoint uses it (below that, the
// corresponding `ProfileObservation` field is left at its neutral default rather than computed
// from too few points to be meaningful).
constexpr int kCausalCheckpointCount = 4;
constexpr int kCausalCheckpointMinSamples = 4;

double mean_prefix(const std::vector<double>& values, int count) {
  count = std::min(count, static_cast<int>(values.size()));
  if (count <= 0) {
    return 0.0;
  }
  double sum = 0.0;
  for (int i = 0; i < count; ++i) {
    sum += values[static_cast<std::size_t>(i)];
  }
  return sum / static_cast<double>(count);
}

double histogram_entropy(const double* values, int n, int n_bins = 16) {
  if (n <= 0) {
    return 0.0;
  }
  const int bins = std::max(1, n_bins);
  double lo = values[0];
  double hi = values[0];
  for (int i = 1; i < n; ++i) {
    lo = std::min(lo, values[i]);
    hi = std::max(hi, values[i]);
  }
  if (hi <= lo) {
    hi = lo + 1.0;
  }
  std::vector<double> hist(static_cast<std::size_t>(bins), 0.0);
  const double width = (hi - lo) / static_cast<double>(bins);
  for (int i = 0; i < n; ++i) {
    int idx = static_cast<int>((values[i] - lo) / width);
    idx = std::clamp(idx, 0, bins - 1);
    hist[static_cast<std::size_t>(idx)] += 1.0;
  }
  double ent = 0.0;
  const double norm = static_cast<double>(n) + kEps * static_cast<double>(bins);
  for (double c : hist) {
    const double p = (c + kEps) / norm;
    ent -= p * std::log(p);
  }
  return ent;
}

double log_probs_entropy(const double* log_probs, int vocab_size) {
  if (log_probs == nullptr || vocab_size <= 0) {
    return 0.0;
  }
  double max_lp = log_probs[0];
  for (int i = 1; i < vocab_size; ++i) {
    max_lp = std::max(max_lp, log_probs[i]);
  }
  double sum = 0.0;
  for (int i = 0; i < vocab_size; ++i) {
    sum += std::exp(log_probs[i] - max_lp);
  }
  const double log_z = max_lp + std::log(sum + kEps);
  double h = 0.0;
  for (int i = 0; i < vocab_size; ++i) {
    const double p = std::exp(log_probs[i] - log_z);
    if (p > kEps) {
      h -= p * std::log(p + kEps);
    }
  }
  return h;
}

double compute_step_alpha(const std::vector<double>& input_embed, const std::vector<double>& log_probs) {
  if (log_probs.empty()) {
    return 0.5;
  }
  const double h_out =
      log_probs_entropy(log_probs.data(), static_cast<int>(log_probs.size()));
  double h_in = h_out + kEps;
  if (!input_embed.empty()) {
    h_in = histogram_entropy(input_embed.data(), static_cast<int>(input_embed.size()));
  }
  if (h_in > kEps) {
    return std::clamp(1.0 - h_out / h_in, 0.0, 1.0);
  }
  return 0.5;
}

double field_magnitude(const double* field, int n_dims) {
  if (field == nullptr || n_dims <= 0) {
    return 0.0;
  }
  double acc = 0.0;
  for (int d = 0; d < n_dims; ++d) {
    acc += field[d] * field[d];
  }
  return std::sqrt(acc);
}

double mean_alpha(const std::vector<double>& step_alphas) {
  if (step_alphas.empty()) {
    return 0.5;
  }
  const double sum = std::accumulate(step_alphas.begin(), step_alphas.end(), 0.0);
  return std::clamp(sum / static_cast<double>(step_alphas.size()), 0.0, 1.0);
}

int argmax_log_probs(const std::vector<double>& log_probs, int vocab_size) {
  const int n = std::min(static_cast<int>(log_probs.size()), vocab_size);
  if (n <= 0) {
    return 0;
  }
  int argmax = 0;
  for (int i = 1; i < n; ++i) {
    if (log_probs[static_cast<std::size_t>(i)] > log_probs[static_cast<std::size_t>(argmax)]) {
      argmax = i;
    }
  }
  return argmax;
}

double max_log_prob_confidence(const std::vector<double>& log_probs, int vocab_size) {
  const int n = std::min(static_cast<int>(log_probs.size()), vocab_size);
  if (n <= 0) {
    return 0.5;
  }
  double max_lp = log_probs[0];
  for (int i = 1; i < n; ++i) {
    max_lp = std::max(max_lp, log_probs[static_cast<std::size_t>(i)]);
  }
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += std::exp(log_probs[static_cast<std::size_t>(i)] - max_lp);
  }
  const double log_z = max_lp + std::log(sum + kEps);
  const double confidence = std::exp(max_lp - log_z);
  // std::clamp passes NaN through unchanged (all bound comparisons are false), so a
  // diverged/NaN log_probs vector (e.g. from an unstable training run) would otherwise
  // propagate a NaN confidence into downstream binning (compute_calibration), which casts
  // it to an int index. Guard explicitly rather than relying on clamp for NaN safety.
  if (!std::isfinite(confidence)) {
    return 0.5;
  }
  return std::clamp(confidence, 0.0, 1.0);
}

}  // namespace

void LmIntelligenceMonitor::reset() {
  field_dim_ = 0;
  embed_dim_ = 0;
  field_count_ = 0;
  field_history_.clear();
  embed_history_.clear();
  sequence_trace_.clear();
  confidences_.clear();
  correct_.clear();
  step_alphas_.clear();
  base_fields_.clear();
  perturbed_fields_.clear();
  perturbations_.clear();
  pair_count_ = 0;
  epistemic_sum_ = 0.0;
  aleatoric_sum_ = 0.0;
  variance_steps_ = 0;
  epistemic_history_.clear();
  aleatoric_history_.clear();
  prev_field_.clear();
}

void LmIntelligenceMonitor::trim_field_history() {
  while (field_count_ > kMaxFieldHistory && field_dim_ > 0) {
    field_history_.erase(field_history_.begin(),
                         field_history_.begin() + static_cast<std::ptrdiff_t>(field_dim_));
    if (!embed_history_.empty() && embed_dim_ > 0) {
      embed_history_.erase(embed_history_.begin(),
                           embed_history_.begin() + static_cast<std::ptrdiff_t>(embed_dim_));
    }
    --field_count_;
  }
  while (static_cast<int>(sequence_trace_.size()) > kMaxFieldHistory) {
    sequence_trace_.erase(sequence_trace_.begin());
  }
}

void LmIntelligenceMonitor::append_perturbation_pair(const double* base, const double* perturbed) {
  if (base == nullptr || perturbed == nullptr || field_dim_ <= 0) {
    return;
  }
  const std::size_t offset = static_cast<std::size_t>(pair_count_ * field_dim_);
  base_fields_.resize(offset + static_cast<std::size_t>(field_dim_));
  perturbed_fields_.resize(offset + static_cast<std::size_t>(field_dim_));
  perturbations_.resize(offset + static_cast<std::size_t>(field_dim_));
  for (int d = 0; d < field_dim_; ++d) {
    const std::size_t idx = offset + static_cast<std::size_t>(d);
    base_fields_[idx] = base[d];
    perturbed_fields_[idx] = perturbed[d];
    const double delta = perturbed[d] - base[d];
    perturbations_[idx] = delta + (delta == 0.0 ? kPerturbationEps : 0.0);
  }
  ++pair_count_;
  while (pair_count_ > kMaxFieldHistory - 1) {
    base_fields_.erase(base_fields_.begin(),
                       base_fields_.begin() + static_cast<std::ptrdiff_t>(field_dim_));
    perturbed_fields_.erase(perturbed_fields_.begin(),
                            perturbed_fields_.begin() + static_cast<std::ptrdiff_t>(field_dim_));
    perturbations_.erase(perturbations_.begin(),
                         perturbations_.begin() + static_cast<std::ptrdiff_t>(field_dim_));
    --pair_count_;
  }
}

void LmIntelligenceMonitor::observe_token(const std::vector<double>& input_embed,
                                          const std::vector<double>& field_hidden,
                                          const std::vector<double>& log_probs, double epistemic_var,
                                          double aleatoric_var, std::int64_t next_token_id,
                                          int vocab_size) {
  if (!log_probs.empty()) {
    step_alphas_.push_back(compute_step_alpha(input_embed, log_probs));
    while (static_cast<int>(step_alphas_.size()) > kMaxFieldHistory) {
      step_alphas_.erase(step_alphas_.begin());
    }

    const int vocab = vocab_size > 0 ? vocab_size : static_cast<int>(log_probs.size());
    const int pred_id = argmax_log_probs(log_probs, vocab);
    confidences_.push_back(max_log_prob_confidence(log_probs, vocab));
    if (next_token_id >= 0 && next_token_id < vocab) {
      correct_.push_back(pred_id == static_cast<int>(next_token_id) ? 1 : 0);
    } else {
      correct_.push_back(0);
    }
    while (static_cast<int>(confidences_.size()) > kMaxFieldHistory) {
      confidences_.erase(confidences_.begin());
      correct_.erase(correct_.begin());
    }
  }

  if (epistemic_var > 0.0 || aleatoric_var > 0.0) {
    epistemic_sum_ += epistemic_var;
    aleatoric_sum_ += aleatoric_var;
    ++variance_steps_;
    epistemic_history_.push_back(epistemic_var);
    aleatoric_history_.push_back(aleatoric_var);
    while (static_cast<int>(epistemic_history_.size()) > kMaxFieldHistory) {
      epistemic_history_.erase(epistemic_history_.begin());
      aleatoric_history_.erase(aleatoric_history_.begin());
    }
  }

  if (field_hidden.empty()) {
    return;
  }

  if (field_dim_ > 0 && static_cast<int>(field_hidden.size()) != field_dim_) {
    reset();
  }
  if (field_dim_ == 0) {
    field_dim_ = static_cast<int>(field_hidden.size());
  }
  if (field_dim_ <= 0) {
    return;
  }

  if (!input_embed.empty()) {
    if (embed_dim_ == 0) {
      embed_dim_ = static_cast<int>(input_embed.size());
    }
    embed_history_.insert(embed_history_.end(), input_embed.begin(), input_embed.end());
  } else if (embed_dim_ > 0) {
    embed_history_.insert(embed_history_.end(), static_cast<std::size_t>(embed_dim_), 0.0);
  }

  field_history_.insert(field_history_.end(), field_hidden.begin(), field_hidden.end());
  ++field_count_;
  sequence_trace_.push_back(field_magnitude(field_hidden.data(), field_dim_));

  if (!prev_field_.empty()) {
    append_perturbation_pair(prev_field_.data(), field_hidden.data());
  }
  prev_field_ = field_hidden;

  trim_field_history();
}

cypha::intelligence::ProfileObservation LmIntelligenceMonitor::compute_observation() const {
  cypha::intelligence::ProfileObservation obs;
  obs.alpha = mean_alpha(step_alphas_);

  if (field_count_ > 0 && field_dim_ > 0) {
    const auto method = use_eigenvalue_d_eff_
                            ? cypha::intelligence::ParticipationRatioMethod::CovarianceEigenvalue
                            : cypha::intelligence::ParticipationRatioMethod::VarianceProxy;
    obs.d_eff = cypha::intelligence::compute_participation_ratio(
        field_history_.data(), field_count_, field_dim_, method);
  }

  if (pair_count_ > 0 && field_dim_ > 0) {
    obs.sigma_branch = cypha::intelligence::compute_branching_ratio_sensitivity(
        base_fields_.data(), perturbed_fields_.data(), perturbations_.data(), pair_count_,
        field_dim_);
    obs.lipschitz = cypha::intelligence::compute_lipschitz_sensitivity(
        base_fields_.data(), perturbed_fields_.data(), pair_count_, field_dim_);
  }

  if (sequence_trace_.size() >= 2) {
    obs.tau = cypha::intelligence::compute_memory_depth_normalized(
        sequence_trace_.data(), static_cast<int>(sequence_trace_.size()), 1, kTauMaxLag,
        kTauMaxSteps);
  }

  if (variance_steps_ > 0) {
    const double epistemic_mean = epistemic_sum_ / static_cast<double>(variance_steps_);
    const double aleatoric_mean = aleatoric_sum_ / static_cast<double>(variance_steps_);
    obs.r_eu = cypha::intelligence::compute_epistemic_ratio(epistemic_mean, aleatoric_mean);
  }

  if (!confidences_.empty() && confidences_.size() == correct_.size()) {
    obs.calibration = cypha::intelligence::compute_calibration(
        confidences_.data(), correct_.data(), static_cast<int>(confidences_.size()));
  }

  return obs;
}

cypha::intelligence::ProfileObservation LmIntelligenceMonitor::snapshot_observation() const {
  return compute_observation();
}

void LmIntelligenceMonitor::feed_causal_checkpoints(
    cypha::intelligence::CausalGraphMonitor& causal) const {
  const int n_alpha = static_cast<int>(step_alphas_.size());
  const int n_seq = static_cast<int>(sequence_trace_.size());
  const int n_var = static_cast<int>(epistemic_history_.size());
  if (n_alpha < kCausalCheckpointMinSamples && n_seq < kCausalCheckpointMinSamples &&
      n_var < kCausalCheckpointMinSamples) {
    return;
  }

  for (int c = 1; c <= kCausalCheckpointCount; ++c) {
    const double frac = static_cast<double>(c) / static_cast<double>(kCausalCheckpointCount);
    const int k_alpha = static_cast<int>(std::llround(static_cast<double>(n_alpha) * frac));
    const int k_seq = static_cast<int>(std::llround(static_cast<double>(n_seq) * frac));
    const int k_var = static_cast<int>(std::llround(static_cast<double>(n_var) * frac));

    cypha::intelligence::ProfileObservation obs;
    if (k_alpha >= kCausalCheckpointMinSamples) {
      obs.alpha = mean_prefix(step_alphas_, k_alpha);
      if (static_cast<int>(confidences_.size()) >= k_alpha &&
          static_cast<int>(correct_.size()) >= k_alpha) {
        obs.calibration =
            cypha::intelligence::compute_calibration(confidences_.data(), correct_.data(), k_alpha);
      }
    }
    if (k_seq >= kCausalCheckpointMinSamples) {
      obs.tau = cypha::intelligence::compute_memory_depth_normalized(
          sequence_trace_.data(), k_seq, 1, kTauMaxLag, kTauMaxSteps);
    }
    if (k_var >= kCausalCheckpointMinSamples) {
      const double e_mean = mean_prefix(epistemic_history_, k_var);
      const double a_mean = mean_prefix(aleatoric_history_, k_var);
      obs.r_eu = cypha::intelligence::compute_epistemic_ratio(e_mean, a_mean);
    }
    causal.observe_profile(obs);
  }
}

void LmIntelligenceMonitor::flush_to_profiler(
    cypha::intelligence::IntelligenceProfiler& profiler) {
  cypha::intelligence::ProfileBatch batch;
  batch.tau_max_lag = kTauMaxLag;
  batch.tau_max_steps = kTauMaxSteps;

  if (field_count_ > 0 && field_dim_ > 0) {
    batch.output = field_history_.data();
    batch.n_samples = field_count_;
    batch.n_dims = field_dim_;
    // batch.input must share n_dims with batch.output for compute_alpha_gria; embed_dim often
    // differs from field_dim (e.g. d17 d_embed=64, field_dim=160).
    if (embed_dim_ > 0 && embed_dim_ == field_dim_ &&
        static_cast<int>(embed_history_.size()) == field_count_ * embed_dim_) {
      batch.input = embed_history_.data();
    }
    // σ and L come from pair_count consecutive-field deltas; do not pass mismatched
    // perturbed buffers into update_from_batch (field_count != pair_count → overread).
  }

  // τ is supplied via batch.tau from compute_observation() (sequence_trace_ is 1-D).
  // Do not pass batch.sequence here: update_from_batch would use batch.n_dims
  // (field_dim_) and overread sequence_trace_ in compute_memory_depth_normalized.

  if (!confidences_.empty() && confidences_.size() == correct_.size()) {
    batch.confidences = confidences_.data();
    batch.correct = correct_.data();
    batch.n_labels = static_cast<int>(confidences_.size());
  }

  if (variance_steps_ > 0) {
    batch.epistemic_var = epistemic_sum_ / static_cast<double>(variance_steps_);
    batch.aleatoric_var = aleatoric_sum_ / static_cast<double>(variance_steps_);
  }

  const cypha::intelligence::ProfileObservation snapshot = compute_observation();
  batch.sigma_branch = snapshot.sigma_branch;
  batch.tau = snapshot.tau;
  batch.lipschitz = snapshot.lipschitz;

  if (batch.n_samples > 0 || batch.n_labels > 0 || batch.sequence != nullptr ||
      batch.epistemic_var.has_value()) {
    profiler.update_from_batch(batch);
  }

  cypha::intelligence::ProfileObservation hook_obs;
  hook_obs.alpha = snapshot.alpha;
  hook_obs.d_eff = snapshot.d_eff;
  hook_obs.r_eu = snapshot.r_eu;
  hook_obs.calibration = snapshot.calibration;
  profiler.update(hook_obs);

  // Feed the profiler's persistent causal graph with real, genuinely time-varying checkpoints
  // reconstructed from this monitor's own per-token history (see `feed_causal_checkpoints` and
  // docs/reports/SOFT_WORLD_CAUSAL_GRAPH_PLAN.md §9.7). This is what lets
  // `CausalGraphMonitor::causal_fidelity()` become non-degenerate within a single bench run:
  // the two profiler updates above are both derived from the same summed per-token statistics
  // (calibration and r_eu are numerically identical between them), so they alone cannot produce
  // a non-zero online correlation no matter how many times this function is called.
  feed_causal_checkpoints(profiler.causal_graph());
}

}  // namespace cypha::cyphalm
