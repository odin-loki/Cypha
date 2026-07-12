#include "cypha/rpsm/rpsm_sequence_layer.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace cypha::rpsm {

namespace {

constexpr double kEps = 1e-12;
constexpr std::uint64_t kIzaacSeedOffset = 991;

void matvec_row_major(const double* mat, int rows, int cols, const double* x, double* y) {
  for (int r = 0; r < rows; ++r) {
    double acc = 0.0;
    for (int c = 0; c < cols; ++c) {
      acc += mat[static_cast<std::size_t>(r * cols + c)] * x[static_cast<std::size_t>(c)];
    }
    y[static_cast<std::size_t>(r)] = acc;
  }
}

void matvec_transpose_row_major(const double* mat, int rows, int cols, const double* x, double* y) {
  for (int c = 0; c < cols; ++c) {
    double acc = 0.0;
    for (int r = 0; r < rows; ++r) {
      acc += mat[static_cast<std::size_t>(r * cols + c)] * x[static_cast<std::size_t>(r)];
    }
    y[static_cast<std::size_t>(c)] = acc;
  }
}

void init_normal_matrix(std::vector<double>& out, int rows, int cols, std::mt19937_64& rng, double scale) {
  std::normal_distribution<double> nd(0.0, scale);
  out.assign(static_cast<std::size_t>(rows * cols), 0.0);
  for (auto& v : out) {
    v = nd(rng);
  }
}

void init_orthogonal_matrix(std::vector<double>& out, int dim, std::mt19937_64& rng) {
  std::normal_distribution<double> nd(0.0, 1.0);
  out.assign(static_cast<std::size_t>(dim * dim), 0.0);
  std::vector<double> a(static_cast<std::size_t>(dim * dim));
  for (auto& v : a) {
    v = nd(rng);
  }
  std::vector<double> col(static_cast<std::size_t>(dim));
  for (int j = 0; j < dim; ++j) {
    for (int i = 0; i < dim; ++i) {
      col[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i * dim + j)];
    }
    for (int k = 0; k < j; ++k) {
      double dot = 0.0;
      for (int i = 0; i < dim; ++i) {
        dot += col[static_cast<std::size_t>(i)] * out[static_cast<std::size_t>(i * dim + k)];
      }
      for (int i = 0; i < dim; ++i) {
        col[static_cast<std::size_t>(i)] -= dot * out[static_cast<std::size_t>(i * dim + k)];
      }
    }
    double norm = 0.0;
    for (int i = 0; i < dim; ++i) {
      norm += col[static_cast<std::size_t>(i)] * col[static_cast<std::size_t>(i)];
    }
    norm = std::sqrt(std::max(norm, 1e-18));
    for (int i = 0; i < dim; ++i) {
      out[static_cast<std::size_t>(i * dim + j)] = col[static_cast<std::size_t>(i)] / norm;
    }
  }
}

std::uint64_t matrix_seed(const RpsmSequenceConfig& cfg, int matrix_id) {
  if (cfg.use_izaac_init) {
    return cfg.seed + kIzaacSeedOffset + static_cast<std::uint64_t>(matrix_id);
  }
  return cfg.seed + static_cast<std::uint64_t>(matrix_id);
}

void init_psi_matrices(PsiMatrices& psi, const RpsmSequenceConfig& cfg) {
  const int d = cfg.feat_dim;
  const int k = cfg.n_classes;
  psi.feat_dim = d;
  psi.n_classes = k;
  psi.mu.assign(static_cast<std::size_t>((1 + k) * d), 0.0);
  psi.inv_var.assign(static_cast<std::size_t>(d), 1.0);
  psi.counts.assign(static_cast<std::size_t>(k), 1.0);
  psi.v_mean = 1.0;
  std::mt19937_64 rng(matrix_seed(cfg, 0));
  std::normal_distribution<double> nd(0.0, 0.05);
  for (int j = 0; j < d; ++j) {
    psi.mu[static_cast<std::size_t>(j)] = nd(rng);
  }
  for (int c = 0; c < k; ++c) {
    for (int j = 0; j < d; ++j) {
      psi.mu[static_cast<std::size_t>((1 + c) * d + j)] = nd(rng);
    }
  }
}

double raw_activation_mix(IzaacActivationMix mix, double x) {
  switch (mix) {
    case IzaacActivationMix::TanhOnly:
      return std::tanh(x);
    case IzaacActivationMix::TanhRelu:
      return std::tanh(std::max(0.0, x));
    case IzaacActivationMix::TanhSigmoid:
      return std::tanh(1.0 / (1.0 + std::exp(-x)));
    case IzaacActivationMix::ReluTanh:
      return std::max(0.0, std::tanh(x));
    case IzaacActivationMix::LinearTanh:
      return 0.5 * x + 0.5 * std::tanh(x);
    default:
      return std::tanh(x);
  }
}

double activation_derivative(IzaacActivationMix mix, double x) {
  switch (mix) {
    case IzaacActivationMix::TanhOnly: {
      const double t = std::tanh(x);
      return 1.0 - t * t;
    }
    case IzaacActivationMix::TanhRelu: {
      if (x <= 0.0) {
        return 0.0;
      }
      const double t = std::tanh(x);
      return 1.0 - t * t;
    }
    case IzaacActivationMix::TanhSigmoid: {
      const double s = 1.0 / (1.0 + std::exp(-x));
      const double t = std::tanh(s);
      return (1.0 - t * t) * s * (1.0 - s);
    }
    case IzaacActivationMix::ReluTanh: {
      const double t = std::tanh(x);
      return (t > 0.0 ? 1.0 : 0.0) * (1.0 - t * t);
    }
    case IzaacActivationMix::LinearTanh: {
      const double t = std::tanh(x);
      return 0.5 + 0.5 * (1.0 - t * t);
    }
    default: {
      const double t = std::tanh(x);
      return 1.0 - t * t;
    }
  }
}

}  // namespace

double gria_alpha_spectral(const double* psi_row_major, int n_levels, int state_dim) {
  const int L = std::max(1, n_levels);
  const int D = std::max(1, state_dim);

  // Gram matrix G = Psi @ Psi^T (L x L, symmetric PSD) — cheap since L << D for every
  // configured tier (Tiny/Small/Medium/Large all use L/D = 1/32, RPSM_IMPLEMENTATION.md:97-102).
  std::vector<double> gram(static_cast<std::size_t>(L) * static_cast<std::size_t>(L), 0.0);
  for (int i = 0; i < L; ++i) {
    const double* ri = psi_row_major + static_cast<std::size_t>(i) * static_cast<std::size_t>(D);
    for (int j = i; j < L; ++j) {
      const double* rj = psi_row_major + static_cast<std::size_t>(j) * static_cast<std::size_t>(D);
      double acc = 0.0;
      for (int c = 0; c < D; ++c) {
        acc += ri[static_cast<std::size_t>(c)] * rj[static_cast<std::size_t>(c)];
      }
      gram[static_cast<std::size_t>(i) * static_cast<std::size_t>(L) + static_cast<std::size_t>(j)] = acc;
      gram[static_cast<std::size_t>(j) * static_cast<std::size_t>(L) + static_cast<std::size_t>(i)] = acc;
    }
  }

  // Power iteration for the top eigenvalue of the symmetric PSD Gram matrix; converges to
  // sigma_max(Psi)^2 since G = Psi Psi^T. 32 iterations is ample for L <= 32 (Large tier).
  std::vector<double> v(static_cast<std::size_t>(L), 1.0 / std::sqrt(static_cast<double>(L)));
  double lambda_max = 0.0;
  for (int iter = 0; iter < 32; ++iter) {
    std::vector<double> w(static_cast<std::size_t>(L), 0.0);
    for (int i = 0; i < L; ++i) {
      double acc = 0.0;
      for (int j = 0; j < L; ++j) {
        acc += gram[static_cast<std::size_t>(i) * static_cast<std::size_t>(L) + static_cast<std::size_t>(j)] *
               v[static_cast<std::size_t>(j)];
      }
      w[static_cast<std::size_t>(i)] = acc;
    }
    double norm_sq = 0.0;
    for (double x : w) {
      norm_sq += x * x;
    }
    const double norm = std::sqrt(norm_sq);
    if (norm < 1e-12) {
      lambda_max = 0.0;
      break;
    }
    for (int i = 0; i < L; ++i) {
      v[static_cast<std::size_t>(i)] = w[static_cast<std::size_t>(i)] / norm;
    }
    lambda_max = norm;
  }

  const double sigma_max = std::sqrt(std::max(lambda_max, 0.0));
  const double normalized = sigma_max / std::sqrt(static_cast<double>(D));
  // Edge-of-chaos squash: at init (Psi ~ N(0,1)), normalized ~ 1 + sqrt(L/D) for every tier
  // (constant L/D ratio across Tiny/Small/Medium/Large), which lands near alpha=0.476 — inside
  // the spec's [0.3, 0.6] target band (RPSM_IMPLEMENTATION.md:44) and close to the existing
  // 0.485 "edge of chaos" convention already used elsewhere in this codebase (H10 cell
  // hypothesis, cypha_cell_hypothesis.cpp:118).
  const double centered = normalized - 1.0;
  const double sigmoid = 1.0 / (1.0 + std::exp(-2.0 * centered));
  return 0.3 + 0.3 * sigmoid;
}

IzaacActivationMix select_izaac_activation_mix(std::uint64_t seed) {
  const auto idx = static_cast<int>(seed % static_cast<std::uint64_t>(kIzaacActivationMixCount));
  return static_cast<IzaacActivationMix>(idx);
}

RpsmGlobalMemory::RpsmGlobalMemory(int n_slots, int dim)
    : n_slots_(std::max(1, n_slots)), dim_(std::max(1, dim)) {
  slots_.assign(static_cast<std::size_t>(n_slots_ * dim_), 0.0);
}

void RpsmGlobalMemory::reset() {
  std::fill(slots_.begin(), slots_.end(), 0.0);
  write_head_ = 0;
}

void RpsmGlobalMemory::soft_read(const double* query, int query_dim, double* out) const {
  const int qd = std::min(dim_, query_dim);
  std::fill(out, out + dim_, 0.0);
  if (n_slots_ <= 0 || dim_ <= 0) {
    return;
  }

  double max_logit = -1e30;
  std::vector<double> logits(static_cast<std::size_t>(n_slots_), 0.0);
  for (int s = 0; s < n_slots_; ++s) {
    const double* slot = slots_.data() + static_cast<std::size_t>(s * dim_);
    double dot = 0.0;
    for (int i = 0; i < qd; ++i) {
      dot += query[i] * slot[i];
    }
    logits[static_cast<std::size_t>(s)] = dot;
    max_logit = std::max(max_logit, dot);
  }

  double sum = 0.0;
  for (int s = 0; s < n_slots_; ++s) {
    const double w = std::exp(logits[static_cast<std::size_t>(s)] - max_logit);
    logits[static_cast<std::size_t>(s)] = w;
    sum += w;
  }
  const double inv = 1.0 / (sum + kEps);
  for (int s = 0; s < n_slots_; ++s) {
    const double w = logits[static_cast<std::size_t>(s)] * inv;
    const double* slot = slots_.data() + static_cast<std::size_t>(s * dim_);
    for (int i = 0; i < dim_; ++i) {
      out[i] += w * slot[i];
    }
  }
}

void RpsmGlobalMemory::ring_write(const double* vec, int vec_dim, double surprise) {
  if (n_slots_ <= 0 || dim_ <= 0 || vec == nullptr) {
    return;
  }
  if (!std::isfinite(surprise) || surprise <= 0.0) {
    return;
  }
  const double gate = std::tanh(surprise);
  if (gate <= 1e-6) {
    return;
  }
  const int copy = std::min(dim_, vec_dim);
  double* slot = slots_.data() + static_cast<std::size_t>(write_head_ * dim_);
  std::fill(slot, slot + dim_, 0.0);
  for (int i = 0; i < copy; ++i) {
    slot[i] = gate * vec[i];
  }
  write_head_ = (write_head_ + 1) % n_slots_;
}

RpsmSequenceLayer::RpsmSequenceLayer(RpsmSequenceConfig cfg)
    : cfg_(cfg),
      global_memory_(cfg.n_memory_slots, cfg.state_dim) {
  const int sd = cfg_.state_dim;
  const int nl = std::max(1, cfg_.n_levels);

  if (cfg_.use_izaac_init) {
    activation_mix_ = select_izaac_activation_mix(cfg_.seed);
  } else {
    activation_mix_ = cfg_.activation_mix;
  }

  init_psi_matrices(psi_, cfg_);

  h_levels_.assign(static_cast<std::size_t>(nl), std::vector<double>(static_cast<std::size_t>(sd), 0.0));
  psi_rows_.assign(static_cast<std::size_t>(nl * sd), 0.0);

  if (cfg_.use_izaac_init) {
    std::mt19937_64 rng_up(matrix_seed(cfg_, 1));
    init_orthogonal_matrix(w_up_, sd, rng_up);
    std::mt19937_64 rng_enc(matrix_seed(cfg_, 2));
    init_normal_matrix(w_enc_, cfg_.feat_dim, sd, rng_enc, 0.02);
    std::mt19937_64 rng_carry(matrix_seed(cfg_, 3));
    init_normal_matrix(w_carry_, cfg_.feat_dim, sd, rng_carry, 0.02);
  } else {
    std::mt19937_64 rng(cfg_.seed);
    init_normal_matrix(w_up_, sd, sd, rng, 0.02);
    init_normal_matrix(w_enc_, cfg_.feat_dim, sd, rng, 0.02);
    init_normal_matrix(w_carry_, cfg_.feat_dim, sd, rng, 0.02);
  }

  feat_buf_.assign(static_cast<std::size_t>(cfg_.feat_dim), 0.0);
  llr_buf_.assign(static_cast<std::size_t>(cfg_.n_classes), 0.0);
  work_up_.assign(static_cast<std::size_t>(sd), 0.0);
  work_down_.assign(static_cast<std::size_t>(sd), 0.0);
  work_err_.assign(static_cast<std::size_t>(sd), 0.0);
  mem_read_.assign(static_cast<std::size_t>(sd), 0.0);
  enc_pre_.assign(static_cast<std::size_t>(cfg_.feat_dim), 0.0);
  enc_grad_.assign(static_cast<std::size_t>(cfg_.feat_dim), 0.0);
  input_grad_.assign(static_cast<std::size_t>(sd), 0.0);

  // §15: world_m2_ starts at 1.0/dim (not 0.0), matching memory_train.cpp's own
  // `world_M2.assign(d_latent, 1.0)` (memory_train.cpp:151) -- a deliberate regularising prior on
  // the Welford variance accumulator, not an oversight; ported as-is. world_v_ starts at 1.0 to
  // match psi_.inv_var's own init (init_psi_matrices: `inv_var.assign(d, 1.0)`), so enabling this
  // flag is a no-op on step 0 and only diverges from the frozen-at-1.0 baseline once real feature
  // statistics accumulate.
  world_n_ = 0;
  world_m2_.assign(static_cast<std::size_t>(cfg_.feat_dim), 1.0);
  world_v_.assign(static_cast<std::size_t>(cfg_.feat_dim), 1.0);

  bptt_cache_.assign(static_cast<std::size_t>(std::max(1, cfg_.bptt_window)), BpttStepCache{});
  bptt_fill_ = 0;
}

const std::vector<double>& RpsmSequenceLayer::level_hidden(int level) const {
  static const std::vector<double> kEmpty;
  if (level < 0 || level >= static_cast<int>(h_levels_.size())) {
    return kEmpty;
  }
  return h_levels_[static_cast<std::size_t>(level)];
}

void RpsmSequenceLayer::reset() {
  for (auto& row : h_levels_) {
    std::fill(row.begin(), row.end(), 0.0);
  }
  std::fill(psi_rows_.begin(), psi_rows_.end(), 0.0);
  global_memory_.reset();
  last_surprise_ = 0.0;
  // Discard any partial in-flight BPTT window rather than flushing it: the cached `h_inj`
  // snapshots are only valid relative to the state trajectory that produced them, and that
  // trajectory just got reset to zero. This mirrors CyphaLMModel::reset_context()'s own
  // `bptt_buffer_.clear()` (rather than flush) on the same kind of boundary.
  bptt_fill_ = 0;
  bptt_flushed_this_call_ = false;
  bptt_flush_input_grads_.clear();
}

double RpsmSequenceLayer::apply_activation(double x) const {
  return raw_activation_mix(activation_mix_, x);
}

void RpsmSequenceLayer::log_softmax_row(const double* logits, int k, double* log_out) {
  double max_z = logits[0];
  for (int i = 1; i < k; ++i) {
    max_z = std::max(max_z, logits[i]);
  }
  double sum = 0.0;
  for (int i = 0; i < k; ++i) {
    sum += std::exp(logits[i] - max_z);
  }
  const double log_z = max_z + std::log(sum + kEps);
  for (int i = 0; i < k; ++i) {
    log_out[i] = logits[i] - log_z;
  }
}

void RpsmSequenceLayer::inject_input_multilevel(const double* input, int input_dim,
                                                 BpttStepCache* cache) {
  const int sd = cfg_.state_dim;
  const int d = cfg_.feat_dim;
  const int in_n = std::max(0, input_dim);
  const int nl = static_cast<int>(h_levels_.size());

  if (cache != nullptr) {
    cache->in_n = in_n;
    cache->inj_acc.assign(static_cast<std::size_t>(sd), 0.0);
  }

  for (int l = 0; l < nl; ++l) {
    const double scale = 1.0 / static_cast<double>(l + 1);
    auto& h = h_levels_[static_cast<std::size_t>(l)];
    for (int i = 0; i < sd; ++i) {
      double acc = 0.0;
      for (int j = 0; j < in_n && j < sd; ++j) {
        acc += w_carry_[static_cast<std::size_t>((j % d) * sd + i)] * input[j];
      }
      // `acc` does not depend on `l` (only the post-activation blend `scale` below does), so
      // one snapshot per step suffices -- record it on the first level's pass.
      if (cache != nullptr && l == 0) {
        cache->inj_acc[static_cast<std::size_t>(i)] = acc;
      }
      h[static_cast<std::size_t>(i)] += scale * apply_activation(acc);
    }
  }
}

void RpsmSequenceLayer::encode_level0_features(const double* input, int input_dim) {
  const int d = cfg_.feat_dim;
  const int sd = cfg_.state_dim;
  const int in_n = std::max(0, input_dim);
  const auto& h0 = h_levels_[0];

  for (int j = 0; j < d; ++j) {
    double acc = 0.0;
    for (int i = 0; i < in_n && i < sd; ++i) {
      acc += w_enc_[static_cast<std::size_t>(j * sd + i)] * input[i];
    }
    for (int i = 0; i < sd; ++i) {
      acc += w_carry_[static_cast<std::size_t>(j * sd + i)] * h0[static_cast<std::size_t>(i)];
    }
    enc_pre_[static_cast<std::size_t>(j)] = acc;
    feat_buf_[static_cast<std::size_t>(j)] = apply_activation(acc);
  }
}

double RpsmSequenceLayer::hierarchy_update(BpttStepCache* cache) {
  const int sd = cfg_.state_dim;
  const int nl = static_cast<int>(h_levels_.size());
  // Fix 1 (spectral alpha): compute from psi_rows_ as it stood entering this call (i.e. the
  // previous step's Psi), before this loop overwrites it below. Opt-in; defaults to the fixed
  // cfg_.alpha_carry to preserve Phase 0/0b's exact measured behaviour when disabled.
  const double a = cfg_.use_spectral_alpha ? gria_alpha_spectral(psi_rows_.data(), nl, sd)
                                            : cfg_.alpha_carry;
  const double beta = cfg_.beta_memory;
  double hierarchy_loss = 0.0;

  if (cache != nullptr) {
    cache->alpha = a;
    cache->h_inj.assign(static_cast<std::size_t>(nl), std::vector<double>(static_cast<std::size_t>(sd)));
    cache->up_pre.assign(static_cast<std::size_t>(nl), std::vector<double>(static_cast<std::size_t>(sd)));
    cache->down_pre.assign(static_cast<std::size_t>(nl), std::vector<double>(static_cast<std::size_t>(sd)));
    cache->blended_pre.assign(static_cast<std::size_t>(nl), std::vector<double>(static_cast<std::size_t>(sd)));
  }

  for (int l = 0; l < nl; ++l) {
    auto& h = h_levels_[static_cast<std::size_t>(l)];
    // Snapshot the state *entering* this level's transform (h_inj: post-injection/encode,
    // pre-hierarchy) -- this is the exact tensor the reverse pass needs `d(w_up_)`,
    // `d(mem query)` etc. to be taken with respect to. Must be a copy, not a reference: `h` is
    // mutated below, in place, before this function returns.
    if (cache != nullptr) {
      cache->h_inj[static_cast<std::size_t>(l)] = h;
    }
    matvec_row_major(w_up_.data(), sd, sd, h.data(), work_up_.data());
    if (cache != nullptr) {
      cache->up_pre[static_cast<std::size_t>(l)].assign(work_up_.begin(), work_up_.end());
    }
    for (int i = 0; i < sd; ++i) {
      work_up_[static_cast<std::size_t>(i)] = apply_activation(work_up_[static_cast<std::size_t>(i)]);
    }
    matvec_transpose_row_major(w_up_.data(), sd, sd, h.data(), work_down_.data());
    if (cache != nullptr) {
      cache->down_pre[static_cast<std::size_t>(l)].assign(work_down_.begin(), work_down_.end());
    }
    for (int i = 0; i < sd; ++i) {
      work_down_[static_cast<std::size_t>(i)] = apply_activation(work_down_[static_cast<std::size_t>(i)]);
    }
    double level_err_sq = 0.0;
    for (int i = 0; i < sd; ++i) {
      work_err_[static_cast<std::size_t>(i)] =
          work_up_[static_cast<std::size_t>(i)] - work_down_[static_cast<std::size_t>(i)];
      level_err_sq += work_err_[static_cast<std::size_t>(i)] * work_err_[static_cast<std::size_t>(i)];
    }
    hierarchy_loss += level_err_sq;

    global_memory_.soft_read(h.data(), sd, mem_read_.data());

    for (int i = 0; i < sd; ++i) {
      const double blended =
          (1.0 - a) * h[static_cast<std::size_t>(i)] + a * work_err_[static_cast<std::size_t>(i)] +
          beta * mem_read_[static_cast<std::size_t>(i)];
      if (cache != nullptr) {
        cache->blended_pre[static_cast<std::size_t>(l)][static_cast<std::size_t>(i)] = blended;
      }
      h[static_cast<std::size_t>(i)] = apply_activation(blended);
      psi_rows_[static_cast<std::size_t>(l * sd + i)] = h[static_cast<std::size_t>(i)];
    }

    const double level_surprise = std::sqrt(level_err_sq / static_cast<double>(sd) + kEps);
    if (level_surprise >= cfg_.surprise_threshold) {
      global_memory_.ring_write(h.data(), sd, level_surprise);
    }
  }

  last_surprise_ = std::sqrt(hierarchy_loss / static_cast<double>(nl * sd) + kEps);
  return hierarchy_loss / static_cast<double>(nl * sd);
}

namespace {
constexpr double kWorldMinVar = 1e-4;
constexpr int kWorldWelfordSteps = 20;
}  // namespace

void RpsmSequenceLayer::update_world_stats(const double* feat) {
  // §15 (RPSM_UPGRADE_PLAN.md): ported from `memory_train.cpp`'s `world_update()`
  // (`native/src/memory_train.cpp:41-121`), specialised to this layer's single `psi_.mu` row-0
  // (`mu0`) / `psi_.inv_var` state (that file's `world_log_norm`/`world_D_LOG2PI` are not used
  // anywhere in `batched_llr_gemm`'s simpler LLR formula, so they are intentionally not ported).
  const int d = cfg_.feat_dim;
  double* mu0 = psi_.mu.data();  // row 0 of Psi_mu.
  ++world_n_;

  if (world_n_ <= kWorldWelfordSteps) {
    // Welford's online mean/variance -- exact for the first kWorldWelfordSteps observations,
    // matching memory_train.cpp:47-82 term-for-term (including its order of operations: mu0 is
    // updated in place *before* M2 accumulates, using the post-update mu0 in the second delta).
    for (int j = 0; j < d; ++j) {
      const double delta0 = feat[j] - mu0[j];
      mu0[j] += delta0 / static_cast<double>(world_n_);
      world_m2_[static_cast<std::size_t>(j)] += delta0 * (feat[j] - mu0[j]);
    }
    if (world_n_ > 1) {
      double v_sum = 0.0;
      for (int j = 0; j < d; ++j) {
        const double vj = std::max(
            world_m2_[static_cast<std::size_t>(j)] / static_cast<double>(world_n_ - 1), kWorldMinVar);
        world_v_[static_cast<std::size_t>(j)] = vj;
        psi_.inv_var[static_cast<std::size_t>(j)] = std::min(1.0 / vj, cfg_.inv_var_max_scale);
        v_sum += vj;
      }
      psi_.v_mean = v_sum / static_cast<double>(d);
    }
    return;
  }

  // EMA regime (memory_train.cpp:84-111): delta is captured w.r.t. mu0 *before* mu0's own update,
  // matching that file's `delta_em`/`world_buf` ordering exactly.
  const double lr = cfg_.world_stats_lr;
  double v_sum = 0.0;
  for (int j = 0; j < d; ++j) {
    const double delta = feat[j] - mu0[j];
    mu0[j] += lr * delta;
    double vj = (1.0 - lr) * world_v_[static_cast<std::size_t>(j)] + lr * delta * delta;
    vj = std::max(vj, kWorldMinVar);
    world_v_[static_cast<std::size_t>(j)] = vj;
    psi_.inv_var[static_cast<std::size_t>(j)] = std::min(1.0 / vj, cfg_.inv_var_max_scale);
    v_sum += vj;
  }
  psi_.v_mean = v_sum / static_cast<double>(d);
}

double RpsmSequenceLayer::step(const double* input, int input_dim, double* log_probs_out) {
  inject_input_multilevel(input, input_dim);
  encode_level0_features(input, input_dim);

  batched_llr_gemm(feat_buf_.data(), 1, psi_, nullptr, llr_buf_.data());
  log_softmax_row(llr_buf_.data(), cfg_.n_classes, log_probs_out);

  hierarchy_update();

  double norm_sq = 0.0;
  for (double v : h_levels_[0]) {
    norm_sq += v * v;
  }
  return std::sqrt(norm_sq + kEps);
}

RpsmTrainStepMetrics RpsmSequenceLayer::train_step(const double* input, int input_dim, int target_class,
                                                    double lr) {
  RpsmTrainStepMetrics metrics;
  const int d = cfg_.feat_dim;
  const int k = cfg_.n_classes;
  const int sd = cfg_.state_dim;
  const int tgt = std::clamp(target_class, 0, std::max(0, k - 1));
  const int in_n = std::max(0, input_dim);

  bptt_flushed_this_call_ = false;
  BpttStepCache& cache = bptt_cache_[static_cast<std::size_t>(bptt_fill_)];

  inject_input_multilevel(input, input_dim, &cache);
  encode_level0_features(input, input_dim);

  batched_llr_gemm(feat_buf_.data(), 1, psi_, nullptr, llr_buf_.data());

  std::vector<double> log_probs(static_cast<std::size_t>(k));
  log_softmax_row(llr_buf_.data(), k, log_probs.data());
  metrics.nll = -log_probs[static_cast<std::size_t>(tgt)];

  const double hierarchy_loss = hierarchy_update(&cache);
  metrics.hierarchy_loss = hierarchy_loss;
  metrics.surprise = last_surprise_;
  metrics.loss = metrics.nll + cfg_.hierarchy_loss_weight * hierarchy_loss;

  // Fix 2 (normalised eta): eta = eta_base / (||E||_F + eps), RPSM_IMPLEMENTATION.md:46-53.
  // ||E||_F is the Frobenius norm of the full multi-level prediction-error matrix; hierarchy_loss
  // is already that quantity's mean-square (sum over all L*sd elements, divided by L*sd), so the
  // raw sum recovers exactly via hierarchy_loss * n_levels * sd. Opt-in; defaults to plain `lr`
  // to preserve Phase 0/0b's exact measured behaviour when disabled. Clamped (not part of the
  // original spec formula) to avoid a divide-by-near-zero blowup once the hierarchy error
  // converges toward 0 later in training.
  double effective_lr = lr;
  if (cfg_.use_normalized_eta) {
    const double frob_err = std::sqrt(std::max(0.0, hierarchy_loss) *
                                       static_cast<double>(cfg_.n_levels * sd));
    const double scale = std::clamp(1.0 / (frob_err + 1e-8), 0.0, cfg_.eta_norm_max_scale);
    effective_lr = lr * scale;
  }
  cache.effective_lr = effective_lr;

  std::vector<double> grad_logits(static_cast<std::size_t>(k), 0.0);
  double max_z = llr_buf_[0];
  for (int i = 1; i < k; ++i) {
    max_z = std::max(max_z, llr_buf_[static_cast<std::size_t>(i)]);
  }
  double sum = 0.0;
  for (int i = 0; i < k; ++i) {
    sum += std::exp(llr_buf_[static_cast<std::size_t>(i)] - max_z);
  }
  const double inv = 1.0 / (sum + kEps);
  for (int i = 0; i < k; ++i) {
    grad_logits[static_cast<std::size_t>(i)] =
        std::exp(llr_buf_[static_cast<std::size_t>(i)] - max_z) * inv;
  }
  grad_logits[static_cast<std::size_t>(tgt)] -= 1.0;

  std::fill(enc_grad_.begin(), enc_grad_.end(), 0.0);
  for (int c = 0; c < k; ++c) {
    const double* delta = psi_.mu.data() + static_cast<std::size_t>((1 + c) * d);
    const double gc = grad_logits[static_cast<std::size_t>(c)];
    for (int j = 0; j < d; ++j) {
      enc_grad_[static_cast<std::size_t>(j)] += gc * psi_.inv_var[static_cast<std::size_t>(j)] * delta[j];
    }
  }

  // Ψ_mu output-classifier gradient (rows 1..K = per-class discriminative directions delta_k).
  // llr_k = sum_j inv_var[j]*delta_k[j]*(feat[j] - mu0[j] - 0.5*delta_k[j]) - u_k + ctx_k, so
  // d(llr_k)/d(delta_k[j]) = inv_var[j] * (feat[j] - mu0[j] - delta_k[j]); combined with the
  // softmax/cross-entropy grad_logits[k] via the chain rule this is the same SGD update the
  // hybrid LSTM head applies to Wy every step (char_lstm.cpp backward_step/apply_gradients).
  // Without this, delta_k stays frozen at its random init forever (see RPSM_UPGRADE_PLAN.md Finding #1).
  const double* mu0 = psi_.mu.data();
  for (int c = 0; c < k; ++c) {
    double* delta = psi_.mu.data() + static_cast<std::size_t>((1 + c) * d);
    const double gc = grad_logits[static_cast<std::size_t>(c)];
    for (int j = 0; j < d; ++j) {
      const double grad = gc * psi_.inv_var[static_cast<std::size_t>(j)] *
                           (feat_buf_[static_cast<std::size_t>(j)] - mu0[j] - delta[j]);
      delta[j] -= effective_lr * grad;
    }
  }

  // §14: this is the pre-existing single-step-local gradient (unchanged formula/value from
  // before §14 -- it never depended on `h0`/weight-update ordering to begin with, see the git
  // history for this block), kept for exact backward compatibility with `input_grad()`'s
  // documented contract (native_rpsm_embed_grad_finite_diff, `rpsm_embed_backprop`'s per-step
  // immediate embedding update). It intentionally does *not* include the hierarchy/injection
  // paths' contribution to d(loss)/d(input) -- that deeper signal is new in §14 and is exposed
  // separately, once per window, via `bptt_window_input_grads()`.
  std::fill(input_grad_.begin(), input_grad_.end(), 0.0);
  for (int j = 0; j < d; ++j) {
    const double pre = enc_pre_[static_cast<std::size_t>(j)];
    const double chain =
        enc_grad_[static_cast<std::size_t>(j)] * activation_derivative(activation_mix_, pre);
    for (int i = 0; i < sd; ++i) {
      input_grad_[static_cast<std::size_t>(i)] +=
          chain * w_enc_[static_cast<std::size_t>(j * sd + i)];
    }
  }

  // §14: `w_up_`/`w_enc_`/`w_carry_` are no longer updated here from this step's local
  // activations alone (the old per-step block computed `w_up_`'s update from a *stale*
  // `work_err_` left over from hierarchy_update()'s last-level loop iteration -- a latent bug,
  // not just "no BPTT" -- and `w_carry_`'s update from `h_levels_[0]` *after* hierarchy_update()
  // had already overwritten it with the next step's carry, not the h_inj value actually used by
  // encode_level0_features(); both are fixed by construction in bptt_backward_and_apply(), which
  // reads the correctly-timed `cache.h_inj`/per-level `work_err_` snapshots instead). Cache what
  // the reverse pass needs and apply the real update once per `cfg_.bptt_window` steps.
  cache.enc_pre = enc_pre_;
  cache.enc_grad = enc_grad_;
  cache.input.assign(static_cast<std::size_t>(sd), 0.0);
  for (int i = 0; i < sd && i < in_n; ++i) {
    cache.input[static_cast<std::size_t>(i)] = input[i];
  }

  ++bptt_fill_;
  if (bptt_fill_ >= static_cast<int>(bptt_cache_.size())) {
    bptt_backward_and_apply();
    bptt_fill_ = 0;
    bptt_flushed_this_call_ = true;
  }

  if (tgt >= 0 && tgt < k) {
    psi_.counts[static_cast<std::size_t>(tgt)] += 1.0;
  }

  // §15: update mu0/inv_var *after* this step's classifier gradient and BPTT cache have already
  // used the pre-update values (mirrors memory_train.cpp's own `world_update(*this, h, world_lr)`
  // call at the end of `memory_train()`, memory_train.cpp:368). Opt-in; default-off preserves
  // every prior phase's exact measured behaviour.
  if (cfg_.use_online_world_stats) {
    update_world_stats(feat_buf_.data());
  }

  return metrics;
}

void RpsmSequenceLayer::bptt_backward_and_apply() {
  // §14 core mechanism (RPSM_UPGRADE_PLAN.md §13.6/§13.7(a)): reverse-time pass through the
  // `bptt_window` steps just cached, propagating the recursive gradient of the *whole window's*
  // loss through the hierarchy's state carry `h_carry_{t+1} = act(blended_pre_t)`, exactly the
  // dependency `train_step`'s old per-step-local update never accounted for.
  //
  // Notation per cached step t, per level l (see hierarchy_update()/inject_input_multilevel()):
  //   h_inj_t[l]        = state entering hierarchy_update this step (post-injection/encode).
  //   up_pre_t[l]   = W_up   . h_inj_t[l];  up_t[l]   = act(up_pre_t[l])
  //   down_pre_t[l] = W_up^T . h_inj_t[l];  down_t[l] = act(down_pre_t[l])
  //   err_t[l]      = up_t[l] - down_t[l]
  //   blended_pre_t[l] = (1-a)*h_inj_t[l] + a*err_t[l] + beta*mem_read_t[l]
  //   h_carry_{t+1}[l]  = act(blended_pre_t[l])   -- becomes h_inj_{t+1}[l]'s *base*, i.e. the
  //                                                  h_pre that inject_input_multilevel adds to.
  //   (level 0 only) enc_pre_t = W_enc.input_t + W_carry.h_inj_t[0]; feat_t = act(enc_pre_t)
  //
  // Backward recursion, t = N-1 .. 0 (standard truncated-BPTT boundary: the gradient entering
  // "after step N-1" -- i.e. from steps beyond this window, not yet processed -- is exactly
  // zero, and the gradient leaving "before step 0" is discarded rather than chased into a
  // previous window, since window boundaries reuse the *forward* state continuously but
  // deliberately truncate the *backward* pass, matching the hybrid path's own bptt_ssm_update):
  //   grad_h_next[l]  = d(window loss)/d(h_carry_{t+1}[l])            (0 at t = N-1)
  //   g_blend[l]      = grad_h_next[l] * act'(blended_pre_t[l])
  //   d(h_inj_t[l])  += (1-a) * g_blend[l]                             [blend's direct term]
  //   d(err_t[l])     = a * g_blend[l] + hierarchy_loss_weight * 2*err_t[l] / (n_levels*state_dim)
  //                                                                    [+ hierarchy_loss's own
  //                                                                     direct gradient, since
  //                                                                     hierarchy_loss_t is
  //                                                                     itself part of the loss]
  //   d(up_pre_t[l])   =  d(err_t[l]) * act'(up_pre_t[l])
  //   d(down_pre_t[l]) = -d(err_t[l]) * act'(down_pre_t[l])
  //   d(w_up_)        += outer(d(up_pre_t[l]), h_inj_t[l])   [from up   = act(W_up   . h_inj)]
  //                    + outer(h_inj_t[l], d(down_pre_t[l])) [from down = act(W_up^T . h_inj),
  //                                                            note the transposed index order]
  //   d(h_inj_t[l])  += W_up^T . d(up_pre_t[l])  +  W_up . d(down_pre_t[l])
  //   (level 0 only, classifier->encode path, using the *cached* enc_grad_t so a later step's
  //    classifier SGD on psi_.mu -- which still runs every step, unchanged -- cannot corrupt an
  //    earlier step's gradient):
  //     d(enc_pre_t)    = enc_grad_t * act'(enc_pre_t)
  //     d(h_inj_t[0])  += W_carry^T . d(enc_pre_t)
  //     d(w_carry_)    += outer(d(enc_pre_t), h_inj_t[0])
  //     d(w_enc_)      += outer(d(enc_pre_t), input_t)
  //   (injection path, shared W_carry_ across all levels via the level-independent
  //    `inj_acc_t = sum_j W_carry_[(j%d)*sd+i] * input_t[j]`; only its *scale* varies by level):
  //     d(acc_t[i])     = sum_l d(h_inj_t[l][i]) * (1/(l+1)) * act'(inj_acc_t[i])
  //     d(w_carry_)    += outer(d(acc_t), input_t)   [wrapped through (j % feat_dim), matching
  //                                                    inject_input_multilevel's own indexing]
  //   d(input_t)      += (encode path) W_enc^T . d(enc_pre_t)
  //                    + (injection path) sum_i d(acc_t[i]) * W_carry_[(j%d)*sd+i]
  //                      [both bounded to input_t's actually-used length `cache.in_n`, matching
  //                       the exact bounds inject_input_multilevel()/encode_level0_features()
  //                       use forward -- input positions beyond that bound have *zero* forward
  //                       influence, so their analytic gradient must be exactly zero, not merely
  //                       small, to match a finite-difference check bit-for-bit]
  //   grad_h_next[l] (for step t-1) := d(h_inj_t[l])   [h_pre_t = h_carry_t = h_inj_t - inj_t is
  //                                                      purely additive, so d(h_pre_t) is the
  //                                                      *complete* d(h_inj_t) computed above]
  //
  // Deliberate, explicitly-scoped simplification: `mem_read_t[l]`'s contribution to
  // `blended_pre_t[l]` is *not* backpropagated into `d(h_inj_t[l])` (i.e. M_slots' read path is
  // treated as a stop-gradient w.r.t. its query). This is not a silent truncation of the path
  // this section is actually about (the hierarchy/injection/encode recurrence above is exact,
  // full N-step chain rule, with no truncation beyond the window boundary itself); it is a
  // separate, bounded, and measured design choice: RPSM_UPGRADE_PLAN.md §13.3 measured
  // ||mem_read|| at ~1% of ||h|| and beta_memory=0.1, so this path's contribution to the blend
  // is ~0.1% of the state -- and its *content* is a raw, non-learned activation snapshot
  // (RpsmGlobalMemory::ring_write copies `gate*h`, not a projected key/value), so even an exact
  // query-side gradient here could not address the content-quality issue §13.3 already found.
  // Backpropagating through the attention weights exactly would additionally require caching a
  // full slot-matrix snapshot at every step (slots mutate mid-window on a `ring_write`), for a
  // provably tiny and, per §13.4, previously BPC-negative-or-neutral term. The finite-difference
  // test (native_rpsm_bptt_grad_finite_diff) sets `beta_memory=0` so this scope decision cannot
  // silently mask a real error in the terms that *are* verified here.
  const int sd = cfg_.state_dim;
  const int d = cfg_.feat_dim;
  const int nl = static_cast<int>(h_levels_.size());
  const int window = static_cast<int>(bptt_cache_.size());
  if (window <= 0 || sd <= 0) {
    return;
  }

  std::vector<double> d_w_up(static_cast<std::size_t>(sd) * static_cast<std::size_t>(sd), 0.0);
  std::vector<double> d_w_enc(static_cast<std::size_t>(d) * static_cast<std::size_t>(sd), 0.0);
  std::vector<double> d_w_carry(static_cast<std::size_t>(d) * static_cast<std::size_t>(sd), 0.0);

  std::vector<std::vector<double>> grad_h_next(
      static_cast<std::size_t>(nl), std::vector<double>(static_cast<std::size_t>(sd), 0.0));

  bptt_flush_input_grads_.assign(static_cast<std::size_t>(window),
                                  std::vector<double>(static_cast<std::size_t>(sd), 0.0));

  double lr_sum = 0.0;
  std::vector<double> tmp_a(static_cast<std::size_t>(sd));
  std::vector<double> tmp_b(static_cast<std::size_t>(sd));

  for (int t = window - 1; t >= 0; --t) {
    const BpttStepCache& c = bptt_cache_[static_cast<std::size_t>(t)];
    lr_sum += c.effective_lr;
    auto& ig = bptt_flush_input_grads_[static_cast<std::size_t>(t)];

    std::vector<std::vector<double>> d_h_inj(
        static_cast<std::size_t>(nl), std::vector<double>(static_cast<std::size_t>(sd), 0.0));

    const double hier_norm = (cfg_.hierarchy_loss_weight > 0.0)
        ? (2.0 * cfg_.hierarchy_loss_weight / static_cast<double>(nl * sd))
        : 0.0;

    for (int l = 0; l < nl; ++l) {
      const auto& h_inj = c.h_inj[static_cast<std::size_t>(l)];
      const auto& up_pre = c.up_pre[static_cast<std::size_t>(l)];
      const auto& down_pre = c.down_pre[static_cast<std::size_t>(l)];
      const auto& blended_pre = c.blended_pre[static_cast<std::size_t>(l)];
      const auto& g_next = grad_h_next[static_cast<std::size_t>(l)];
      auto& d_hi = d_h_inj[static_cast<std::size_t>(l)];

      std::vector<double> d_up_pre(static_cast<std::size_t>(sd));
      std::vector<double> d_down_pre(static_cast<std::size_t>(sd));
      for (int i = 0; i < sd; ++i) {
        const double g_blend =
            g_next[static_cast<std::size_t>(i)] *
            activation_derivative(activation_mix_, blended_pre[static_cast<std::size_t>(i)]);
        const double up_i = apply_activation(up_pre[static_cast<std::size_t>(i)]);
        const double down_i = apply_activation(down_pre[static_cast<std::size_t>(i)]);
        const double err_i = up_i - down_i;
        d_hi[static_cast<std::size_t>(i)] += (1.0 - c.alpha) * g_blend;
        const double d_err = c.alpha * g_blend + hier_norm * err_i;
        d_up_pre[static_cast<std::size_t>(i)] =
            d_err * activation_derivative(activation_mix_, up_pre[static_cast<std::size_t>(i)]);
        d_down_pre[static_cast<std::size_t>(i)] =
            -d_err * activation_derivative(activation_mix_, down_pre[static_cast<std::size_t>(i)]);
      }

      // d(h_inj) += W_up^T . d(up_pre)   [up_pre[i] = sum_c W_up[i*sd+c] * h_inj[c]]
      matvec_transpose_row_major(w_up_.data(), sd, sd, d_up_pre.data(), tmp_a.data());
      // d(h_inj) += W_up . d(down_pre)   [down_pre[i] = sum_r W_up[r*sd+i] * h_inj[r]]
      matvec_row_major(w_up_.data(), sd, sd, d_down_pre.data(), tmp_b.data());
      for (int i = 0; i < sd; ++i) {
        d_hi[static_cast<std::size_t>(i)] +=
            tmp_a[static_cast<std::size_t>(i)] + tmp_b[static_cast<std::size_t>(i)];
      }

      for (int i = 0; i < sd; ++i) {
        const double dup = d_up_pre[static_cast<std::size_t>(i)];
        const double ddown = d_down_pre[static_cast<std::size_t>(i)];
        for (int cc = 0; cc < sd; ++cc) {
          d_w_up[static_cast<std::size_t>(i * sd + cc)] +=
              dup * h_inj[static_cast<std::size_t>(cc)];
          d_w_up[static_cast<std::size_t>(cc * sd + i)] +=
              h_inj[static_cast<std::size_t>(cc)] * ddown;
        }
      }
    }

    const int bound = std::min(c.in_n, sd);

    // Level-0 classifier->encode path.
    {
      const auto& h_inj0 = c.h_inj[0];
      std::vector<double> d_enc_pre(static_cast<std::size_t>(d));
      for (int j = 0; j < d; ++j) {
        d_enc_pre[static_cast<std::size_t>(j)] =
            c.enc_grad[static_cast<std::size_t>(j)] *
            activation_derivative(activation_mix_, c.enc_pre[static_cast<std::size_t>(j)]);
      }
      for (int j = 0; j < d; ++j) {
        const double dep = d_enc_pre[static_cast<std::size_t>(j)];
        for (int i = 0; i < sd; ++i) {
          d_h_inj[0][static_cast<std::size_t>(i)] += w_carry_[static_cast<std::size_t>(j * sd + i)] * dep;
          d_w_carry[static_cast<std::size_t>(j * sd + i)] += dep * h_inj0[static_cast<std::size_t>(i)];
        }
        for (int i = 0; i < bound; ++i) {
          d_w_enc[static_cast<std::size_t>(j * sd + i)] += dep * c.input[static_cast<std::size_t>(i)];
          ig[static_cast<std::size_t>(i)] += w_enc_[static_cast<std::size_t>(j * sd + i)] * dep;
        }
      }
    }

    // Injection path (shared W_carry_ across all levels, level-independent `inj_acc`).
    {
      std::vector<double> d_acc(static_cast<std::size_t>(sd), 0.0);
      for (int l = 0; l < nl; ++l) {
        const double scale = 1.0 / static_cast<double>(l + 1);
        for (int i = 0; i < sd; ++i) {
          d_acc[static_cast<std::size_t>(i)] +=
              d_h_inj[static_cast<std::size_t>(l)][static_cast<std::size_t>(i)] * scale *
              activation_derivative(activation_mix_, c.inj_acc[static_cast<std::size_t>(i)]);
        }
      }
      for (int j = 0; j < bound; ++j) {
        double acc_grad = 0.0;
        for (int i = 0; i < sd; ++i) {
          acc_grad += d_acc[static_cast<std::size_t>(i)] *
                      w_carry_[static_cast<std::size_t>((j % d) * sd + i)];
        }
        ig[static_cast<std::size_t>(j)] += acc_grad;
      }
      for (int j = 0; j < sd; ++j) {
        for (int i = 0; i < sd; ++i) {
          d_w_carry[static_cast<std::size_t>((j % d) * sd + i)] +=
              d_acc[static_cast<std::size_t>(i)] * c.input[static_cast<std::size_t>(j)];
        }
      }
    }

    grad_h_next = std::move(d_h_inj);
  }

  const double mean_lr = lr_sum / static_cast<double>(window);
  const double inv_n = 1.0 / static_cast<double>(window);
  for (std::size_t idx = 0; idx < w_up_.size(); ++idx) {
    w_up_[idx] -= mean_lr * d_w_up[idx] * inv_n;
  }
  for (std::size_t idx = 0; idx < w_enc_.size(); ++idx) {
    w_enc_[idx] -= mean_lr * d_w_enc[idx] * inv_n;
  }
  for (std::size_t idx = 0; idx < w_carry_.size(); ++idx) {
    w_carry_[idx] -= mean_lr * d_w_carry[idx] * inv_n;
  }
}

}  // namespace cypha::rpsm
