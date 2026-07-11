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

void RpsmSequenceLayer::inject_input_multilevel(const double* input, int input_dim) {
  const int sd = cfg_.state_dim;
  const int d = cfg_.feat_dim;
  const int in_n = std::max(0, input_dim);
  const int nl = static_cast<int>(h_levels_.size());

  for (int l = 0; l < nl; ++l) {
    const double scale = 1.0 / static_cast<double>(l + 1);
    auto& h = h_levels_[static_cast<std::size_t>(l)];
    for (int i = 0; i < sd; ++i) {
      double acc = 0.0;
      for (int j = 0; j < in_n && j < sd; ++j) {
        acc += w_carry_[static_cast<std::size_t>((j % d) * sd + i)] * input[j];
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

double RpsmSequenceLayer::hierarchy_update() {
  const int sd = cfg_.state_dim;
  const int nl = static_cast<int>(h_levels_.size());
  const double a = cfg_.alpha_carry;
  const double beta = cfg_.beta_memory;
  double hierarchy_loss = 0.0;

  for (int l = 0; l < nl; ++l) {
    auto& h = h_levels_[static_cast<std::size_t>(l)];
    matvec_row_major(w_up_.data(), sd, sd, h.data(), work_up_.data());
    for (int i = 0; i < sd; ++i) {
      work_up_[static_cast<std::size_t>(i)] = apply_activation(work_up_[static_cast<std::size_t>(i)]);
    }
    matvec_transpose_row_major(w_up_.data(), sd, sd, h.data(), work_down_.data());
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

  inject_input_multilevel(input, input_dim);
  encode_level0_features(input, input_dim);

  batched_llr_gemm(feat_buf_.data(), 1, psi_, nullptr, llr_buf_.data());

  std::vector<double> log_probs(static_cast<std::size_t>(k));
  log_softmax_row(llr_buf_.data(), k, log_probs.data());
  metrics.nll = -log_probs[static_cast<std::size_t>(tgt)];

  const double hierarchy_loss = hierarchy_update();
  metrics.hierarchy_loss = hierarchy_loss;
  metrics.surprise = last_surprise_;
  metrics.loss = metrics.nll + cfg_.hierarchy_loss_weight * hierarchy_loss;

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
      delta[j] -= lr * grad;
    }
  }

  const auto& h0 = h_levels_[0];
  std::fill(input_grad_.begin(), input_grad_.end(), 0.0);
  for (int j = 0; j < d; ++j) {
    const double pre = enc_pre_[static_cast<std::size_t>(j)];
    const double chain =
        enc_grad_[static_cast<std::size_t>(j)] * activation_derivative(activation_mix_, pre);
    for (int i = 0; i < sd; ++i) {
      input_grad_[static_cast<std::size_t>(i)] +=
          chain * w_enc_[static_cast<std::size_t>(j * sd + i)];
      double x_in = 0.0;
      if (i < in_n) {
        x_in = input[i];
      }
      w_enc_[static_cast<std::size_t>(j * sd + i)] -= lr * chain * x_in;
      w_carry_[static_cast<std::size_t>(j * sd + i)] -= lr * chain * h0[static_cast<std::size_t>(i)];
    }
  }

  const double hier_w = cfg_.hierarchy_loss_weight;
  if (hier_w > 0.0 && hierarchy_loss > kEps) {
    const double hier_scale = 2.0 * hier_w * lr / static_cast<double>(cfg_.n_levels);
    for (int l = 0; l < cfg_.n_levels; ++l) {
      const auto& h = h_levels_[static_cast<std::size_t>(l)];
      for (int i = 0; i < sd; ++i) {
        const double err = work_err_[static_cast<std::size_t>(i)];
        for (int j = 0; j < sd; ++j) {
          w_up_[static_cast<std::size_t>(i * sd + j)] -= hier_scale * err * h[static_cast<std::size_t>(j)];
        }
      }
    }
  }

  if (tgt >= 0 && tgt < k) {
    psi_.counts[static_cast<std::size_t>(tgt)] += 1.0;
  }

  return metrics;
}

}  // namespace cypha::rpsm
