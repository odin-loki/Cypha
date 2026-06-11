#include "cypha/cyphalm/gria_lowrank.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace cypha {
namespace cyphalm {

namespace {

constexpr double kLogEps = 1e-12;

void matvec_rowmajor(const double* M, int rows, int cols, const double* x, double* y) {
  for (int r = 0; r < rows; ++r) {
    double s = 0.0;
    const double* row = M + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
    for (int c = 0; c < cols; ++c) {
      s += row[c] * x[c];
    }
    y[r] = s;
  }
}

void outer_rowmajor(const double* a, int na, const double* b, int nb, double* out) {
  for (int i = 0; i < na; ++i) {
    for (int j = 0; j < nb; ++j) {
      out[static_cast<std::size_t>(i) * static_cast<std::size_t>(nb) + static_cast<std::size_t>(j)] =
          a[i] * b[j];
    }
  }
}

void softmax_logits(const double* logits, int n, double* probs) {
  double mx = logits[0];
  for (int i = 1; i < n; ++i) {
    mx = std::max(mx, logits[i]);
  }
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    probs[i] = std::exp(logits[i] - mx);
    sum += probs[i];
  }
  for (int i = 0; i < n; ++i) {
    probs[i] /= (sum + kLogEps);
  }
}

}  // namespace

GRIALowRank::GRIALowRank(int field_dim_in, int vocab_size_in, int rank_in, double alpha_init, bool alpha_learnable_in,
                           std::uint64_t seed) {
  field_dim = field_dim_in;
  vocab_size = vocab_size_in;
  rank = rank_in;
  alpha_learnable = alpha_learnable_in;
  U.assign(static_cast<std::size_t>(field_dim) * static_cast<std::size_t>(rank), 0.0);
  V.assign(static_cast<std::size_t>(rank) * static_cast<std::size_t>(vocab_size), 0.0);
  alpha.assign(static_cast<std::size_t>(vocab_size), alpha_init);
  bias.assign(static_cast<std::size_t>(vocab_size), 0.0);

  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  constexpr double kScale = 0.01;
  for (auto& v : U) v = nd(rng) * kScale;
  for (auto& v : V) v = nd(rng) * kScale;
}

void GRIALowRank::logits(const double* v, double* z_out) const {
  std::vector<double> h(static_cast<std::size_t>(rank));
  for (int r = 0; r < rank; ++r) {
    double s = 0.0;
    for (int j = 0; j < field_dim; ++j) {
      s += U[static_cast<std::size_t>(j) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)] * v[j];
    }
    h[static_cast<std::size_t>(r)] = s;
  }
  for (int k = 0; k < vocab_size; ++k) {
    double s = 0.0;
    for (int r = 0; r < rank; ++r) {
      s += V[static_cast<std::size_t>(r) * static_cast<std::size_t>(vocab_size) + static_cast<std::size_t>(k)] *
           h[static_cast<std::size_t>(r)];
    }
    z_out[k] = alpha[static_cast<std::size_t>(k)] * s +
               (1.0 - alpha[static_cast<std::size_t>(k)]) * bias[static_cast<std::size_t>(k)];
  }
}

void GRIALowRank::forward(const double* v, double* log_probs_out) const {
  std::vector<double> z(static_cast<std::size_t>(vocab_size));
  logits(v, z.data());
  std::vector<double> probs(static_cast<std::size_t>(vocab_size));
  softmax_logits(z.data(), vocab_size, probs.data());
  for (int k = 0; k < vocab_size; ++k) {
    log_probs_out[k] = std::log(probs[static_cast<std::size_t>(k)] + kLogEps);
  }
}

GRIALowRankGrad GRIALowRank::cross_entropy_gradients(const double* v, int target_id) const {
  if (target_id < 0 || target_id >= vocab_size) {
    throw std::runtime_error("gria_lowrank: target_id out of range");
  }
  GRIALowRankGrad out;
  out.dU.assign(static_cast<std::size_t>(field_dim) * static_cast<std::size_t>(rank), 0.0);
  out.dV.assign(static_cast<std::size_t>(rank) * static_cast<std::size_t>(vocab_size), 0.0);
  out.d_alpha.assign(static_cast<std::size_t>(vocab_size), 0.0);
  out.d_bias.assign(static_cast<std::size_t>(vocab_size), 0.0);
  out.dv.assign(static_cast<std::size_t>(field_dim), 0.0);

  std::vector<double> h(static_cast<std::size_t>(rank));
  for (int r = 0; r < rank; ++r) {
    double s = 0.0;
    for (int j = 0; j < field_dim; ++j) {
      s += U[static_cast<std::size_t>(j) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)] * v[j];
    }
    h[static_cast<std::size_t>(r)] = s;
  }

  std::vector<double> z(static_cast<std::size_t>(vocab_size));
  for (int k = 0; k < vocab_size; ++k) {
    double s = 0.0;
    for (int r = 0; r < rank; ++r) {
      s += V[static_cast<std::size_t>(r) * static_cast<std::size_t>(vocab_size) + static_cast<std::size_t>(k)] *
           h[static_cast<std::size_t>(r)];
    }
    z[static_cast<std::size_t>(k)] = s;
  }

  std::vector<double> logits(static_cast<std::size_t>(vocab_size));
  for (int k = 0; k < vocab_size; ++k) {
    logits[static_cast<std::size_t>(k)] =
        alpha[static_cast<std::size_t>(k)] * z[static_cast<std::size_t>(k)] +
        (1.0 - alpha[static_cast<std::size_t>(k)]) * bias[static_cast<std::size_t>(k)];
  }

  std::vector<double> probs(static_cast<std::size_t>(vocab_size));
  softmax_logits(logits.data(), vocab_size, probs.data());

  std::vector<double> d_logits = probs;
  d_logits[static_cast<std::size_t>(target_id)] -= 1.0;

  std::vector<double> dz(static_cast<std::size_t>(vocab_size));
  for (int k = 0; k < vocab_size; ++k) {
    dz[static_cast<std::size_t>(k)] = d_logits[static_cast<std::size_t>(k)] * alpha[static_cast<std::size_t>(k)];
    out.d_alpha[static_cast<std::size_t>(k)] =
        d_logits[static_cast<std::size_t>(k)] * (z[static_cast<std::size_t>(k)] - bias[static_cast<std::size_t>(k)]);
    out.d_bias[static_cast<std::size_t>(k)] =
        d_logits[static_cast<std::size_t>(k)] * (1.0 - alpha[static_cast<std::size_t>(k)]);
  }

  outer_rowmajor(h.data(), rank, dz.data(), vocab_size, out.dV.data());

  std::vector<double> dh(static_cast<std::size_t>(rank));
  for (int r = 0; r < rank; ++r) {
    double s = 0.0;
    for (int k = 0; k < vocab_size; ++k) {
      s += V[static_cast<std::size_t>(r) * static_cast<std::size_t>(vocab_size) + static_cast<std::size_t>(k)] *
           dz[static_cast<std::size_t>(k)];
    }
    dh[static_cast<std::size_t>(r)] = s;
  }

  outer_rowmajor(v, field_dim, dh.data(), rank, out.dU.data());

  for (int j = 0; j < field_dim; ++j) {
    double s = 0.0;
    for (int r = 0; r < rank; ++r) {
      s += U[static_cast<std::size_t>(j) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)] *
           dh[static_cast<std::size_t>(r)];
    }
    out.dv[static_cast<std::size_t>(j)] = s;
  }

  return out;
}

void GRIALowRank::update_weights(const GRIALowRankGrad& grad, double lr) {
  for (std::size_t i = 0; i < U.size(); ++i) U[i] -= lr * grad.dU[i];
  for (std::size_t i = 0; i < V.size(); ++i) V[i] -= lr * grad.dV[i];
}

void GRIALowRank::update_alpha(const GRIALowRankGrad& grad, double lr) {
  if (!alpha_learnable) {
    return;
  }
  for (int k = 0; k < vocab_size; ++k) {
    alpha[static_cast<std::size_t>(k)] -= lr * grad.d_alpha[static_cast<std::size_t>(k)];
    alpha[static_cast<std::size_t>(k)] = std::max(0.01, std::min(0.99, alpha[static_cast<std::size_t>(k)]));
  }
}

void GRIALowRank::update_bias(const GRIALowRankGrad& grad, double lr) {
  for (int k = 0; k < vocab_size; ++k) {
    bias[static_cast<std::size_t>(k)] -= lr * grad.d_bias[static_cast<std::size_t>(k)];
  }
}

double GRIALowRank::train_step(const double* v, int target_id, double lr) {
  std::vector<double> log_probs(static_cast<std::size_t>(vocab_size));
  forward(v, log_probs.data());
  const double loss = -log_probs[static_cast<std::size_t>(target_id)];
  GRIALowRankGrad grad = cross_entropy_gradients(v, target_id);
  update_weights(grad, lr);
  update_alpha(grad, lr);
  update_bias(grad, lr);
  return loss;
}

void GRIALowRank::set_laplace_prior(const double* token_counts, int n, double smoothing) {
  const int m = std::min(n, vocab_size);
  double sum = 0.0;
  for (int k = 0; k < vocab_size; ++k) {
    const double c = (k < m && token_counts) ? token_counts[k] : 1.0;
    const double sm = c + smoothing;
    bias[static_cast<std::size_t>(k)] = std::log(sm + 1e-12);
    sum += sm;
  }
  const double log_z = std::log(sum + 1e-12);
  for (int k = 0; k < vocab_size; ++k) {
    bias[static_cast<std::size_t>(k)] -= log_z;
  }
}

std::vector<double> GRIALowRank::grad_v_cross_entropy(const double* v, int target_id) const {
  return cross_entropy_gradients(v, target_id).dv;
}

void GRIALowRank::load_state(const std::vector<double>& u, const std::vector<double>& v,
                             const std::vector<double>& alpha_in, const std::vector<double>& bias_in,
                             bool alpha_learnable_in) {
  U = u;
  V = v;
  alpha = alpha_in;
  bias = bias_in;
  alpha_learnable = alpha_learnable_in;
  if (!U.empty() && !V.empty()) {
    rank = static_cast<int>(U.size()) / std::max(1, field_dim);
  }
}

void GRIALowRank::load_from_full_w(const std::vector<double>& w_vocab_x_d, int d_input, int target_rank) {
  const int vs = vocab_size;
  const int d = d_input;
  rank = std::max(1, std::min({target_rank, d, vs}));
  U.assign(static_cast<std::size_t>(d) * static_cast<std::size_t>(rank), 0.0);
  V.assign(static_cast<std::size_t>(rank) * static_cast<std::size_t>(vs), 0.0);

  std::mt19937_64 rng(42);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (auto& v : U) v = nd(rng);

  auto dot_col_u = [&](int r, int r2) {
    double s = 0.0;
    for (int j = 0; j < d; ++j) {
      s += U[static_cast<std::size_t>(j) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)] *
           U[static_cast<std::size_t>(j) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r2)];
    }
    return s;
  };

  for (int iter = 0; iter < 24; ++iter) {
    std::vector<double> vtmp(static_cast<std::size_t>(vs) * static_cast<std::size_t>(rank), 0.0);
    for (int k = 0; k < vs; ++k) {
      for (int r = 0; r < rank; ++r) {
        double s = 0.0;
        for (int j = 0; j < d; ++j) {
          const double wkj = w_vocab_x_d[static_cast<std::size_t>(k) * static_cast<std::size_t>(d) +
                                         static_cast<std::size_t>(j)];
          s += wkj * U[static_cast<std::size_t>(j) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)];
        }
        vtmp[static_cast<std::size_t>(k) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)] = s;
      }
    }
    for (int r = 0; r < rank; ++r) {
      for (int rp = 0; rp < r; ++rp) {
        double proj = 0.0;
        for (int k = 0; k < vs; ++k) {
          proj += vtmp[static_cast<std::size_t>(k) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)] *
                  vtmp[static_cast<std::size_t>(k) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(rp)];
        }
        for (int k = 0; k < vs; ++k) {
          vtmp[static_cast<std::size_t>(k) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)] -=
              proj * vtmp[static_cast<std::size_t>(k) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(rp)];
        }
      }
      double norm = 0.0;
      for (int k = 0; k < vs; ++k) {
        const double v =
            vtmp[static_cast<std::size_t>(k) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)];
        norm += v * v;
      }
      norm = std::sqrt(norm + 1e-12);
      for (int k = 0; k < vs; ++k) {
        V[static_cast<std::size_t>(r) * static_cast<std::size_t>(vs) + static_cast<std::size_t>(k)] =
            vtmp[static_cast<std::size_t>(k) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)] / norm;
      }
    }
    for (int r = 0; r < rank; ++r) {
      for (int j = 0; j < d; ++j) {
        double s = 0.0;
        for (int k = 0; k < vs; ++k) {
          const double wkj = w_vocab_x_d[static_cast<std::size_t>(k) * static_cast<std::size_t>(d) +
                                         static_cast<std::size_t>(j)];
          s += wkj * V[static_cast<std::size_t>(r) * static_cast<std::size_t>(vs) + static_cast<std::size_t>(k)];
        }
        U[static_cast<std::size_t>(j) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)] = s;
      }
    }
    for (int r = 0; r < rank; ++r) {
      for (int rp = 0; rp < r; ++rp) {
        const double proj = dot_col_u(r, rp);
        for (int j = 0; j < d; ++j) {
          U[static_cast<std::size_t>(j) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)] -=
              proj * U[static_cast<std::size_t>(j) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(rp)];
        }
      }
      double norm = 0.0;
      for (int j = 0; j < d; ++j) {
        const double u = U[static_cast<std::size_t>(j) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)];
        norm += u * u;
      }
      norm = std::sqrt(norm + 1e-12);
      for (int j = 0; j < d; ++j) {
        U[static_cast<std::size_t>(j) * static_cast<std::size_t>(rank) + static_cast<std::size_t>(r)] /= norm;
      }
    }
  }
}

std::map<std::string, double> GRIALowRank::alpha_spectrum() const {
  if (alpha.empty()) {
    return {{"mean", 0.5}, {"std", 0.0}, {"min", 0.5}, {"max", 0.5}};
  }
  double sum = 0.0;
  double sum_sq = 0.0;
  double mn = alpha[0];
  double mx = alpha[0];
  for (double a : alpha) {
    sum += a;
    sum_sq += a * a;
    mn = std::min(mn, a);
    mx = std::max(mx, a);
  }
  const double n = static_cast<double>(alpha.size());
  const double mean = sum / n;
  const double var = std::max(0.0, sum_sq / n - mean * mean);
  return {{"mean", mean},
          {"std", std::sqrt(var)},
          {"min", mn},
          {"max", mx}};
}

}  // namespace cyphalm
}  // namespace cypha
