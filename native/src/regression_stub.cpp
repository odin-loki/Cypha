// M4 — mixture-of-experts regression helpers (see `include/cypha/regression_stub.hpp`).
#include <cmath>
#include <cstddef>
#include <vector>

#include "cypha/infer_cpu.hpp"
#include "cypha/regression_stub.hpp"

namespace {

bool cholesky_factor_spd(const std::vector<double>& A, int n, std::vector<double>& L) {
  L.assign(static_cast<std::size_t>(n * n), 0.0);
  for (int j = 0; j < n; ++j) {
    double sum = 0.0;
    for (int k = 0; k < j; ++k) {
      sum += L[static_cast<std::size_t>(j * n + k)] * L[static_cast<std::size_t>(j * n + k)];
    }
    const double ajj = A[static_cast<std::size_t>(j * n + j)] - sum;
    if (ajj <= 1e-30) {
      return false;
    }
    L[static_cast<std::size_t>(j * n + j)] = std::sqrt(ajj);
    for (int i = j + 1; i < n; ++i) {
      sum = 0.0;
      for (int k = 0; k < j; ++k) {
        sum += L[static_cast<std::size_t>(i * n + k)] * L[static_cast<std::size_t>(j * n + k)];
      }
      L[static_cast<std::size_t>(i * n + j)] =
          (A[static_cast<std::size_t>(i * n + j)] - sum) / L[static_cast<std::size_t>(j * n + j)];
    }
  }
  return true;
}

void forward_subst_lower(const double* L, int n, double* x) {
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int k = 0; k < i; ++k) {
      s += L[static_cast<std::size_t>(i * n + k)] * x[k];
    }
    x[i] = (x[i] - s) / L[static_cast<std::size_t>(i * n + i)];
  }
}

void backward_subst_lt(const double* L, int n, double* x) {
  for (int i = n - 1; i >= 0; --i) {
    double s = 0.0;
    for (int k = i + 1; k < n; ++k) {
      s += L[static_cast<std::size_t>(k * n + i)] * x[k];
    }
    x[i] = (x[i] - s) / L[static_cast<std::size_t>(i * n + i)];
  }
}

}  // namespace

namespace cypha::regression {

int native_regression_milestone() { return k_native_regression_milestone; }

void predict_mixture_scalar(const double* probs, const double* mu, const double* var_ema,
                            std::size_t k, double& out_y_pred, double& out_uncertainty) {
  double y = 0.0;
  double v = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    y += probs[i] * mu[i];
    v += probs[i] * var_ema[i];
  }
  out_y_pred = y;
  out_uncertainty = std::sqrt(std::max(v, 0.0));
}

void predict_mixture_batch(const double* probs, int n, int k, int d, const double* mu_mat,
                           const double* var_vec, double* out_y, double* out_unc) {
  for (int i = 0; i < n; ++i) {
    const double* pi = probs + static_cast<std::size_t>(i) * static_cast<std::size_t>(k);
    for (int j = 0; j < d; ++j) {
      double s = 0.0;
      for (int c = 0; c < k; ++c) {
        s += pi[static_cast<std::size_t>(c)] * mu_mat[static_cast<std::size_t>(c * d + j)];
      }
      out_y[static_cast<std::size_t>(i * d + j)] = s;
    }
    double v = 0.0;
    for (int c = 0; c < k; ++c) {
      v += pi[static_cast<std::size_t>(c)] * var_vec[static_cast<std::size_t>(c)];
    }
    out_unc[static_cast<std::size_t>(i)] = std::sqrt(std::max(v, 0.0));
  }
}

void expert_target_ema_step(std::vector<double>& mu, double& var_ema, int& n_updates, const double* y, int d,
                            double lr) {
  if (d < 1) {
    return;
  }
  if (mu.size() != static_cast<std::size_t>(d)) {
    mu.assign(y, y + d);
    var_ema = 0.0;
    n_updates = 1;
    return;
  }
  double dot = 0.0;
  for (int j = 0; j < d; ++j) {
    const std::size_t jj = static_cast<std::size_t>(j);
    double delta = y[j] - mu[jj];
    dot += delta * delta;
    mu[jj] += lr * delta;
  }
  var_ema = (1.0 - lr) * var_ema + lr * dot;
  n_updates += 1;
}

void rff_encode_batch_rowmajor(const double* X, int n, int d_in, const double* W, const double* b, int D,
                               double* out_phi) {
  if (n < 1 || d_in < 1 || D < 1) {
    return;
  }
  const double scale = std::sqrt(2.0 / static_cast<double>(D));
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < D; ++d) {
      double z = b[static_cast<std::size_t>(d)];
      for (int j = 0; j < d_in; ++j) {
        z += X[static_cast<std::size_t>(i * d_in + j)] *
             W[static_cast<std::size_t>(d * d_in + j)];
      }
      out_phi[static_cast<std::size_t>(i * D + d)] = scale * std::cos(z);
    }
  }
}

bool ridge_fit_bias(const double* Phi, int n, int d_feat, double lam, const double* y_norm, double* out_coef) {
  if (n < 1 || d_feat < 1) {
    return false;
  }
  const int p = d_feat + 1;
  std::vector<double> A(static_cast<std::size_t>(p * p), 0.0);
  std::vector<double> rhs(static_cast<std::size_t>(p), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int a = 0; a < p; ++a) {
      const double va = (a < d_feat) ? Phi[static_cast<std::size_t>(i * d_feat + a)] : 1.0;
      for (int b = 0; b < p; ++b) {
        const double vb = (b < d_feat) ? Phi[static_cast<std::size_t>(i * d_feat + b)] : 1.0;
        A[static_cast<std::size_t>(a * p + b)] += va * vb;
      }
      rhs[static_cast<std::size_t>(a)] += va * y_norm[i];
    }
  }
  for (int j = 0; j < d_feat; ++j) {
    A[static_cast<std::size_t>(j * p + j)] += lam;
  }
  std::vector<double> L;
  if (!cholesky_factor_spd(A, p, L)) {
    return false;
  }
  for (int j = 0; j < p; ++j) {
    out_coef[j] = rhs[static_cast<std::size_t>(j)];
  }
  forward_subst_lower(L.data(), p, out_coef);
  backward_subst_lt(L.data(), p, out_coef);
  return true;
}

void linear_predict_with_bias(const double* Phi, int n, int d_feat, const double* coef, double* out_pred) {
  const double bias = coef[static_cast<std::size_t>(d_feat)];
  for (int i = 0; i < n; ++i) {
    double s = bias;
    for (int j = 0; j < d_feat; ++j) {
      s += Phi[static_cast<std::size_t>(i * d_feat + j)] * coef[static_cast<std::size_t>(j)];
    }
    out_pred[static_cast<std::size_t>(i)] = s;
  }
}

void mke_expert_linear_dots(const double* phi, int d_feat, int K, const double* W_rowmajor, double* out_dots) {
  for (int k = 0; k < K; ++k) {
    double s = 0.0;
    for (int j = 0; j < d_feat; ++j) {
      s += phi[static_cast<std::size_t>(j)] * W_rowmajor[static_cast<std::size_t>(k * d_feat + j)];
    }
    out_dots[static_cast<std::size_t>(k)] = s;
  }
}

double rff_rls_train_step(const double* phi, int D, double* w, double* b, double* P_rowmajor, double y_raw,
                          double y_mean, double y_std) {
  if (D < 1) {
    return 0.0;
  }
  const int Dp = D + 1;
  std::vector<double> phi_b(static_cast<std::size_t>(Dp));
  for (int i = 0; i < D; ++i) {
    phi_b[static_cast<std::size_t>(i)] = phi[static_cast<std::size_t>(i)];
  }
  phi_b[static_cast<std::size_t>(D)] = 1.0;

  const double yn = (y_raw - y_mean) / y_std;
  double pred = *b;
  for (int i = 0; i < D; ++i) {
    pred += phi[static_cast<std::size_t>(i)] * w[static_cast<std::size_t>(i)];
  }
  const double err = yn - pred;

  std::vector<double> Pp(static_cast<std::size_t>(Dp));
  for (int i = 0; i < Dp; ++i) {
    double s = 0.0;
    for (int j = 0; j < Dp; ++j) {
      s += P_rowmajor[static_cast<std::size_t>(i * Dp + j)] * phi_b[static_cast<std::size_t>(j)];
    }
    Pp[static_cast<std::size_t>(i)] = s;
  }
  double denom = 1.0;
  for (int j = 0; j < Dp; ++j) {
    denom += phi_b[static_cast<std::size_t>(j)] * Pp[static_cast<std::size_t>(j)];
  }
  if (denom <= 0.0) {
    denom = 1e-30;
  }
  for (int i = 0; i < Dp; ++i) {
    for (int j = 0; j < Dp; ++j) {
      P_rowmajor[static_cast<std::size_t>(i * Dp + j)] -=
          Pp[static_cast<std::size_t>(i)] * Pp[static_cast<std::size_t>(j)] / denom;
    }
  }
  for (int i = 0; i < Dp; ++i) {
    const double delta_i = Pp[static_cast<std::size_t>(i)] / denom * err;
    if (i < D) {
      w[static_cast<std::size_t>(i)] += delta_i;
    } else {
      *b += delta_i;
    }
  }
  return err * err * y_std * y_std;
}

void mke_expert_rls_scalar_step(const double* phi, int D, double pi, double gh_scale, double err,
                                double forgetting_factor, double* w, double* P_rowmajor) {
  if (D < 1 || pi < 0.02) {
    return;
  }
  if (forgetting_factor > 0.0 && forgetting_factor < 1.0) {
    const double inv_ff = 1.0 / forgetting_factor;
    for (int i = 0; i < D * D; ++i) {
      P_rowmajor[static_cast<std::size_t>(i)] *= inv_ff;
    }
  }
  std::vector<double> Pphi(static_cast<std::size_t>(D));
  for (int i = 0; i < D; ++i) {
    double s = 0.0;
    for (int j = 0; j < D; ++j) {
      s += P_rowmajor[static_cast<std::size_t>(i * D + j)] * phi[static_cast<std::size_t>(j)];
    }
    Pphi[static_cast<std::size_t>(i)] = s;
  }
  double phi_Pphi = 0.0;
  for (int j = 0; j < D; ++j) {
    phi_Pphi += phi[static_cast<std::size_t>(j)] * Pphi[static_cast<std::size_t>(j)];
  }
  double denom = 1.0 + pi * phi_Pphi;
  if (denom <= 0.0) {
    denom = 1e-30;
  }
  std::vector<double> Kg(static_cast<std::size_t>(D));
  for (int i = 0; i < D; ++i) {
    Kg[static_cast<std::size_t>(i)] = pi * Pphi[static_cast<std::size_t>(i)] / denom * gh_scale;
    w[static_cast<std::size_t>(i)] += Kg[static_cast<std::size_t>(i)] * err;
  }
  // P -= pi * outer(Kg, phi) @ P
  std::vector<double> OP(static_cast<std::size_t>(D * D), 0.0);
  for (int i = 0; i < D; ++i) {
    for (int j = 0; j < D; ++j) {
      double s = 0.0;
      for (int m = 0; m < D; ++m) {
        s += Kg[static_cast<std::size_t>(i)] * phi[static_cast<std::size_t>(m)] *
             P_rowmajor[static_cast<std::size_t>(m * D + j)];
      }
      OP[static_cast<std::size_t>(i * D + j)] = pi * s;
    }
  }
  for (int i = 0; i < D * D; ++i) {
    P_rowmajor[static_cast<std::size_t>(i)] -= OP[static_cast<std::size_t>(i)];
  }
}

double two_stage_dif_predict(const double* llr, int K, const double* x, int d_in, const double* w1, double b1,
                             const double* phi2, int D2, const double* w2, double b2, double y_mean,
                             double y_std) {
  double y_norm = b1;
  for (int i = 0; i < K; ++i) {
    y_norm += llr[static_cast<std::size_t>(i)] * w1[static_cast<std::size_t>(i)];
  }
  for (int j = 0; j < d_in; ++j) {
    y_norm += x[static_cast<std::size_t>(j)] * w1[static_cast<std::size_t>(K + j)];
  }
  double s2 = b2;
  for (int j = 0; j < D2; ++j) {
    s2 += phi2[static_cast<std::size_t>(j)] * w2[static_cast<std::size_t>(j)];
  }
  y_norm += s2;
  return y_norm * y_std + y_mean;
}

void two_stage_dif_predict_batch(const double* llr, int n, int K, const double* X, int d_in, const double* w1,
                                 double b1, const double* phi2, int D2, const double* w2, double b2, double y_mean,
                                 double y_std, double* y_out) {
  if (n < 1 || y_out == nullptr) {
    return;
  }
  for (int i = 0; i < n; ++i) {
    y_out[static_cast<std::size_t>(i)] = two_stage_dif_predict(
        llr + static_cast<std::size_t>(i * K), K, X + static_cast<std::size_t>(i * d_in), d_in, w1, b1,
        phi2 + static_cast<std::size_t>(i * D2), D2, w2, b2, y_mean, y_std);
  }
}

bool two_stage_dif_ridge_fit_from_llr(const double* llr_rowmajor, int n, int K, const double* X_rowmajor, int d_in,
                                    const double* y_raw, double y_mean, double y_std, double lam1, double lam2,
                                    const double* enc2_W, const double* enc2_b, int D2, double* out_w1, double* out_b1,
                                    double* out_w2, double* out_b2) {
  if (n < 1 || K < 1 || d_in < 1 || D2 < 1 || llr_rowmajor == nullptr || X_rowmajor == nullptr ||
      y_raw == nullptr || enc2_W == nullptr || enc2_b == nullptr || out_w1 == nullptr || out_b1 == nullptr ||
      out_w2 == nullptr || out_b2 == nullptr) {
    return false;
  }
  const int p1 = K + d_in + 1;
  const double ys = std::max(y_std, 1e-8);
  std::vector<double> yn(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    yn[static_cast<std::size_t>(i)] = (y_raw[static_cast<std::size_t>(i)] - y_mean) / ys;
  }

  std::vector<double> A1(static_cast<std::size_t>(p1 * p1), 0.0);
  std::vector<double> r1(static_cast<std::size_t>(p1), 0.0);
  const double lam1s = lam1 * static_cast<double>(n);
  std::vector<double> f(static_cast<std::size_t>(p1));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < K; ++j) {
      f[static_cast<std::size_t>(j)] = llr_rowmajor[static_cast<std::size_t>(i * K + j)];
    }
    for (int j = 0; j < d_in; ++j) {
      f[static_cast<std::size_t>(K + j)] = X_rowmajor[static_cast<std::size_t>(i * d_in + j)];
    }
    f[static_cast<std::size_t>(p1 - 1)] = 1.0;
    const double yi = yn[static_cast<std::size_t>(i)];
    for (int a = 0; a < p1; ++a) {
      const double fa = f[static_cast<std::size_t>(a)];
      r1[static_cast<std::size_t>(a)] += fa * yi;
      for (int b = 0; b < p1; ++b) {
        A1[static_cast<std::size_t>(a * p1 + b)] += fa * f[static_cast<std::size_t>(b)];
      }
    }
  }
  for (int j = 0; j < p1; ++j) {
    A1[static_cast<std::size_t>(j * p1 + j)] += lam1s;
  }

  std::vector<double> coef1 = r1;
  std::vector<double> L1;
  if (!cholesky_factor_spd(A1, p1, L1)) {
    return false;
  }
  forward_subst_lower(L1.data(), p1, coef1.data());
  backward_subst_lt(L1.data(), p1, coef1.data());

  for (int j = 0; j < K + d_in; ++j) {
    out_w1[static_cast<std::size_t>(j)] = coef1[static_cast<std::size_t>(j)];
  }
  *out_b1 = coef1[static_cast<std::size_t>(p1 - 1)];

  std::vector<double> res(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    double pred = coef1[static_cast<std::size_t>(p1 - 1)];
    for (int j = 0; j < K; ++j) {
      pred += llr_rowmajor[static_cast<std::size_t>(i * K + j)] * coef1[static_cast<std::size_t>(j)];
    }
    for (int j = 0; j < d_in; ++j) {
      pred += X_rowmajor[static_cast<std::size_t>(i * d_in + j)] * coef1[static_cast<std::size_t>(K + j)];
    }
    res[static_cast<std::size_t>(i)] = yn[static_cast<std::size_t>(i)] - pred;
  }

  std::vector<double> Phi(static_cast<std::size_t>(n * D2));
  rff_encode_batch_rowmajor(X_rowmajor, n, d_in, enc2_W, enc2_b, D2, Phi.data());

  const int p2 = D2 + 1;
  std::vector<double> A2(static_cast<std::size_t>(p2 * p2), 0.0);
  std::vector<double> r2(static_cast<std::size_t>(p2), 0.0);
  const double lam2s = lam2 * static_cast<double>(n);

  for (int i = 0; i < n; ++i) {
    for (int a = 0; a < p2; ++a) {
      const double fa =
          (a < D2) ? Phi[static_cast<std::size_t>(i * D2 + a)] : 1.0;
      r2[static_cast<std::size_t>(a)] += fa * res[static_cast<std::size_t>(i)];
      for (int b = 0; b < p2; ++b) {
        const double fb =
            (b < D2) ? Phi[static_cast<std::size_t>(i * D2 + b)] : 1.0;
        A2[static_cast<std::size_t>(a * p2 + b)] += fa * fb;
      }
    }
  }
  for (int j = 0; j < p2; ++j) {
    A2[static_cast<std::size_t>(j * p2 + j)] += lam2s;
  }

  std::vector<double> coef2 = r2;
  std::vector<double> L2;
  if (!cholesky_factor_spd(A2, p2, L2)) {
    return false;
  }
  forward_subst_lower(L2.data(), p2, coef2.data());
  backward_subst_lt(L2.data(), p2, coef2.data());

  for (int j = 0; j < D2; ++j) {
    out_w2[static_cast<std::size_t>(j)] = coef2[static_cast<std::size_t>(j)];
  }
  *out_b2 = coef2[static_cast<std::size_t>(D2)];

  return true;
}

namespace {

void softmax_scaled_llr_row(const double* llr, int k, double temperature, double eps, double* out) {
  std::vector<double> z(static_cast<std::size_t>(k));
  const double inv_t = 1.0 / (temperature + eps);
  for (int i = 0; i < k; ++i) {
    z[static_cast<std::size_t>(i)] = llr[static_cast<std::size_t>(i)] * inv_t;
  }
  if (k <= 8) {
    double mx = z[0];
    for (int i = 1; i < k; ++i) {
      mx = std::max(mx, z[static_cast<std::size_t>(i)]);
    }
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
      out[i] = std::exp(z[static_cast<std::size_t>(i)] - mx);
      sum += out[i];
    }
    sum += eps;
    for (int i = 0; i < k; ++i) {
      out[i] /= sum;
    }
    return;
  }
  double mx = z[0];
  for (int i = 1; i < k; ++i) {
    mx = std::max(mx, z[static_cast<std::size_t>(i)]);
  }
  double sum = 0.0;
  for (int i = 0; i < k; ++i) {
    out[i] = std::exp(z[static_cast<std::size_t>(i)] - mx);
    sum += out[i];
  }
  for (int i = 0; i < k; ++i) {
    out[i] /= (sum + eps);
  }
}

}  // namespace

void router_softmax_from_llr(const double* llr, int K, double temperature, double eps, double* probs_out) {
  if (K < 1) {
    return;
  }
  softmax_scaled_llr_row(llr, K, temperature, eps, probs_out);
}

double mke_routing_entropy(const double* probs, int K, double eps) {
  double s = 0.0;
  for (int j = 0; j < K; ++j) {
    const double p = probs[static_cast<std::size_t>(j)];
    s -= p * std::log(p + eps);
  }
  return s;
}

double mke_scalar_predict_from_llr(const double* llr, int K, double temperature, double eps,
                                   const double* expert_mu, double* out_entropy) {
  if (K < 1) {
    return 0.0;
  }
  std::vector<double> probs(static_cast<std::size_t>(K));
  softmax_scaled_llr_row(llr, K, temperature, eps, probs.data());
  double y = 0.0;
  for (int k = 0; k < K; ++k) {
    y += probs[static_cast<std::size_t>(k)] * expert_mu[static_cast<std::size_t>(k)];
  }
  if (out_entropy != nullptr) {
    *out_entropy = mke_routing_entropy(probs.data(), K, eps);
  }
  return y;
}

double two_stage_dif_predict_with_clf(const cypha::CyphaInferModel& clf, const double* x_row_major, int d_in,
                                      const double* enc2_W, const double* enc2_b, int D2, const double* w1, double b1,
                                      const double* w2, double b2, double y_mean, double y_std) {
  if (d_in != clf.d_latent || D2 < 1) {
    return 0.0;
  }
  const int K = static_cast<int>(clf.labels.size());
  if (K < 1) {
    return 0.0;
  }
  std::vector<double> h;
  cypha::batch_encode(clf, x_row_major, 1, h);
  std::vector<double> llr;
  cypha::score_matrix_use_field(clf, h.data(), 1, llr);
  if (static_cast<int>(llr.size()) != K) {
    return 0.0;
  }
  std::vector<double> phi2(static_cast<std::size_t>(D2));
  rff_encode_batch_rowmajor(x_row_major, 1, d_in, enc2_W, enc2_b, D2, phi2.data());
  return two_stage_dif_predict(llr.data(), K, x_row_major, d_in, w1, b1, phi2.data(), D2, w2, b2, y_mean, y_std);
}

}  // namespace cypha::regression
