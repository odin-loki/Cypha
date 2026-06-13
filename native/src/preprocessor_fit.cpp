#include "cypha/numpy_default_rng.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/regression_stub.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cypha {

namespace {

constexpr double kJacobiTol = 1e-14;
constexpr int kJacobiMaxSweeps = 120;
constexpr double kStdFloor = 1e-8;
constexpr double kGammaEps = 1e-12;
constexpr double kCvReg = 1e-5;
constexpr double kCvValFrac = 0.2;
constexpr int kCvFolds = 5;

const std::vector<double>& default_gamma_cv_grid() {
  static const std::vector<double> g = {0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0};
  return g;
}

bool cholesky_spd(std::vector<double>& a, int n, std::vector<double>& l) {
  l.assign(static_cast<std::size_t>(n * n), 0.0);
  for (int j = 0; j < n; ++j) {
    double sum = 0.0;
    for (int k = 0; k < j; ++k) {
      sum += l[static_cast<std::size_t>(j * n + k)] * l[static_cast<std::size_t>(j * n + k)];
    }
    const double ajj = a[static_cast<std::size_t>(j * n + j)] - sum;
    if (ajj <= 1e-30) {
      return false;
    }
    l[static_cast<std::size_t>(j * n + j)] = std::sqrt(ajj);
    for (int i = j + 1; i < n; ++i) {
      sum = 0.0;
      for (int k = 0; k < j; ++k) {
        sum += l[static_cast<std::size_t>(i * n + k)] * l[static_cast<std::size_t>(j * n + k)];
      }
      l[static_cast<std::size_t>(i * n + j)] =
          (a[static_cast<std::size_t>(i * n + j)] - sum) / l[static_cast<std::size_t>(j * n + j)];
    }
  }
  return true;
}

void forward_subst(const double* l, int n, double* x) {
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int k = 0; k < i; ++k) {
      s += l[static_cast<std::size_t>(i * n + k)] * x[k];
    }
    x[i] = (x[i] - s) / l[static_cast<std::size_t>(i * n + i)];
  }
}

void backward_subst_lt(const double* l, int n, double* x) {
  for (int i = n - 1; i >= 0; --i) {
    double s = 0.0;
    for (int k = i + 1; k < n; ++k) {
      s += l[static_cast<std::size_t>(k * n + i)] * x[k];
    }
    x[i] = (x[i] - s) / l[static_cast<std::size_t>(i * n + i)];
  }
}

bool ridge_solve_no_bias(const double* phi, int n, int d_feat, double lam, const double* y, double* coef) {
  if (n < 1 || d_feat < 1) {
    return false;
  }
  std::vector<double> a(static_cast<std::size_t>(d_feat * d_feat), 0.0);
  std::vector<double> rhs(static_cast<std::size_t>(d_feat), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int a_idx = 0; a_idx < d_feat; ++a_idx) {
      const double va = phi[static_cast<std::size_t>(i * d_feat + a_idx)];
      for (int b_idx = 0; b_idx < d_feat; ++b_idx) {
        a[static_cast<std::size_t>(a_idx * d_feat + b_idx)] +=
            va * phi[static_cast<std::size_t>(i * d_feat + b_idx)];
      }
      rhs[static_cast<std::size_t>(a_idx)] += va * y[i];
    }
  }
  for (int j = 0; j < d_feat; ++j) {
    a[static_cast<std::size_t>(j * d_feat + j)] += lam;
  }
  std::vector<double> l;
  if (!cholesky_spd(a, d_feat, l)) {
    return false;
  }
  for (int j = 0; j < d_feat; ++j) {
    coef[j] = rhs[static_cast<std::size_t>(j)];
  }
  forward_subst(l.data(), d_feat, coef);
  backward_subst_lt(l.data(), d_feat, coef);
  return true;
}

double ridge_val_rmse(const double* phi_fit, int n_fit, const double* phi_val, int n_val, int d_feat,
                      double lam, const double* y_fit, const double* y_val, int y_stride) {
  std::vector<double> coef(static_cast<std::size_t>(d_feat), 0.0);
  std::vector<double> y_fit_scalar(static_cast<std::size_t>(n_fit), 0.0);
  for (int i = 0; i < n_fit; ++i) {
    y_fit_scalar[static_cast<std::size_t>(i)] = y_fit[static_cast<std::size_t>(i * y_stride)];
  }
  if (!ridge_solve_no_bias(phi_fit, n_fit, d_feat, lam, y_fit_scalar.data(), coef.data())) {
    return std::numeric_limits<double>::infinity();
  }
  double sse = 0.0;
  for (int i = 0; i < n_val; ++i) {
    double pred = 0.0;
    for (int j = 0; j < d_feat; ++j) {
      pred += phi_val[static_cast<std::size_t>(i * d_feat + j)] * coef[static_cast<std::size_t>(j)];
    }
    const double err = pred - y_val[static_cast<std::size_t>(i * y_stride)];
    sse += err * err;
  }
  return std::sqrt(sse / static_cast<double>(n_val));
}

void init_rff_weights(NumpyDefaultRng& rng, double gamma, int rff_dim, int d_in, std::vector<double>& w_flat,
                      std::vector<double>& b) {
  w_flat.resize(static_cast<std::size_t>(rff_dim * d_in));
  for (int r = 0; r < rff_dim; ++r) {
    for (int c = 0; c < d_in; ++c) {
      w_flat[static_cast<std::size_t>(r * d_in + c)] = rng.normal(0.0, gamma);
    }
  }
  b.resize(static_cast<std::size_t>(rff_dim));
  constexpr double kTwoPi = 2.0 * 3.14159265358979323846264338328;
  for (int r = 0; r < rff_dim; ++r) {
    b[static_cast<std::size_t>(r)] = rng.uniform(0.0, kTwoPi);
  }
}

double cv_score_gamma_holdout_y(const std::vector<double>& x, int n_rows, int d_in, int rff_dim, int seed,
                                double gamma, const std::vector<double>& y, int y_cols) {
  NumpyDefaultRng rng(seed);
  const std::vector<int> perm = rng.permutation(n_rows);
  const int n_val = std::max(1, static_cast<int>(n_rows * kCvValFrac));
  const int n_fit = n_rows - n_val;
  if (n_fit < 1) {
    return std::numeric_limits<double>::infinity();
  }

  std::vector<int> fit_idx(static_cast<std::size_t>(n_fit));
  std::vector<int> val_idx(static_cast<std::size_t>(n_val));
  for (int i = 0; i < n_val; ++i) {
    val_idx[static_cast<std::size_t>(i)] = perm[static_cast<std::size_t>(i)];
  }
  for (int i = 0; i < n_fit; ++i) {
    fit_idx[static_cast<std::size_t>(i)] = perm[static_cast<std::size_t>(n_val + i)];
  }

  std::vector<double> x_fit(static_cast<std::size_t>(n_fit * d_in));
  std::vector<double> x_val(static_cast<std::size_t>(n_val * d_in));
  for (int i = 0; i < n_fit; ++i) {
    const int src = fit_idx[static_cast<std::size_t>(i)];
    for (int c = 0; c < d_in; ++c) {
      x_fit[static_cast<std::size_t>(i * d_in + c)] = x[static_cast<std::size_t>(src * d_in + c)];
    }
  }
  for (int i = 0; i < n_val; ++i) {
    const int src = val_idx[static_cast<std::size_t>(i)];
    for (int c = 0; c < d_in; ++c) {
      x_val[static_cast<std::size_t>(i * d_in + c)] = x[static_cast<std::size_t>(src * d_in + c)];
    }
  }

  std::vector<double> w;
  std::vector<double> b;
  init_rff_weights(rng, gamma, rff_dim, d_in, w, b);
  std::vector<double> phi_fit(static_cast<std::size_t>(n_fit * rff_dim));
  std::vector<double> phi_val(static_cast<std::size_t>(n_val * rff_dim));
  regression::rff_encode_batch_rowmajor(x_fit.data(), n_fit, d_in, w.data(), b.data(), rff_dim, phi_fit.data());
  regression::rff_encode_batch_rowmajor(x_val.data(), n_val, d_in, w.data(), b.data(), rff_dim, phi_val.data());

  if (y_cols == 1) {
    std::vector<double> y_fit(static_cast<std::size_t>(n_fit));
    std::vector<double> y_val(static_cast<std::size_t>(n_val));
    for (int i = 0; i < n_fit; ++i) {
      y_fit[static_cast<std::size_t>(i)] = y[static_cast<std::size_t>(fit_idx[static_cast<std::size_t>(i)])];
    }
    for (int i = 0; i < n_val; ++i) {
      y_val[static_cast<std::size_t>(i)] = y[static_cast<std::size_t>(val_idx[static_cast<std::size_t>(i)])];
    }
    return ridge_val_rmse(phi_fit.data(), n_fit, phi_val.data(), n_val, rff_dim, kCvReg, y_fit.data(), y_val.data(),
                          1);
  }

  double sse = 0.0;
  int count = 0;
  for (int col = 0; col < y_cols; ++col) {
    std::vector<double> y_fit(static_cast<std::size_t>(n_fit));
    std::vector<double> y_val(static_cast<std::size_t>(n_val));
    for (int i = 0; i < n_fit; ++i) {
      y_fit[static_cast<std::size_t>(i)] =
          y[static_cast<std::size_t>(fit_idx[static_cast<std::size_t>(i)] * y_cols + col)];
    }
    for (int i = 0; i < n_val; ++i) {
      y_val[static_cast<std::size_t>(i)] =
          y[static_cast<std::size_t>(val_idx[static_cast<std::size_t>(i)] * y_cols + col)];
    }
    const double rmse =
        ridge_val_rmse(phi_fit.data(), n_fit, phi_val.data(), n_val, rff_dim, kCvReg, y_fit.data(), y_val.data(), 1);
    if (!std::isfinite(rmse)) {
      return std::numeric_limits<double>::infinity();
    }
    sse += rmse * rmse * static_cast<double>(n_val);
    count += n_val;
  }
  return std::sqrt(sse / static_cast<double>(count));
}

double cv_score_gamma_reconstruct_x(const std::vector<double>& x, int n_rows, int d_in, int rff_dim, int seed,
                                    double gamma, int fold) {
  const int fold_size = n_rows / kCvFolds;
  const int val_start = fold * fold_size;
  const int val_end = (fold == kCvFolds - 1) ? n_rows : (fold + 1) * fold_size;
  const int n_val = val_end - val_start;
  const int n_fit = n_rows - n_val;
  if (n_fit < 1 || n_val < 1) {
    return std::numeric_limits<double>::infinity();
  }

  std::vector<double> x_fit(static_cast<std::size_t>(n_fit * d_in));
  std::vector<double> x_val(static_cast<std::size_t>(n_val * d_in));
  int fit_i = 0;
  for (int i = 0; i < n_rows; ++i) {
    if (i >= val_start && i < val_end) {
      continue;
    }
    for (int c = 0; c < d_in; ++c) {
      x_fit[static_cast<std::size_t>(fit_i * d_in + c)] = x[static_cast<std::size_t>(i * d_in + c)];
    }
    ++fit_i;
  }
  for (int i = val_start; i < val_end; ++i) {
    const int vi = i - val_start;
    for (int c = 0; c < d_in; ++c) {
      x_val[static_cast<std::size_t>(vi * d_in + c)] = x[static_cast<std::size_t>(i * d_in + c)];
    }
  }

  NumpyDefaultRng rng(seed);
  std::vector<double> w;
  std::vector<double> b;
  init_rff_weights(rng, gamma, rff_dim, d_in, w, b);
  std::vector<double> phi_fit(static_cast<std::size_t>(n_fit * rff_dim));
  std::vector<double> phi_val(static_cast<std::size_t>(n_val * rff_dim));
  regression::rff_encode_batch_rowmajor(x_fit.data(), n_fit, d_in, w.data(), b.data(), rff_dim, phi_fit.data());
  regression::rff_encode_batch_rowmajor(x_val.data(), n_val, d_in, w.data(), b.data(), rff_dim, phi_val.data());

  double mse_sum = 0.0;
  for (int col = 0; col < d_in; ++col) {
    std::vector<double> y_fit(static_cast<std::size_t>(n_fit));
    std::vector<double> y_val(static_cast<std::size_t>(n_val));
    fit_i = 0;
    for (int i = 0; i < n_rows; ++i) {
      if (i >= val_start && i < val_end) {
        continue;
      }
      y_fit[static_cast<std::size_t>(fit_i++)] = x[static_cast<std::size_t>(i * d_in + col)];
    }
    for (int i = val_start; i < val_end; ++i) {
      y_val[static_cast<std::size_t>(i - val_start)] = x[static_cast<std::size_t>(i * d_in + col)];
    }
    const double rmse =
        ridge_val_rmse(phi_fit.data(), n_fit, phi_val.data(), n_val, rff_dim, kCvReg, y_fit.data(), y_val.data(), 1);
    if (!std::isfinite(rmse)) {
      return std::numeric_limits<double>::infinity();
    }
    mse_sum += rmse * rmse;
  }
  return std::sqrt(mse_sum / static_cast<double>(d_in));
}

double cv_score_gamma(const std::vector<double>& x, int n_rows, int d_in, int rff_dim, int seed, double gamma,
                      const std::vector<double>* y_rowmajor, int y_cols) {
  if (y_rowmajor != nullptr && !y_rowmajor->empty()) {
    return cv_score_gamma_holdout_y(x, n_rows, d_in, rff_dim, seed, gamma, *y_rowmajor, y_cols);
  }
  double sum = 0.0;
  for (int fold = 0; fold < kCvFolds; ++fold) {
    sum += cv_score_gamma_reconstruct_x(x, n_rows, d_in, rff_dim, seed, gamma, fold);
  }
  return sum / static_cast<double>(kCvFolds);
}

double estimate_rff_gamma_median_pairwise(const std::vector<double>& x, int n_rows, int n_cols,
                                          int max_samples = 500) {
  if (n_rows <= 1 || n_cols <= 0) {
    return 1.0;
  }
  const int n = std::min(n_rows, max_samples);
  std::vector<double> dists;
  dists.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n - 1) / 2);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double sq = 0.0;
      for (int c = 0; c < n_cols; ++c) {
        const double diff =
            x[static_cast<std::size_t>(i * n_cols + c)] - x[static_cast<std::size_t>(j * n_cols + c)];
        sq += diff * diff;
      }
      dists.push_back(std::sqrt(sq));
    }
  }
  if (dists.empty()) {
    return 1.0;
  }
  const std::size_t mid = dists.size() / 2;
  std::nth_element(dists.begin(), dists.begin() + static_cast<std::ptrdiff_t>(mid), dists.end());
  const double med = std::max(dists[mid], kGammaEps);
  return 1.0 / med;
}

// Symmetric A (n×n row-major). Overwrites A with nearly diagonal; v (n×n row-major) accumulates
// eigenvectors as columns: v[i*n+j] = component i of eigenvector j.
void jacobi_diagonalize(std::vector<double>& a, int n, std::vector<double>& v) {
  v.assign(static_cast<std::size_t>(n * n), 0.0);
  for (int i = 0; i < n; ++i) {
    v[static_cast<std::size_t>(i * n + i)] = 1.0;
  }
  for (int sweep = 0; sweep < kJacobiMaxSweeps; ++sweep) {
    int p = 0;
    int q = 1;
    double max_val = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        double val = std::abs(a[static_cast<std::size_t>(i * n + j)]);
        if (val > max_val) {
          max_val = val;
          p = i;
          q = j;
        }
      }
    }
    if (max_val < kJacobiTol) {
      break;
    }
    double app = a[static_cast<std::size_t>(p * n + p)];
    double aqq = a[static_cast<std::size_t>(q * n + q)];
    double apq = a[static_cast<std::size_t>(p * n + q)];
    double theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
    double c = std::cos(theta);
    double s = std::sin(theta);
    for (int i = 0; i < n; ++i) {
      if (i == p || i == q) {
        continue;
      }
      std::size_t ip = static_cast<std::size_t>(i * n + p);
      std::size_t iq = static_cast<std::size_t>(i * n + q);
      double aip = a[ip];
      double aiq = a[iq];
      double nip = c * aip - s * aiq;
      double niq = s * aip + c * aiq;
      a[ip] = a[static_cast<std::size_t>(p * n + i)] = nip;
      a[iq] = a[static_cast<std::size_t>(q * n + i)] = niq;
    }
    double app_n = c * c * app - 2.0 * c * s * apq + s * s * aqq;
    double aqq_n = s * s * app + 2.0 * c * s * apq + c * c * aqq;
    double apq_n = (c * c - s * s) * apq + c * s * (app - aqq);
    a[static_cast<std::size_t>(p * n + p)] = app_n;
    a[static_cast<std::size_t>(q * n + q)] = aqq_n;
    a[static_cast<std::size_t>(p * n + q)] = a[static_cast<std::size_t>(q * n + p)] = apq_n;
    for (int i = 0; i < n; ++i) {
      std::size_t ip = static_cast<std::size_t>(i * n + p);
      std::size_t iq = static_cast<std::size_t>(i * n + q);
      double vip = v[ip];
      double viq = v[iq];
      v[ip] = c * vip - s * viq;
      v[iq] = s * vip + c * viq;
    }
  }
}

}  // namespace

const std::vector<double>& default_rff_gamma_cv_grid() { return default_gamma_cv_grid(); }

double estimate_rff_gamma_cv(const std::vector<double>& x_rowmajor, int n_rows, int n_cols, int rff_dim, int seed,
                             const std::vector<double>* y_rowmajor, int y_cols) {
  if (n_rows <= 0 || n_cols <= 0 || rff_dim <= 0) {
    return 1.0;
  }
  const auto& gammas = default_gamma_cv_grid();
  double best_g = gammas.front();
  double best_score = std::numeric_limits<double>::infinity();
  for (double g : gammas) {
    const double score =
        cv_score_gamma(x_rowmajor, n_rows, n_cols, rff_dim, seed, g, y_rowmajor, y_cols);
    if (score < best_score) {
      best_score = score;
      best_g = g;
    }
  }
  return best_g;
}

void PreprocessorState::fit_from_design_matrix(const std::vector<double>& row_major, int n_rows, int n_cols,
                                                const std::vector<double>* y_rowmajor, int y_cols) {
  if (n_rows <= 0 || n_cols <= 0) {
    throw std::invalid_argument("fit_from_design_matrix: n_rows and n_cols must be positive");
  }
  if (static_cast<int>(row_major.size()) != n_rows * n_cols) {
    throw std::invalid_argument("fit_from_design_matrix: row_major size != n_rows * n_cols");
  }
  mean.clear();
  stddev.clear();
  pca_components.clear();
  pca_mean.clear();
  rff_w.clear();
  rff_b.clear();
  fitted = false;
  input_dim = n_cols;

  std::vector<double> x = row_major;
  int d = n_cols;

  if (scale) {
    mean.assign(static_cast<std::size_t>(d), 0.0);
    stddev.assign(static_cast<std::size_t>(d), 0.0);
    for (int j = 0; j < d; ++j) {
      double s = 0.0;
      for (int i = 0; i < n_rows; ++i) {
        s += x[static_cast<std::size_t>(i * d + j)];
      }
      mean[static_cast<std::size_t>(j)] = s / static_cast<double>(n_rows);
    }
    for (int j = 0; j < d; ++j) {
      double mu = mean[static_cast<std::size_t>(j)];
      double vsum = 0.0;
      for (int i = 0; i < n_rows; ++i) {
        double t = x[static_cast<std::size_t>(i * d + j)] - mu;
        vsum += t * t;
      }
      double stdv = std::sqrt(vsum / static_cast<double>(n_rows));
      stddev[static_cast<std::size_t>(j)] = std::max(stdv, kStdFloor);
    }
    for (int i = 0; i < n_rows; ++i) {
      for (int j = 0; j < d; ++j) {
        x[static_cast<std::size_t>(i * d + j)] =
            (x[static_cast<std::size_t>(i * d + j)] - mean[static_cast<std::size_t>(j)]) /
            stddev[static_cast<std::size_t>(j)];
      }
    }
  }

  int d_work = d;
  if (pca_dim >= 0 && pca_dim < d_work) {
    pca_mean.assign(static_cast<std::size_t>(d_work), 0.0);
    for (int j = 0; j < d_work; ++j) {
      double s = 0.0;
      for (int i = 0; i < n_rows; ++i) {
        s += x[static_cast<std::size_t>(i * d_work + j)];
      }
      pca_mean[static_cast<std::size_t>(j)] = s / static_cast<double>(n_rows);
    }
    std::vector<double> xc(static_cast<std::size_t>(n_rows * d_work));
    for (int i = 0; i < n_rows; ++i) {
      for (int j = 0; j < d_work; ++j) {
        xc[static_cast<std::size_t>(i * d_work + j)] =
            x[static_cast<std::size_t>(i * d_work + j)] - pca_mean[static_cast<std::size_t>(j)];
      }
    }
    std::vector<double> m(static_cast<std::size_t>(d_work * d_work), 0.0);
    for (int r = 0; r < d_work; ++r) {
      for (int c = 0; c < d_work; ++c) {
        double acc = 0.0;
        for (int i = 0; i < n_rows; ++i) {
          acc += xc[static_cast<std::size_t>(i * d_work + r)] * xc[static_cast<std::size_t>(i * d_work + c)];
        }
        m[static_cast<std::size_t>(r * d_work + c)] = acc;
      }
    }
    std::vector<double> v;
    jacobi_diagonalize(m, d_work, v);
    std::vector<int> order(static_cast<std::size_t>(d_work));
    for (int i = 0; i < d_work; ++i) {
      order[static_cast<std::size_t>(i)] = i;
    }
    std::sort(order.begin(), order.end(), [&](int ia, int ib) {
      return m[static_cast<std::size_t>(ia * d_work + ia)] > m[static_cast<std::size_t>(ib * d_work + ib)];
    });
    pca_components.resize(static_cast<std::size_t>(pca_dim));
    for (int k = 0; k < pca_dim; ++k) {
      int col = order[static_cast<std::size_t>(k)];
      pca_components[static_cast<std::size_t>(k)].resize(static_cast<std::size_t>(d_work));
      for (int j = 0; j < d_work; ++j) {
        pca_components[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] =
            v[static_cast<std::size_t>(j * d_work + col)];
      }
    }
    output_dim = pca_dim;
    d_work = pca_dim;
  } else {
    output_dim = d_work;
  }

  if (rff_dim > 0) {
    double gamma = rff_gamma;
    if (auto_rff_gamma_cv) {
      gamma = estimate_rff_gamma_cv(x, n_rows, d_work, rff_dim, seed, y_rowmajor, y_cols);
      rff_gamma = gamma;
    } else if (auto_rff_gamma) {
      gamma = estimate_rff_gamma_median_pairwise(x, n_rows, d_work);
      rff_gamma = gamma;
    }
    NumpyDefaultRng rng(seed);
    rff_w.resize(static_cast<std::size_t>(rff_dim));
    for (int r = 0; r < rff_dim; ++r) {
      rff_w[static_cast<std::size_t>(r)].resize(static_cast<std::size_t>(d_work));
      for (int c = 0; c < d_work; ++c) {
        rff_w[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = rng.normal(0.0, gamma);
      }
    }
    rff_b.resize(static_cast<std::size_t>(rff_dim));
    constexpr double kTwoPi = 2.0 * 3.14159265358979323846264338328;
    for (int r = 0; r < rff_dim; ++r) {
      rff_b[static_cast<std::size_t>(r)] = rng.uniform(0.0, kTwoPi);
    }
    output_dim = rff_dim;
  }

  fitted = true;
}

}  // namespace cypha
