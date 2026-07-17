#include "cypha/rff_features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "cypha/regression_stub.hpp"

namespace cypha {

namespace {

constexpr double kTwoPi = 2.0 * 3.14159265358979323846264338328;

double sq_dist(const double* a, const double* b, int d) {
  double s = 0.0;
  for (int i = 0; i < d; ++i) {
    const double diff = a[i] - b[i];
    s += diff * diff;
  }
  return s;
}

double rbf_value(const double* a, const double* b, int d, double gamma) {
  return std::exp(-gamma * sq_dist(a, b, d));
}

void fwht_core(double* v, int n) {
  if (n <= 1) {
    return;
  }
  for (int len = 1; len < n; len <<= 1) {
    for (int i = 0; i < n; i += 2 * len) {
      for (int j = 0; j < len; ++j) {
        const double a = v[i + j];
        const double b = v[i + j + len];
        v[i + j] = a + b;
        v[i + j + len] = a - b;
      }
    }
  }
  const double scale = 1.0 / std::sqrt(static_cast<double>(n));
  for (int i = 0; i < n; ++i) {
    v[i] *= scale;
  }
}

int pow2_ceil(int n) {
  int p = 1;
  while (p < std::max(n, 1)) {
    p <<= 1;
  }
  return p;
}

void init_orthogonal_rows(std::mt19937& rng, int d_in, std::vector<double>& rows) {
  rows.assign(static_cast<std::size_t>(d_in) * static_cast<std::size_t>(d_in), 0.0);
  std::vector<double> a(static_cast<std::size_t>(d_in) * static_cast<std::size_t>(d_in));
  std::normal_distribution<double> nd(0.0, 1.0);
  for (auto& v : a) {
    v = nd(rng);
  }
  std::vector<double> col(static_cast<std::size_t>(d_in));
  for (int j = 0; j < d_in; ++j) {
    for (int i = 0; i < d_in; ++i) {
      col[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i * d_in + j)];
    }
    for (int k = 0; k < j; ++k) {
      double dot = 0.0;
      for (int i = 0; i < d_in; ++i) {
        dot += col[static_cast<std::size_t>(i)] * rows[static_cast<std::size_t>(i * d_in + k)];
      }
      for (int i = 0; i < d_in; ++i) {
        col[static_cast<std::size_t>(i)] -= dot * rows[static_cast<std::size_t>(i * d_in + k)];
      }
    }
    double norm = 0.0;
    for (int i = 0; i < d_in; ++i) {
      norm += col[static_cast<std::size_t>(i)] * col[static_cast<std::size_t>(i)];
    }
    norm = std::sqrt(std::max(norm, 1e-18));
    for (int i = 0; i < d_in; ++i) {
      rows[static_cast<std::size_t>(i * d_in + j)] = col[static_cast<std::size_t>(i)] / norm;
    }
  }
}

bool cholesky_lower(const double* sym_row_major, int n, double ridge, double* L_row_major) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      L_row_major[i * n + j] = sym_row_major[i * n + j] + (i == j ? ridge : 0.0);
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = L_row_major[i * n + j];
      for (int k = 0; k < j; ++k) {
        sum -= L_row_major[i * n + k] * L_row_major[j * n + k];
      }
      if (i == j) {
        if (sum <= 1e-12) {
          return false;
        }
        L_row_major[i * n + j] = std::sqrt(sum);
      } else {
        L_row_major[i * n + j] = sum / L_row_major[j * n + j];
      }
    }
    for (int j = i + 1; j < n; ++j) {
      L_row_major[i * n + j] = 0.0;
    }
  }
  return true;
}

void llt_solve_lower(const double* L, int n, const double* b, double* x) {
  for (int i = 0; i < n; ++i) {
    double s = b[i];
    for (int k = 0; k < i; ++k) {
      s -= L[i * n + k] * x[k];
    }
    x[i] = s / L[i * n + i];
  }
  for (int i = n - 1; i >= 0; --i) {
    double s = x[i];
    for (int k = i + 1; k < n; ++k) {
      s -= L[k * n + i] * x[k];
    }
    x[i] = s / L[i * n + i];
  }
}

void build_kernel_matrix(const double* X, int n, int d, double gamma, std::vector<double>& K) {
  K.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    const double* xi = X + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
    for (int j = 0; j < n; ++j) {
      const double* xj = X + static_cast<std::size_t>(j) * static_cast<std::size_t>(d);
      K[static_cast<std::size_t>(i * n + j)] = (i == j) ? 1.0 : rbf_value(xi, xj, d, gamma);
    }
  }
}

void invert_lower_triangular(const double* L, int n, double* Linv) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      Linv[i * n + j] = 0.0;
    }
  }
  for (int i = 0; i < n; ++i) {
    Linv[i * n + i] = 1.0 / L[i * n + i];
    for (int j = 0; j < i; ++j) {
      double sum = 0.0;
      for (int k = j; k < i; ++k) {
        sum += L[i * n + k] * Linv[k * n + j];
      }
      Linv[i * n + j] = -sum / L[i * n + i];
    }
  }
}

void transpose(const double* A, int rows, int cols, double* At) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      At[j * rows + i] = A[i * cols + j];
    }
  }
}

void matmul(const double* A, int ar, int ac, const double* B, int bc, double* C) {
  for (int i = 0; i < ar; ++i) {
    for (int j = 0; j < bc; ++j) {
      double s = 0.0;
      for (int k = 0; k < ac; ++k) {
        s += A[i * ac + k] * B[k * bc + j];
      }
      C[i * bc + j] = s;
    }
  }
}

}  // namespace

void fwht_inplace(double* v, int n) {
  if (n <= 1) {
    return;
  }
  if ((n & (n - 1)) != 0) {
    throw std::invalid_argument("fwht_inplace: n must be a power of two");
  }
  fwht_core(v, n);
}

int next_pow2(int n) { return pow2_ceil(n); }

void init_rff_weights_iid(std::mt19937& rng, double gamma, int D, int d_in, std::vector<double>& w_flat,
                          std::vector<double>& b, bool kernel_memory_scale) {
  w_flat.assign(static_cast<std::size_t>(D) * static_cast<std::size_t>(d_in), 0.0);
  b.assign(static_cast<std::size_t>(D), 0.0);
  const double sigma =
      kernel_memory_scale ? std::sqrt(std::max(2.0 * gamma, 1e-12)) : std::sqrt(std::max(gamma, 1e-12));
  std::normal_distribution<double> ndist(0.0, sigma);
  std::uniform_real_distribution<double> udist(0.0, kTwoPi);
  for (int r = 0; r < D; ++r) {
    for (int c = 0; c < d_in; ++c) {
      w_flat[static_cast<std::size_t>(r * d_in + c)] = ndist(rng);
    }
    b[static_cast<std::size_t>(r)] = udist(rng);
  }
}

void init_rff_weights_sorf(std::mt19937& rng, double gamma, int D, int d_in, std::vector<double>& w_flat,
                           std::vector<double>& b, bool kernel_memory_scale) {
  w_flat.assign(static_cast<std::size_t>(D) * static_cast<std::size_t>(d_in), 0.0);
  b.assign(static_cast<std::size_t>(D), 0.0);
  const double row_scale =
      (kernel_memory_scale ? std::sqrt(std::max(2.0 * gamma, 1e-12)) : std::sqrt(std::max(gamma, 1e-12))) *
      std::sqrt(static_cast<double>(std::max(d_in, 1)));
  std::uniform_real_distribution<double> udist(0.0, kTwoPi);
  int r = 0;
  while (r < D) {
    std::vector<double> Q;
    init_orthogonal_rows(rng, d_in, Q);
    for (int i = 0; i < d_in && r < D; ++i, ++r) {
      for (int c = 0; c < d_in; ++c) {
        w_flat[static_cast<std::size_t>(r * d_in + c)] =
            row_scale * Q[static_cast<std::size_t>(i * d_in + c)];
      }
      b[static_cast<std::size_t>(r)] = udist(rng);
    }
  }
}

void init_rff_weights(RffProjectionKind kind, std::mt19937& rng, double gamma, int D, int d_in,
                      std::vector<double>& w_flat, std::vector<double>& b, bool kernel_memory_scale) {
  if (kind == RffProjectionKind::Sorf) {
    init_rff_weights_sorf(rng, gamma, D, d_in, w_flat, b, kernel_memory_scale);
  } else {
    init_rff_weights_iid(rng, gamma, D, d_in, w_flat, b, kernel_memory_scale);
  }
}

double rbf_kernel_value(const double* a, const double* b, int d, double gamma) {
  return rbf_value(a, b, d, gamma);
}

double rff_kernel_frobenius_error(const double* X_rowmajor, int n, int d, const double* W, const double* b,
                                  int D, double gamma) {
  if (n <= 0) {
    return 0.0;
  }
  std::vector<double> phi(static_cast<std::size_t>(n) * static_cast<std::size_t>(D));
  regression::rff_encode_batch_rowmajor(X_rowmajor, n, d, W, b, D, phi.data());

  double err_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double* xi = X_rowmajor + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
    for (int j = 0; j < n; ++j) {
      const double* xj = X_rowmajor + static_cast<std::size_t>(j) * static_cast<std::size_t>(d);
      const double k_exact = rbf_value(xi, xj, d, gamma);
      double k_hat = 0.0;
      for (int t = 0; t < D; ++t) {
        k_hat += phi[static_cast<std::size_t>(i * D + t)] * phi[static_cast<std::size_t>(j * D + t)];
      }
      const double diff = k_exact - k_hat;
      err_sq += diff * diff;
    }
  }
  return std::sqrt(err_sq);
}

double nystrom_kernel_frobenius_error(const double* X_rowmajor, int n, int d,
                                      const double* landmarks_rowmajor, int m, double gamma, double ridge) {
  if (n <= 0 || m <= 0) {
    return 0.0;
  }
  std::vector<double> K_exact;
  build_kernel_matrix(X_rowmajor, n, d, gamma, K_exact);

  std::vector<double> C_nm(static_cast<std::size_t>(n) * static_cast<std::size_t>(m), 0.0);
  for (int i = 0; i < n; ++i) {
    const double* xi = X_rowmajor + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
    for (int j = 0; j < m; ++j) {
      const double* lj = landmarks_rowmajor + static_cast<std::size_t>(j) * static_cast<std::size_t>(d);
      C_nm[static_cast<std::size_t>(i * m + j)] = rbf_value(xi, lj, d, gamma);
    }
  }

  std::vector<double> K_mm(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
  for (int i = 0; i < m; ++i) {
    const double* li = landmarks_rowmajor + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
    for (int j = 0; j <= i; ++j) {
      const double* lj = landmarks_rowmajor + static_cast<std::size_t>(j) * static_cast<std::size_t>(d);
      const double kij = (i == j) ? 1.0 : rbf_value(li, lj, d, gamma);
      K_mm[static_cast<std::size_t>(i * m + j)] = kij;
      K_mm[static_cast<std::size_t>(j * m + i)] = kij;
    }
  }

  std::vector<double> L(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
  if (!cholesky_lower(K_mm.data(), m, ridge, L.data())) {
    return std::numeric_limits<double>::infinity();
  }
  std::vector<double> Linv(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
  invert_lower_triangular(L.data(), m, Linv.data());
  std::vector<double> W(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
  transpose(Linv.data(), m, m, W.data());

  std::vector<double> C_W(static_cast<std::size_t>(n) * static_cast<std::size_t>(m), 0.0);
  matmul(C_nm.data(), n, m, W.data(), m, C_W.data());

  double err_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double k_hat = 0.0;
      for (int t = 0; t < m; ++t) {
        k_hat += C_W[static_cast<std::size_t>(i * m + t)] * C_W[static_cast<std::size_t>(j * m + t)];
      }
      const double diff = K_exact[static_cast<std::size_t>(i * n + j)] - k_hat;
      err_sq += diff * diff;
    }
  }
  return std::sqrt(err_sq);
}

void select_leverage_landmark_indices(const double* X_rowmajor, int n, int d, int m, double gamma,
                                      double ridge, std::uint64_t rng_seed, std::vector<int>& out_indices) {
  out_indices.clear();
  if (n <= 0 || m <= 0) {
    return;
  }
  m = std::min(m, n);

  std::vector<double> K;
  build_kernel_matrix(X_rowmajor, n, d, gamma, K);

  std::vector<double> scores(static_cast<std::size_t>(n), 0.0);
  std::vector<double> L(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
  if (!cholesky_lower(K.data(), n, ridge, L.data())) {
    for (int i = 0; i < n; ++i) {
      scores[static_cast<std::size_t>(i)] = 1.0 / static_cast<double>(n);
    }
  } else {
    for (int i = 0; i < n; ++i) {
      std::vector<double> k_i(static_cast<std::size_t>(n));
      for (int j = 0; j < n; ++j) {
        k_i[static_cast<std::size_t>(j)] = K[static_cast<std::size_t>(i * n + j)];
      }
      std::vector<double> sol(static_cast<std::size_t>(n), 0.0);
      llt_solve_lower(L.data(), n, k_i.data(), sol.data());
      double lev = 0.0;
      for (int j = 0; j < n; ++j) {
        lev += k_i[static_cast<std::size_t>(j)] * sol[static_cast<std::size_t>(j)];
      }
      scores[static_cast<std::size_t>(i)] = std::max(lev, 1e-12);
    }
  }

  std::mt19937 rng(static_cast<std::uint32_t>(rng_seed & 0xffffffffu));
  std::vector<int> remaining(n);
  std::iota(remaining.begin(), remaining.end(), 0);
  out_indices.reserve(static_cast<std::size_t>(m));

  for (int pick = 0; pick < m; ++pick) {
    double total = 0.0;
    for (int idx : remaining) {
      total += scores[static_cast<std::size_t>(idx)];
    }
    if (total <= 0.0) {
      out_indices.push_back(remaining[static_cast<std::size_t>(pick % static_cast<int>(remaining.size()))]);
      continue;
    }
    std::uniform_real_distribution<double> udist(0.0, total);
    const double u = udist(rng);
    double acc = 0.0;
    int chosen = remaining.back();
    for (int idx : remaining) {
      acc += scores[static_cast<std::size_t>(idx)];
      if (acc >= u) {
        chosen = idx;
        break;
      }
    }
    out_indices.push_back(chosen);
    remaining.erase(std::remove(remaining.begin(), remaining.end(), chosen), remaining.end());
    scores[static_cast<std::size_t>(chosen)] = 0.0;
  }
}

}  // namespace cypha
