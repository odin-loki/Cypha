#include "cypha/preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cypha {

namespace {

constexpr double kJacobiTol = 1e-14;
constexpr int kJacobiMaxSweeps = 120;
constexpr double kStdFloor = 1e-8;

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

void PreprocessorState::fit_from_design_matrix(const std::vector<double>& row_major, int n_rows, int n_cols) {
  if (n_rows <= 0 || n_cols <= 0) {
    throw std::invalid_argument("fit_from_design_matrix: n_rows and n_cols must be positive");
  }
  if (static_cast<int>(row_major.size()) != n_rows * n_cols) {
    throw std::invalid_argument("fit_from_design_matrix: row_major size != n_rows * n_cols");
  }
  if (rff_dim > 0) {
    throw std::runtime_error(
        "fit_from_design_matrix: RFF fitting is not implemented natively; fit in Python and load preprocessor.json");
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
  } else {
    output_dim = d_work;
  }

  fitted = true;
}

}  // namespace cypha
