#include "cypha/nig_field.hpp"

#include "cypha/load_cypha.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace cypha {

namespace {

constexpr double kEps = 1e-8;
constexpr double kHCap = 50.0;

}  // namespace

void field_diag_a(int fd, std::vector<double>& a_out) {
  a_out.assign(static_cast<std::size_t>(fd), 0.0);
  if (fd <= 0) {
    return;
  }
  const int g = fd / 5;
  const int r = fd - 4 * g;
  int i = 0;
  for (int k = 0; k < g; ++k) {
    a_out[static_cast<std::size_t>(i++)] = 0.30;
  }
  for (int k = 0; k < g; ++k) {
    a_out[static_cast<std::size_t>(i++)] = 0.60;
  }
  for (int k = 0; k < g; ++k) {
    a_out[static_cast<std::size_t>(i++)] = 0.85;
  }
  for (int k = 0; k < g; ++k) {
    a_out[static_cast<std::size_t>(i++)] = 0.95;
  }
  for (int k = 0; k < r; ++k) {
    a_out[static_cast<std::size_t>(i++)] = 0.99;
  }
}

void recompute_field_a_eff(int fd, const std::vector<double>& w_t_row_major, std::vector<float>& a_eff_out) {
  std::vector<double> a;
  field_diag_a(fd, a);
  a_eff_out.resize(static_cast<std::size_t>(fd * fd));
  for (int i = 0; i < fd; ++i) {
    for (int j = 0; j < fd; ++j) {
      // Match Python `(np.diag(self.field._a) + self.field._W_T).astype(np.float32)` per element:
      // float64 add, then round to float32 (not float32 + float32).
      double v64 = w_t_row_major[static_cast<std::size_t>(i * fd + j)];
      if (i == j) {
        v64 += a[static_cast<std::size_t>(i)];
      }
      a_eff_out[static_cast<std::size_t>(i * fd + j)] = static_cast<float>(v64);
    }
  }
}

void patch_field_a_eff_into_root(CNode& root) {
  const CNode* fwt = map_get(root, "field_W_T");
  if (fwt == nullptr || fwt->kind != CNode::Tensor || fwt->shape.size() != 2 ||
      fwt->shape[0] != fwt->shape[1]) {
    return;
  }
  const int fd = static_cast<int>(fwt->shape[0]);
  if (fd <= 0 || static_cast<int>(fwt->tensor.size()) != fd * fd) {
    return;
  }
  std::vector<float> aeff;
  recompute_field_a_eff(fd, fwt->tensor, aeff);
  CNode t;
  t.kind = CNode::Tensor;
  t.shape = {static_cast<std::uint32_t>(fd), static_cast<std::uint32_t>(fd)};
  t.tensor.resize(static_cast<std::size_t>(fd * fd));
  for (int i = 0; i < fd * fd; ++i) {
    t.tensor[static_cast<std::size_t>(i)] = static_cast<double>(aeff[static_cast<std::size_t>(i)]);
  }
  for (auto& kv : root.map) {
    if (kv.first == "field_a_eff") {
      kv.second = std::move(t);
      return;
    }
  }
  root.map.emplace_back("field_a_eff", std::move(t));
}

void nig_field_inject(std::vector<double>& field_h, const double* signal, int fd, double strength) {
  if (fd <= 0 || static_cast<int>(field_h.size()) != fd) {
    return;
  }
  double h_sq = 0.0;
  double s_sq = 0.0;
  for (int t = 0; t < fd; ++t) {
    h_sq += field_h[static_cast<std::size_t>(t)] * field_h[static_cast<std::size_t>(t)];
    double sv = signal[static_cast<std::size_t>(t)];
    s_sq += sv * sv;
  }
  if (!std::isfinite(s_sq) || s_sq == 0.0) {
    return;
  }
  double h_mag = std::sqrt(std::max(h_sq, 0.0)) + kEps;
  double s_mag = std::sqrt(s_sq) + kEps;
  double scale = strength * (h_mag / s_mag);
  for (int t = 0; t < fd; ++t) {
    field_h[static_cast<std::size_t>(t)] += scale * signal[static_cast<std::size_t>(t)];
  }
  h_sq = 0.0;
  for (int t = 0; t < fd; ++t) {
    double v = field_h[static_cast<std::size_t>(t)];
    h_sq += v * v;
  }
  if (!std::isfinite(h_sq)) {
    std::fill(field_h.begin(), field_h.end(), 0.0);
  } else if (h_sq > kHCap * kHCap) {
    double t = kHCap / std::sqrt(h_sq);
    for (double& v : field_h) {
      v *= t;
    }
  }
}

void nig_field_evolve(const std::vector<float>& a_eff, int fd, const double* h_in, std::vector<double>& field_h) {
  if (fd <= 0 || static_cast<int>(a_eff.size()) != fd * fd) {
    return;
  }
  field_h.resize(static_cast<std::size_t>(fd));
  std::vector<float> x(static_cast<std::size_t>(fd));
  std::vector<float> y(static_cast<std::size_t>(fd));
  for (int j = 0; j < fd; ++j) {
    x[static_cast<std::size_t>(j)] = static_cast<float>(h_in[static_cast<std::size_t>(j)]);
  }
  for (int i = 0; i < fd; ++i) {
    float s = 0.f;
    for (int j = 0; j < fd; ++j) {
      s += a_eff[static_cast<std::size_t>(i * fd + j)] * x[static_cast<std::size_t>(j)];
    }
    y[static_cast<std::size_t>(i)] = s;
  }
  double y_sq = 0.0;
  for (int i = 0; i < fd; ++i) {
    double v = static_cast<double>(y[static_cast<std::size_t>(i)]);
    if (!std::isfinite(v)) {
      std::fill(field_h.begin(), field_h.end(), 0.0);
      return;
    }
    field_h[static_cast<std::size_t>(i)] = v;
    y_sq += v * v;
  }
  if (y_sq > kHCap * kHCap) {
    double t = kHCap / std::sqrt(y_sq);
    for (double& v : field_h) {
      v *= t;
    }
  }
}

void nig_field_update_causal(std::vector<double>& w_t_row_major, int fd, const double* h_t, const double* h_target,
                             double lr, std::vector<float>& sr_vec, std::vector<float>& a_eff_out) {
  if (fd <= 0 || static_cast<int>(w_t_row_major.size()) != fd * fd) {
    return;
  }
  if (static_cast<int>(sr_vec.size()) != fd) {
    sr_vec.assign(static_cast<std::size_t>(fd), 1.0f / static_cast<float>(std::sqrt(static_cast<double>(fd))));
  }
  std::vector<double> err(static_cast<std::size_t>(fd), 0.0);
  for (int i = 0; i < fd; ++i) {
    double s = 0.0;
    for (int j = 0; j < fd; ++j) {
      s += w_t_row_major[static_cast<std::size_t>(i * fd + j)] * h_t[static_cast<std::size_t>(j)];
    }
    err[static_cast<std::size_t>(i)] = s - h_target[static_cast<std::size_t>(i)];
  }
  const double d = static_cast<double>(fd);
  for (int i = 0; i < fd; ++i) {
    double ei = err[static_cast<std::size_t>(i)];
    for (int j = 0; j < fd; ++j) {
      w_t_row_major[static_cast<std::size_t>(i * fd + j)] -= lr * ei * h_t[static_cast<std::size_t>(j)] / d;
    }
  }
  std::vector<float> v = sr_vec;
  for (int it = 0; it < 3; ++it) {
    std::vector<float> nv(static_cast<std::size_t>(fd), 0.f);
    for (int i = 0; i < fd; ++i) {
      float s = 0.f;
      for (int j = 0; j < fd; ++j) {
        s += static_cast<float>(w_t_row_major[static_cast<std::size_t>(i * fd + j)]) * v[static_cast<std::size_t>(j)];
      }
      nv[static_cast<std::size_t>(i)] = s;
    }
    double nv_sq = 0.0;
    for (int i = 0; i < fd; ++i) {
      nv_sq += static_cast<double>(nv[static_cast<std::size_t>(i)]) * static_cast<double>(nv[static_cast<std::size_t>(i)]);
    }
    nv_sq = std::sqrt(nv_sq);
    if (nv_sq < kEps) {
      break;
    }
    for (int i = 0; i < fd; ++i) {
      v[static_cast<std::size_t>(i)] = nv[static_cast<std::size_t>(i)] / static_cast<float>(nv_sq);
    }
  }
  sr_vec = v;
  double sr = 0.0;
  for (int i = 0; i < fd; ++i) {
    double s = 0.0;
    for (int j = 0; j < fd; ++j) {
      s += w_t_row_major[static_cast<std::size_t>(i * fd + j)] * static_cast<double>(v[static_cast<std::size_t>(j)]);
    }
    sr += s * s;
  }
  sr = std::sqrt(sr);
  if (sr > 0.85) {
    double t = 0.85 / sr;
    for (double& w : w_t_row_major) {
      w *= t;
    }
  }
  recompute_field_a_eff(fd, w_t_row_major, a_eff_out);
}

bool latent_to_field_signal(const double* h, int d, const std::vector<double>& w_inject_row_major, int fd,
                            std::vector<double>& signal_out) {
  signal_out.assign(static_cast<std::size_t>(fd), 0.0);
  if (d <= 0 || fd <= 0 || h == nullptr) {
    return false;
  }
  double hn = 0.0;
  for (int j = 0; j < d; ++j) {
    double v = h[static_cast<std::size_t>(j)];
    hn += v * v;
  }
  hn = std::sqrt(hn) + kEps;
  if (w_inject_row_major.empty()) {
    if (d != fd) {
      return false;
    }
    for (int i = 0; i < fd; ++i) {
      signal_out[static_cast<std::size_t>(i)] = h[static_cast<std::size_t>(i)] / hn;
    }
    return true;
  }
  if (static_cast<int>(w_inject_row_major.size()) != fd * d) {
    return false;
  }
  for (int i = 0; i < fd; ++i) {
    double s = 0.0;
    for (int j = 0; j < d; ++j) {
      s += w_inject_row_major[static_cast<std::size_t>(i * d + j)] * (h[static_cast<std::size_t>(j)] / hn);
    }
    signal_out[static_cast<std::size_t>(i)] = s;
  }
  return true;
}

}  // namespace cypha
