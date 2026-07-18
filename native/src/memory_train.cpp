#include "cypha/memory_train.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "cypha/class_gmm.hpp"
#include "cypha/em_step.hpp"
#include "cypha/load_cypha.hpp"

namespace cypha {

namespace {

constexpr double kEps = 1e-8;
constexpr double kMinVar = 1e-4;
constexpr double kMdlLambda = 0.001;
constexpr int kMdlColdStart = 8;
constexpr double kRepulseCap = 0.5;
constexpr double kLog2Pi = 1.8378770664093453;  // log(2*pi)
constexpr double kGmmPiEma = 0.05;

double as_double(const CNode& n) {
  if (n.kind == CNode::Float) {
    return n.f;
  }
  if (n.kind == CNode::Int) {
    return static_cast<double>(n.i);
  }
  throw std::runtime_error("expected number");
}

std::int64_t as_int64(const CNode& n) {
  if (n.kind == CNode::Int) {
    return n.i;
  }
  if (n.kind == CNode::Float) {
    return static_cast<std::int64_t>(n.f);
  }
  throw std::runtime_error("expected int");
}

void world_update(CyphaDifMemoryState& s, const double* h, double lr) {
  const int d = s.d_latent;
  if (static_cast<int>(s.world_mu.size()) != d) {
    throw std::runtime_error("world dim");
  }
  s.world_n += 1;
  if (s.world_n <= 20) {
    std::vector<double> delta0(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
      delta0[static_cast<std::size_t>(j)] = h[j] - s.world_mu[static_cast<std::size_t>(j)];
    }
    for (int j = 0; j < d; ++j) {
      double delta = delta0[static_cast<std::size_t>(j)];
      s.world_mu[static_cast<std::size_t>(j)] += delta / static_cast<double>(s.world_n);
      s.world_M2[static_cast<std::size_t>(j)] +=
          delta * (h[j] - s.world_mu[static_cast<std::size_t>(j)]);
    }
    if (s.world_n > 1) {
      for (int j = 0; j < d; ++j) {
        double vj = std::max(s.world_M2[static_cast<std::size_t>(j)] / static_cast<double>(s.world_n - 1),
                             kMinVar);
        s.world_v[static_cast<std::size_t>(j)] = vj;
        s.world_inv_v[static_cast<std::size_t>(j)] = 1.0 / vj;
      }
      double sumlog = 0.0;
      for (int j = 0; j < d; ++j) {
        sumlog += std::log(std::max(s.world_v[static_cast<std::size_t>(j)], kMinVar));
      }
      s.world_log_norm = s.world_D_LOG2PI - 0.5 * sumlog;
      s.world_v_mean = 0.0;
      for (int j = 0; j < d; ++j) {
        s.world_v_mean += s.world_v[static_cast<std::size_t>(j)];
      }
      s.world_v_mean /= static_cast<double>(d);
    }
    double drift_n = 0.0;
    for (int j = 0; j < d; ++j) {
      double a = delta0[static_cast<std::size_t>(j)];
      drift_n += a * a;
    }
    drift_n = std::sqrt(drift_n) / static_cast<double>(s.world_n);
    s.world_drift_ema = 0.95 * s.world_drift_ema + 0.05 * drift_n;
  } else {
    std::vector<double> delta_em(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
      delta_em[static_cast<std::size_t>(j)] = h[j] - s.world_mu[static_cast<std::size_t>(j)];
      s.world_mu[static_cast<std::size_t>(j)] += lr * delta_em[static_cast<std::size_t>(j)];
    }
    double drift_norm = 0.0;
    for (int j = 0; j < d; ++j) {
      double a = delta_em[static_cast<std::size_t>(j)];
      drift_norm += a * a;
    }
    drift_norm = std::sqrt(drift_norm);
    s.world_drift_ema = 0.95 * s.world_drift_ema + 0.05 * lr * drift_norm;
    // buf uses delta w.r.t. μ *before* the μ update (same `delta` tensor as Python).
    for (int j = 0; j < d; ++j) {
      double d0 = delta_em[static_cast<std::size_t>(j)];
      s.world_buf[static_cast<std::size_t>(j)] = d0 * d0 * lr;
    }
    for (int j = 0; j < d; ++j) {
      s.world_v[static_cast<std::size_t>(j)] =
          (1.0 - lr) * s.world_v[static_cast<std::size_t>(j)] + s.world_buf[static_cast<std::size_t>(j)];
      s.world_v[static_cast<std::size_t>(j)] = std::max(s.world_v[static_cast<std::size_t>(j)], kMinVar);
      s.world_inv_v[static_cast<std::size_t>(j)] = 1.0 / s.world_v[static_cast<std::size_t>(j)];
    }
    s.world_v_mean = 0.0;
    for (int j = 0; j < d; ++j) {
      s.world_v_mean += s.world_v[static_cast<std::size_t>(j)] * s.world_inv_d[static_cast<std::size_t>(j)];
    }
    s.world_log_n_ctr += 1;
    if (s.world_log_n_ctr >= 8) {
      double sumlog = 0.0;
      for (int j = 0; j < d; ++j) {
        sumlog += std::log(std::max(s.world_v[static_cast<std::size_t>(j)], kMinVar));
      }
      s.world_log_norm = s.world_D_LOG2PI - 0.5 * sumlog;
      s.world_log_n_ctr = 0;
    }
  }
}

}  // namespace

void CyphaDifMemoryState::refresh_world_log_norm_from_v() {
  constexpr double kMinVarStep = 1e-4;
  double sumlog = 0.0;
  for (int j = 0; j < d_latent; ++j) {
    sumlog += std::log(std::max(world_v[static_cast<std::size_t>(j)], kMinVarStep));
  }
  world_log_norm = world_D_LOG2PI - 0.5 * sumlog;
  world_log_n_ctr = 0;
}

CyphaDifMemoryState CyphaDifMemoryState::from_cypha_root(const CNode& root, const double* f_field_row_major,
                                                        int field_dim_in) {
  CyphaDifMemoryState s;
  const CNode& enc = map_get_required(root, "enc_W");
  if (enc.kind != CNode::Tensor || enc.shape.size() != 2 || enc.shape[0] != enc.shape[1]) {
    throw std::runtime_error("enc_W");
  }
  s.d_latent = static_cast<int>(enc.shape[0]);

  const CNode& world = map_get_required(root, "world");
  const CNode& wmu = map_get_required(world, "mu");
  const CNode& wv = map_get_required(world, "v");
  s.world_mu = wmu.tensor;
  s.world_v = wv.tensor;
  s.d_latent = static_cast<int>(s.world_mu.size());
  s.world_inv_v.resize(static_cast<std::size_t>(s.d_latent));
  s.world_M2.assign(static_cast<std::size_t>(s.d_latent), 1.0);
  s.world_buf.assign(static_cast<std::size_t>(s.d_latent), 0.0);
  s.world_inv_d.assign(static_cast<std::size_t>(s.d_latent), 1.0 / static_cast<double>(s.d_latent));
  s.world_v_mean = 0.0;
  for (int j = 0; j < s.d_latent; ++j) {
    double vj = s.world_v[static_cast<std::size_t>(j)];
    s.world_v_mean += vj;
    s.world_inv_v[static_cast<std::size_t>(j)] = 1.0 / std::max(vj, kMinVar);
  }
  s.world_v_mean /= static_cast<double>(s.d_latent);
  const CNode& wn = map_get_required(world, "n");
  s.world_n = static_cast<std::int64_t>(as_int64(wn));
  const CNode& wd = map_get_required(world, "drift_ema");
  s.world_drift_ema = as_double(wd);
  s.world_D_LOG2PI = -0.5 * static_cast<double>(s.d_latent) * kLog2Pi;
  double sumlog = 0.0;
  for (int j = 0; j < s.d_latent; ++j) {
    sumlog += std::log(std::max(s.world_v[static_cast<std::size_t>(j)], kMinVar));
  }
  s.world_log_norm = s.world_D_LOG2PI - 0.5 * sumlog;

  const CNode& fh = map_get_required(root, "field_h");
  s.field_dim = static_cast<int>(fh.shape[0]);
  if (s.field_dim != field_dim_in) {
    throw std::runtime_error("field_dim mismatch");
  }
  const int expected_f = s.d_latent * s.field_dim;
  const CNode* wff = map_get(world, "F_field");
  if (wff != nullptr && wff->kind == CNode::Tensor && wff->shape.size() == 2 &&
      static_cast<int>(wff->shape[0]) == s.d_latent && static_cast<int>(wff->shape[1]) == s.field_dim &&
      static_cast<int>(wff->tensor.size()) == expected_f) {
    s.f_field = wff->tensor;
  } else {
    if (f_field_row_major == nullptr) {
      throw std::runtime_error(
          "world.F_field missing or wrong shape in .cypha; pass external f_field row-major buffer");
    }
    s.f_field.assign(f_field_row_major, f_field_row_major + expected_f);
  }

  const CNode& classes = map_get_required(root, "classes");
  s.cypha_format = read_cypha_format(root);
  s.use_class_gmm = read_class_gmm_enabled(root);
  s.class_gmm_m = read_class_gmm_m(root, s.use_class_gmm);
  const int max_m = s.gmm_max_m();
  const int n_classes = static_cast<int>(classes.map.size());
  if (s.use_class_gmm) {
    class_gmm_ensure_capacity(n_classes, s.d_latent, max_m, s.D, s.class_pi, s.class_n_comp);
  } else {
    s.D.assign(static_cast<std::size_t>(n_classes) * static_cast<std::size_t>(s.d_latent), 0.0);
  }
  int k_idx = 0;
  for (const auto& pr : classes.map) {
    const std::string& lbl = pr.first;
    int k = static_cast<int>(s.labels.size());
    s.label_index[lbl] = k;
    s.labels.push_back(lbl);
    const CNode& cnode = pr.second;
    load_class_delta_from_cnode(cnode, s.d_latent, max_m, k_idx, s.D, s.class_pi, s.class_n_comp);
    const CNode& no = map_get_required(cnode, "n_obs");
    s.n_obs_buf.push_back(static_cast<double>(as_int64(no)));
    const CNode* nc = map_get(cnode, "n_correct");
    std::int64_t ncor = 0;
    if (nc != nullptr && nc->kind != CNode::Nil) {
      ncor = as_int64(*nc);
    }
    s.n_correct.push_back(ncor);
    ++k_idx;
  }
  if (!s.use_class_gmm && !s.D.empty()) {
    // Legacy layout: compact K×d (class_n_comp unused).
  }
  return s;
}

bool CyphaDifMemoryState::class_mean_and_variance(const std::string& label, std::vector<double>& mu_out,
                                                  std::vector<double>& v_out) const {
  auto it = label_index.find(label);
  if (it == label_index.end()) {
    return false;
  }
  int k = it->second;
  mu_out.resize(static_cast<std::size_t>(d_latent));
  v_out.resize(static_cast<std::size_t>(d_latent));
  const int max_m = gmm_max_m();
  for (int j = 0; j < d_latent; ++j) {
    mu_out[static_cast<std::size_t>(j)] =
        world_mu[static_cast<std::size_t>(j)] + D[class_gmm_d_offset(k, 0, d_latent, max_m) + static_cast<std::size_t>(j)];
    v_out[static_cast<std::size_t>(j)] = world_v[static_cast<std::size_t>(j)];
  }
  return true;
}

ClassGmmStorage CyphaDifMemoryState::gmm_view() const {
  ClassGmmStorage g;
  g.enabled = use_class_gmm;
  g.max_m = gmm_max_m();
  g.d = d_latent;
  g.K = static_cast<int>(labels.size());
  g.D = D.data();
  g.class_pi = class_pi.data();
  g.class_n_comp = class_n_comp.empty() ? nullptr : class_n_comp.data();
  return g;
}

double CyphaDifMemoryState::memory_train(const double* h, const std::string& label, const double* h_field,
                                         const std::unordered_map<std::string, double>& context_prior,
                                         double temperature, double /*ood_sigma*/, double world_lr,
                                         double delta_lr, MemoryTrainMeta* meta_out) {
  const int d = d_latent;

  const int max_m = gmm_max_m();

  auto get_or_create = [&](const std::string& lbl) -> int {
    auto it = label_index.find(lbl);
    if (it != label_index.end()) {
      return it->second;
    }
    int k = static_cast<int>(labels.size());
    labels.push_back(lbl);
    label_index[lbl] = k;
    n_obs_buf.push_back(0.0);
    n_correct.push_back(0);
    if (use_class_gmm) {
      class_gmm_ensure_capacity(k + 1, d, max_m, D, class_pi, class_n_comp);
      class_gmm_init_class_row(k, d, max_m, std::min(class_gmm_m, max_m),
                               static_cast<std::uint64_t>(k + n_obs_buf.size()), D, class_pi, class_n_comp);
    } else {
      D.resize(static_cast<std::size_t>((k + 1) * d), 0.0);
    }
    return k;
  };

  const int k_idx = get_or_create(label);
  const int K = static_cast<int>(labels.size());
  // Refresh after get_or_create — D/class_pi may have reallocated.
  const ClassGmmStorage gmm = gmm_view();

  std::vector<double> mu0(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) {
    mu0[static_cast<std::size_t>(j)] = world_mu[static_cast<std::size_t>(j)];
  }
  std::vector<double> ff_proj(static_cast<std::size_t>(d), 0.0);
  if (h_field != nullptr) {
    double h_sq = 0.0;
    for (int t = 0; t < field_dim; ++t) {
      h_sq += h_field[t] * h_field[t];
    }
    if (std::isfinite(h_sq) && h_sq <= 1e8) {
      for (int j = 0; j < d; ++j) {
        double acc = 0.0;
        for (int t = 0; t < field_dim; ++t) {
          acc += f_field[static_cast<std::size_t>(j * field_dim + t)] * h_field[t];
        }
        ff_proj[static_cast<std::size_t>(j)] = acc;
        mu0[static_cast<std::size_t>(j)] += acc;
      }
    }
  }

  const double log_norm = world_log_norm;

  std::vector<double> h_mu0(static_cast<std::size_t>(d));
  std::vector<double> r(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) {
    h_mu0[static_cast<std::size_t>(j)] = h[j] - mu0[static_cast<std::size_t>(j)];
    r[static_cast<std::size_t>(j)] = h_mu0[static_cast<std::size_t>(j)] * world_inv_v[static_cast<std::size_t>(j)];
  }

  std::vector<double> cross(static_cast<std::size_t>(K), 0.0);
  std::vector<double> d_sq(static_cast<std::size_t>(K), 0.0);
  std::vector<double> scores(static_cast<std::size_t>(K), 0.0);
  if (!use_class_gmm) {
    for (int k = 0; k < K; ++k) {
      double ck = 0.0;
      double sk = 0.0;
      for (int j = 0; j < d; ++j) {
        double Dkj = D[static_cast<std::size_t>(k * d + j)];
        ck += Dkj * r[static_cast<std::size_t>(j)];
        sk += Dkj * Dkj * world_inv_v[static_cast<std::size_t>(j)];
      }
      cross[static_cast<std::size_t>(k)] = ck;
      d_sq[static_cast<std::size_t>(k)] = sk;
      scores[static_cast<std::size_t>(k)] = ck - 0.5 * sk;
    }
  } else {
    for (int k = 0; k < K; ++k) {
      scores[static_cast<std::size_t>(k)] = class_gmm_logsumexp_score(gmm, k, r.data(), world_inv_v.data());
      const double* delta0 = D.data() + class_gmm_d_offset(k, 0, d, max_m);
      cross[static_cast<std::size_t>(k)] = class_delta_llr_fragment(delta0, r.data(), world_inv_v.data(), d);
      d_sq[static_cast<std::size_t>(k)] = 0.0;
      for (int j = 0; j < d; ++j) {
        double Dkj = delta0[static_cast<std::size_t>(j)];
        d_sq[static_cast<std::size_t>(k)] += Dkj * Dkj * world_inv_v[static_cast<std::size_t>(j)];
      }
    }
  }

  std::vector<double> ctx_arr(static_cast<std::size_t>(K), 0.0);
  for (int k = 0; k < K; ++k) {
    auto it = context_prior.find(labels[static_cast<std::size_t>(k)]);
    if (it != context_prior.end()) {
      ctx_arr[static_cast<std::size_t>(k)] = it->second;
    }
    scores[static_cast<std::size_t>(k)] += ctx_arr[static_cast<std::size_t>(k)];
  }
  int best_idx = 0;
  for (int k = 1; k < K; ++k) {
    if (scores[static_cast<std::size_t>(k)] > scores[static_cast<std::size_t>(best_idx)]) {
      best_idx = k;
    }
  }
  const std::string& pred = labels[static_cast<std::size_t>(best_idx)];
  bool correct = (pred == label);
  if (correct) {
    n_correct[static_cast<std::size_t>(k_idx)] += 1;
  }

  const double cross_k = cross[static_cast<std::size_t>(k_idx)];
  const double d_sq_k = d_sq[static_cast<std::size_t>(k_idx)];

  n_obs_buf[static_cast<std::size_t>(k_idx)] += 1.0;

  std::vector<double> scales(static_cast<std::size_t>(K), 0.0);
  std::vector<double> s_e(static_cast<std::size_t>(K), 0.0);
  double sum_e = 0.0;
  double s_mx = scores[static_cast<std::size_t>(best_idx)];
  for (int k = 0; k < K; ++k) {
    s_e[static_cast<std::size_t>(k)] = std::exp(scores[static_cast<std::size_t>(k)] - s_mx);
    sum_e += s_e[static_cast<std::size_t>(k)];
  }
  sum_e += kEps;
  for (int k = 0; k < K; ++k) {
    scales[static_cast<std::size_t>(k)] =
        -delta_lr * std::min(s_e[static_cast<std::size_t>(k)] / sum_e, kRepulseCap);
  }
  scales[static_cast<std::size_t>(k_idx)] = delta_lr;

  if (!task_prefix_protect.empty()) {
    for (int k = 0; k < K; ++k) {
      const std::string& lbl = labels[static_cast<std::size_t>(k)];
      const auto us = lbl.find('_');
      const std::string pref = (us == std::string::npos) ? lbl : lbl.substr(0, us);
      if (pref != task_prefix_protect) {
        scales[static_cast<std::size_t>(k)] = 0.0;
      }
    }
    // Keep attracting the true target even if its prefix somehow mismatched.
    scales[static_cast<std::size_t>(k_idx)] = delta_lr;
  }

  if (!use_class_gmm) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < d; ++j) {
        double& Dkj = D[static_cast<std::size_t>(k * d + j)];
        Dkj += scales[static_cast<std::size_t>(k)] * (h_mu0[static_cast<std::size_t>(j)] - Dkj);
      }
    }
  } else {
    std::vector<double> comp_resp(static_cast<std::size_t>(max_m));
    for (int k = 0; k < K; ++k) {
      const double scale_k = scales[static_cast<std::size_t>(k)];
      const int M = std::max(1, std::min(class_n_comp[static_cast<std::size_t>(k)], max_m));
      if (k == k_idx && M > 1) {
        class_gmm_component_responsibilities(gmm, k, r.data(), world_inv_v.data(), temperature, comp_resp.data());
        class_gmm_update_pi_ema(k, max_m, comp_resp.data(), M, kGmmPiEma, class_pi);
      } else {
        for (int m = 0; m < M; ++m) {
          comp_resp[static_cast<std::size_t>(m)] = (m == 0) ? 1.0 : 0.0;
        }
      }
      for (int m = 0; m < M; ++m) {
        const double wm = comp_resp[static_cast<std::size_t>(m)] * scale_k;
        double* delta = D.data() + class_gmm_d_offset(k, m, d, max_m);
        for (int j = 0; j < d; ++j) {
          delta[static_cast<std::size_t>(j)] +=
              wm * (h_mu0[static_cast<std::size_t>(j)] - delta[static_cast<std::size_t>(j)]);
        }
      }
    }
  }

  double v_m = world_v_mean;
  double snr_fac = 1.0 / std::max(v_m, 1.0);
  std::vector<double> lam_eff(static_cast<std::size_t>(K), 0.0);
  for (int k = 0; k < K; ++k) {
    double nobs = n_obs_buf[static_cast<std::size_t>(k)];
    double cold = (nobs >= static_cast<double>(kMdlColdStart)) ? 1.0 : 0.0;
    double w = nobs / (nobs + 16.0);
    lam_eff[static_cast<std::size_t>(k)] =
        kMdlLambda * snr_fac * std::max(0.00025, w) * cold;
  }
  for (int k = 0; k < K; ++k) {
    double f = 1.0 - lam_eff[static_cast<std::size_t>(k)];
    if (!use_class_gmm) {
      for (int j = 0; j < d; ++j) {
        D[static_cast<std::size_t>(k * d + j)] *= f;
      }
    } else {
      const int M = std::max(1, std::min(class_n_comp[static_cast<std::size_t>(k)], max_m));
      for (int m = 0; m < M; ++m) {
        double* delta = D.data() + class_gmm_d_offset(k, m, d, max_m);
        for (int j = 0; j < d; ++j) {
          delta[static_cast<std::size_t>(j)] *= f;
        }
      }
    }
  }

  world_update(*this, h, world_lr);

  if (meta_out != nullptr) {
    meta_out->pred_label = pred;
    meta_out->correct = correct;
    std::vector<double> mu0p(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
      mu0p[static_cast<std::size_t>(j)] = world_mu[static_cast<std::size_t>(j)];
    }
    if (h_field != nullptr) {
      double h_sq = 0.0;
      for (int t = 0; t < field_dim; ++t) {
        h_sq += h_field[t] * h_field[t];
      }
      if (std::isfinite(h_sq) && h_sq <= 1e8) {
        for (int j = 0; j < d; ++j) {
          double acc = 0.0;
          for (int t = 0; t < field_dim; ++t) {
            acc += f_field[static_cast<std::size_t>(j * field_dim + t)] * h_field[t];
          }
          mu0p[static_cast<std::size_t>(j)] += acc;
        }
      }
    }
    std::vector<double> rp(static_cast<std::size_t>(d));
    for (int j = 0; j < d; ++j) {
      rp[static_cast<std::size_t>(j)] =
          (h[j] - mu0p[static_cast<std::size_t>(j)]) * world_inv_v[static_cast<std::size_t>(j)];
    }
    std::vector<double> crossp(static_cast<std::size_t>(K), 0.0);
    std::vector<double> dsqp(static_cast<std::size_t>(K), 0.0);
    std::vector<double> llrp(static_cast<std::size_t>(K), 0.0);
    double v_mean_p = world_v_mean;
    if (!use_class_gmm) {
      for (int k = 0; k < K; ++k) {
        double ck = 0.0;
        double sk = 0.0;
        for (int j = 0; j < d; ++j) {
          double Dkj = D[static_cast<std::size_t>(k * d + j)];
          ck += Dkj * rp[static_cast<std::size_t>(j)];
          sk += Dkj * Dkj * world_inv_v[static_cast<std::size_t>(j)];
        }
        crossp[static_cast<std::size_t>(k)] = ck;
        dsqp[static_cast<std::size_t>(k)] = sk;
      }
    } else {
      const ClassGmmStorage gmm_p = gmm_view();
      for (int k = 0; k < K; ++k) {
        crossp[static_cast<std::size_t>(k)] = class_gmm_logsumexp_score(gmm_p, k, rp.data(), world_inv_v.data());
        dsqp[static_cast<std::size_t>(k)] = 0.0;
      }
    }
    int best_llr = 0;
    for (int k = 0; k < K; ++k) {
      double u_p = v_mean_p / (n_obs_buf[static_cast<std::size_t>(k)] + 1.0);
      if (!use_class_gmm) {
        llrp[static_cast<std::size_t>(k)] =
            crossp[static_cast<std::size_t>(k)] - 0.5 * dsqp[static_cast<std::size_t>(k)] - u_p +
            ctx_arr[static_cast<std::size_t>(k)];
      } else {
        llrp[static_cast<std::size_t>(k)] =
            crossp[static_cast<std::size_t>(k)] - u_p + ctx_arr[static_cast<std::size_t>(k)];
      }
      if (llrp[static_cast<std::size_t>(k)] > llrp[static_cast<std::size_t>(best_llr)]) {
        best_llr = k;
      }
    }
    int second_llr = -1;
    for (int k = 0; k < K; ++k) {
      if (k == best_llr) {
        continue;
      }
      if (second_llr < 0 || llrp[static_cast<std::size_t>(k)] > llrp[static_cast<std::size_t>(second_llr)]) {
        second_llr = k;
      }
    }
    double ml_p = llrp[static_cast<std::size_t>(best_llr)];
    double T_inv = 1.0 / (temperature + kEps);
    double sum_exp = kEps;
    for (int k = 0; k < K; ++k) {
      sum_exp += std::exp((llrp[static_cast<std::size_t>(k)] - ml_p) * T_inv);
    }
    meta_out->post_conf = 1.0 / sum_exp;
    meta_out->llr_rank0 = labels[static_cast<std::size_t>(best_llr)];
    meta_out->llr_rank1.clear();
    if (second_llr >= 0) {
      meta_out->llr_rank1 = labels[static_cast<std::size_t>(second_llr)];
    }
    meta_out->post_llr_max = ml_p;
  }

  double r_dot_hmu = 0.0;
  for (int j = 0; j < d; ++j) {
    r_dot_hmu += r[static_cast<std::size_t>(j)] * h_mu0[static_cast<std::size_t>(j)];
  }
  return -log_norm + 0.5 * r_dot_hmu - cross_k + 0.5 * d_sq_k;
}

void CyphaDifMemoryState::dedup_check(const std::string& label) {
  constexpr double kDedupThresh = 0.60;
  if (static_cast<int>(labels.size()) < 2) {
    return;
  }
  auto itk = label_index.find(label);
  if (itk == label_index.end()) {
    return;
  }
  const int k = itk->second;
  const int d = d_latent;
  auto dot_delta = [&](int a, int b) {
    double s = 0.0;
    const int max_m = gmm_max_m();
    for (int t = 0; t < d; ++t) {
      s += D[class_gmm_d_offset(a, 0, d, max_m) + static_cast<std::size_t>(t)] *
           D[class_gmm_d_offset(b, 0, d, max_m) + static_cast<std::size_t>(t)];
    }
    return s;
  };
  auto norm_delta = [&](int a) {
    double s = 0.0;
    const int max_m = gmm_max_m();
    for (int t = 0; t < d; ++t) {
      const double v = D[class_gmm_d_offset(a, 0, d, max_m) + static_cast<std::size_t>(t)];
      s += v * v;
    }
    return std::sqrt(std::max(s, 0.0)) + kEps;
  };
  const double nk = norm_delta(k);
  const int K = static_cast<int>(labels.size());
  for (int j = 0; j < K; ++j) {
    if (j == k) {
      continue;
    }
    const double nj = norm_delta(j);
    const double cos_sim = dot_delta(k, j) / (nk * nj);
    if (cos_sim > kDedupThresh) {
      const double overlap = cos_sim - kDedupThresh;
      for (int t = 0; t < d; ++t) {
        const std::size_t ak = class_gmm_d_offset(k, 0, d, gmm_max_m()) + static_cast<std::size_t>(t);
        const std::size_t aj = class_gmm_d_offset(j, 0, d, gmm_max_m()) + static_cast<std::size_t>(t);
        const double push_t = overlap * 0.5 * D[aj] / nj;
        D[ak] -= push_t;
        D[aj] -= push_t;
      }
    }
  }
}

double memory_max_classify_llr(const CyphaDifMemoryState& s, const double* h, const double* h_field,
                               const std::unordered_map<std::string, double>& context_prior) {
  const int d = s.d_latent;
  const int K = static_cast<int>(s.labels.size());
  if (K == 0) {
    return 0.0;
  }
  std::vector<double> mu0(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) {
    mu0[static_cast<std::size_t>(j)] = s.world_mu[static_cast<std::size_t>(j)];
  }
  if (h_field != nullptr) {
    double h_sq = 0.0;
    for (int t = 0; t < s.field_dim; ++t) {
      h_sq += h_field[t] * h_field[t];
    }
    if (std::isfinite(h_sq) && h_sq <= 1e8) {
      for (int j = 0; j < d; ++j) {
        double acc = 0.0;
        for (int t = 0; t < s.field_dim; ++t) {
          acc += s.f_field[static_cast<std::size_t>(j * s.field_dim + t)] * h_field[t];
        }
        mu0[static_cast<std::size_t>(j)] += acc;
      }
    }
  }
  std::vector<double> d_h(static_cast<std::size_t>(d));
  std::vector<double> rp(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) {
    d_h[static_cast<std::size_t>(j)] = h[j] - mu0[static_cast<std::size_t>(j)];
    rp[static_cast<std::size_t>(j)] = d_h[static_cast<std::size_t>(j)] * s.world_inv_v[static_cast<std::size_t>(j)];
  }
  double mx = -1e300;
  const ClassGmmStorage gmm = s.gmm_view();
  for (int k = 0; k < K; ++k) {
    double llr = 0.0;
    if (!s.use_class_gmm) {
      double cross = 0.0;
      double d_sq = 0.0;
      for (int j = 0; j < d; ++j) {
        double Dkj = s.D[static_cast<std::size_t>(k * d + j)];
        cross += Dkj * rp[static_cast<std::size_t>(j)];
        d_sq += Dkj * Dkj * s.world_inv_v[static_cast<std::size_t>(j)];
      }
      double ctx = 0.0;
      auto it = context_prior.find(s.labels[static_cast<std::size_t>(k)]);
      if (it != context_prior.end()) {
        ctx = it->second;
      }
      double u_arr = s.world_v_mean / (s.n_obs_buf[static_cast<std::size_t>(k)] + 1.0);
      llr = cross - 0.5 * d_sq - u_arr + ctx;
    } else {
      double ctx = 0.0;
      auto it = context_prior.find(s.labels[static_cast<std::size_t>(k)]);
      if (it != context_prior.end()) {
        ctx = it->second;
      }
      const double u_arr = s.world_v_mean / (s.n_obs_buf[static_cast<std::size_t>(k)] + 1.0);
      llr = class_llr_for_k(gmm, k, rp.data(), s.world_inv_v.data(), u_arr, ctx);
    }
    mx = std::max(mx, llr);
  }
  return mx;
}

double memory_mahal_world_scalar(const CyphaDifMemoryState& s, const double* h) {
  const int d = s.d_latent;
  double sum = 0.0;
  for (int j = 0; j < d; ++j) {
    double dj = h[j] - s.world_mu[static_cast<std::size_t>(j)];
    sum += dj * dj * s.world_inv_v[static_cast<std::size_t>(j)];
  }
  return sum / (static_cast<double>(d) + kEps);
}

namespace {

CNode tensor_1d(const std::vector<double>& data) {
  CNode t;
  t.kind = CNode::Tensor;
  t.shape = {static_cast<std::uint32_t>(data.size())};
  t.tensor = data;
  return t;
}

CNode tensor_2d_rowmajor(const std::vector<double>& data, int rows, int cols) {
  CNode t;
  t.kind = CNode::Tensor;
  t.shape = {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)};
  t.tensor = data;
  return t;
}

CNode node_int64(std::int64_t v) {
  CNode n;
  n.kind = CNode::Int;
  n.i = v;
  return n;
}

CNode node_f64(double v) {
  CNode n;
  n.kind = CNode::Float;
  n.f = v;
  return n;
}

CNode build_classes_map(const CyphaDifMemoryState& s) {
  CNode m;
  m.kind = CNode::Map;
  const int max_m = s.gmm_max_m();
  for (const std::string& lbl : s.labels) {
    auto it = s.label_index.find(lbl);
    if (it == s.label_index.end()) {
      throw std::runtime_error("merge_state_into_root: label_index");
    }
    const int k = it->second;
    CNode c;
    c.kind = CNode::Map;
    const int n_comp =
        s.use_class_gmm && !s.class_n_comp.empty()
            ? std::max(1, std::min(s.class_n_comp[static_cast<std::size_t>(k)], max_m))
            : 1;
    if (n_comp <= 1) {
      std::vector<double> dm(static_cast<std::size_t>(s.d_latent));
      for (int j = 0; j < s.d_latent; ++j) {
        dm[static_cast<std::size_t>(j)] =
            s.D[class_gmm_d_offset(k, 0, s.d_latent, max_m) + static_cast<std::size_t>(j)];
      }
      c.map.emplace_back("delta_mu", tensor_1d(dm));
    } else {
      CNode dm2;
      dm2.kind = CNode::Tensor;
      dm2.shape = {static_cast<std::uint32_t>(n_comp), static_cast<std::uint32_t>(s.d_latent)};
      dm2.tensor.resize(static_cast<std::size_t>(n_comp * s.d_latent));
      for (int cm = 0; cm < n_comp; ++cm) {
        for (int j = 0; j < s.d_latent; ++j) {
          dm2.tensor[static_cast<std::size_t>(cm * s.d_latent + j)] =
              s.D[class_gmm_d_offset(k, cm, s.d_latent, max_m) + static_cast<std::size_t>(j)];
        }
      }
      c.map.emplace_back("delta_mu", std::move(dm2));
      std::vector<double> mix(static_cast<std::size_t>(n_comp));
      for (int cm = 0; cm < n_comp; ++cm) {
        mix[static_cast<std::size_t>(cm)] = s.class_pi[class_gmm_d_offset(k, cm, 1, max_m)];
      }
      c.map.emplace_back("mixing", tensor_1d(mix));
    }
    const double nobs = s.n_obs_buf[static_cast<std::size_t>(k)];
    const double rn = std::round(nobs);
    if (std::abs(nobs - rn) < 1e-9 && rn >= static_cast<double>(std::numeric_limits<std::int64_t>::min()) &&
        rn <= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
      c.map.emplace_back("n_obs", node_int64(static_cast<std::int64_t>(rn)));
    } else {
      c.map.emplace_back("n_obs", node_f64(nobs));
    }
    c.map.emplace_back("n_correct", node_int64(s.n_correct[static_cast<std::size_t>(k)]));
    m.map.emplace_back(lbl, std::move(c));
  }
  return m;
}

CNode patch_world_node(const CNode& world_in, const CyphaDifMemoryState& s) {
  CNode w = clone_cnode(world_in);
  for (auto& kv : w.map) {
    if (kv.first == "mu") {
      kv.second = tensor_1d(s.world_mu);
    } else if (kv.first == "v") {
      kv.second = tensor_1d(s.world_v);
    } else if (kv.first == "n") {
      kv.second = node_int64(s.world_n);
    } else if (kv.first == "drift_ema") {
      kv.second = node_f64(s.world_drift_ema);
    } else if (kv.first == "F_field") {
      if (static_cast<int>(s.f_field.size()) == s.d_latent * s.field_dim) {
        kv.second = tensor_2d_rowmajor(s.f_field, s.d_latent, s.field_dim);
      }
    }
  }
  return w;
}

}  // namespace

double class_fisher_rao_norm(const double* delta_mu, int d, const double* v0) {
  double sum = 0.0;
  for (int j = 0; j < d; ++j) {
    const double vj = std::max(v0[static_cast<std::size_t>(j)], kMinVar);
    const double dm = delta_mu[static_cast<std::size_t>(j)];
    sum += dm * dm / vj;
  }
  return sum;
}

std::vector<std::string> memory_merge_from(CyphaDifMemoryState& self, const CyphaDifMemoryState& other,
                                           double weight_self, double weight_other) {
  if (self.d_latent != other.d_latent || self.field_dim != other.field_dim) {
    throw std::runtime_error("memory_merge_from: dimension mismatch");
  }

  std::vector<double> self_v0 = self.world_v;
  std::vector<double> other_v0 = other.world_v;

  const double n_self = static_cast<double>(self.world_n);
  const double n_other = static_cast<double>(other.world_n);
  const double n_total = n_self + n_other + kEps;
  const int d = self.d_latent;

  for (int j = 0; j < d; ++j) {
    self.world_mu[static_cast<std::size_t>(j)] =
        (n_self * self.world_mu[static_cast<std::size_t>(j)] + n_other * other.world_mu[static_cast<std::size_t>(j)]) /
        n_total;
    self.world_v[static_cast<std::size_t>(j)] =
        (n_self * self.world_v[static_cast<std::size_t>(j)] + n_other * other.world_v[static_cast<std::size_t>(j)]) /
        n_total;
    self.world_inv_v[static_cast<std::size_t>(j)] =
        1.0 / std::max(self.world_v[static_cast<std::size_t>(j)], kMinVar);
  }
  self.world_n = self.world_n + other.world_n;
  self.refresh_world_log_norm_from_v();
  double sum_v = 0.0;
  for (int j = 0; j < d; ++j) {
    sum_v += self.world_v[static_cast<std::size_t>(j)];
  }
  self.world_v_mean = sum_v / static_cast<double>(d);

  auto get_or_create = [&](const std::string& lbl) -> int {
    auto it = self.label_index.find(lbl);
    if (it != self.label_index.end()) {
      return it->second;
    }
    const int k = static_cast<int>(self.labels.size());
    self.labels.push_back(lbl);
    self.label_index[lbl] = k;
    self.n_obs_buf.push_back(0.0);
    self.n_correct.push_back(0);
    if (self.use_class_gmm) {
      class_gmm_ensure_capacity(k + 1, d, self.gmm_max_m(), self.D, self.class_pi, self.class_n_comp);
      class_gmm_init_class_row(k, d, self.gmm_max_m(), std::min(self.class_gmm_m, self.gmm_max_m()),
                               static_cast<std::uint64_t>(k + 1), self.D, self.class_pi, self.class_n_comp);
    } else {
      self.D.resize(static_cast<std::size_t>((k + 1) * d), 0.0);
    }
    return k;
  };

  const int max_m_self = self.gmm_max_m();
  const int max_m_other = other.gmm_max_m();

  std::vector<std::string> new_labels;
  for (const std::string& lbl : other.labels) {
    auto oit = other.label_index.find(lbl);
    if (oit == other.label_index.end()) {
      continue;
    }
    const int ok = oit->second;
    if (self.label_index.find(lbl) != self.label_index.end()) {
      const int sk = self.label_index.at(lbl);
      double norm_s = class_gmm_fisher_rao_norm(self.gmm_view(), sk, self_v0.data());
      double norm_o = class_gmm_fisher_rao_norm(other.gmm_view(), ok, other_v0.data());
      const double total = norm_s * weight_self + norm_o * weight_other + kEps;
      const double w_s = norm_s * weight_self / total;
      const double w_o = norm_o * weight_other / total;
      const int Ms = self.use_class_gmm ? std::max(1, std::min(self.class_n_comp[static_cast<std::size_t>(sk)], max_m_self)) : 1;
      const int Mo = other.use_class_gmm ? std::max(1, std::min(other.class_n_comp[static_cast<std::size_t>(ok)], max_m_other)) : 1;
      for (int m = 0; m < std::max(Ms, Mo); ++m) {
        if (m >= Ms || m >= Mo) {
          continue;
        }
        for (int j = 0; j < d; ++j) {
          self.D[class_gmm_d_offset(sk, m, d, max_m_self) + static_cast<std::size_t>(j)] =
              w_s * self.D[class_gmm_d_offset(sk, m, d, max_m_self) + static_cast<std::size_t>(j)] +
              w_o * other.D[class_gmm_d_offset(ok, m, d, max_m_other) + static_cast<std::size_t>(j)];
        }
      }
      self.n_obs_buf[static_cast<std::size_t>(sk)] += other.n_obs_buf[static_cast<std::size_t>(ok)];
      self.n_correct[static_cast<std::size_t>(sk)] += other.n_correct[static_cast<std::size_t>(ok)];
    } else {
      const int nk = get_or_create(lbl);
      const int Mo = other.use_class_gmm ? std::max(1, std::min(other.class_n_comp[static_cast<std::size_t>(ok)], max_m_other)) : 1;
      for (int m = 0; m < Mo; ++m) {
        for (int j = 0; j < d; ++j) {
          self.D[class_gmm_d_offset(nk, m, d, max_m_self) + static_cast<std::size_t>(j)] =
              other.D[class_gmm_d_offset(ok, m, d, max_m_other) + static_cast<std::size_t>(j)];
        }
        if (self.use_class_gmm && !other.class_pi.empty()) {
          self.class_pi[class_gmm_d_offset(nk, m, 1, max_m_self)] =
              other.class_pi[class_gmm_d_offset(ok, m, 1, max_m_other)];
        }
      }
      if (self.use_class_gmm && !other.class_n_comp.empty()) {
        self.class_n_comp[static_cast<std::size_t>(nk)] = other.class_n_comp[static_cast<std::size_t>(ok)];
      }
      self.n_obs_buf[static_cast<std::size_t>(nk)] = other.n_obs_buf[static_cast<std::size_t>(ok)];
      self.n_correct[static_cast<std::size_t>(nk)] = other.n_correct[static_cast<std::size_t>(ok)];
      new_labels.push_back(lbl);
    }
  }
  return new_labels;
}

CNode CyphaDifMemoryState::merge_state_into_root_for_save(const CNode& root, const CyphaDifMemoryState& s) {
  CNode out = clone_cnode(root);
  for (auto& kv : out.map) {
    if (kv.first == "world") {
      kv.second = patch_world_node(kv.second, s);
    } else if (kv.first == "classes") {
      kv.second = build_classes_map(s);
    } else if (kv.first == "cypha_format") {
      kv.second = node_int64(s.use_class_gmm ? kCyphaFormatV4 : s.cypha_format);
    } else if (kv.first == "use_class_gmm") {
      kv.second = node_int64(s.use_class_gmm ? 1 : 0);
    } else if (kv.first == "class_gmm") {
      if (s.use_class_gmm) {
        CNode cg;
        cg.kind = CNode::Map;
        cg.map.push_back({"enabled", node_int64(1)});
        cg.map.push_back({"default_m", node_int64(s.class_gmm_m)});
        cg.map.push_back({"max_m", node_int64(kClassGmmMaxM)});
        kv.second = std::move(cg);
      }
    }
  }
  if (s.use_class_gmm && map_get(out, "cypha_format") == nullptr) {
    out.map.push_back({"cypha_format", node_int64(kCyphaFormatV4)});
  }
  if (s.use_class_gmm && map_get(out, "use_class_gmm") == nullptr) {
    out.map.push_back({"use_class_gmm", node_int64(1)});
  }
  if (s.use_class_gmm && map_get(out, "class_gmm") == nullptr) {
    CNode cg;
    cg.kind = CNode::Map;
    cg.map.push_back({"enabled", node_int64(1)});
    cg.map.push_back({"default_m", node_int64(s.class_gmm_m)});
    cg.map.push_back({"max_m", node_int64(kClassGmmMaxM)});
    out.map.push_back({"class_gmm", std::move(cg)});
  }
  return out;
}

}  // namespace cypha
