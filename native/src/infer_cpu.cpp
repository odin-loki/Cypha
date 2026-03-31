#include "cypha/infer_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "cypha/bessel_table.hpp"
#include "cypha/nig_field.hpp"

namespace cypha {

namespace {

constexpr double kEps = 1e-8;
constexpr double kMinVar = 1e-4;

double as_double(const CNode& n) {
  if (n.kind == CNode::Float) {
    return n.f;
  }
  if (n.kind == CNode::Int) {
    return static_cast<double>(n.i);
  }
  throw std::runtime_error("Expected numeric node");
}

std::int64_t as_int64(const CNode& n) {
  if (n.kind == CNode::Int) {
    return n.i;
  }
  if (n.kind == CNode::Float) {
    return static_cast<std::int64_t>(n.f);
  }
  throw std::runtime_error("Expected integer node");
}

double np_interp(double x) {
  using cypha::detail::kBesselK2K1;
  using cypha::detail::kBesselN;
  using cypha::detail::kBesselX0;
  using cypha::detail::kBesselX1;
  if (x <= kBesselX0) {
    return kBesselK2K1[0];
  }
  if (x >= kBesselX1) {
    return kBesselK2K1[kBesselN - 1];
  }
  const double step = (kBesselX1 - kBesselX0) / static_cast<double>(kBesselN - 1);
  double pos = (x - kBesselX0) / step;
  std::size_t i = static_cast<std::size_t>(pos);
  if (i >= kBesselN - 1) {
    i = kBesselN - 2;
  }
  double t = pos - static_cast<double>(i);
  return kBesselK2K1[i] * (1.0 - t) + kBesselK2K1[i + 1] * t;
}

double gig_e_inv_v_lam_neg1(double chi0, double psi) {
  if (chi0 < kEps || psi < kEps) {
    return psi / std::max(chi0, kEps);
  }
  double chi_g = std::max(chi0, kEps);
  double x = std::sqrt(chi_g * psi);
  if (x < 1e-6) {
    return psi / chi_g;
  }
  double chi_b = chi_g;
  double x_b = x;
  if (x_b <= 120.0) {
    double xt = std::clamp(x_b, cypha::detail::kBesselX0, cypha::detail::kBesselX1);
    double ratio = np_interp(xt);
    return std::sqrt(psi / chi_b) * ratio;
  }
  return psi / chi_b;
}

double np_interp_k0k1(double x) {
  using cypha::detail::kBesselK0K1;
  using cypha::detail::kBesselN;
  using cypha::detail::kBesselX0;
  using cypha::detail::kBesselX1;
  if (x <= kBesselX0) {
    return kBesselK0K1[0];
  }
  if (x >= kBesselX1) {
    return kBesselK0K1[kBesselN - 1];
  }
  const double step = (kBesselX1 - kBesselX0) / static_cast<double>(kBesselN - 1);
  double pos = (x - kBesselX0) / step;
  std::size_t i = static_cast<std::size_t>(pos);
  if (i >= kBesselN - 1) {
    i = kBesselN - 2;
  }
  double t = pos - static_cast<double>(i);
  return kBesselK0K1[i] * (1.0 - t) + kBesselK0K1[i + 1] * t;
}

/// E[V] for V ~ GIG(λ=-1, chi, psi); table path matches Cypha.py `_gig_E_V`.
double gig_e_v_lam_neg1(double chi0, double psi) {
  if (chi0 < kEps || psi < kEps) {
    return chi0 / std::max(psi, kEps);
  }
  double chi_g = std::max(chi0, kEps);
  double x = std::sqrt(chi_g * psi);
  if (x < 1e-6) {
    return chi_g / std::max(psi, kEps);
  }
  double chi_b = chi_g;
  double x_b = x;
  if (x_b <= 120.0) {
    double xt = std::clamp(x_b, cypha::detail::kBesselX0, cypha::detail::kBesselX1);
    double ratio = np_interp_k0k1(xt);
    return std::sqrt(chi_b / psi) * ratio;
  }
  return std::sqrt(chi_b / std::max(psi, kEps));
}

double nig_adapt_chi_impl(double chi, double psi, double innovation_sq, double R, double alpha) {
  double chi_post = chi + innovation_sq / std::max(R, kEps);
  double ev = gig_e_v_lam_neg1(chi_post, psi);
  return std::clamp(alpha * ev, 1e-4, 1e3);
}

double nig_r_eff_scalar(double mp, double r, double chi, double psi) {
  mp = std::max(mp, 0.0);
  double chi_post = chi + mp / std::max(r, kEps);
  double e_inv = gig_e_inv_v_lam_neg1(chi_post, psi);
  return r / std::max(e_inv, kEps);
}

void softmax_row_like_python(const double* x, int k, double eps, double* out) {
  if (k <= 8) {
    double mx = x[0];
    for (int i = 1; i < k; ++i) {
      mx = std::max(mx, x[i]);
    }
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
      out[i] = std::exp(x[i] - mx);
      sum += out[i];
    }
    sum += eps;
    for (int i = 0; i < k; ++i) {
      out[i] /= sum;
    }
    return;
  }
  double mx = x[0];
  for (int i = 1; i < k; ++i) {
    mx = std::max(mx, x[i]);
  }
  double sum = 0.0;
  for (int i = 0; i < k; ++i) {
    out[i] = std::exp(x[i] - mx);
    sum += out[i];
  }
  for (int i = 0; i < k; ++i) {
    out[i] /= (sum + eps);
  }
}

}  // namespace

double gh_train_lr_scale(double mahal_sq, double r_base, double chi, double psi) {
  double R_eff = nig_r_eff_scalar(mahal_sq, r_base, chi, psi);
  return r_base / std::max(R_eff, r_base);
}

double nig_R_eff_gh(double mahal_sq, double r_base, double chi, double psi) {
  return nig_r_eff_scalar(std::max(mahal_sq, 0.0), r_base, chi, psi);
}

std::pair<double, double> nig_adapt_session_chi(double chi, double psi, double innovation_sq, double r_base,
                                                double alpha) {
  double chi_new = nig_adapt_chi_impl(chi, psi, innovation_sq, r_base, alpha);
  return {chi_new, psi};
}

void auto_recalibrate_temperature(CyphaInferModel& m, double decay) {
  constexpr double kEps = 1e-8;
  if (m.llr_scale_n < 50) {
    return;
  }
  if (m.llr_scale_baseline <= 0.0 || !std::isfinite(m.llr_scale_baseline)) {
    m.llr_scale_baseline = std::max(m.llr_scale_ema, kEps);
    return;
  }
  const double base_T =
      (m.base_temp > 0.0 && std::isfinite(m.base_temp)) ? m.base_temp : m.temperature;
  if (!std::isfinite(base_T) || base_T <= 0.0) {
    return;
  }
  const double ratio = m.llr_scale_ema / (m.llr_scale_baseline + kEps);
  double T_adj = base_T / (ratio + kEps);
  const double lo = base_T * 0.2;
  const double hi = base_T * 5.0;
  if (T_adj < lo) {
    T_adj = lo;
  }
  if (T_adj > hi) {
    T_adj = hi;
  }
  if (!std::isfinite(m.temperature)) {
    m.temperature = base_T;
  }
  m.temperature = decay * m.temperature + (1.0 - decay) * T_adj;
}

namespace {

bool ctx_row_nonempty(const std::unordered_map<std::string, std::unordered_map<std::string, double>>& m,
                      const std::string& row_key) {
  auto it = m.find(row_key);
  return it != m.end() && !it->second.empty();
}

double ctx_map2_get(const std::unordered_map<std::string, std::unordered_map<std::string, double>>& m,
                    const std::string& a, const std::string& b) {
  auto it = m.find(a);
  if (it == m.end()) {
    return 0.0;
  }
  auto jt = it->second.find(b);
  if (jt == it->second.end()) {
    return 0.0;
  }
  return jt->second;
}

}  // namespace

void context_record_step(CyphaInferModel& m, const std::string& label, bool correct) {
  constexpr int kShortWin = 32;
  constexpr double decay = 0.98;

  if (!m.ctx_last_label.empty()) {
    const std::string& frm = m.ctx_last_label;
    m.cooccur[frm][label] += 1.0;
    m.cooccur_tot[frm] += 1.0;

    double old_val = m.mid_trans[frm][label];
    double new_val = decay * old_val + (1.0 - decay);
    m.mid_trans[frm][label] = new_val;
    m.mid_trans_tot[frm] += new_val - old_val;
  }

  if (static_cast<int>(m.ctx_history.size()) == kShortWin) {
    const std::string& ev = m.ctx_history.front().first;
    auto it = m.t1_counts.find(ev);
    if (it != m.t1_counts.end()) {
      it->second -= 1.0;
    }
    m.ctx_history.erase(m.ctx_history.begin());
  } else {
    m.t1_total += 1.0;
  }
  m.ctx_history.push_back({label, correct});
  m.t1_counts[label] += 1.0;

  m.mid_n += 1.0;
  m.mid_freq_total *= decay;
  for (auto& pr : m.mid_freq) {
    pr.second *= decay;
  }
  bool found = false;
  for (auto& pr : m.mid_freq) {
    if (pr.first == label) {
      pr.second += (1.0 - decay);
      found = true;
      break;
    }
  }
  if (!found) {
    m.mid_freq.emplace_back(label, 1.0 - decay);
  }
  m.mid_freq_total += (1.0 - decay);

  m.ctx_last_label = label;
}

void context_prior_for_labels(const CyphaInferModel& m, const std::vector<std::string>& classes,
                              std::vector<double>& ctx_out) {
  constexpr double eps = 1e-8;
  const int K = static_cast<int>(classes.size());
  ctx_out.assign(static_cast<std::size_t>(K), 0.0);
  if (K == 0) {
    return;
  }

  const double field_confidence = std::min(m.mid_n / 200.0, 1.0);
  const std::string& last = m.ctx_last_label;

  const double t1_denom = m.t1_total + static_cast<double>(K);

  double co_total = static_cast<double>(K);
  if (!last.empty()) {
    auto ct = m.cooccur_tot.find(last);
    co_total += (ct != m.cooccur_tot.end() ? ct->second : 0.0);
  }

  double mt_total = static_cast<double>(K);
  if (!last.empty()) {
    auto mtt = m.mid_trans_tot.find(last);
    mt_total += (mtt != m.mid_trans_tot.end() ? mtt->second : 0.0) + static_cast<double>(K) * 1e-3;
  }

  const double mid_total = m.mid_freq_total + static_cast<double>(K);

  const bool have_co = !last.empty() && ctx_row_nonempty(m.cooccur, last);
  const bool have_mt = !last.empty() && ctx_row_nonempty(m.mid_trans, last);

  const double alpha_fc = std::min(field_confidence, 0.7);
  const double w1 = 1.0 - alpha_fc;
  const double w2 = alpha_fc;

  std::unordered_map<std::string, double> freq_lookup;
  freq_lookup.reserve(m.mid_freq.size() * 2 + 1);
  for (const auto& pr : m.mid_freq) {
    freq_lookup[pr.first] = pr.second;
  }

  for (int ki = 0; ki < K; ++ki) {
    const std::string& kk = classes[static_cast<std::size_t>(ki)];

    double t1_c = 1.0;
    auto tc = m.t1_counts.find(kk);
    if (tc != m.t1_counts.end()) {
      t1_c += tc->second;
    }
    double t1_log = std::log(t1_c / t1_denom + eps);

    double t1_co = 0.0;
    if (have_co) {
      double co_v = ctx_map2_get(m.cooccur, last, kk) + 1.0;
      t1_co = std::log(co_v / co_total + eps);
    }
    double tier1 = 0.6 * t1_log + 0.4 * t1_co;

    double mid_k = 1e-3;
    auto fk = freq_lookup.find(kk);
    if (fk != freq_lookup.end()) {
      mid_k += fk->second;
    }
    double t2_log = std::log(mid_k / (mid_total + 1e-3) + eps);

    double t2_tr = 0.0;
    if (have_mt) {
      double mt_k = ctx_map2_get(m.mid_trans, last, kk) + 1e-3;
      t2_tr = std::log(mt_k / (mt_total + 1e-3) + eps);
    }
    double tier2 = 0.6 * t2_log + 0.4 * t2_tr;

    ctx_out[static_cast<std::size_t>(ki)] = w1 * tier1 + w2 * tier2;
  }
}

CyphaInferModel CyphaInferModel::from_root(const CNode& root, const double* f_field_row_major,
                                           int field_dim_in) {
  CyphaInferModel m;
  const CNode& enc = map_get_required(root, "enc_W");
  if (enc.kind != CNode::Tensor || enc.shape.size() != 2 || enc.shape[0] != enc.shape[1]) {
    throw std::runtime_error("enc_W must be a square 2D tensor");
  }
  m.d_latent = static_cast<int>(enc.shape[0]);
  m.enc_w = enc.tensor;

  const CNode& world = map_get_required(root, "world");
  const CNode& wmu = map_get_required(world, "mu");
  const CNode& wv = map_get_required(world, "v");
  if (wmu.kind != CNode::Tensor || wmu.shape.size() != 1 ||
      static_cast<int>(wmu.shape[0]) != m.d_latent) {
    throw std::runtime_error("world.mu shape mismatch");
  }
  if (wv.kind != CNode::Tensor || wv.shape.size() != 1 ||
      static_cast<int>(wv.shape[0]) != m.d_latent) {
    throw std::runtime_error("world.v shape mismatch");
  }
  m.mu_world = wmu.tensor;
  m.inv_v.resize(static_cast<std::size_t>(m.d_latent));
  double sum_v = 0.0;
  for (int j = 0; j < m.d_latent; ++j) {
    double vj = wv.tensor[static_cast<std::size_t>(j)];
    sum_v += vj;
    double clamped = std::max(vj, kMinVar);
    m.inv_v[static_cast<std::size_t>(j)] = 1.0 / clamped;
  }
  m.v_mean = sum_v / static_cast<double>(m.d_latent);

  const CNode& fh = map_get_required(root, "field_h");
  if (fh.kind != CNode::Tensor || fh.shape.size() != 1) {
    throw std::runtime_error("field_h must be 1D tensor");
  }
  m.field_dim = static_cast<int>(fh.shape[0]);
  if (m.field_dim != field_dim_in) {
    throw std::runtime_error("field_h length does not match native parity header field_dim");
  }
  m.field_h = fh.tensor;

  const CNode& tnode = map_get_required(root, "temperature");
  m.temperature = as_double(tnode);
  m.base_temp = m.temperature;
  const CNode* btmp = map_get(root, "base_temp");
  if (btmp != nullptr && btmp->kind != CNode::Nil) {
    m.base_temp = as_double(*btmp);
  }

  const CNode* maha = map_get(root, "mahal_ema");
  m.has_mahal_ema = false;
  m.mahal_ema = 0.0;
  if (maha != nullptr && maha->kind != CNode::Nil) {
    m.has_mahal_ema = true;
    m.mahal_ema = as_double(*maha);
  }

  const CNode& midn = map_get_required(root, "mid_n");
  m.mid_n = as_double(midn);

  const CNode& mf = map_get_required(root, "mid_freq");
  if (mf.kind != CNode::Map) {
    throw std::runtime_error("mid_freq must be a dict");
  }
  for (const auto& pr : mf.map) {
    m.mid_freq.emplace_back(pr.first, as_double(pr.second));
  }
  m.mid_freq_total = 0.0;
  for (const auto& pr : m.mid_freq) {
    m.mid_freq_total += pr.second;
  }

  const CNode* mtr = map_get(root, "mid_trans");
  if (mtr != nullptr && mtr->kind == CNode::Map) {
    for (const auto& outer : mtr->map) {
      if (outer.second.kind != CNode::Map) {
        continue;
      }
      double row_sum = 0.0;
      for (const auto& inner : outer.second.map) {
        double v = as_double(inner.second);
        m.mid_trans[outer.first][inner.first] = v;
        row_sum += v;
      }
      m.mid_trans_tot[outer.first] = row_sum;
    }
  }

  const CNode* lse = map_get(root, "llr_scale_ema");
  if (lse != nullptr && lse->kind != CNode::Nil) {
    m.llr_scale_ema = as_double(*lse);
  }
  const CNode* lsn = map_get(root, "llr_scale_n");
  if (lsn != nullptr && lsn->kind != CNode::Nil) {
    m.llr_scale_n = static_cast<int>(as_int64(*lsn));
  }
  const CNode* lsbase = map_get(root, "llr_scale_baseline");
  if (lsbase != nullptr && lsbase->kind != CNode::Nil) {
    m.llr_scale_baseline = as_double(*lsbase);
  }

  const CNode& classes = map_get_required(root, "classes");
  if (classes.kind != CNode::Map) {
    throw std::runtime_error("classes must be dict");
  }
  m.labels.reserve(classes.map.size());
  m.D.clear();
  m.n_obs.clear();
  for (const auto& pr : classes.map) {
    m.labels.push_back(pr.first);
    const CNode& cnode = pr.second;
    if (cnode.kind != CNode::Map) {
      throw std::runtime_error("class entry must be dict");
    }
    const CNode& dm = map_get_required(cnode, "delta_mu");
    if (dm.kind != CNode::Tensor || dm.shape.size() != 1 ||
        static_cast<int>(dm.shape[0]) != m.d_latent) {
      throw std::runtime_error("delta_mu bad shape");
    }
    for (double v : dm.tensor) {
      m.D.push_back(v);
    }
    const CNode& no = map_get_required(cnode, "n_obs");
    m.n_obs.push_back(static_cast<double>(as_int64(no)));
  }

  const int expected_f = m.d_latent * m.field_dim;
  const CNode* wff = map_get(world, "F_field");
  if (wff != nullptr && wff->kind == CNode::Tensor && wff->shape.size() == 2 &&
      static_cast<int>(wff->shape[0]) == m.d_latent && static_cast<int>(wff->shape[1]) == m.field_dim &&
      static_cast<int>(wff->tensor.size()) == expected_f) {
    m.f_field = wff->tensor;
  } else {
    if (f_field_row_major == nullptr) {
      throw std::runtime_error(
          "world.F_field missing or wrong shape in .cypha; pass external f_field row-major buffer");
    }
    m.f_field.assign(f_field_row_major, f_field_row_major + expected_f);
  }

  const CNode* fwt = map_get(root, "field_W_T");
  const CNode* fae = map_get(root, "field_a_eff");
  if (fwt != nullptr && fwt->kind == CNode::Tensor && fwt->shape.size() == 2 &&
      fwt->shape[0] == fwt->shape[1]) {
    int fd_wt = static_cast<int>(fwt->shape[0]);
    if (fd_wt == m.field_dim && static_cast<int>(fwt->tensor.size()) == fd_wt * fd_wt) {
      m.field_w_t = fwt->tensor;
      if (fae != nullptr && fae->kind == CNode::Tensor && fae->shape.size() == 2 &&
          static_cast<int>(fae->shape[0]) == fd_wt && static_cast<int>(fae->shape[1]) == fd_wt &&
          static_cast<int>(fae->tensor.size()) == fd_wt * fd_wt) {
        m.field_a_eff.resize(static_cast<std::size_t>(fd_wt * fd_wt));
        for (int i = 0; i < fd_wt * fd_wt; ++i) {
          m.field_a_eff[static_cast<std::size_t>(i)] =
              static_cast<float>(fae->tensor[static_cast<std::size_t>(i)]);
        }
      } else {
        recompute_field_a_eff(fd_wt, m.field_w_t, m.field_a_eff);
      }
      const float inv_sqrt = 1.f / static_cast<float>(std::sqrt(static_cast<double>(fd_wt)));
      m.field_sr_vec.assign(static_cast<std::size_t>(fd_wt), inv_sqrt);
    }
  }

  const CNode* winj = map_get(root, "w_inject");
  if (winj != nullptr && winj->kind == CNode::Tensor && winj->shape.size() == 2) {
    int fd_w = static_cast<int>(winj->shape[0]);
    int d_w = static_cast<int>(winj->shape[1]);
    if (fd_w == m.field_dim && d_w == m.d_latent &&
        static_cast<int>(winj->tensor.size()) == fd_w * d_w) {
      m.w_inject = winj->tensor;
    }
  }

  const CNode* mstd = map_get(root, "mahal_std_ema");
  if (mstd != nullptr && mstd->kind != CNode::Nil) {
    m.mahal_std_ema = as_double(*mstd);
  }
  const CNode* llre = map_get(root, "llr_ema");
  if (llre != nullptr && llre->kind != CNode::Nil) {
    m.llr_ema = as_double(*llre);
  }
  const CNode* tsp = map_get(root, "total_steps");
  if (tsp != nullptr && tsp->kind != CNode::Nil) {
    m.saved_total_steps = static_cast<int>(as_int64(*tsp));
  }

  const CNode* tcor = map_get(root, "total_correct");
  m.total_correct = 0;
  if (tcor != nullptr && tcor->kind != CNode::Nil) {
    m.total_correct = as_int64(*tcor);
  }

  const CNode* fstep = map_get(root, "field_step");
  m.field_step = 0;
  if (fstep != nullptr && fstep->kind != CNode::Nil) {
    m.field_step = as_int64(*fstep);
  }

  // Tier-1 sliding window + co-occurrence (Python `TieredContextBuffer` in `save_state`).
  const CNode* ctx_hp = map_get(root, "ctx_hist_packed");
  if (ctx_hp != nullptr && ctx_hp->kind == CNode::Map) {
    m.ctx_history.clear();
    m.t1_counts.clear();
    std::vector<std::pair<int, const CNode*>> entries;
    entries.reserve(ctx_hp->map.size());
    for (const auto& pr : ctx_hp->map) {
      try {
        entries.emplace_back(std::stoi(pr.first), &pr.second);
      } catch (const std::exception&) {
        continue;
      }
    }
    std::sort(entries.begin(), entries.end(),
              [](const std::pair<int, const CNode*>& a, const std::pair<int, const CNode*>& b) {
                return a.first < b.first;
              });
    for (const auto& kv : entries) {
      const CNode& em = *kv.second;
      if (em.kind != CNode::Map) {
        continue;
      }
      const CNode* lnode = map_get(em, "l");
      const CNode* cnode = map_get(em, "c");
      if (lnode == nullptr || lnode->kind != CNode::Str || cnode == nullptr || cnode->kind != CNode::Bool) {
        continue;
      }
      m.ctx_history.push_back({lnode->s, cnode->b});
    }
    m.t1_total = static_cast<double>(m.ctx_history.size());
    for (const auto& pr : m.ctx_history) {
      m.t1_counts[pr.first] += 1.0;
    }

    m.cooccur.clear();
    m.cooccur_tot.clear();
    const CNode* cco = map_get(root, "ctx_cooccur");
    if (cco != nullptr && cco->kind == CNode::Map) {
      for (const auto& outer : cco->map) {
        if (outer.second.kind != CNode::Map) {
          continue;
        }
        for (const auto& inner : outer.second.map) {
          m.cooccur[outer.first][inner.first] = as_double(inner.second);
        }
      }
    }
    const CNode* ctot = map_get(root, "ctx_cooccur_tot");
    if (ctot != nullptr && ctot->kind == CNode::Map) {
      for (const auto& pr : ctot->map) {
        m.cooccur_tot[pr.first] = as_double(pr.second);
      }
    }
    const CNode* clbl = map_get(root, "ctx_last_label");
    if (clbl != nullptr && clbl->kind == CNode::Str) {
      m.ctx_last_label = clbl->s;
    } else {
      m.ctx_last_label.clear();
    }
  }

  return m;
}

void batch_encode(const CyphaInferModel& m, const double* x_row_major, int n, std::vector<double>& h_out) {
  const int d = m.d_latent;
  h_out.assign(static_cast<std::size_t>(n * d), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      double acc = 0.0;
      for (int kk = 0; kk < d; ++kk) {
        const double f_ik = x_row_major[i * d + kk];
        const double w_jk = m.enc_w[static_cast<std::size_t>(j * d + kk)];
        acc += f_ik * w_jk;
      }
      h_out[static_cast<std::size_t>(i * d + j)] = acc;
    }
  }
}

void batch_llr_from_x(const CyphaInferModel& m, const double* x_row_major, int n, std::vector<double>& llr_out) {
  std::vector<double> h;
  batch_encode(m, x_row_major, n, h);
  score_matrix_use_field(m, h.data(), n, llr_out);
}

void score_matrix_use_field(const CyphaInferModel& m, const double* h_row_major, int n,
                            std::vector<double>& llr_out) {
  const int d = m.d_latent;
  const int K = static_cast<int>(m.labels.size());
  llr_out.assign(static_cast<std::size_t>(n * K), 0.0);
  if (K == 0) {
    return;
  }

  std::vector<double> mu0(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) {
    mu0[static_cast<std::size_t>(j)] = m.mu_world[static_cast<std::size_t>(j)];
  }
  double h_sq = 0.0;
  for (double v : m.field_h) {
    h_sq += v * v;
  }
  if (std::isfinite(h_sq) && h_sq <= 1e8) {
    for (int j = 0; j < d; ++j) {
      double acc = 0.0;
      for (int t = 0; t < m.field_dim; ++t) {
        acc += m.f_field[static_cast<std::size_t>(j * m.field_dim + t)] * m.field_h[static_cast<std::size_t>(t)];
      }
      mu0[static_cast<std::size_t>(j)] += acc;
    }
  }

  std::vector<double> ctx;
  context_prior_for_labels(m, m.labels, ctx);

  std::vector<double> d_sq(static_cast<std::size_t>(K));
  for (int k = 0; k < K; ++k) {
    double s = 0.0;
    for (int j = 0; j < d; ++j) {
      double Dkj = m.D[static_cast<std::size_t>(k * d + j)];
      s += Dkj * Dkj * m.inv_v[static_cast<std::size_t>(j)];
    }
    d_sq[static_cast<std::size_t>(k)] = s;
  }

  std::vector<double> u_k(static_cast<std::size_t>(K));
  for (int k = 0; k < K; ++k) {
    double nk = m.n_obs[static_cast<std::size_t>(k)];
    u_k[static_cast<std::size_t>(k)] = m.v_mean / (nk + 1.0);
  }

  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < K; ++k) {
      double cross = 0.0;
      for (int j = 0; j < d; ++j) {
        double Rij = (h_row_major[i * d + j] - mu0[static_cast<std::size_t>(j)]) *
                     m.inv_v[static_cast<std::size_t>(j)];
        cross += Rij * m.D[static_cast<std::size_t>(k * d + j)];
      }
      double bias = -0.5 * d_sq[static_cast<std::size_t>(k)] - u_k[static_cast<std::size_t>(k)] +
                    ctx[static_cast<std::size_t>(k)];
      llr_out[static_cast<std::size_t>(i * K + k)] = cross + bias;
    }
  }
}

void softmax_batch_like_python(const double* z_row_major, int n, int k, double eps,
                               std::vector<double>& probs_out) {
  probs_out.assign(static_cast<std::size_t>(n * k), 0.0);
  std::vector<double> row(static_cast<std::size_t>(k));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      row[static_cast<std::size_t>(j)] = z_row_major[i * k + j];
    }
    softmax_row_like_python(row.data(), k, eps, probs_out.data() + i * k);
  }
}

namespace {

double compute_ece_bins(const double* confs, const double* correct, int n, int n_bins) {
  double ece = 0.0;
  for (int b = 0; b < n_bins; ++b) {
    double lo = static_cast<double>(b) / static_cast<double>(n_bins);
    double hi = static_cast<double>(b + 1) / static_cast<double>(n_bins);
    double sum_w = 0.0;
    double sum_c = 0.0;
    double sum_corr = 0.0;
    for (int i = 0; i < n; ++i) {
      if (confs[i] >= lo && confs[i] < hi) {
        sum_w += 1.0;
        sum_c += confs[i];
        sum_corr += correct[i];
      }
    }
    if (sum_w > 0.0) {
      ece += sum_w * std::abs(sum_c / sum_w - sum_corr / sum_w) / static_cast<double>(n);
    }
  }
  return ece;
}

}  // namespace

double adapt_temperature_ece(CyphaInferModel& infer, const double* h_row_major, int n_cal, const int* true_class_idx,
                             int n_grid, double T_min, double T_max, int n_bins) {
  constexpr double kEps = 1e-8;
  const int K = static_cast<int>(infer.labels.size());
  if (n_cal <= 0 || K == 0) {
    return infer.temperature;
  }
  std::vector<double> llr;
  score_matrix_use_field(infer, h_row_major, n_cal, llr);

  std::vector<double> z(static_cast<std::size_t>(n_cal * K));
  std::vector<double> probs;
  std::vector<double> confs(static_cast<std::size_t>(n_cal));
  std::vector<double> corr(static_cast<std::size_t>(n_cal));

  double best_ece = std::numeric_limits<double>::infinity();
  double best_T = infer.temperature;

  auto eval_T = [&](double T) {
    for (int i = 0; i < n_cal; ++i) {
      for (int k = 0; k < K; ++k) {
        z[static_cast<std::size_t>(i * K + k)] =
            llr[static_cast<std::size_t>(i * K + k)] / (T + kEps);
      }
    }
    softmax_batch_like_python(z.data(), n_cal, K, kEps, probs);
    for (int i = 0; i < n_cal; ++i) {
      int bi = 0;
      double pmax = probs[static_cast<std::size_t>(i * K)];
      for (int k = 1; k < K; ++k) {
        double pk = probs[static_cast<std::size_t>(i * K + k)];
        if (pk > pmax) {
          pmax = pk;
          bi = k;
        }
      }
      confs[static_cast<std::size_t>(i)] = pmax;
      corr[static_cast<std::size_t>(i)] = (bi == true_class_idx[i]) ? 1.0 : 0.0;
    }
    return compute_ece_bins(confs.data(), corr.data(), n_cal, n_bins);
  };

  if (n_grid <= 1) {
    double T0 = T_min;
    best_T = T0;
    best_ece = eval_T(T0);
  } else {
    double log_a = std::log(T_min);
    double log_b = std::log(T_max);
    for (int gi = 0; gi < n_grid; ++gi) {
      double log_t = log_a + (log_b - log_a) * static_cast<double>(gi) / static_cast<double>(n_grid - 1);
      double T = std::exp(log_t);
      double ece = eval_T(T);
      if (ece < best_ece) {
        best_ece = ece;
        best_T = T;
      }
    }
  }

  infer.temperature = best_T;
  return best_T;
}

void world_gate_vector_use_field(const CyphaInferModel& m, const double* h_row_major, int n,
                                 double gh_chi, double gh_psi, std::vector<double>& gates_out) {
  const int d = m.d_latent;
  gates_out.assign(static_cast<std::size_t>(n), 0.0);
  if (gh_chi <= 0.0 || gh_psi <= 0.0) {
    throw std::runtime_error("GH gate disabled path not implemented for native parity");
  }

  std::vector<double> mu0(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) {
    mu0[static_cast<std::size_t>(j)] = m.mu_world[static_cast<std::size_t>(j)];
  }
  double h_sq = 0.0;
  for (double v : m.field_h) {
    h_sq += v * v;
  }
  if (std::isfinite(h_sq) && h_sq <= 1e8) {
    for (int j = 0; j < d; ++j) {
      double acc = 0.0;
      for (int t = 0; t < m.field_dim; ++t) {
        acc += m.f_field[static_cast<std::size_t>(j * m.field_dim + t)] * m.field_h[static_cast<std::size_t>(t)];
      }
      mu0[static_cast<std::size_t>(j)] += acc;
    }
  }

  double r_base = m.v_mean;
  if (m.has_mahal_ema && std::isfinite(m.mahal_ema) && m.mahal_ema > kEps) {
    r_base = m.mahal_ema;
  }

  const double denom_d = std::max(d, 1);
  for (int i = 0; i < n; ++i) {
    double mahal_per_dim = 0.0;
    for (int j = 0; j < d; ++j) {
      double diff = h_row_major[i * d + j] - mu0[static_cast<std::size_t>(j)];
      double rj = diff * m.inv_v[static_cast<std::size_t>(j)];
      mahal_per_dim += diff * rj;
    }
    mahal_per_dim /= denom_d;
    double mp = std::max(mahal_per_dim, 0.0);
    double r_eff = nig_r_eff_scalar(mp, r_base, gh_chi, gh_psi);
    gates_out[static_cast<std::size_t>(i)] = r_base / std::max(r_eff, r_base);
  }
}

}  // namespace cypha
