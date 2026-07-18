#include "cypha/class_gmm.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cypha/load_cypha.hpp"

namespace cypha {

namespace {

double as_double_node(const CNode& n) {
  if (n.kind == CNode::Float) {
    return n.f;
  }
  if (n.kind == CNode::Int) {
    return static_cast<double>(n.i);
  }
  throw std::runtime_error("expected number");
}

std::int64_t as_int64_node(const CNode& n) {
  if (n.kind == CNode::Int) {
    return n.i;
  }
  if (n.kind == CNode::Float) {
    return static_cast<std::int64_t>(n.f);
  }
  throw std::runtime_error("expected int");
}

}  // namespace

int read_cypha_format(const CNode& root) {
  const CNode* fv = map_get(root, "cypha_format");
  if (fv != nullptr && fv->kind == CNode::Int) {
    return static_cast<int>(fv->i);
  }
  return kCyphaFormatV3;
}

bool read_class_gmm_enabled(const CNode& root) {
  const CNode* flag = map_get(root, "use_class_gmm");
  if (flag != nullptr && flag->kind == CNode::Int) {
    return flag->i != 0;
  }
  if (flag != nullptr && flag->kind == CNode::Float) {
    return flag->f != 0.0;
  }
  const CNode* cg = map_get(root, "class_gmm");
  if (cg != nullptr && cg->kind == CNode::Map) {
    const CNode* en = map_get(*cg, "enabled");
    if (en != nullptr && en->kind == CNode::Int) {
      return en->i != 0;
    }
  }
  return false;
}

int read_class_gmm_m(const CNode& root, bool enabled) {
  if (!enabled) {
    return 1;
  }
  const CNode* cg = map_get(root, "class_gmm");
  if (cg != nullptr && cg->kind == CNode::Map) {
    const CNode* dm = map_get(*cg, "default_m");
    if (dm != nullptr) {
      return static_cast<int>(as_int64_node(*dm));
    }
  }
  return kClassGmmDefaultM;
}

void load_class_delta_from_cnode(const CNode& cnode, int d, int max_m, int k, std::vector<double>& D,
                                 std::vector<double>& class_pi, std::vector<int>& class_n_comp) {
  const CNode& dm = map_get_required(cnode, "delta_mu");
  int n_comp = 1;
  if (dm.kind == CNode::Tensor && dm.shape.size() == 2) {
    n_comp = static_cast<int>(dm.shape[0]);
    const int dm_d = static_cast<int>(dm.shape[1]);
    if (dm_d != d) {
      throw std::runtime_error("delta_mu cols mismatch d_latent");
    }
    n_comp = std::max(1, std::min(n_comp, max_m));
    if (!class_n_comp.empty()) {
      class_n_comp[static_cast<std::size_t>(k)] = n_comp;
    }
    for (int m = 0; m < n_comp; ++m) {
      for (int j = 0; j < d; ++j) {
        D[class_gmm_d_offset(k, m, d, max_m) + static_cast<std::size_t>(j)] =
            dm.tensor[static_cast<std::size_t>(m * d + j)];
      }
    }
  } else if (dm.kind == CNode::Tensor && dm.shape.size() == 1 && static_cast<int>(dm.shape[0]) == d) {
    if (!class_n_comp.empty()) {
      class_n_comp[static_cast<std::size_t>(k)] = 1;
    }
    for (int j = 0; j < d; ++j) {
      D[class_gmm_d_offset(k, 0, d, max_m) + static_cast<std::size_t>(j)] =
          dm.tensor[static_cast<std::size_t>(j)];
    }
  } else {
    throw std::runtime_error("delta_mu bad shape");
  }

  const CNode* mix = map_get(cnode, "mixing");
  if (mix != nullptr && mix->kind == CNode::Tensor && mix->shape.size() == 1 && !class_pi.empty()) {
    const int mp = static_cast<int>(mix->shape[0]);
    for (int m = 0; m < std::min(mp, n_comp); ++m) {
      class_pi[class_gmm_d_offset(k, m, 1, max_m)] = mix->tensor[static_cast<std::size_t>(m)];
    }
  } else if (!class_pi.empty()) {
    const double inv = 1.0 / static_cast<double>(n_comp);
    for (int m = 0; m < n_comp; ++m) {
      class_pi[class_gmm_d_offset(k, m, 1, max_m)] = inv;
    }
  }
}

namespace {

double log_sum_exp(const double* log_w, int n) {
  if (n <= 0) {
    return -1e300;
  }
  double mx = log_w[0];
  for (int i = 1; i < n; ++i) {
    mx = std::max(mx, log_w[static_cast<std::size_t>(i)]);
  }
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += std::exp(log_w[static_cast<std::size_t>(i)] - mx);
  }
  return mx + std::log(std::max(sum, kEmEps));
}

}  // namespace

double class_gmm_logsumexp_score(const ClassGmmStorage& g, int k, const double* r, const double* inv_v) {
  if (!g.enabled || g.class_n_comp == nullptr) {
    const double* delta = g.D + class_gmm_d_offset(k, 0, g.d, 1);
    return class_delta_llr_fragment(delta, r, inv_v, g.d);
  }
  const int M = std::max(1, std::min(g.class_n_comp[k], g.max_m));
  if (M == 1) {
    const double* delta = g.D + class_gmm_d_offset(k, 0, g.d, g.max_m);
    return class_delta_llr_fragment(delta, r, inv_v, g.d);
  }
  std::vector<double> log_w(static_cast<std::size_t>(M));
  for (int m = 0; m < M; ++m) {
    const double* delta = g.D + class_gmm_d_offset(k, m, g.d, g.max_m);
    const double frag = class_delta_llr_fragment(delta, r, inv_v, g.d);
    const double pi = g.class_pi[class_gmm_d_offset(k, m, 1, g.max_m)];
    log_w[static_cast<std::size_t>(m)] = std::log(std::max(pi, kEmEps)) + frag;
  }
  return log_sum_exp(log_w.data(), M);
}

double class_llr_for_k(const ClassGmmStorage& g, int k, const double* r, const double* inv_v, double u_k,
                       double ctx) {
  double score = 0.0;
  if (!g.enabled || g.class_n_comp == nullptr || g.class_n_comp[k] <= 1) {
    const double* delta = g.D + class_gmm_d_offset(k, 0, g.d, g.enabled ? g.max_m : 1);
    score = class_delta_llr_fragment(delta, r, inv_v, g.d);
  } else {
    const int M = std::min(g.class_n_comp[k], g.max_m);
    std::vector<double> log_w(static_cast<std::size_t>(M));
    for (int m = 0; m < M; ++m) {
      const double* delta = g.D + class_gmm_d_offset(k, m, g.d, g.max_m);
      const double frag = class_delta_llr_fragment(delta, r, inv_v, g.d);
      const double pi = g.class_pi[class_gmm_d_offset(k, m, 1, g.max_m)];
      log_w[static_cast<std::size_t>(m)] = std::log(std::max(pi, kEmEps)) + frag;
    }
    score = log_sum_exp(log_w.data(), M);
  }
  return score - u_k + ctx;
}

void class_gmm_component_responsibilities(const ClassGmmStorage& g, int k, const double* r,
                                          const double* inv_v, double temperature, double* r_out) {
  const int M = std::max(1, g.class_n_comp != nullptr ? std::min(g.class_n_comp[k], g.max_m) : 1);
  if (M <= 1) {
    r_out[0] = 1.0;
    return;
  }
  std::vector<double> loglik(static_cast<std::size_t>(M));
  std::vector<double> prior(static_cast<std::size_t>(M));
  for (int m = 0; m < M; ++m) {
    const double* delta = g.D + class_gmm_d_offset(k, m, g.d, g.max_m);
    loglik[static_cast<std::size_t>(m)] = class_delta_llr_fragment(delta, r, inv_v, g.d);
    prior[static_cast<std::size_t>(m)] = g.class_pi[class_gmm_d_offset(k, m, 1, g.max_m)];
  }
  responsibilities(loglik.data(), prior.data(), M, temperature, kEmEps, r_out);
}

double class_gmm_fisher_rao_norm(const ClassGmmStorage& g, int k, const double* v0) {
  const int M = g.enabled && g.class_n_comp != nullptr ? std::max(1, std::min(g.class_n_comp[k], g.max_m)) : 1;
  const int max_m = g.enabled ? g.max_m : 1;
  double sum = 0.0;
  for (int m = 0; m < M; ++m) {
    const double* delta = g.D + class_gmm_d_offset(k, m, g.d, max_m);
    for (int j = 0; j < g.d; ++j) {
      const double vj = std::max(v0[static_cast<std::size_t>(j)], kEmMinVar);
      const double dm = delta[static_cast<std::size_t>(j)];
      sum += dm * dm / vj;
    }
  }
  return sum;
}

void class_gmm_ensure_capacity(int K, int d, int max_m, std::vector<double>& D, std::vector<double>& class_pi,
                               std::vector<int>& class_n_comp) {
  if (K < 0 || d <= 0 || max_m <= 0) {
    throw std::runtime_error("class_gmm_ensure_capacity: bad dims");
  }
  const std::size_t need_d = static_cast<std::size_t>(K) * static_cast<std::size_t>(max_m) * static_cast<std::size_t>(d);
  if (D.size() < need_d) {
    D.resize(need_d, 0.0);
  }
  const std::size_t need_pi = static_cast<std::size_t>(K) * static_cast<std::size_t>(max_m);
  if (class_pi.size() < need_pi) {
    class_pi.resize(need_pi, 0.0);
  }
  if (static_cast<int>(class_n_comp.size()) < K) {
    const int old_k = static_cast<int>(class_n_comp.size());
    class_n_comp.resize(static_cast<std::size_t>(K), 1);
    for (int k = old_k; k < K; ++k) {
      class_gmm_init_class_row(k, d, max_m, max_m > 1 ? kClassGmmDefaultM : 1, static_cast<std::uint64_t>(k + 1),
                               D, class_pi, class_n_comp);
    }
  }
}

void class_gmm_init_class_row(int k, int d, int max_m, int n_comp, std::uint64_t seed, std::vector<double>& D,
                              std::vector<double>& class_pi, std::vector<int>& class_n_comp) {
  n_comp = std::max(1, std::min(n_comp, max_m));
  class_n_comp[static_cast<std::size_t>(k)] = n_comp;
  for (int m = 0; m < max_m; ++m) {
    double* delta = D.data() + class_gmm_d_offset(k, m, d, max_m);
    for (int j = 0; j < d; ++j) {
      delta[static_cast<std::size_t>(j)] = 0.0;
    }
    class_pi[class_gmm_d_offset(k, m, 1, max_m)] = 0.0;
  }
  const double inv_m = 1.0 / static_cast<double>(n_comp);
  for (int m = 0; m < n_comp; ++m) {
    class_pi[class_gmm_d_offset(k, m, 1, max_m)] = inv_m;
  }
  // Secondary components: small orthogonal perturbations so XOR lobes can split.
  for (int m = 1; m < n_comp; ++m) {
    double* delta = D.data() + class_gmm_d_offset(k, m, d, max_m);
    const std::uint64_t s = seed * 9973u + static_cast<std::uint64_t>(k) * 131u +
                            static_cast<std::uint64_t>(m) * 17u;
    for (int j = 0; j < d; ++j) {
      const double u = static_cast<double>((s + static_cast<std::uint64_t>(j) * 2654435761u) % 10007u) / 5003.5 -
                       1.0;
      delta[static_cast<std::size_t>(j)] = 0.15 * u * static_cast<double>((j % 2 == 0) ? 1 : -1);
    }
  }
}

void class_gmm_update_pi_ema(int k, int max_m, const double* resp, int n_comp, double alpha,
                             std::vector<double>& class_pi) {
  n_comp = std::max(1, std::min(n_comp, max_m));
  double sum = 0.0;
  for (int m = 0; m < n_comp; ++m) {
    const std::size_t idx = class_gmm_d_offset(k, m, 1, max_m);
    class_pi[idx] = (1.0 - alpha) * class_pi[idx] + alpha * resp[static_cast<std::size_t>(m)];
    sum += class_pi[idx];
  }
  sum = std::max(sum, kEmEps);
  for (int m = 0; m < n_comp; ++m) {
    const std::size_t idx = class_gmm_d_offset(k, m, 1, max_m);
    class_pi[idx] /= sum;
  }
}

void class_gmm_hard_split_warmstart(int d, const double* world_mu, const std::vector<std::vector<double>>& h_rows,
                                    const std::vector<int>& class_k, int max_m, int n_comp,
                                    std::vector<double>& D, std::vector<double>& class_pi,
                                    std::vector<int>& class_n_comp) {
  if (d <= 0 || h_rows.empty() || class_k.size() != h_rows.size() || world_mu == nullptr) {
    return;
  }
  n_comp = std::max(1, std::min(n_comp, max_m));
  int K = 0;
  for (int k : class_k) K = std::max(K, k + 1);
  if (K <= 0) return;
  class_gmm_ensure_capacity(K, d, max_m, D, class_pi, class_n_comp);

  for (int k = 0; k < K; ++k) {
    std::vector<std::size_t> idxs;
    for (std::size_t i = 0; i < class_k.size(); ++i) {
      if (class_k[i] == k) idxs.push_back(i);
    }
    if (idxs.empty()) continue;
    class_n_comp[static_cast<std::size_t>(k)] = n_comp;
    // Class mean in field space.
    std::vector<double> mean(static_cast<std::size_t>(d), 0.0);
    for (std::size_t i : idxs) {
      for (int j = 0; j < d; ++j) {
        mean[static_cast<std::size_t>(j)] += h_rows[i][static_cast<std::size_t>(j)];
      }
    }
    const double inv_n = 1.0 / static_cast<double>(idxs.size());
    for (int j = 0; j < d; ++j) mean[static_cast<std::size_t>(j)] *= inv_n;

    int axis = 0;
    double best_var = -1.0;
    for (int j = 0; j < d; ++j) {
      double v = 0.0;
      for (std::size_t i : idxs) {
        const double diff = h_rows[i][static_cast<std::size_t>(j)] - mean[static_cast<std::size_t>(j)];
        v += diff * diff;
      }
      if (v > best_var) {
        best_var = v;
        axis = j;
      }
    }

    std::vector<double> mean0(static_cast<std::size_t>(d), 0.0);
    std::vector<double> mean1(static_cast<std::size_t>(d), 0.0);
    int n0 = 0;
    int n1 = 0;
    for (std::size_t i : idxs) {
      const bool side = h_rows[i][static_cast<std::size_t>(axis)] >= mean[static_cast<std::size_t>(axis)];
      auto& dest = side ? mean1 : mean0;
      int& cnt = side ? n1 : n0;
      for (int j = 0; j < d; ++j) dest[static_cast<std::size_t>(j)] += h_rows[i][static_cast<std::size_t>(j)];
      ++cnt;
    }
    if (n0 == 0 || n1 == 0) {
      // Degenerate split — keep single-mode at class mean.
      class_n_comp[static_cast<std::size_t>(k)] = 1;
      double* delta = D.data() + class_gmm_d_offset(k, 0, d, max_m);
      for (int j = 0; j < d; ++j) {
        delta[static_cast<std::size_t>(j)] = mean[static_cast<std::size_t>(j)] - world_mu[static_cast<std::size_t>(j)];
      }
      class_pi[class_gmm_d_offset(k, 0, 1, max_m)] = 1.0;
      continue;
    }
    const double inv0 = 1.0 / static_cast<double>(n0);
    const double inv1 = 1.0 / static_cast<double>(n1);
    for (int j = 0; j < d; ++j) {
      mean0[static_cast<std::size_t>(j)] *= inv0;
      mean1[static_cast<std::size_t>(j)] *= inv1;
    }
    for (int m = 0; m < max_m; ++m) {
      class_pi[class_gmm_d_offset(k, m, 1, max_m)] = 0.0;
      double* delta = D.data() + class_gmm_d_offset(k, m, d, max_m);
      for (int j = 0; j < d; ++j) delta[static_cast<std::size_t>(j)] = 0.0;
    }
    const std::vector<double>* means[2] = {&mean0, &mean1};
    const int counts[2] = {n0, n1};
    const double total = static_cast<double>(n0 + n1);
    for (int m = 0; m < std::min(2, n_comp); ++m) {
      double* delta = D.data() + class_gmm_d_offset(k, m, d, max_m);
      for (int j = 0; j < d; ++j) {
        delta[static_cast<std::size_t>(j)] =
            (*means[m])[static_cast<std::size_t>(j)] - world_mu[static_cast<std::size_t>(j)];
      }
      class_pi[class_gmm_d_offset(k, m, 1, max_m)] = static_cast<double>(counts[m]) / total;
    }
    class_n_comp[static_cast<std::size_t>(k)] = std::min(2, n_comp);
  }
}

}  // namespace cypha
