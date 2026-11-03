#include "cypha/infer_cpu.hpp"
#include "cypha/retrieval.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cypha {

namespace {

constexpr double kMinVar = 1e-4;
constexpr double kEps = 1e-8;

std::vector<double> world_variances(const CyphaInferModel& m) {
  const int d = m.d_latent;
  std::vector<double> v(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) {
    v[static_cast<std::size_t>(j)] = 1.0 / m.inv_v[static_cast<std::size_t>(j)];
  }
  return v;
}

}  // namespace

double diag_gaussian_logpdf(const double* h, const double* mu, const double* v, int d, double min_var) {
  double acc = 0.0;
  for (int j = 0; j < d; ++j) {
    const double vs = std::max(v[j], min_var);
    const double diff = h[j] - mu[j];
    acc += std::log(vs) + diff * diff / vs;
  }
  return -0.5 * acc;
}

bool class_params_for_label(const CyphaInferModel& m, const std::string& label, std::vector<double>& mu_out,
                            std::vector<double>& v_out) {
  const int d = m.d_latent;
  const int K = static_cast<int>(m.labels.size());
  int ki = -1;
  for (int k = 0; k < K; ++k) {
    if (m.labels[static_cast<std::size_t>(k)] == label) {
      ki = k;
      break;
    }
  }
  if (ki < 0) {
    return false;
  }
  mu_out.resize(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) {
    mu_out[static_cast<std::size_t>(j)] =
        m.mu_world[static_cast<std::size_t>(j)] + m.D[static_cast<std::size_t>(ki * d + j)];
  }
  v_out = world_variances(m);
  return true;
}

std::vector<RetrieveHit> retrieve_at_h(const CyphaInferModel& m, const double* h_query,
                                       const double* database_h, int n_db, int top_k,
                                       const CyphaInferOptions& opt,
                                       const std::optional<std::string>& label) {
  std::vector<RetrieveHit> out;
  const int d = m.d_latent;
  const int K = static_cast<int>(m.labels.size());
  if (K == 0 || d <= 0 || n_db <= 0 || top_k <= 0) {
    return out;
  }

  const double* h_field = opt.use_field ? m.field_h.data() : nullptr;
  std::optional<double> mahal_ema_opt;
  if (m.has_mahal_ema && std::isfinite(m.mahal_ema) && m.mahal_ema > kEps) {
    mahal_ema_opt = m.mahal_ema;
  }

  const ClassifyAtHResult q_cls =
      classify_at_h(m, h_query, h_field, m.temperature, mahal_ema_opt, m.mahal_std_ema, 1.0, 1.0, true);
  const std::string use_label = label.has_value() ? *label : q_cls.label;

  std::vector<double> mu_k;
  std::vector<double> v0;
  if (!class_params_for_label(m, use_label, mu_k, v0)) {
    return out;
  }

  std::vector<std::pair<double, RetrieveHit>> scored;
  scored.reserve(static_cast<std::size_t>(n_db));
  for (int i = 0; i < n_db; ++i) {
    const double* h_i = database_h + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
    const double ll = diag_gaussian_logpdf(h_i, mu_k.data(), v0.data(), d);
    const ClassifyAtHResult item_cls =
        classify_at_h(m, h_i, h_field, m.temperature, mahal_ema_opt, m.mahal_std_ema, 1.0, 1.0, true);
    scored.push_back({ll, RetrieveHit{i, ll, item_cls.label}});
  }

  std::sort(scored.begin(), scored.end(),
            [](const std::pair<double, RetrieveHit>& a, const std::pair<double, RetrieveHit>& b) {
              if (a.first != b.first) {
                return a.first > b.first;
              }
              return a.second.index < b.second.index;
            });

  const int k_actual = std::min(top_k, n_db);
  out.reserve(static_cast<std::size_t>(k_actual));
  for (int i = 0; i < k_actual; ++i) {
    out.push_back(scored[static_cast<std::size_t>(i)].second);
  }
  return out;
}

}  // namespace cypha
