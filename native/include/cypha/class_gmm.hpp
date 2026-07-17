#pragma once

/// Per-class diagonal Gaussian mixtures (Phase 3 optimality).
///
/// When ``use_class_gmm`` is false (default), ``D`` remains ``K×d`` (legacy single-mode).
/// When enabled, ``D`` is ``K×maxM×d`` row-major with ``class_pi`` (``K×maxM``) and
/// ``class_n_comp[k]`` in ``[1, kClassGmmMaxM]``.

#include "cypha/em_step.hpp"
#include "cypha/load_cypha.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cypha {

constexpr int kClassGmmDefaultM = 2;
constexpr int kClassGmmMaxM = 4;
constexpr int kCyphaFormatV3 = 3;
constexpr int kCyphaFormatV4 = 4;

/// Read-only view of per-class mixture storage on ``CyphaDifMemoryState`` / ``CyphaInferModel``.
struct ClassGmmStorage {
  bool enabled{false};
  int max_m{kClassGmmMaxM};
  int d{0};
  int K{0};
  const double* D{nullptr};
  const double* class_pi{nullptr};
  const int* class_n_comp{nullptr};
};

inline std::size_t class_gmm_d_offset(int k, int m, int d, int max_m) {
  return (static_cast<std::size_t>(k) * static_cast<std::size_t>(max_m) + static_cast<std::size_t>(m)) *
         static_cast<std::size_t>(d);
}

/// Single-mode LLR fragment: ``cross − 0.5·d_sq`` (no uncertainty / context).
inline double class_delta_llr_fragment(const double* delta, const double* r, const double* inv_v, int d) {
  double cross = 0.0;
  double d_sq = 0.0;
  for (int j = 0; j < d; ++j) {
    const double Dkj = delta[static_cast<std::size_t>(j)];
    cross += Dkj * r[static_cast<std::size_t>(j)];
    const double w = inv_v != nullptr ? inv_v[static_cast<std::size_t>(j)] : 1.0;
    d_sq += Dkj * Dkj * w;
  }
  return cross - 0.5 * d_sq;
}

/// ``log Σ_m π_m · exp(fragment_m)`` for class ``k`` (log-sum-exp stable).
double class_gmm_logsumexp_score(const ClassGmmStorage& g, int k, const double* r, const double* inv_v);

/// Full class LLR including MDL uncertainty ``u_k`` and context (GMM or single-mode).
double class_llr_for_k(const ClassGmmStorage& g, int k, const double* r, const double* inv_v, double u_k,
                       double ctx);

/// E-step responsibilities over components of class ``k``; writes ``r_out`` length ``class_n_comp[k]``.
void class_gmm_component_responsibilities(const ClassGmmStorage& g, int k, const double* r, const double* inv_v,
                                          double temperature, double* r_out);

/// Fisher–Rao norm summed over active components (MDL cap target).
double class_gmm_fisher_rao_norm(const ClassGmmStorage& g, int k, const double* v0);

/// Ensure ``D`` / ``class_pi`` / ``class_n_comp`` capacity for ``K`` classes × ``max_m`` components.
void class_gmm_ensure_capacity(int K, int d, int max_m, std::vector<double>& D, std::vector<double>& class_pi,
                               std::vector<int>& class_n_comp);

/// Init a newly created class row (``k``) with ``n_comp`` components; ``seed`` perturbs secondary modes.
void class_gmm_init_class_row(int k, int d, int max_m, int n_comp, std::uint64_t seed, std::vector<double>& D,
                              std::vector<double>& class_pi, std::vector<int>& class_n_comp);

/// Slow EMA update of mixing weights from component responsibilities.
void class_gmm_update_pi_ema(int k, int max_m, const double* resp, int n_comp, double alpha,
                             std::vector<double>& class_pi);

int read_cypha_format(const CNode& root);
bool read_class_gmm_enabled(const CNode& root);
int read_class_gmm_m(const CNode& root, bool enabled);

/// Load one class row from a ``classes`` entry (legacy ``M=1`` or ``M×d`` tensor).
void load_class_delta_from_cnode(const CNode& cnode, int d, int max_m, int k, std::vector<double>& D,
                                 std::vector<double>& class_pi, std::vector<int>& class_n_comp);

}  // namespace cypha
