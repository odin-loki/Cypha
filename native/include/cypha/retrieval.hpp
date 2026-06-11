#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cypha {

struct CyphaInferModel;
struct CyphaInferOptions;

/// One ranked retrieval hit: ``(database_index, log_likelihood, predicted_label)``.
struct RetrieveHit {
  int index{};
  double log_likelihood{};
  std::string predicted_label;
};

/// Python ``_diag_gaussian_logpdf(h, mu, v)`` with ``v`` variances (not inv_v).
double diag_gaussian_logpdf(const double* h, const double* mu, const double* v, int d,
                            double min_var = 1e-4);

/// Class mean ``mu_k`` and diagonal variance ``v0`` (Python ``DIFMemory.get_class_params``).
/// Returns false when ``label`` is unknown.
bool class_params_for_label(const CyphaInferModel& m, const std::string& label, std::vector<double>& mu_out,
                            std::vector<double>& v_out);

/// Python ``CyphaDIF.retrieve`` on pre-encoded latent rows.
/// ``database_h`` is ``n_db × d_latent`` row-major; ``h_query`` length ``d_latent``.
std::vector<RetrieveHit> retrieve_at_h(const CyphaInferModel& m, const double* h_query,
                                       const double* database_h, int n_db, int top_k,
                                       const CyphaInferOptions& opt,
                                       const std::optional<std::string>& label = std::nullopt);

}  // namespace cypha
