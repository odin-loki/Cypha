#pragma once

#include <vector>

namespace cypha {

struct CyphaInferModel;

namespace rpsm {

/// Unified CyphaDIF state matrices (Option A scaffold).
/// Row 0 of ``mu`` is world prior μ (field-adjusted); rows 1..K are class differentials Δk.
/// ``inv_var`` is diagonal Ψ_var precision (shared NIG field; length ``feat_dim``).
struct PsiMatrices {
  int feat_dim{};
  int n_classes{};
  std::vector<double> mu;       // (1 + n_classes) × feat_dim row-major
  std::vector<double> inv_var;  // feat_dim
  std::vector<double> counts;   // n_obs per class
  double v_mean{};
};

/// Build Ψ from ``CyphaInferModel`` (world prior + field injection on row 0, ``D`` on rows 1..K).
PsiMatrices build_psi_from_model(const CyphaInferModel& m);

/// In-place variant for infer hot-path scratch reuse (same numerics as ``build_psi_from_model``).
void build_psi_from_model_into(PsiMatrices& psi, const CyphaInferModel& m);

/// Batched LLR via a single GEMM-shaped multiply: ``llr_out`` is ``n × n_classes`` row-major.
/// ``ctx`` length ``n_classes`` (Tier-1 context prior); may be nullptr → zeros.
void batched_llr_gemm(const double* h_row_major, int n, const PsiMatrices& psi, const double* ctx,
                      double* llr_out);

}  // namespace rpsm
}  // namespace cypha
