#pragma once

#include <vector>

#include <nlohmann/json.hpp>

namespace cypha::cyphalm {

/// Diagonal NIG expert (matches ``cypha_lm.expert_field.nig_expert.NIGExpert``).
class NIGExpert {
 public:
    NIGExpert(double kappa0, double alpha0, double beta0, int dim);

    int dim() const { return dim_; }
    void update(const double* y);
    void predictive_mean(std::vector<double>& out) const;
    /// Per-dimension ``1 / predictive_variance`` (clamped).
    void predictive_inv_variance(std::vector<double>& out) const;
    double epistemic_variance_mean() const;
    double aleatoric_variance_mean() const;
    double predictive_log_prob(const double* x) const;
    double predictive_entropy() const;

    nlohmann::json state_dict() const;
    void load_state_dict(const nlohmann::json& state);
    int dim_;
    double kappa0_;
    double alpha0_;
    double beta0_;
    std::vector<double> mu_n_;
    std::vector<double> kappa_n_;
    std::vector<double> alpha_n_;
    std::vector<double> beta_n_;
};

}  // namespace cypha::cyphalm
