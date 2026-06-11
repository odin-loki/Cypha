#pragma once

#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_nig_expert.hpp"

namespace cypha::cyphalm {

struct DIFPredictOutput {
    std::vector<double> mean;
    double epistemic_var = 0.0;
    double aleatoric_var = 0.0;
    std::vector<double> routing_probs;
    int active_experts = 0;
};

/// CyphaDIF expert field for native CyphaLM (routing + NIG experts).
class CyphaDIF {
 public:
    explicit CyphaDIF(const CyphaLMConfig& cfg);

    void reset();
    DIFPredictOutput predict(const double* x, int dim);
    void train_step(const double* x, int x_dim, const double* y, int y_dim);
    int expert_count() const { return static_cast<int>(experts_.size()); }
    std::vector<double> alpha_per_expert() const;

    nlohmann::json get_state() const;
    void set_state(const nlohmann::json& state);

    /// Expert-centered output means (K×field_dim row-major) + pooled inv variance for U4 feedback.
    bool discriminative_state(std::vector<double>& delta_mu_rows,
                              std::vector<double>& inv_v) const;

 private:
    struct Expert {
        NIGExpert input_nig;
        NIGExpert output_nig;
        std::vector<std::vector<double>> activation_history;
        Expert(double k0, double a0, double b0, int in_dim, int out_dim);
        double input_entropy() const;
        void record_activation(const double* x, int dim, int max_history = 256);
    };

    int field_dim_;
    int max_experts_;
    double kappa0_;
    double alpha0_;
    double beta0_;
    int input_dim_{-1};
    std::vector<Expert> experts_;

    double prior_log_prob(const double* x, int dim) const;
    std::vector<double> route(const double* x, int dim);
    void ensure_input_dim(int dim);
    Expert make_expert();
};

}  // namespace cypha::cyphalm
