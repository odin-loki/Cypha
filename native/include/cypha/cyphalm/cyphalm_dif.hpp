#pragma once

#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_nig_expert.hpp"
#include "cypha/kernel_memory.hpp"

namespace cypha::cyphalm {

struct DIFPredictOutput {
    std::vector<double> mean;
    double epistemic_var = 0.0;
    double aleatoric_var = 0.0;
    std::vector<double> routing_probs;
    int active_experts = 0;
};

/// CyphaDIF expert field for native CyphaLM (routing + NIG experts).
/// When ``use_kernel_llr`` is set (H04), ``route`` blends linear LLR softmax with
/// ``KernelMemory`` Nyström scores (RBF proxy fallback until ``n_basis >= 4``).
class CyphaDIF {
 public:
    explicit CyphaDIF(const CyphaLMConfig& cfg);

    void reset();
    DIFPredictOutput predict(const double* x, int dim);
    void train_step(const double* x, int x_dim, const double* y, int y_dim);
    int expert_count() const { return static_cast<int>(experts_.size()); }
    int kernel_n_basis() const;
    std::vector<double> alpha_per_expert() const;

    nlohmann::json get_state() const;
    void set_state(const nlohmann::json& state);

    /// Override routing kernel blend for the current train/route step (Phase 38 κ scaling).
    void set_runtime_kernel_blend(double blend) { kernel_blend_ = blend; }
    double runtime_kernel_blend() const { return kernel_blend_; }

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
    bool use_soft_expert_updates_{false};
    bool use_kernel_llr_{false};
    double kernel_blend_{0.25};
    double kernel_lr_scale_{1.0};
    int input_dim_{-1};
    std::unique_ptr<cypha::KernelMemory> kernel_mem_;
    std::vector<Expert> experts_;

    double prior_log_prob(const double* x, int dim) const;
    std::vector<double> route(const double* x, int dim);
    void ensure_input_dim(int dim);
    Expert make_expert();
    static double field_magnitude_score(const double* x, int dim);
    std::vector<std::string> expert_labels() const;
    std::vector<double> kernel_proxy_routing_probs(const double* x, int dim) const;
    void blend_routing_with_kernel_proxy(std::vector<double>& probs, const double* x, int dim) const;
    void blend_routing_probs_with_kernel(std::vector<double>& probs, const double* x, int dim) const;
};

}  // namespace cypha::cyphalm
