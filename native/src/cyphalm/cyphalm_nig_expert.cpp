#include "cypha/cyphalm/cyphalm_nig_expert.hpp"

#include <cmath>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace cypha::cyphalm {

NIGExpert::NIGExpert(double kappa0, double alpha0, double beta0, int dim)
    : dim_(dim), kappa0_(kappa0), alpha0_(alpha0), beta0_(beta0) {
    if (dim_ < 1 || kappa0_ <= 0.0 || alpha0_ <= 1.0 || beta0_ <= 0.0) {
        throw std::invalid_argument("NIGExpert: invalid hyperparameters");
    }
    mu_n_.assign(static_cast<std::size_t>(dim_), 0.0);
    kappa_n_.assign(static_cast<std::size_t>(dim_), kappa0_);
    alpha_n_.assign(static_cast<std::size_t>(dim_), alpha0_);
    beta_n_.assign(static_cast<std::size_t>(dim_), beta0_);
}

void NIGExpert::update(const double* y) {
    for (int i = 0; i < dim_; ++i) {
        const double mu0 = mu_n_[static_cast<std::size_t>(i)];
        const double k0 = kappa_n_[static_cast<std::size_t>(i)];
        const double a0 = alpha_n_[static_cast<std::size_t>(i)];
        const double b0 = beta_n_[static_cast<std::size_t>(i)];
        const double yi = y[i];
        const double kn = k0 + 1.0;
        mu_n_[static_cast<std::size_t>(i)] = (k0 * mu0 + yi) / kn;
        kappa_n_[static_cast<std::size_t>(i)] = kn;
        alpha_n_[static_cast<std::size_t>(i)] = a0 + 0.5;
        beta_n_[static_cast<std::size_t>(i)] = b0 + (k0 * (yi - mu0) * (yi - mu0)) / (2.0 * kn);
    }
}

void NIGExpert::predictive_mean(std::vector<double>& out) const {
    out = mu_n_;
}

double NIGExpert::epistemic_variance_mean() const {
    double s = 0.0;
    for (int i = 0; i < dim_; ++i) {
        const double kn = kappa_n_[static_cast<std::size_t>(i)];
        const double an = alpha_n_[static_cast<std::size_t>(i)];
        const double bn = beta_n_[static_cast<std::size_t>(i)];
        s += bn / (kn * (an - 1.0));
    }
    return s / static_cast<double>(dim_);
}

double NIGExpert::aleatoric_variance_mean() const {
    double s = 0.0;
    for (int i = 0; i < dim_; ++i) {
        const double an = alpha_n_[static_cast<std::size_t>(i)];
        const double bn = beta_n_[static_cast<std::size_t>(i)];
        s += bn / (an - 1.0);
    }
    return s / static_cast<double>(dim_);
}

double NIGExpert::predictive_log_prob(const double* x) const {
    double lp = 0.0;
    for (int i = 0; i < dim_; ++i) {
        const double mu = mu_n_[static_cast<std::size_t>(i)];
        const double kn = kappa_n_[static_cast<std::size_t>(i)];
        const double an = alpha_n_[static_cast<std::size_t>(i)];
        const double bn = beta_n_[static_cast<std::size_t>(i)];
        const double nu = 2.0 * an;
        const double scale = std::sqrt(bn * (kn + 1.0) / (an * kn));
        const double z = (x[i] - mu) / (scale + 1e-12);
        lp += -0.5 * (nu + 1.0) * std::log1p(z * z / nu) - std::log(scale + 1e-12);
    }
    return lp;
}

nlohmann::json NIGExpert::state_dict() const {
    nlohmann::json j;
    j["mu_n"] = mu_n_;
    j["kappa_n"] = kappa_n_;
    j["alpha_n"] = alpha_n_;
    j["beta_n"] = beta_n_;
    return j;
}

void NIGExpert::load_state_dict(const nlohmann::json& state) {
    if (state.contains("mu_n")) {
        const auto& m = state.at("mu_n");
        if (m.is_array()) {
            mu_n_ = m.get<std::vector<double>>();
        } else {
            mu_n_.assign(1, m.get<double>());
        }
    }
    if (state.contains("kappa_n")) kappa_n_ = state.at("kappa_n").get<std::vector<double>>();
    if (state.contains("alpha_n")) alpha_n_ = state.at("alpha_n").get<std::vector<double>>();
    if (state.contains("beta_n")) beta_n_ = state.at("beta_n").get<std::vector<double>>();
    dim_ = static_cast<int>(mu_n_.size());
}

}  // namespace cypha::cyphalm
