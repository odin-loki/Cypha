#include "cypha/cyphalm/cyphalm_dif.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace cypha::cyphalm {

namespace {

constexpr double kNoveltyThreshold = 0.05;
constexpr double kActiveThreshold = 0.001;

std::vector<double> softmax(const std::vector<double>& logits) {
    if (logits.empty()) return {};
    double mx = logits[0];
    for (double v : logits) mx = std::max(mx, v);
    std::vector<double> out = logits;
    double sum = 0.0;
    for (double& v : out) {
        v = std::exp(v - mx);
        sum += v;
    }
    if (sum <= 0.0) {
        const double u = 1.0 / static_cast<double>(out.size());
        std::fill(out.begin(), out.end(), u);
        return out;
    }
    for (double& v : out) v /= sum;
    return out;
}

}  // namespace

CyphaDIF::Expert::Expert(double k0, double a0, double b0, int in_dim, int out_dim)
    : input_nig(k0, a0, b0, in_dim), output_nig(k0, a0, b0, out_dim) {}

CyphaDIF::CyphaDIF(const CyphaLMConfig& cfg)
    : field_dim_(cfg.field_dim),
      max_experts_(std::max(1, cfg.max_experts)),
      kappa0_(cfg.nig_kappa0),
      alpha0_(cfg.nig_alpha0),
      beta0_(cfg.nig_beta0) {
    if (cfg.n_experts > 0) {
        input_dim_ = field_dim_;
        for (int i = 0; i < cfg.n_experts; ++i) {
            experts_.emplace_back(make_expert());
        }
    }
}

CyphaDIF::Expert CyphaDIF::make_expert() {
    if (input_dim_ < 0) {
        throw std::runtime_error("CyphaDIF: input dim unknown");
    }
    return Expert(kappa0_, alpha0_, beta0_, input_dim_, field_dim_);
}

void CyphaDIF::ensure_input_dim(int dim) {
    if (input_dim_ < 0) {
        input_dim_ = dim;
    } else if (input_dim_ != dim) {
        throw std::invalid_argument("CyphaDIF: input dim mismatch");
    }
}

void CyphaDIF::reset() {
    const int warm = static_cast<int>(experts_.size());
    experts_.clear();
    input_dim_ = warm > 0 ? field_dim_ : -1;
    for (int i = 0; i < warm; ++i) {
        experts_.push_back(make_expert());
    }
}

double CyphaDIF::prior_log_prob(const double* x, int dim) const {
    NIGExpert prior(kappa0_, alpha0_, beta0_, dim);
    return prior.predictive_log_prob(x);
}

std::vector<double> CyphaDIF::route(const double* x, int dim) {
    ensure_input_dim(dim);
    while (true) {
        if (experts_.empty()) {
            experts_.push_back(make_expert());
        }
        if (experts_.size() == 1) {
            const double prior = prior_log_prob(x, dim);
            const double llr = experts_[0].input_nig.predictive_log_prob(x) - prior;
            if (llr < std::log(kNoveltyThreshold + 1e-12) &&
                static_cast<int>(experts_.size()) < max_experts_) {
                experts_.push_back(make_expert());
                continue;
            }
            return std::vector<double>{1.0};
        }
        std::vector<double> llrs;
        llrs.reserve(experts_.size());
        const double prior = prior_log_prob(x, dim);
        for (const auto& ex : experts_) {
            llrs.push_back(ex.input_nig.predictive_log_prob(x) - prior);
        }
        auto probs = softmax(llrs);
        const double max_p = *std::max_element(probs.begin(), probs.end());
        if ((max_p < kNoveltyThreshold || llrs.empty()) &&
            static_cast<int>(experts_.size()) < max_experts_) {
            experts_.push_back(make_expert());
            continue;
        }
        return probs;
    }
}

DIFPredictOutput CyphaDIF::predict(const double* x, int dim) {
    DIFPredictOutput out;
    out.mean.assign(static_cast<std::size_t>(field_dim_), 0.0);
    const auto probs = route(x, dim);
    out.routing_probs = probs;
    if (experts_.empty()) {
        return out;
    }
    if (experts_.size() == 1) {
        experts_[0].output_nig.predictive_mean(out.mean);
        out.epistemic_var = experts_[0].output_nig.epistemic_variance_mean();
        out.aleatoric_var = experts_[0].output_nig.aleatoric_variance_mean();
        out.active_experts = 1;
        return out;
    }
    std::vector<std::vector<double>> means(experts_.size());
    double epi = 0.0;
    double ale = 0.0;
    int active = 0;
    for (std::size_t k = 0; k < experts_.size(); ++k) {
        means[k].resize(static_cast<std::size_t>(field_dim_));
        experts_[k].output_nig.predictive_mean(means[k]);
        const double p = probs[k];
        for (int i = 0; i < field_dim_; ++i) {
            out.mean[static_cast<std::size_t>(i)] += p * means[k][static_cast<std::size_t>(i)];
        }
        epi += p * experts_[k].output_nig.epistemic_variance_mean();
        ale += p * experts_[k].output_nig.aleatoric_variance_mean();
        if (p > kActiveThreshold) ++active;
    }
    out.epistemic_var = epi;
    out.aleatoric_var = ale;
    out.active_experts = active;
    return out;
}

void CyphaDIF::train_step(const double* x, int x_dim, const double* y, int y_dim) {
    if (y_dim != field_dim_) {
        throw std::invalid_argument("CyphaDIF::train_step: y dim mismatch");
    }
    const auto probs = route(x, x_dim);
    int winner = 0;
    for (int i = 1; i < static_cast<int>(probs.size()); ++i) {
        if (probs[static_cast<std::size_t>(i)] > probs[static_cast<std::size_t>(winner)]) {
            winner = i;
        }
    }
    auto& ex = experts_[static_cast<std::size_t>(winner)];
    ex.input_nig.update(x);
    ex.output_nig.update(y);
}

nlohmann::json CyphaDIF::get_state() const {
    nlohmann::json j;
    if (input_dim_ < 0) {
        j["input_dim"] = nullptr;
    } else {
        j["input_dim"] = input_dim_;
    }
    nlohmann::json experts = nlohmann::json::array();
    for (const auto& ex : experts_) {
        nlohmann::json row;
        row["input_nig"] = ex.input_nig.state_dict();
        row["output_nig"] = ex.output_nig.state_dict();
        row["activation_history"] = nlohmann::json::array();
        experts.push_back(row);
    }
    j["experts"] = experts;
    return j;
}

void CyphaDIF::set_state(const nlohmann::json& state) {
    experts_.clear();
    if (state.contains("input_dim") && !state.at("input_dim").is_null()) {
        input_dim_ = state.at("input_dim").get<int>();
    } else {
        input_dim_ = -1;
    }
    if (!state.contains("experts") || input_dim_ < 0) return;
    for (const auto& ex : state.at("experts")) {
        Expert expert = make_expert();
        expert.input_nig.load_state_dict(ex.at("input_nig"));
        expert.output_nig.load_state_dict(ex.at("output_nig"));
        experts_.push_back(std::move(expert));
    }
}

}  // namespace cypha::cyphalm
