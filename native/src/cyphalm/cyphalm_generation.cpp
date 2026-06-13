#include "cypha/cyphalm/cyphalm_generation.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace cypha::cyphalm {

namespace {

int argmax_log_probs(const std::vector<double>& lp) {
    int best = 0;
    double mx = lp.empty() ? -1e30 : lp[0];
    for (int i = 1; i < static_cast<int>(lp.size()); ++i) {
        if (lp[static_cast<std::size_t>(i)] > mx) {
            mx = lp[static_cast<std::size_t>(i)];
            best = i;
        }
    }
    return best;
}

int sample_from_probs(const std::vector<double>& probs, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    const double r = ud(rng);
    double acc = 0.0;
    for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
        acc += probs[static_cast<std::size_t>(i)];
        if (r <= acc) return i;
    }
    return static_cast<int>(probs.size()) - 1;
}

int sample_top_k(const std::vector<double>& lp, double temperature, int top_k, std::mt19937_64& rng) {
    const int n = static_cast<int>(lp.size());
    const int kk = std::max(1, std::min(top_k, n));
    std::vector<int> idx(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) {
        return lp[static_cast<std::size_t>(a)] > lp[static_cast<std::size_t>(b)];
    });
    double mx = lp[static_cast<std::size_t>(idx[0])];
    std::vector<double> probs(static_cast<std::size_t>(kk));
    double sum = 0.0;
    for (int i = 0; i < kk; ++i) {
        probs[static_cast<std::size_t>(i)] =
            std::exp((lp[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])] - mx) / temperature);
        sum += probs[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < kk; ++i) probs[static_cast<std::size_t>(i)] /= sum + 1e-12;
    const int pick = sample_from_probs(probs, rng);
    return idx[static_cast<std::size_t>(pick)];
}

int sample_top_p(const std::vector<double>& lp, double temperature, double top_p, std::mt19937_64& rng) {
    const int n = static_cast<int>(lp.size());
    std::vector<int> order(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return lp[static_cast<std::size_t>(a)] > lp[static_cast<std::size_t>(b)];
    });
    std::vector<double> probs(static_cast<std::size_t>(n));
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        probs[static_cast<std::size_t>(i)] = std::exp(lp[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])] / temperature);
        sum += probs[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < n; ++i) probs[static_cast<std::size_t>(i)] /= sum + 1e-12;
    double cum = 0.0;
    int cutoff = 1;
    for (int i = 0; i < n; ++i) {
        cum += probs[static_cast<std::size_t>(i)];
        cutoff = i + 1;
        if (cum >= top_p) break;
    }
    std::vector<double> nucleus(static_cast<std::size_t>(cutoff));
    double nsum = 0.0;
    for (int i = 0; i < cutoff; ++i) {
        nucleus[static_cast<std::size_t>(i)] = probs[static_cast<std::size_t>(i)];
        nsum += nucleus[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < cutoff; ++i) nucleus[static_cast<std::size_t>(i)] /= nsum + 1e-12;
    const int pick = sample_from_probs(nucleus, rng);
    return order[static_cast<std::size_t>(pick)];
}

int sample_full_vocab(const std::vector<double>& lp, double temperature, std::mt19937_64& rng) {
    const int n = static_cast<int>(lp.size());
    double mx = lp.empty() ? 0.0 : *std::max_element(lp.begin(), lp.end());
    std::vector<double> probs(static_cast<std::size_t>(n));
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        probs[static_cast<std::size_t>(i)] = std::exp((lp[static_cast<std::size_t>(i)] - mx) / temperature);
        sum += probs[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < n; ++i) probs[static_cast<std::size_t>(i)] /= sum + 1e-12;
    return sample_from_probs(probs, rng);
}

int sample_token(const std::vector<double>& lp, const DecodeParams& params, std::mt19937_64& rng) {
    const double temp = std::max(params.temperature, 1e-6);
    if (params.strategy == DecodeStrategy::Greedy || params.temperature <= 1e-6) {
        return argmax_log_probs(lp);
    }
    if (params.strategy == DecodeStrategy::TopK) {
        return sample_top_k(lp, temp, params.top_k, rng);
    }
    if (params.strategy == DecodeStrategy::TopP) {
        return sample_top_p(lp, temp, params.top_p, rng);
    }
    return sample_full_vocab(lp, temp, rng);
}

void consume_prompt(CyphaLMModel& model, const std::vector<int>& prompt_ids) {
    model.reset_context();
    if (prompt_ids.size() <= 1) return;
    for (std::size_t i = 0; i + 1 < prompt_ids.size(); ++i) {
        model.predict_next(static_cast<std::uint32_t>(prompt_ids[i]));
    }
}

DecodeStrategy effective_sample_strategy(const DecodeParams& params) {
    if (params.strategy == DecodeStrategy::UncertaintyGated) return DecodeStrategy::Temperature;
    return params.strategy;
}

GenerateStep step_from_pred(const PredictNextOutput& pred, int token_id, double loss, bool halted = false) {
    GenerateStep s;
    s.token_id = token_id;
    s.loss = loss;
    s.epistemic_var = pred.epistemic_var;
    s.aleatoric_var = pred.aleatoric_var;
    s.halted = halted;
    return s;
}

nlohmann::json step_record_json(const GenerateStep& s, int index, bool done, bool halted_on_uncertainty) {
    nlohmann::json row;
    row["index"] = index;
    row["done"] = done;
    if (done) {
        row["halted_on_uncertainty"] = halted_on_uncertainty;
        if (halted_on_uncertainty) {
            row["epistemic_var"] = s.epistemic_var;
            row["active_experts"] = s.active_experts;
        }
        return row;
    }
    row["token_id"] = s.token_id;
    row["loss"] = s.loss;
    row["epistemic_var"] = s.epistemic_var;
    row["aleatoric_var"] = s.aleatoric_var;
    row["active_experts"] = s.active_experts;
    row["halted_on_uncertainty"] = false;
    return row;
}

bool uncertainty_halt(const DecodeParams& params, double epistemic_var) {
    return params.strategy == DecodeStrategy::UncertaintyGated && params.uncertainty_threshold.has_value() &&
           epistemic_var > *params.uncertainty_threshold;
}

}  // namespace

DecodeStrategy decode_strategy_from_string(const std::string& name) {
    if (name == "greedy") return DecodeStrategy::Greedy;
    if (name == "top_k") return DecodeStrategy::TopK;
    if (name == "top_p") return DecodeStrategy::TopP;
    if (name == "uncertainty_gated") return DecodeStrategy::UncertaintyGated;
    return DecodeStrategy::Temperature;
}

GenerateOutput generate_decode(CyphaLMModel& model, const std::vector<int>& prompt_ids, int max_tokens,
                               const DecodeParams& params) {
    GenerateOutput out;
    out.strategy = params.strategy;
    consume_prompt(model, prompt_ids);
    int last = prompt_ids.empty() ? 0 : prompt_ids.back();
    std::mt19937_64 rng(params.seed);
    DecodeParams sample_params = params;
    sample_params.strategy = effective_sample_strategy(params);

    for (int i = 0; i < max_tokens; ++i) {
        const auto pred = model.predict_next(static_cast<std::uint32_t>(last));
        if (uncertainty_halt(params, pred.epistemic_var)) {
            out.halted_on_uncertainty = true;
            GenerateStep halt_step;
            halt_step.epistemic_var = pred.epistemic_var;
            halt_step.aleatoric_var = pred.aleatoric_var;
            halt_step.halted = true;
            out.per_step.push_back(halt_step);
            break;
        }
        const int tok = sample_token(pred.log_probs, sample_params, rng);
        const double loss = -pred.log_probs[static_cast<std::size_t>(tok)];
        out.generated_ids.push_back(tok);
        out.per_step.push_back(step_from_pred(pred, tok, loss));
        last = tok;
    }
    return out;
}

GenerateOutput generate_greedy(CyphaLMModel& model, const std::vector<int>& prompt_ids, int max_tokens) {
    DecodeParams p;
    p.strategy = DecodeStrategy::Greedy;
    p.temperature = 0.0;
    return generate_decode(model, prompt_ids, max_tokens, p);
}

GenerateOutput generate_sample(CyphaLMModel& model, const std::vector<int>& prompt_ids, int max_tokens,
                               double temperature, int top_k, std::uint64_t seed) {
    DecodeParams p;
    p.strategy = DecodeStrategy::TopK;
    p.temperature = temperature;
    p.top_k = top_k;
    p.seed = seed;
    return generate_decode(model, prompt_ids, max_tokens, p);
}

void stream_generate(CyphaLMModel& model, const std::vector<int>& prompt_ids, int max_tokens,
                     const DecodeParams& params, const std::function<bool(const nlohmann::json&)>& cb) {
    consume_prompt(model, prompt_ids);
    int last = prompt_ids.empty() ? 0 : prompt_ids.back();
    std::mt19937_64 rng(params.seed);
    DecodeParams sample_params = params;
    sample_params.strategy = effective_sample_strategy(params);
    int index = 0;

    for (int i = 0; i < max_tokens; ++i) {
        const auto pred = model.predict_next(static_cast<std::uint32_t>(last));
        if (uncertainty_halt(params, pred.epistemic_var)) {
            GenerateStep halt_step;
            halt_step.epistemic_var = pred.epistemic_var;
            halt_step.aleatoric_var = pred.aleatoric_var;
            if (!cb(step_record_json(halt_step, index, true, true))) return;
            return;
        }
        const int tok = sample_token(pred.log_probs, sample_params, rng);
        const double loss = -pred.log_probs[static_cast<std::size_t>(tok)];
        GenerateStep step = step_from_pred(pred, tok, loss);
        if (!cb(step_record_json(step, index, false, false))) return;
        last = tok;
        ++index;
    }
    GenerateStep end_step;
    cb(step_record_json(end_step, index, true, false));
}

nlohmann::json predict_next_json(CyphaLMModel& model, int token_id) {
    const auto pred = model.predict_next(static_cast<std::uint32_t>(token_id));
    nlohmann::json j;
    j["token_id"] = token_id;
    j["log_probs"] = pred.log_probs;
    j["epistemic_var"] = pred.epistemic_var;
    j["aleatoric_var"] = pred.aleatoric_var;
    j["top_k_tokens"] = pred.top_k_tokens;
    j["top_k_probs"] = pred.top_k_probs;
    j["active_experts"] = 0;
    j["dominant_expert"] = 0;
    j["routing_probs"] = nlohmann::json::array();
    return j;
}

nlohmann::json lm_summary_json(const CyphaLMModel& model, const std::string& source_path, int n_generations) {
    nlohmann::json j;
    j["loaded"] = true;
    j["source_path"] = source_path;
    j["vocab_size"] = model.config().vocab_size;
    j["field_dim"] = model.config().field_dim;
    j["context_mode"] = context_mode_string(model.config().context_mode);
    j["n_generations"] = n_generations;
    j["hybrid_gria_weight"] = model.hybrid_gria_weight();
    return j;
}

}  // namespace cypha::cyphalm
