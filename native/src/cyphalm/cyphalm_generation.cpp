#include "cypha/cyphalm/cyphalm_generation.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "cypha/curriculum.hpp"
#include "cypha/cyphalm/cyphalm_intelligence_hook.hpp"
#include "cypha/cyphalm/lm_intelligence_monitor.hpp"
#include "cypha/intelligence/measurers.hpp"

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

double r_eu_from_pred(const PredictNextOutput& pred) {
    return cypha::intelligence::compute_epistemic_ratio(pred.epistemic_var, pred.aleatoric_var);
}

bool epistemic_should_halt(const DecodeParams& params, const PredictNextOutput& pred,
                           cypha::intelligence::EpistemicThreshold* threshold) {
    if (!params.epistemic_halt) {
        return false;
    }
    const double r_eu = r_eu_from_pred(pred);
    if (threshold != nullptr) {
        return threshold->should_correct(r_eu);
    }
    if (params.uncertainty_threshold.has_value()) {
        return r_eu > *params.uncertainty_threshold;
    }
    return r_eu > 0.5;
}

constexpr int kMaxSelfCorrectPasses = 3;

double pred_max_confidence(const PredictNextOutput& pred) {
    if (pred.log_probs.empty()) {
        return 0.0;
    }
    const int n = static_cast<int>(pred.log_probs.size());
    std::vector<double> probs(static_cast<std::size_t>(n));
    double mx = pred.log_probs[0];
    for (int i = 1; i < n; ++i) {
        mx = std::max(mx, pred.log_probs[static_cast<std::size_t>(i)]);
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        probs[static_cast<std::size_t>(i)] =
            std::exp(pred.log_probs[static_cast<std::size_t>(i)] - mx);
        sum += probs[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < n; ++i) {
        probs[static_cast<std::size_t>(i)] /= sum + 1e-12;
    }
    return cypha::row_max_softmax_confidence(probs.data(), n);
}

PredictNextOutput repredict_with_temperature(const PredictNextOutput& pred, double temperature) {
    PredictNextOutput out = pred;
    if (pred.log_probs.empty()) {
        return out;
    }
    const int n = static_cast<int>(pred.log_probs.size());
    const double temp = std::max(temperature, 1e-6);
    double mx = pred.log_probs[0];
    for (int i = 1; i < n; ++i) {
        mx = std::max(mx, pred.log_probs[static_cast<std::size_t>(i)]);
    }
    double sum = 0.0;
    out.log_probs.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        out.log_probs[static_cast<std::size_t>(i)] =
            std::exp((pred.log_probs[static_cast<std::size_t>(i)] - mx) / temp);
        sum += out.log_probs[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < n; ++i) {
        out.log_probs[static_cast<std::size_t>(i)] =
            std::log(out.log_probs[static_cast<std::size_t>(i)] / (sum + 1e-12) + 1e-12);
    }
    return out;
}

PredictNextOutput self_correct_predict(CyphaLMModel& model, const PredictNextOutput& initial,
                                       const DecodeParams& params,
                                       cypha::intelligence::EpistemicThreshold* threshold,
                                       int& passes_out) {
    PredictNextOutput best = initial;
    passes_out = 1;
    if (!params.self_correct || !epistemic_should_halt(params, best, threshold)) {
        return best;
    }

    const double saved_blend = model.hybrid_blend_logit();
    double pass_blend = saved_blend;
    double pass_temp = std::max(params.temperature, 1e-6);
    const bool hybrid = model.config().context_mode == ContextMode::Hybrid;
    double best_conf = pred_max_confidence(best);

    while (passes_out < kMaxSelfCorrectPasses &&
           epistemic_should_halt(params, best, threshold)) {
        pass_blend = pass_blend * 0.82 + 0.05;
        pass_temp = std::min(2.0, pass_temp * 1.08);
        PredictNextOutput retry = best;
        if (hybrid) {
            retry = model.repredict_hybrid_blend(pass_blend);
            retry.epistemic_var = best.epistemic_var;
            retry.aleatoric_var = best.aleatoric_var;
        } else {
            retry = repredict_with_temperature(best, pass_temp);
        }
        const double retry_conf = pred_max_confidence(retry);
        if (retry_conf > best_conf ||
            !epistemic_should_halt(params, retry, threshold)) {
            best = retry;
            best_conf = retry_conf;
        }
        ++passes_out;
    }

    model.set_hybrid_blend_logit(saved_blend);
    if (threshold != nullptr) {
        threshold->update(r_eu_from_pred(best), passes_out > 1);
    }
    return best;
}

std::vector<double> safe_embed(CyphaLMModel& model, std::uint32_t token_id) {
    try {
        return model.embed_vector(token_id);
    } catch (const std::exception&) {
        return {};
    }
}

void observe_decode_step(CyphaLMModel& model, cypha::intelligence::IntelligenceProfiler* profiler,
                         LmIntelligenceMonitor* monitor, std::uint32_t context_token,
                         const PredictNextOutput& pred, std::uint32_t next_token_id) {
    if (profiler == nullptr || monitor == nullptr) {
        return;
    }
    monitor->observe_token(safe_embed(model, context_token), model.field_vector(), pred.log_probs,
                           pred.epistemic_var, pred.aleatoric_var,
                           static_cast<std::int64_t>(next_token_id),
                           static_cast<int>(model.config().vocab_size));
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
                               const DecodeParams& params,
                               cypha::intelligence::EpistemicThreshold* epistemic_threshold,
                               cypha::intelligence::IntelligenceProfiler* profiler,
                               LmIntelligenceMonitor* monitor) {
    GenerateOutput out;
    out.strategy = params.strategy;
    consume_prompt(model, prompt_ids);
    int last = prompt_ids.empty() ? 0 : prompt_ids.back();
    std::mt19937_64 rng(params.seed);
    DecodeParams sample_params = params;
    sample_params.strategy = effective_sample_strategy(params);
    LmIntelligenceMonitor local_monitor;
    LmIntelligenceMonitor* active_monitor = monitor != nullptr ? monitor : nullptr;
    if (profiler != nullptr && active_monitor == nullptr) {
        active_monitor = &local_monitor;
    }

    for (int i = 0; i < max_tokens; ++i) {
        auto pred = model.predict_next(static_cast<std::uint32_t>(last));
        if (uncertainty_halt(params, pred.epistemic_var)) {
            out.halted_on_uncertainty = true;
            out.r_eu_proxy = r_eu_from_pred(pred);
            GenerateStep halt_step;
            halt_step.epistemic_var = pred.epistemic_var;
            halt_step.aleatoric_var = pred.aleatoric_var;
            halt_step.halted = true;
            out.per_step.push_back(halt_step);
            break;
        }
        if (epistemic_should_halt(params, pred, epistemic_threshold)) {
            if (params.self_correct) {
                int passes = 1;
                pred = self_correct_predict(model, pred, params, epistemic_threshold, passes);
                out.self_corrected = true;
                out.self_correct_passes = std::max(out.self_correct_passes, passes);
                const std::uint32_t ctx = static_cast<std::uint32_t>(last);
                last = argmax_log_probs(pred.log_probs);
                const double loss =
                    pred.log_probs.empty() ? 0.0 : -pred.log_probs[static_cast<std::size_t>(last)];
                observe_decode_step(model, profiler, active_monitor, ctx, pred,
                                    static_cast<std::uint32_t>(last));
                out.generated_ids.push_back(last);
                out.per_step.push_back(step_from_pred(pred, last, loss));
                continue;
            }
            const double r_eu = r_eu_from_pred(pred);
            out.halted_on_epistemic = true;
            out.halted_on_uncertainty = true;
            out.r_eu_proxy = r_eu;
            GenerateStep halt_step;
            halt_step.epistemic_var = pred.epistemic_var;
            halt_step.aleatoric_var = pred.aleatoric_var;
            halt_step.halted = true;
            out.per_step.push_back(halt_step);
            if (epistemic_threshold != nullptr) {
                epistemic_threshold->update(r_eu, false);
            }
            break;
        }
        const int tok = sample_token(pred.log_probs, sample_params, rng);
        const double loss = -pred.log_probs[static_cast<std::size_t>(tok)];
        observe_decode_step(model, profiler, active_monitor, static_cast<std::uint32_t>(last), pred,
                            static_cast<std::uint32_t>(tok));
        out.generated_ids.push_back(tok);
        out.per_step.push_back(step_from_pred(pred, tok, loss));
        last = tok;
    }
    if (profiler != nullptr && active_monitor != nullptr) {
        active_monitor->flush_to_profiler(*profiler);
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
                     const DecodeParams& params, const std::function<bool(const nlohmann::json&)>& cb,
                     cypha::intelligence::EpistemicThreshold* epistemic_threshold,
                     cypha::intelligence::IntelligenceProfiler* profiler,
                     LmIntelligenceMonitor* monitor) {
    consume_prompt(model, prompt_ids);
    int last = prompt_ids.empty() ? 0 : prompt_ids.back();
    std::mt19937_64 rng(params.seed);
    DecodeParams sample_params = params;
    sample_params.strategy = effective_sample_strategy(params);
    int index = 0;
    LmIntelligenceMonitor local_monitor;
    LmIntelligenceMonitor* active_monitor = monitor != nullptr ? monitor : nullptr;
    if (profiler != nullptr && active_monitor == nullptr) {
        active_monitor = &local_monitor;
    }

    for (int i = 0; i < max_tokens; ++i) {
        auto pred = model.predict_next(static_cast<std::uint32_t>(last));
        if (uncertainty_halt(params, pred.epistemic_var)) {
            GenerateStep halt_step;
            halt_step.epistemic_var = pred.epistemic_var;
            halt_step.aleatoric_var = pred.aleatoric_var;
            if (!cb(step_record_json(halt_step, index, true, true))) return;
            if (profiler != nullptr && active_monitor != nullptr) {
                active_monitor->flush_to_profiler(*profiler);
            }
            return;
        }
        if (epistemic_should_halt(params, pred, epistemic_threshold)) {
            if (params.self_correct) {
                int passes = 1;
                pred = self_correct_predict(model, pred, params, epistemic_threshold, passes);
                const std::uint32_t ctx = static_cast<std::uint32_t>(last);
                last = argmax_log_probs(pred.log_probs);
                const double loss =
                    pred.log_probs.empty() ? 0.0 : -pred.log_probs[static_cast<std::size_t>(last)];
                observe_decode_step(model, profiler, active_monitor, ctx, pred,
                                    static_cast<std::uint32_t>(last));
                GenerateStep step = step_from_pred(pred, last, loss);
                if (!cb(step_record_json(step, index, false, false))) return;
                ++index;
                continue;
            }
            GenerateStep halt_step;
            halt_step.epistemic_var = pred.epistemic_var;
            halt_step.aleatoric_var = pred.aleatoric_var;
            if (!cb(step_record_json(halt_step, index, true, true))) return;
            if (profiler != nullptr && active_monitor != nullptr) {
                active_monitor->flush_to_profiler(*profiler);
            }
            return;
        }
        const int tok = sample_token(pred.log_probs, sample_params, rng);
        const double loss = -pred.log_probs[static_cast<std::size_t>(tok)];
        observe_decode_step(model, profiler, active_monitor, static_cast<std::uint32_t>(last), pred,
                            static_cast<std::uint32_t>(tok));
        GenerateStep step = step_from_pred(pred, tok, loss);
        if (!cb(step_record_json(step, index, false, false))) return;
        last = tok;
        ++index;
    }
    if (profiler != nullptr && active_monitor != nullptr) {
        active_monitor->flush_to_profiler(*profiler);
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
