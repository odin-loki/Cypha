#include "cypha/cyphalm/cyphalm_generation.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>

#include "cypha/curriculum.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
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

constexpr double kLogNegInf = -1e30;

bool decode_modifiers_active(const DecodeParams& params) {
    return params.ban_last_k > 0 || params.repetition_penalty > 1.0 + 1e-9 ||
           params.text_like_prior > 0.0 || params.min_p > 0.0 || params.no_repeat_ngram > 1;
}

double text_like_log_bonus(int byte, double strength) {
    if (strength <= 0.0) {
        return 0.0;
    }
    if (byte >= 32 && byte <= 126) {
        return strength;
    }
    if (byte == '\n' || byte == '\t' || byte == '\r') {
        return strength * 0.5;
    }
    if (byte < 32 || byte == 127) {
        return -strength;
    }
    return 0.0;
}

void apply_decode_modifiers(std::vector<double>& lp, const std::vector<int>& recent,
                            const DecodeParams& params) {
    const int n = static_cast<int>(lp.size());
    if (n <= 0) {
        return;
    }
    const int rep_win = params.repetition_window > 0 ? params.repetition_window : 32;
    const int rep_start = std::max(0, static_cast<int>(recent.size()) - rep_win);

    if (params.ban_last_k > 0) {
        const int ban_start =
            std::max(0, static_cast<int>(recent.size()) - params.ban_last_k);
        for (int i = ban_start; i < static_cast<int>(recent.size()); ++i) {
            const int b = recent[static_cast<std::size_t>(i)];
            if (b >= 0 && b < n) {
                lp[static_cast<std::size_t>(b)] = kLogNegInf;
            }
        }
    }

    if (params.repetition_penalty > 1.0) {
        const double pen = std::log(params.repetition_penalty);
        for (int i = rep_start; i < static_cast<int>(recent.size()); ++i) {
            const int b = recent[static_cast<std::size_t>(i)];
            if (b >= 0 && b < n) {
                lp[static_cast<std::size_t>(b)] -= pen;
            }
        }
    }

    if (params.text_like_prior > 0.0) {
        for (int b = 0; b < n; ++b) {
            lp[static_cast<std::size_t>(b)] += text_like_log_bonus(b, params.text_like_prior);
        }
    }

    // No-repeat n-gram: if the last n-1 bytes occurred earlier in the window, ban
    // the byte that followed them there. Skipped if it would ban every byte.
    const int ng = params.no_repeat_ngram;
    const int sz = static_cast<int>(recent.size());
    if (ng > 1 && sz >= ng - 1) {
        const int win_start = std::max(0, sz - std::max(ng, params.no_repeat_window));
        std::vector<int> banned;
        for (int i = win_start; i + ng - 1 < sz; ++i) {
            bool same = true;
            for (int k = 0; k < ng - 1 && same; ++k) {
                same = recent[static_cast<std::size_t>(i + k)] ==
                       recent[static_cast<std::size_t>(sz - (ng - 1) + k)];
            }
            if (same) banned.push_back(recent[static_cast<std::size_t>(i + ng - 1)]);
        }
        int allowed = 0;
        for (int b = 0; b < n; ++b) {
            if (lp[static_cast<std::size_t>(b)] > kLogNegInf / 2 &&
                std::find(banned.begin(), banned.end(), b) == banned.end()) {
                ++allowed;
            }
        }
        if (allowed > 0) {
            for (int b : banned) {
                if (b >= 0 && b < n) lp[static_cast<std::size_t>(b)] = kLogNegInf;
            }
        }
    }

    // min-p: keep bytes with p >= min_p * p_max.
    if (params.min_p > 0.0) {
        const double mx = *std::max_element(lp.begin(), lp.end());
        const double floor_lp = mx + std::log(params.min_p);
        for (double& v : lp) {
            if (v < floor_lp) v = kLogNegInf;
        }
    }
}

void prime_serve_context(CyphaLMModel& model, const std::vector<int>& warmup_ids,
                         const std::vector<int>& prompt_ids) {
    // New stream on the trained model. (reset_context() here used to replace the
    // predictor with an untrained one, so generation ignored all training.) The
    // byte history is kept: match models copy from it, and wiping it cost
    // 0.02-0.03 bits/byte held-out (CYPHALM_LM_QUALITY_REPORT.md).
    model.reset_stream(/*keep_history=*/true);
    model.set_serve_mode(true);
    for (int id : warmup_ids) {
        model.serve_advance(static_cast<std::uint32_t>(id));
    }
    if (prompt_ids.size() <= 1) {
        return;
    }
    for (std::size_t i = 0; i + 1 < prompt_ids.size(); ++i) {
        model.serve_advance(static_cast<std::uint32_t>(prompt_ids[i]));
    }
}

std::vector<int> build_recent_context(const std::vector<int>& warmup_ids,
                                      const std::vector<int>& prompt_ids,
                                      const std::vector<int>& generated_ids) {
    std::vector<int> recent;
    recent.reserve(warmup_ids.size() + prompt_ids.size() + generated_ids.size());
    recent.insert(recent.end(), warmup_ids.begin(), warmup_ids.end());
    recent.insert(recent.end(), prompt_ids.begin(), prompt_ids.end());
    recent.insert(recent.end(), generated_ids.begin(), generated_ids.end());
    return recent;
}

int pick_token_from_pred(const PredictNextOutput& pred, const std::vector<int>& recent,
                         const DecodeParams& sample_params, std::mt19937_64& rng) {
    std::vector<double> lp = pred.log_probs;
    apply_decode_modifiers(lp, recent, sample_params);
    if (sample_params.strategy == DecodeStrategy::Greedy || sample_params.temperature <= 1e-6) {
        return argmax_log_probs(lp);
    }
    return sample_token(lp, sample_params, rng);
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
    const bool hybrid = model.config().context_mode == ContextMode::Hybrid ||
                        model.config().context_mode == ContextMode::Hp;
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
    if (name == "beam") return DecodeStrategy::Beam;
    if (name == "uncertainty_gated") return DecodeStrategy::UncertaintyGated;
    return DecodeStrategy::Temperature;
}

/// Restores online learning when a decode loop exits by any path.
struct LearningGuard {
    explicit LearningGuard(HpSequenceBackend& hp) : hp_(hp) {}
    ~LearningGuard() { hp_.set_learning(true); }
    LearningGuard(const LearningGuard&) = delete;
    LearningGuard& operator=(const LearningGuard&) = delete;
    HpSequenceBackend& hp_;
};

GenerateOutput generate_beam(CyphaLMModel& model, const std::vector<int>& prompt_ids, int max_bytes,
                             const DecodeParams& params) {
    GenerateOutput out;
    out.strategy = DecodeStrategy::Beam;
    const int width = std::max(1, params.beam_width);
    if (max_bytes <= 0) {
        return out;
    }

    prime_serve_context(model, params.warmup_ids, prompt_ids);
    if (!prompt_ids.empty()) {
        model.serve_advance(static_cast<std::uint32_t>(prompt_ids.back()));
    }
    const int vocab = model.config().vocab_size;
    HpSequenceBackend& hp = model.hp_backend();
    const std::unique_ptr<hp::Predictor> root = hp.predictor_snapshot();
    hp::Predictor work(root->config());
    auto replay_tokens = [&](const std::vector<int>& tokens) {
        work.copy_state_from(*root);
        work.set_learning(params.learn_from_output);
        for (int t : tokens) {
            HpSequenceBackend::consume_byte_on(work, static_cast<std::uint8_t>(t));
        }
    };

    struct BeamHypothesis {
        double score = 0.0;
        std::vector<int> tokens;
    };

    std::vector<BeamHypothesis> beam;
    beam.push_back({});

    const int expand_k = std::min(vocab, std::max(2 * width, 16));

    for (int step = 0; step < max_bytes; ++step) {
        struct Candidate {
            BeamHypothesis hyp;
            double score = 0.0;
        };
        std::vector<Candidate> candidates;

        for (const BeamHypothesis& hyp : beam) {
            replay_tokens(hyp.tokens);
            std::vector<double> lp = HpSequenceBackend::byte_log_probs_bit_tree(work, vocab);
            const std::vector<int> recent =
                build_recent_context(params.warmup_ids, prompt_ids, hyp.tokens);
            apply_decode_modifiers(lp, recent, params);
            std::vector<int> order(static_cast<std::size_t>(vocab));
            for (int i = 0; i < vocab; ++i) {
                order[static_cast<std::size_t>(i)] = i;
            }
            const int kk = std::min(expand_k, vocab);
            std::partial_sort(order.begin(), order.begin() + kk, order.end(), [&](int a, int b) {
                return lp[static_cast<std::size_t>(a)] > lp[static_cast<std::size_t>(b)];
            });
            for (int i = 0; i < kk; ++i) {
                const int b = order[static_cast<std::size_t>(i)];
                Candidate cand;
                cand.hyp.tokens = hyp.tokens;
                cand.hyp.tokens.push_back(b);
                cand.hyp.score = hyp.score + lp[static_cast<std::size_t>(b)];
                cand.score = cand.hyp.score;
                candidates.push_back(std::move(cand));
            }
        }

        const int keep = std::min(width, static_cast<int>(candidates.size()));
        std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                          [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
        beam.clear();
        for (int i = 0; i < keep; ++i) {
            beam.push_back(std::move(candidates[static_cast<std::size_t>(i)].hyp));
        }
    }

    if (beam.empty()) {
        return out;
    }
    std::sort(beam.begin(), beam.end(),
              [](const BeamHypothesis& a, const BeamHypothesis& b) { return a.score > b.score; });
    const BeamHypothesis& best = beam[0];
    out.generated_ids = best.tokens;
    for (int id : best.tokens) {
        GenerateStep step_row;
        step_row.token_id = id;
        step_row.loss = -hp.log_prob_byte(static_cast<std::uint8_t>(id));
        out.per_step.push_back(step_row);
        model.serve_advance(static_cast<std::uint32_t>(id));
    }
    return out;
}

namespace {

/// True if a ``n``-byte window ending inside ``tail`` (appended to ``ctx``)
/// already occurs earlier in ``ctx`` + ``tail``.
bool repeats_ngram(const std::vector<int>& ctx, const std::vector<int>& tail, int n) {
    if (n <= 0) return false;
    std::vector<int> all(ctx);
    const std::size_t base = all.size();
    all.insert(all.end(), tail.begin(), tail.end());
    const std::size_t kn = static_cast<std::size_t>(n);
    for (std::size_t e = base; e < all.size(); ++e) {
        if (e + 1 < kn) continue;
        const std::size_t st = e + 1 - kn;
        for (std::size_t j = 0; j < st; ++j) {
            if (std::equal(all.begin() + static_cast<std::ptrdiff_t>(j),
                           all.begin() + static_cast<std::ptrdiff_t>(j + kn),
                           all.begin() + static_cast<std::ptrdiff_t>(st))) {
                return true;
            }
        }
    }
    return false;
}

bool is_word_byte(int b) {
    return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') || b >= 0x80;
}

}  // namespace

GenerateOutput generate_word_lookahead(CyphaLMModel& model, const std::vector<int>& prompt_ids,
                                       int max_bytes, const DecodeParams& params) {
    GenerateOutput out;
    out.strategy = params.strategy;
    if (max_bytes <= 0) return out;
    prime_serve_context(model, params.warmup_ids, prompt_ids);
    HpSequenceBackend& hp = model.hp_backend();
    LearningGuard learning_guard(hp);
    // The prompt's last byte is learned like the rest of the prompt.
    if (!prompt_ids.empty()) model.serve_advance(static_cast<std::uint32_t>(prompt_ids.back()));
    hp.set_learning(false);

    const int vocab = model.config().vocab_size;
    const int k_cands = std::max(1, params.word_candidates);
    constexpr int kMaxWordBytes = 24;
    constexpr std::size_t kRepeatWindow = 1024;
    std::mt19937_64 rng(params.seed);
    DecodeParams sp = params;
    sp.strategy = effective_sample_strategy(params);

    struct Cand {
        std::vector<int> bytes;
        std::vector<double> lp;
        double sum = 0.0;
    };
    std::vector<int>& gen = out.generated_ids;
    while (static_cast<int>(gen.size()) < max_bytes) {
        std::vector<Cand> cands;
        {
            hp::StreamRewind rewind(hp.predictor());
            for (int k = 0; k < k_cands; ++k) {
                Cand c;
                bool seen_word = false;
                std::vector<int> so_far = gen;
                for (int j = 0; j < kMaxWordBytes &&
                                static_cast<int>(gen.size() + c.bytes.size()) < max_bytes;
                     ++j) {
                    const std::vector<double> lp = hp.serve_next_byte_log_probs(vocab);
                    std::vector<double> mod = lp;
                    apply_decode_modifiers(mod, build_recent_context(params.warmup_ids, prompt_ids, so_far), sp);
                    const int b = (sp.strategy == DecodeStrategy::Greedy || sp.temperature <= 1e-6)
                                      ? argmax_log_probs(mod)
                                      : sample_token(mod, sp, rng);
                    c.bytes.push_back(b);
                    c.lp.push_back(lp[static_cast<std::size_t>(b)]);
                    c.sum += lp[static_cast<std::size_t>(b)];
                    so_far.push_back(b);
                    hp.serve_advance_byte(static_cast<std::uint8_t>(b));
                    if (is_word_byte(b)) seen_word = true;
                    else if (seen_word) break;  // the word and its delimiter
                }
                cands.push_back(std::move(c));
                rewind.rewind();
            }
        }
        // Highest mean log p among candidates that do not repeat recent text.
        std::vector<int> recent = build_recent_context(params.warmup_ids, prompt_ids, gen);
        if (recent.size() > kRepeatWindow) {
            recent.erase(recent.begin(), recent.end() - static_cast<std::ptrdiff_t>(kRepeatWindow));
        }
        std::size_t best = 0;
        double best_score = -1e300;
        for (std::size_t i = 0; i < cands.size(); ++i) {
            if (cands[i].bytes.empty() || repeats_ngram(recent, cands[i].bytes, params.word_no_repeat)) continue;
            const double score = cands[i].sum / static_cast<double>(cands[i].bytes.size());
            if (score > best_score) {
                best_score = score;
                best = i;
            }
        }
        const Cand& pick = cands[best];  // all repeat: the first sample
        if (pick.bytes.empty()) break;
        for (std::size_t j = 0; j < pick.bytes.size(); ++j) {
            hp.serve_advance_byte(static_cast<std::uint8_t>(pick.bytes[j]));
            gen.push_back(pick.bytes[j]);
            GenerateStep step;
            step.token_id = pick.bytes[j];
            step.loss = -pick.lp[j];
            out.per_step.push_back(step);
        }
    }
    return out;
}

GenerateOutput generate_decode(CyphaLMModel& model, const std::vector<int>& prompt_ids, int max_tokens,
                               const DecodeParams& params,
                               cypha::intelligence::EpistemicThreshold* epistemic_threshold,
                               cypha::intelligence::IntelligenceProfiler* profiler,
                               LmIntelligenceMonitor* monitor) {
    const int beam_width =
        params.strategy == DecodeStrategy::Beam ? std::max(2, params.beam_width) : params.beam_width;
    if (beam_width > 1) {
        return generate_beam(model, prompt_ids, max_tokens, params);
    }
    if (params.word_candidates > 1 && !params.learn_from_output) {
        return generate_word_lookahead(model, prompt_ids, max_tokens, params);
    }

    GenerateOutput out;
    out.strategy = params.strategy;
    prime_serve_context(model, params.warmup_ids, prompt_ids);
    int last = prompt_ids.empty() ? 0 : prompt_ids.back();
    std::mt19937_64 rng(params.seed);
    DecodeParams sample_params = params;
    sample_params.strategy = effective_sample_strategy(params);
    const bool use_fast_greedy = !params.exact_greedy &&
        params.strategy == DecodeStrategy::Greedy && params.temperature <= 1e-6 &&
        !decode_modifiers_active(params);
    LmIntelligenceMonitor local_monitor;
    LmIntelligenceMonitor* active_monitor = monitor != nullptr ? monitor : nullptr;
    if (profiler != nullptr && active_monitor == nullptr) {
        active_monitor = &local_monitor;
    }

    LearningGuard learning_guard(model.hp_backend());
    for (int i = 0; i < max_tokens; ++i) {
        // The prompt's last byte is always learned like the rest of the prompt;
        // generated bytes follow DecodeParams::learn_from_output.
        model.hp_backend().set_learning(i == 0 || params.learn_from_output);
        PredictNextOutput pred;
        int tok = 0;
        const std::vector<int> recent =
            build_recent_context(params.warmup_ids, prompt_ids, out.generated_ids);
        if (use_fast_greedy) {
            tok = static_cast<int>(model.serve_greedy_next(static_cast<std::uint32_t>(last)));
        } else {
            pred = model.serve_predict_next(static_cast<std::uint32_t>(last));
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
                    tok = pick_token_from_pred(pred, recent, sample_params, rng);
                    last = tok;
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
            tok = pick_token_from_pred(pred, recent, sample_params, rng);
        }
        const double loss =
            use_fast_greedy
                ? -model.hp_backend().log_prob_byte(static_cast<std::uint8_t>(tok))
                : (pred.log_probs.empty() || tok < 0 ||
                   tok >= static_cast<int>(pred.log_probs.size()))
                      ? 0.0
                      : -pred.log_probs[static_cast<std::size_t>(tok)];
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
    if (params.word_candidates > 1 && !params.learn_from_output) {
        // Word lookahead picks whole words; emit its bytes once decoded.
        const GenerateOutput g = generate_word_lookahead(model, prompt_ids, max_tokens, params);
        for (std::size_t i = 0; i < g.per_step.size(); ++i) {
            if (!cb(step_record_json(g.per_step[i], static_cast<int>(i), false, false))) return;
        }
        (void)cb(step_record_json(GenerateStep{}, static_cast<int>(g.per_step.size()), true, false));
        return;
    }
    prime_serve_context(model, params.warmup_ids, prompt_ids);
    int last = prompt_ids.empty() ? 0 : prompt_ids.back();
    std::mt19937_64 rng(params.seed);
    DecodeParams sample_params = params;
    sample_params.strategy = effective_sample_strategy(params);
    const bool use_fast_greedy = !params.exact_greedy &&
        params.strategy == DecodeStrategy::Greedy && params.temperature <= 1e-6 &&
        !decode_modifiers_active(params);
    int index = 0;
    std::vector<int> generated_ids;
    LmIntelligenceMonitor local_monitor;
    LmIntelligenceMonitor* active_monitor = monitor != nullptr ? monitor : nullptr;
    if (profiler != nullptr && active_monitor == nullptr) {
        active_monitor = &local_monitor;
    }

    LearningGuard learning_guard(model.hp_backend());
    for (int i = 0; i < max_tokens; ++i) {
        // The prompt's last byte is always learned like the rest of the prompt;
        // generated bytes follow DecodeParams::learn_from_output.
        model.hp_backend().set_learning(i == 0 || params.learn_from_output);
        PredictNextOutput pred;
        int tok = 0;
        const std::vector<int> recent =
            build_recent_context(params.warmup_ids, prompt_ids, generated_ids);
        if (use_fast_greedy) {
            tok = static_cast<int>(model.serve_greedy_next(static_cast<std::uint32_t>(last)));
        } else {
            pred = model.serve_predict_next(static_cast<std::uint32_t>(last));
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
                    tok = pick_token_from_pred(pred, recent, sample_params, rng);
                    last = tok;
                    const double loss =
                        pred.log_probs.empty() ? 0.0 : -pred.log_probs[static_cast<std::size_t>(last)];
                    observe_decode_step(model, profiler, active_monitor, ctx, pred,
                                        static_cast<std::uint32_t>(last));
                    GenerateStep step = step_from_pred(pred, last, loss);
                    generated_ids.push_back(last);
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
            tok = pick_token_from_pred(pred, recent, sample_params, rng);
        }
        const double loss =
            use_fast_greedy
                ? -model.hp_backend().log_prob_byte(static_cast<std::uint8_t>(tok))
                : (pred.log_probs.empty() || tok < 0 ||
                   tok >= static_cast<int>(pred.log_probs.size()))
                      ? 0.0
                      : -pred.log_probs[static_cast<std::size_t>(tok)];
        observe_decode_step(model, profiler, active_monitor, static_cast<std::uint32_t>(last), pred,
                            static_cast<std::uint32_t>(tok));
        GenerateStep step = step_from_pred(pred, tok, loss);
        generated_ids.push_back(tok);
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
    const auto pred = model.serve_predict_next(static_cast<std::uint32_t>(token_id));
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

std::vector<int> load_warmup_bytes(const std::string& path, int max_bytes, int vocab_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open warmup file: " + path);
    }
    std::vector<int> out;
    if (max_bytes <= 0) {
        return out;
    }
    out.reserve(static_cast<std::size_t>(max_bytes));
    for (int i = 0; i < max_bytes; ++i) {
        char ch = 0;
        if (!in.get(ch)) {
            break;
        }
        const unsigned char c = static_cast<unsigned char>(ch);
        if (static_cast<int>(c) < vocab_size) {
            out.push_back(static_cast<int>(c));
        }
    }
    return out;
}

void warmup_serve_context(CyphaLMModel& model, const std::vector<int>& warmup_ids) {
    for (int id : warmup_ids) {
        model.serve_advance(static_cast<std::uint32_t>(id));
    }
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
