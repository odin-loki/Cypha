#include "cypha/lm/predictive_codec.hpp"

#include "cypha/online/adaptive_predictor_mixer.hpp"
#include "cypha/lm/arithmetic_coder.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace cypha::cyphalm {
namespace {

constexpr double kLn2 = 0.693147180559945309417;

AdaptiveMixerConfig mixer_cfg_from(const CyphaLMModel& model, const PredictiveCodecOptions& opt) {
    AdaptiveMixerConfig c;
    c.vocab_size = model.config().vocab_size;
    c.max_order = std::max(0, opt.ngram_max_order);
    c.mixer_lr = opt.mixer_lr;
    c.laplace = opt.laplace;
    c.use_match_expert = opt.use_match_expert;
    c.use_knn_expert = opt.use_knn_expert;
    c.use_bias_expert = opt.use_bias_expert;
    c.neural_prior = opt.neural_prior;
    c.neural_weight_floor = std::clamp(opt.neural_weight_floor, 0.0, 0.99);
    c.min_context_count = std::max(0, opt.min_context_count);
    return c;
}

/// Mix neural log-probs with hidden-kNN log-probs in probability space.
void blend_hidden_knn(CyphaLMModel& model, std::vector<double>& neural_lp, double mix_w) {
    if (!(mix_w > 0.0) || mix_w >= 1.0) return;
    auto knn = model.hidden_knn_log_probs();
    if (knn.size() != neural_lp.size()) return;
    const int V = static_cast<int>(neural_lp.size());
    double max_n = neural_lp[0];
    double max_k = knn[0];
    for (int i = 1; i < V; ++i) {
        max_n = std::max(max_n, neural_lp[static_cast<std::size_t>(i)]);
        max_k = std::max(max_k, knn[static_cast<std::size_t>(i)]);
    }
    double zn = 0.0;
    double zk = 0.0;
    std::vector<double> pn(static_cast<std::size_t>(V));
    std::vector<double> pk(static_cast<std::size_t>(V));
    for (int i = 0; i < V; ++i) {
        pn[static_cast<std::size_t>(i)] = std::exp(neural_lp[static_cast<std::size_t>(i)] - max_n);
        pk[static_cast<std::size_t>(i)] = std::exp(knn[static_cast<std::size_t>(i)] - max_k);
        zn += pn[static_cast<std::size_t>(i)];
        zk += pk[static_cast<std::size_t>(i)];
    }
    const double a = std::clamp(mix_w, 0.0, 1.0);
    for (int i = 0; i < V; ++i) {
        const double p = (1.0 - a) * (pn[static_cast<std::size_t>(i)] / zn) +
                         a * (pk[static_cast<std::size_t>(i)] / zk);
        neural_lp[static_cast<std::size_t>(i)] = std::log(p + 1e-12);
    }
}

}  // namespace

PredictiveCodecResult compress_tokens(CyphaLMModel& model, const std::vector<std::uint32_t>& tokens,
                                      const PredictiveCodecOptions& opt) {
    PredictiveCodecResult out;
    out.n_tokens = tokens.size();
    if (tokens.size() < 2) {
        out.detail = "need at least 2 tokens (seed + 1 coded)";
        return out;
    }
    const int vocab = model.config().vocab_size;
    if (vocab <= 0) {
        out.detail = "invalid vocab_size";
        return out;
    }

    model.reset_context();
    // Checkpoints omit Adam moments; post-train warm Adam on the encoder would diverge from a
    // freshly loaded decoder under online_adapt. Both sides start cold and stay locked.
    model.reset_optim_state();
    AdaptivePredictorMixer mixer(mixer_cfg_from(model, opt));
    if (opt.use_mixer) mixer.reset();

    ArithmeticEncoder enc;
    double nll_bits = 0.0;
    double neural_bits = 0.0;
    std::size_t coded = 0;
    std::vector<double> mix_lp;

    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
        const std::uint32_t ctx = tokens[i];
        const std::uint32_t nxt = tokens[i + 1];
        if (ctx >= static_cast<std::uint32_t>(vocab) || nxt >= static_cast<std::uint32_t>(vocab)) {
            out.detail = "token id out of vocab";
            return out;
        }
        auto pred = model.predict_next(ctx);
        if (pred.log_probs.size() != static_cast<std::size_t>(vocab)) {
            out.detail = "predict_next failed";
            return out;
        }
        const double neural_lp = pred.log_probs[nxt];
        if (!std::isfinite(neural_lp)) {
            out.detail = "non-finite log_prob";
            return out;
        }
        neural_bits += -neural_lp / kLn2;

        if (opt.use_hidden_knn && opt.hidden_knn_mix > 0.0) {
            blend_hidden_knn(model, pred.log_probs, opt.hidden_knn_mix);
        }
        const double* use_lp = pred.log_probs.data();
        if (opt.use_mixer) {
            mixer.predict(pred.log_probs.data(), mix_lp);
            use_lp = mix_lp.data();
        }
        const double lp = use_lp[nxt];
        if (!std::isfinite(lp)) {
            out.detail = "non-finite mix log_prob";
            return out;
        }
        nll_bits += -lp / kLn2;

        std::vector<double> lp_vec(use_lp, use_lp + vocab);
        SymbolCdf cdf = cdf_from_log_probs(lp_vec);
        enc.encode(nxt, cdf);
        if (opt.use_mixer) mixer.observe(nxt);
        if (opt.online_adapt) {
            (void)model.adapt_after_predict(nxt, opt.online_adapt_lr_scale);
        }
        ++coded;
    }

    out.bytes = enc.finish();
    out.n_coded = coded;
    out.model_bpc = coded > 0 ? nll_bits / static_cast<double>(coded) : 0.0;
    out.neural_bpc = coded > 0 ? neural_bits / static_cast<double>(coded) : 0.0;
    out.coded_bpc =
        coded > 0 ? (8.0 * static_cast<double>(out.bytes.size())) / static_cast<double>(coded) : 0.0;
    return out;
}

std::vector<std::uint32_t> decompress_tokens(CyphaLMModel& model, const std::vector<std::uint8_t>& bytes,
                                             std::uint32_t seed, std::size_t n_tokens,
                                             std::string* detail, const PredictiveCodecOptions& opt) {
    std::vector<std::uint32_t> tokens;
    if (n_tokens < 2) {
        if (detail) *detail = "need at least 2 tokens";
        return tokens;
    }
    const int vocab = model.config().vocab_size;
    if (seed >= static_cast<std::uint32_t>(vocab)) {
        if (detail) *detail = "seed out of vocab";
        return tokens;
    }

    model.reset_context();
    model.reset_optim_state();
    AdaptivePredictorMixer mixer(mixer_cfg_from(model, opt));
    if (opt.use_mixer) mixer.reset();

    ArithmeticDecoder dec(bytes);
    tokens.reserve(n_tokens);
    tokens.push_back(seed);
    std::vector<double> mix_lp;

    for (std::size_t i = 0; i + 1 < n_tokens; ++i) {
        const std::uint32_t ctx = tokens.back();
        auto pred = model.predict_next(ctx);
        if (pred.log_probs.size() != static_cast<std::size_t>(vocab)) {
            if (detail) *detail = "predict_next failed";
            tokens.clear();
            return tokens;
        }
        if (opt.use_hidden_knn && opt.hidden_knn_mix > 0.0) {
            blend_hidden_knn(model, pred.log_probs, opt.hidden_knn_mix);
        }
        const double* use_lp = pred.log_probs.data();
        if (opt.use_mixer) {
            mixer.predict(pred.log_probs.data(), mix_lp);
            use_lp = mix_lp.data();
        }
        std::vector<double> lp_vec(use_lp, use_lp + vocab);
        SymbolCdf cdf = cdf_from_log_probs(lp_vec);
        const std::uint32_t sym = dec.decode(cdf);
        if (sym >= static_cast<std::uint32_t>(vocab)) {
            if (detail) *detail = "decoded symbol out of vocab";
            tokens.clear();
            return tokens;
        }
        if (opt.use_mixer) mixer.observe(sym);
        if (opt.online_adapt) {
            (void)model.adapt_after_predict(sym, opt.online_adapt_lr_scale);
        }
        tokens.push_back(sym);
    }
    return tokens;
}

std::vector<std::uint32_t> generate_via_bits(CyphaLMModel& model, const std::vector<std::uint32_t>& prefix,
                                            std::size_t n_new, std::uint64_t rng_seed,
                                            std::string* detail, const PredictiveCodecOptions& opt) {
    std::vector<std::uint32_t> out = prefix;
    if (prefix.empty() || n_new == 0) {
        if (detail) *detail = "need non-empty prefix and n_new > 0";
        return out;
    }
    const int vocab = model.config().vocab_size;
    const std::size_t nbytes = 64 + n_new * 8;
    std::mt19937_64 rng(rng_seed);
    std::vector<std::uint8_t> bits(nbytes);
    for (std::size_t i = 0; i < nbytes; ++i) {
        bits[i] = static_cast<std::uint8_t>(rng() & 0xffu);
    }

    model.reset_context();
    model.reset_optim_state();
    AdaptivePredictorMixer mixer(mixer_cfg_from(model, opt));
    if (opt.use_mixer) mixer.reset();
    // Sync neural (+ mixer) state on prefix before sampling new tokens.
    for (std::size_t i = 0; i + 1 < prefix.size(); ++i) {
        auto pred = model.predict_next(prefix[i]);
        if (opt.use_mixer) {
            std::vector<double> sync_lp;
            mixer.predict(pred.log_probs.data(), sync_lp);
            mixer.observe(prefix[i + 1]);
        }
        if (opt.online_adapt) {
            model.adapt_after_predict(prefix[i + 1], opt.online_adapt_lr_scale);
        }
    }

    ArithmeticDecoder dec(bits);
    std::vector<double> mix_lp;
    for (std::size_t k = 0; k < n_new; ++k) {
        const std::uint32_t ctx = out.back();
        if (ctx >= static_cast<std::uint32_t>(vocab)) {
            if (detail) *detail = "context out of vocab";
            return out;
        }
        auto pred = model.predict_next(ctx);
        if (pred.log_probs.size() != static_cast<std::size_t>(vocab)) {
            if (detail) *detail = "predict_next failed";
            return out;
        }
        if (opt.use_hidden_knn && opt.hidden_knn_mix > 0.0) {
            blend_hidden_knn(model, pred.log_probs, opt.hidden_knn_mix);
        }
        const double* use_lp = pred.log_probs.data();
        if (opt.use_mixer) {
            mixer.predict(pred.log_probs.data(), mix_lp);
            use_lp = mix_lp.data();
        }
        std::vector<double> lp_vec(use_lp, use_lp + vocab);
        SymbolCdf cdf = cdf_from_log_probs(lp_vec);
        const std::uint32_t sym = dec.decode(cdf);
        if (opt.use_mixer) mixer.observe(sym);
        if (opt.online_adapt) {
            (void)model.adapt_after_predict(sym, opt.online_adapt_lr_scale);
        }
        out.push_back(sym);
    }
    return out;
}

}  // namespace cypha::cyphalm
