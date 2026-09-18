/// Comprehensive smoke: arithmetic coder unit tests + hybrid default + predictive roundtrip.
#include "cypha/cypha.hpp"
#include "cypha/cyphalm/arithmetic_coder.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/predictive_codec.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

int test_arithmetic_roundtrip() {
    using cypha::cyphalm::ArithmeticDecoder;
    using cypha::cyphalm::ArithmeticEncoder;
    using cypha::cyphalm::SymbolCdf;

    SymbolCdf cdf;
    // Alphabet {0,1,2} with masses 1,2,5 → total 8
    cdf.cum = {0, 1, 3, 8};
    const std::vector<std::uint32_t> msg = {2, 0, 1, 2, 2, 1, 0, 2, 1, 2};

    ArithmeticEncoder enc;
    for (auto s : msg) enc.encode(s, cdf);
    const auto bytes = enc.finish();
    if (bytes.empty()) return fail("encoder produced empty bitstream");

    ArithmeticDecoder dec(bytes);
    for (std::size_t i = 0; i < msg.size(); ++i) {
        const auto got = dec.decode(cdf);
        if (got != msg[i]) {
            return fail("arithmetic roundtrip mismatch at " + std::to_string(i));
        }
    }
    std::cout << "arithmetic_roundtrip OK bytes=" << bytes.size() << "\n";
    return 0;
}

int test_cdf_from_log_probs() {
    using cypha::cyphalm::cdf_from_log_probs;
    std::vector<double> lp = {std::log(0.1), std::log(0.2), std::log(0.7)};
    auto cdf = cdf_from_log_probs(lp, 1000);
    if (cdf.alphabet() != 3 || cdf.total() != 1000) {
        return fail("cdf size/total mismatch");
    }
    if (cdf.cum[0] != 0 || cdf.cum[3] != 1000) return fail("cdf endpoints");
    // Mode should get the largest mass.
    const std::uint32_t c0 = cdf.cum[1] - cdf.cum[0];
    const std::uint32_t c1 = cdf.cum[2] - cdf.cum[1];
    const std::uint32_t c2 = cdf.cum[3] - cdf.cum[2];
    if (!(c2 >= c1 && c1 >= c0)) return fail("cdf mass order unexpected");
    if (c0 < 1 || c1 < 1 || c2 < 1) return fail("zero mass symbol");
    std::cout << "cdf_from_log_probs OK masses=" << c0 << "," << c1 << "," << c2 << "\n";
    return 0;
}

std::vector<std::uint32_t> make_pattern_tokens(int n, int period) {
    std::vector<std::uint32_t> tokens;
    tokens.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        tokens.push_back(static_cast<std::uint32_t>((i % period) + 1));
    }
    return tokens;
}

void train_patterned(cypha::Cypha& model, const std::vector<std::uint32_t>& tokens) {
    for (int ep = 0; ep < 3; ++ep) {
        for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
            (void)model.train_token(tokens[i], tokens[i + 1]);
        }
    }
}

int test_hybrid_default_and_codec() {
    cypha::Cypha model;
    if (!model.init_default_sequence(64, 32)) return fail("init_default_sequence");
    auto* lm = model.sequence();
    if (!lm) return fail("sequence null");
    const auto& cfg = lm->config();
    if (cfg.context_mode != cypha::cyphalm::ContextMode::Hp) {
        return fail("expected hp context_mode, got " +
                    cypha::cyphalm::context_mode_name(cfg.context_mode));
    }
    if (cfg.use_pgm_cell || cfg.use_unified_context) {
        return fail("PGM/unified flags should be off on production default");
    }
    if (!cfg.ngram_fuse_split) return fail("ngram_fuse_split should be true");
    if (cfg.use_ngram_count_prior) return fail("use_ngram_count_prior should be false");
    std::cout << "hp_default OK mode=" << cypha::cyphalm::context_mode_name(cfg.context_mode)
              << " lstm_hidden=" << cfg.lstm_hidden << "\n";

    const auto tokens = make_pattern_tokens(256, 7);
    train_patterned(model, tokens);

    std::vector<int> ids(tokens.begin(), tokens.end());
    const double eval_bpc = lm->eval_bpc(ids, static_cast<int>(ids.size()));

    // Snapshot after train: online_adapt mutates weights during compress.
    const auto ckpt = std::filesystem::temp_directory_path() / "cypha_codec_smoke_seq";
    const std::string ckpt_json = ckpt.string() + ".json";
    model.save_sequence(ckpt.string());

    // Neural-only, frozen weights: model_bpc should match eval_bpc.
    cypha::cyphalm::PredictiveCodecOptions neural_only;
    neural_only.use_mixer = false;
    neural_only.online_adapt = false;
    neural_only.use_hidden_knn = false;  // identity with eval_bpc
    auto packed_n = model.compress_tokens(tokens, neural_only);
    if (!packed_n.detail.empty()) return fail("compress neural: " + packed_n.detail);
    if (std::abs(packed_n.model_bpc - eval_bpc) > 0.05) {
        return fail("neural model_bpc=" + std::to_string(packed_n.model_bpc) +
                    " vs eval_bpc=" + std::to_string(eval_bpc));
    }

    // Mixer path without neural adapt (same-instance roundtrip-safe).
    if (!model.load_sequence(ckpt_json)) return fail("reload after neural compress");
    cypha::cyphalm::PredictiveCodecOptions mixer_frozen;
    mixer_frozen.online_adapt = false;
    auto packed = model.compress_tokens(tokens, mixer_frozen);
    if (!packed.detail.empty()) return fail("compress: " + packed.detail);
    if (packed.n_coded != tokens.size() - 1) return fail("n_coded mismatch");
    if (packed.bytes.empty()) return fail("empty compressed bytes");
    if (packed.model_bpc > packed.neural_bpc + 0.15) {
        return fail("mixer dilutes trained neural too much");
    }

    // Cold (untrained) patterned stream: online n-gram experts should help.
    {
        cypha::Cypha cold;
        if (!cold.init_default_sequence(64, 32)) return fail("cold init");
        const auto pat = make_pattern_tokens(200, 5);
        cypha::cyphalm::PredictiveCodecOptions cold_opt;
        cold_opt.online_adapt = false;
        auto cold_pack = cold.compress_tokens(pat, cold_opt);
        if (!cold_pack.detail.empty()) return fail("cold compress: " + cold_pack.detail);
        if (!(cold_pack.model_bpc + 0.05 < cold_pack.neural_bpc)) {
            return fail("mixer should beat cold neural on repetitive stream mix=" +
                        std::to_string(cold_pack.model_bpc) +
                        " neural=" + std::to_string(cold_pack.neural_bpc));
        }
        std::cout << "mixer_cold_gain OK mix=" << cold_pack.model_bpc
                  << " neural=" << cold_pack.neural_bpc << "\n";
    }
    if (packed.coded_bpc > packed.model_bpc + 2.0) {
        return fail("coded_bpc far above model_bpc (coder waste)");
    }

    std::string detail;
    auto decoded = model.decompress_tokens(packed.bytes, tokens.front(), tokens.size(), &detail,
                                           mixer_frozen);
    if (!detail.empty()) return fail("decompress: " + detail);
    if (decoded != tokens) return fail("token roundtrip mismatch");

    if (!model.load_sequence(ckpt_json)) return fail("reload for generate");
    cypha::cyphalm::PredictiveCodecOptions gen_opt;
    gen_opt.online_adapt = false;
    auto gen = model.generate_via_bits({tokens.front(), tokens[1]}, 16, 12345ull, &detail, gen_opt);
    if (!detail.empty()) return fail("generate_via_bits: " + detail);
    if (gen.size() != 18) return fail("generate_via_bits length");
    for (auto t : gen) {
        if (t >= 64) return fail("generated token out of vocab");
    }

    std::cout << "predictive_codec OK mix_bpc=" << packed.model_bpc
              << " neural_bpc=" << packed.neural_bpc << " coded_bpc=" << packed.coded_bpc
              << " bytes=" << packed.bytes.size() << " eval_bpc=" << eval_bpc << "\n";
    return 0;
}

int test_online_adapt_roundtrip() {
    const auto tokens = make_pattern_tokens(128, 7);

    cypha::Cypha trained;
    if (!trained.init_default_sequence(64, 32)) return fail("online_adapt init");
    train_patterned(trained, tokens);

    const auto ckpt = std::filesystem::temp_directory_path() / "cypha_codec_online_adapt";
    const std::string ckpt_json = ckpt.string() + ".json";
    trained.save_sequence(ckpt.string());

    cypha::cyphalm::PredictiveCodecOptions opt;  // online_adapt=true by default
    if (!opt.online_adapt) return fail("online_adapt default should be true");

    cypha::Cypha enc;
    if (!enc.load_sequence(ckpt_json)) return fail("online_adapt enc load");
    auto packed = enc.compress_tokens(tokens, opt);
    if (!packed.detail.empty()) return fail("online_adapt compress: " + packed.detail);

    cypha::Cypha dec;
    if (!dec.load_sequence(ckpt_json)) return fail("online_adapt dec load");
    std::string detail;
    auto decoded = dec.decompress_tokens(packed.bytes, tokens.front(), tokens.size(), &detail, opt);
    if (!detail.empty()) return fail("online_adapt decompress: " + detail);
    if (decoded != tokens) return fail("online_adapt roundtrip mismatch");

    // Bit-identical: same checkpoint state → compress twice must match.
    cypha::Cypha a;
    cypha::Cypha b;
    if (!a.load_sequence(ckpt_json) || !b.load_sequence(ckpt_json)) {
        return fail("online_adapt bit-identical load");
    }
    auto p1 = a.compress_tokens(tokens, opt);
    auto p2 = b.compress_tokens(tokens, opt);
    if (!p1.detail.empty() || !p2.detail.empty()) return fail("online_adapt bit-identical compress");
    if (p1.bytes != p2.bytes) {
        return fail("online_adapt compress not bit-identical from same checkpoint");
    }

    std::cout << "online_adapt OK bytes=" << packed.bytes.size()
              << " model_bpc=" << packed.model_bpc << "\n";
    return 0;
}

}  // namespace

int main() {
    if (const int rc = test_arithmetic_roundtrip()) return rc;
    if (const int rc = test_cdf_from_log_probs()) return rc;
    if (const int rc = test_hybrid_default_and_codec()) return rc;
    if (const int rc = test_online_adapt_roundtrip()) return rc;
    std::cout << "predictive_codec_smoke OK\n";
    return 0;
}
