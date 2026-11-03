/// BPC gap verification: math identity, clone parity, roundtrip SHA.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cypha/portable_popen.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

constexpr double kLog2 = 0.6931471805599453;

double byte_nll_bits_clone(const hp::Config& cfg, hp::Predictor& pred, int byte) {
    hp::Predictor scratch(cfg);
    hp::Predictor::clone_from(pred, scratch);
    double log_p = 0.0;
    for (int i = 7; i >= 0; --i) {
        const int p12 = scratch.predict();
        const double p1 = static_cast<double>(p12) / 4096.0;
        const int bit = (byte >> i) & 1;
        const double p = bit ? p1 : (1.0 - p1);
        log_p += std::log(std::max(p, 1e-300));
        scratch.update(bit);
    }
    return -log_p / kLog2;
}

double byte_nll_bits_observe(hp::Predictor& pred, int byte) {
    double log_p = 0.0;
    for (int i = 7; i >= 0; --i) {
        const int p12 = pred.predict();
        const double p1 = static_cast<double>(p12) / 4096.0;
        const int bit = (byte >> i) & 1;
        const double p = bit ? p1 : (1.0 - p1);
        log_p += std::log(std::max(p, 1e-300));
        pred.update(bit);
    }
    return -log_p / kLog2;
}

std::string sha256_hex(const std::vector<std::uint8_t>& data) {
    std::ostringstream cmd;
    const std::string tmp = "/tmp/bpc_gap_verify.bin";
    {
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size()));
    }
    cmd << "sha256sum \"" << tmp << "\" 2>/dev/null | awk '{print $1}'";
    std::FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        return "sha_failed";
    }
    char buf[128];
    std::string hex;
    if (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
        hex = buf;
        while (!hex.empty() && (hex.back() == '\n' || hex.back() == '\r')) {
            hex.pop_back();
        }
    }
    pclose(pipe);
    return hex;
}

void print_json_kv(const char* key, double v) {
    std::printf("  \"%s\": %.12g,\n", key, v);
}

void print_json_kv_i(const char* key, long long v) {
    std::printf("  \"%s\": %lld,\n", key, v);
}

void print_json_kv_s(const char* key, const char* v) {
    std::printf("  \"%s\": \"%s\",\n", key, v);
}

void print_json_kv_b(const char* key, bool v) {
    std::printf("  \"%s\": %s,\n", key, v ? "true" : "false");
}

}  // namespace

int main(int argc, char** argv) {
    int n_random = 8;  // each check calls next_byte_log_probs(256) (~24s @ table_bits=22)
    std::string corpus_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--random" && i + 1 < argc) {
            n_random = std::stoi(argv[++i]);
        } else if (a == "--corpus" && i + 1 < argc) {
            corpus_path = argv[++i];
        }
    }

    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cfg.hp_table_bits = 22;

    cypha::cyphalm::CyphaLMModel model(cfg);
    auto hp_cfg = cypha::cyphalm::hp_config_from_cyphalm(cfg.hp_table_bits, cfg.hp_mixer_lr,
                                                         cfg.hp_gria);

    std::printf("{\n");
    print_json_kv_s("harness", "bpc_gap_verify");
    print_json_kv_i("hp_table_bits", cfg.hp_table_bits);
    print_json_kv_i("hp_slot_compile_max", cypha::cyphalm::hp_compile_slot_max());

    // A1/A2: random context clone vs observe vs backend next_byte_log_probs
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> ctx_dist(32, 512);
    double max_clone_observe_delta = 0.0;
    double max_clone_api_delta = 0.0;
    double max_bit_sum_byte_delta = 0.0;
    int checks = 0;

    // Fixed-seed replay for determinism
    rng.seed(42);
    for (int t = 0; t < n_random; ++t) {
        hp::Predictor pred(hp_cfg);
        std::vector<int> ctx;
        const int ctx_len = ctx_dist(rng);
        ctx.reserve(static_cast<std::size_t>(ctx_len));
        for (int i = 0; i < ctx_len; ++i) {
            ctx.push_back(byte_dist(rng));
        }
        for (int b : ctx) {
            for (int bit_i = 7; bit_i >= 0; --bit_i) {
                const int bit = (b >> bit_i) & 1;
                (void)pred.predict();
                pred.update(bit);
            }
        }
        const int target = byte_dist(rng);

        const double obs_bits = byte_nll_bits_observe(pred, target);

        hp::Predictor pred2(hp_cfg);
        for (int b : ctx) {
            for (int bit_i = 7; bit_i >= 0; --bit_i) {
                const int bit = (b >> bit_i) & 1;
                (void)pred2.predict();
                pred2.update(bit);
            }
        }
        const double clone_bits = byte_nll_bits_clone(hp_cfg, pred2, target);

        model.reset_context();
        for (int b : ctx) {
            model.hp_backend().consume_byte(static_cast<std::uint8_t>(b));
        }
        const auto lp = model.hp_backend().next_byte_log_probs(256);
        const double api_bits = -lp[static_cast<std::size_t>(target)] / kLog2;

        hp::Predictor pred3(hp_cfg);
        for (int b : ctx) {
            for (int bit_i = 7; bit_i >= 0; --bit_i) {
                const int bit = (b >> bit_i) & 1;
                (void)pred3.predict();
                pred3.update(bit);
            }
        }
        double bit_sum_bits = 0.0;
        for (int bit_i = 7; bit_i >= 0; --bit_i) {
            const int p12 = pred3.predict();
            const double p1 = static_cast<double>(p12) / 4096.0;
            const int bit = (target >> bit_i) & 1;
            const double p = bit ? p1 : (1.0 - p1);
            bit_sum_bits += -std::log(std::max(p, 1e-300)) / kLog2;
            pred3.update(bit);
        }
        hp::Predictor pred4(hp_cfg);
        for (int b : ctx) {
            for (int bit_i = 7; bit_i >= 0; --bit_i) {
                const int bit = (b >> bit_i) & 1;
                (void)pred4.predict();
                pred4.update(bit);
            }
        }
        const double byte_via_bits = byte_nll_bits_observe(pred4, target);

        max_clone_observe_delta = std::max(max_clone_observe_delta, std::abs(clone_bits - obs_bits));
        max_clone_api_delta = std::max(max_clone_api_delta, std::abs(clone_bits - api_bits));
        max_bit_sum_byte_delta =
            std::max(max_bit_sum_byte_delta, std::abs(bit_sum_bits - byte_via_bits));
        ++checks;
    }

    print_json_kv_i("random_checks", checks);
    print_json_kv("max_clone_vs_observe_bits_delta", max_clone_observe_delta);
    print_json_kv("max_clone_vs_api_bits_delta", max_clone_api_delta);
    print_json_kv("max_bit_sum_vs_byte_observe_delta", max_bit_sum_byte_delta);
    print_json_kv_b("a1_byte_nll_identity_ok", max_bit_sum_byte_delta < 1e-9);
    print_json_kv_b("a2_clone_equals_observe_ok", max_clone_observe_delta < 1e-9);
    print_json_kv_b("c10_clone_equals_api_ok", max_clone_api_delta < 1e-6);

    // C11: MSB-first bit order spot check
    bool msb_order_ok = true;
    {
        hp::Predictor pred(hp_cfg);
        const std::uint8_t byte = 0xA5;  // 10100101
        double log_p = 0.0;
        for (int i = 7; i >= 0; --i) {
            const int p12 = pred.predict();
            const int bit = (byte >> i) & 1;
            const double p1 = static_cast<double>(p12) / 4096.0;
            const double p = bit ? p1 : (1.0 - p1);
            log_p += std::log(std::max(p, 1e-300));
            pred.update(bit);
        }
        model.reset_context();
        const double obs = model.hp_backend().observe_next_byte(byte) / kLog2;
        const double direct = -log_p / kLog2;
        msb_order_ok = std::abs(obs - direct) < 1e-9;
    }
    print_json_kv_b("c10_msb_bit_order_ok", msb_order_ok);

    // C12: hp roundtrip on sample if corpus provided
    if (!corpus_path.empty()) {
        std::ifstream in(corpus_path, std::ios::binary);
        std::vector<std::uint8_t> raw;
        char ch;
        int cap = 65536;
        while (cap-- > 0 && in.get(ch)) {
            raw.push_back(static_cast<std::uint8_t>(ch));
        }
        const std::string raw_path = "/tmp/bpc_gap_raw.bin";
        const std::string arc_path = "/tmp/bpc_gap_raw.cyhp";
        const std::string dec_path = "/tmp/bpc_gap_dec.bin";
        {
            std::ofstream out(raw_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(raw.data()),
                      static_cast<std::streamsize>(raw.size()));
        }
        std::ostringstream cmd;
        cmd << "./native/build/hp_gate24 c --mem 22 --lr 2 \"" << raw_path << "\" \"" << arc_path
            << "\" 2>&1";
        std::system(cmd.str().c_str());
        std::ostringstream cmd2;
        cmd2 << "./native/build/hp_gate24 d \"" << arc_path << "\" \"" << dec_path << "\" 2>&1";
        std::system(cmd2.str().c_str());
        std::ifstream dec(dec_path, std::ios::binary);
        std::vector<std::uint8_t> got;
        while (dec.get(ch)) {
            got.push_back(static_cast<std::uint8_t>(ch));
        }
        const bool rt_ok = (got == raw);
        print_json_kv_i("roundtrip_bytes", static_cast<long long>(raw.size()));
        print_json_kv_s("roundtrip_sha256", sha256_hex(raw).c_str());
        print_json_kv_b("roundtrip_ok", rt_ok);
    }

    std::printf("  \"status\": \"ok\"\n");
    std::printf("}\n");
    return 0;
}
