/// gate24 serve/infer microbench: predict_next, bit-tree, legacy, RSS (measured only).
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

#if defined(__linux__)
#include <fstream>
#endif

namespace {

struct ProcStatus {
    long vm_rss_kb = -1;
    long vm_hwm_kb = -1;
};

ProcStatus read_proc_status() {
    ProcStatus s;
#if defined(__linux__)
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &s.vm_rss_kb);
        } else if (line.rfind("VmHWM:", 0) == 0) {
            std::sscanf(line.c_str(), "VmHWM: %ld kB", &s.vm_hwm_kb);
        }
    }
#endif
    return s;
}

double elapsed_us(std::chrono::steady_clock::time_point t0, int iters) {
    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (sec / static_cast<double>(iters)) * 1e6;
}

}  // namespace

int main(int argc, char** argv) {
    int table_bits = 22;
    int warmup_bytes = 65536;
    int iters = 3;
    bool serve_compact = false;
    bool json_out = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--table-bits" && i + 1 < argc) {
            table_bits = std::stoi(argv[++i]);
        } else if (a == "--warmup-bytes" && i + 1 < argc) {
            warmup_bytes = std::stoi(argv[++i]);
        } else if (a == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (a == "--serve-compact") {
            serve_compact = true;
        } else if (a == "--json") {
            json_out = true;
        }
    }

    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cfg.hp_table_bits = table_bits;
    cfg.hp_serve_compact = serve_compact;

    const auto rss_construct = read_proc_status();
    cypha::cyphalm::CyphaLMModel model(cfg);
    const auto rss_after_model = read_proc_status();

    std::mt19937 rng(42);
    model.reset_context();
    for (int i = 0; i < warmup_bytes; ++i) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(rng() & 0xff));
    }
    const auto rss_after_warmup = read_proc_status();

    auto& hp = model.hp_backend();
    const std::uint32_t tok = 65;
    const int k = std::max(1, iters);

    using Clock = std::chrono::steady_clock;

    for (int i = 0; i < 2; ++i) {
        (void)model.predict_next(tok);
    }
    const auto t_predict = Clock::now();
    for (int i = 0; i < k; ++i) {
        (void)model.predict_next(tok);
    }
    const double predict_next_us = elapsed_us(t_predict, k);

    for (int i = 0; i < 2; ++i) {
        (void)hp.next_byte_log_probs(256);
    }
    const auto t_default = Clock::now();
    for (int i = 0; i < k; ++i) {
        (void)hp.next_byte_log_probs(256);
    }
    const double next_byte_log_probs_us = elapsed_us(t_default, k);

    for (int i = 0; i < 2; ++i) {
        (void)hp.next_byte_log_probs_bit_tree(256);
    }
    const auto t_tree = Clock::now();
    for (int i = 0; i < k; ++i) {
        (void)hp.next_byte_log_probs_bit_tree(256);
    }
    const double bit_tree_us = elapsed_us(t_tree, k);

    for (int i = 0; i < 2; ++i) {
        (void)hp.serve_greedy_next_byte();
    }
    const auto t_greedy = Clock::now();
    for (int i = 0; i < k; ++i) {
        (void)hp.serve_greedy_next_byte();
    }
    const double serve_greedy_us = elapsed_us(t_greedy, k);

    double legacy_us = -1.0;
    if (k <= 2) {
        for (int i = 0; i < 1; ++i) {
            (void)hp.next_byte_log_probs_legacy(256);
        }
        const auto t_legacy = Clock::now();
        for (int i = 0; i < k; ++i) {
            (void)hp.next_byte_log_probs_legacy(256);
        }
        legacy_us = elapsed_us(t_legacy, k);
    }

    const auto rss_after_bench = read_proc_status();

    if (json_out) {
        std::printf("{\n");
        std::printf("  \"profile\": \"gate24\",\n");
        std::printf("  \"hp_table_bits\": %d,\n", table_bits);
        std::printf("  \"hp_serve_compact\": %s,\n", serve_compact ? "true" : "false");
        std::printf("  \"warmup_bytes\": %d,\n", warmup_bytes);
        std::printf("  \"iters\": %d,\n", k);
        std::printf("  \"predict_next_us\": %.1f,\n", predict_next_us);
        std::printf("  \"next_byte_log_probs_us\": %.1f,\n", next_byte_log_probs_us);
        std::printf("  \"bit_tree_us\": %.1f,\n", bit_tree_us);
        std::printf("  \"serve_greedy_next_byte_us\": %.1f,\n", serve_greedy_us);
        if (legacy_us >= 0.0) {
            std::printf("  \"legacy_clone_us\": %.1f,\n", legacy_us);
        }
        std::printf("  \"vm_rss_construct_kb\": %ld,\n", rss_construct.vm_rss_kb);
        std::printf("  \"vm_rss_after_model_kb\": %ld,\n", rss_after_model.vm_rss_kb);
        std::printf("  \"vm_rss_after_warmup_kb\": %ld,\n", rss_after_warmup.vm_rss_kb);
        std::printf("  \"vm_hwm_after_bench_kb\": %ld,\n", rss_after_bench.vm_hwm_kb);
        std::printf("  \"status\": \"ok\"\n");
        std::printf("}\n");
    } else {
        std::printf(
            "hp_inference_bench OK table_bits=%d serve_compact=%d warmup=%d iters=%d\n"
            "  predict_next_us=%.0f\n"
            "  next_byte_log_probs_us=%.0f\n"
            "  bit_tree_us=%.0f\n"
            "  serve_greedy_us=%.0f\n",
            table_bits, serve_compact ? 1 : 0, warmup_bytes, k, predict_next_us,
            next_byte_log_probs_us, bit_tree_us, serve_greedy_us);
        if (legacy_us >= 0.0) {
            std::printf("  legacy_clone_us=%.0f\n", legacy_us);
        }
        std::printf("  vm_rss_construct_kb=%ld vm_rss_after_model_kb=%ld "
                    "vm_rss_after_warmup_kb=%ld vm_hwm_after_bench_kb=%ld\n",
                    rss_construct.vm_rss_kb, rss_after_model.vm_rss_kb,
                    rss_after_warmup.vm_rss_kb, rss_after_bench.vm_hwm_kb);
    }
    return 0;
}
