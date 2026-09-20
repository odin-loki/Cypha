/// Gate24 hp hot-path profile — measured stage timings (no invented metrics).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_us(Clock::time_point t0, int iters) {
    const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    return (sec / static_cast<double>(iters)) * 1e6;
}

long read_vmhwm_kb() {
#if defined(__linux__)
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            long kb = 0;
            if (std::sscanf(line.c_str(), "VmHWM: %ld kB", &kb) == 1) {
                return kb;
            }
        }
    }
#endif
    return -1;
}

std::vector<int> load_bytes(const std::string& path, int max_n) {
    std::ifstream in(path, std::ios::binary);
    std::vector<int> out;
    int ch = 0;
    while (in && (max_n <= 0 || static_cast<int>(out.size()) < max_n)) {
        ch = in.get();
        if (ch == std::char_traits<char>::eof()) break;
        out.push_back(ch & 0xff);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpus;
    int max_bytes = 65536;
    int table_bits = 16;
    int warm = 8;
    int iters = 32;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--corpus" && i + 1 < argc) {
            corpus = argv[++i];
        } else if (arg == "--max-bytes" && i + 1 < argc) {
            max_bytes = std::stoi(argv[++i]);
        } else if (arg == "--table-bits" && i + 1 < argc) {
            table_bits = std::stoi(argv[++i]);
        } else if (arg == "--warm" && i + 1 < argc) {
            warm = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::fprintf(stderr,
                         "usage: cyphalm_hp_hotpath_profile [--corpus path] [--max-bytes N] "
                         "[--table-bits 16] [--warm 8] [--iters 32]\n");
            return 0;
        }
    }

    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.vocab_size = 256;
    cfg.hp_table_bits = table_bits;
    cypha::cyphalm::apply_hp_production_recipe(cfg);

    cypha::cyphalm::CyphaLMModel model(cfg);
    std::vector<int> ids;
    if (!corpus.empty()) {
        ids = load_bytes(corpus, max_bytes);
    } else {
        ids.reserve(static_cast<std::size_t>(max_bytes));
        for (int i = 0; i < max_bytes; ++i) {
            ids.push_back((i * 7 + 13) & 0xff);
        }
    }

    if (!ids.empty()) {
        model.train_sequence(ids, std::min(static_cast<int>(ids.size()) - 1, 512), 1);
    }

    auto& hp = model.hp_backend();
    nlohmann::json out;
    out["hp_profile"] = "gate24";
    out["hp_table_bits"] = cfg.hp_table_bits;
    out["warm"] = warm;
    out["iters"] = iters;
    out["vm_hwm_kb_construct"] = read_vmhwm_kb();

    for (int i = 0; i < warm; ++i) {
        hp.observe_next_byte(static_cast<std::uint8_t>(ids.empty() ? 65 : ids[i % ids.size()]));
    }
    const auto t_observe = Clock::now();
    for (int i = 0; i < iters; ++i) {
        hp.observe_next_byte(static_cast<std::uint8_t>(ids.empty() ? 66 : ids[i % ids.size()]));
    }
    out["observe_next_byte_us"] = elapsed_us(t_observe, iters);

    for (int i = 0; i < warm; ++i) {
        (void)hp.log_prob_byte(65);
    }
    const auto t_single = Clock::now();
    for (int i = 0; i < iters; ++i) {
        (void)hp.log_prob_byte(static_cast<std::uint8_t>(65 + (i & 7)));
    }
    out["log_prob_byte_us"] = elapsed_us(t_single, iters);

    for (int i = 0; i < std::min(warm, 2); ++i) {
        (void)hp.next_byte_log_probs(cfg.vocab_size);
    }
    const auto t_vocab = Clock::now();
    for (int i = 0; i < std::max(1, iters / 8); ++i) {
        (void)hp.next_byte_log_probs(cfg.vocab_size);
    }
    const int vocab_iters = std::max(1, iters / 8);
    out["next_byte_log_probs256_us"] = elapsed_us(t_vocab, vocab_iters);
    out["next_byte_log_probs256_iters"] = vocab_iters;

    if (!ids.empty()) {
        cypha::cyphalm::CyphaLMModel bpc_model(cfg);
        bpc_model.train_sequence(ids, std::min(static_cast<int>(ids.size()) - 1, 512), 1);
        const double bpc = bpc_model.eval_bpc(ids, std::min(static_cast<int>(ids.size()), 256));
        out["observe_bpc_sample"] = bpc;
    }

    out["vm_hwm_kb_after"] = read_vmhwm_kb();
    out["next_byte_log_probs32_us"] = [&]() {
        for (int i = 0; i < std::min(warm, 2); ++i) {
            (void)hp.next_byte_log_probs(32);
        }
        const auto t32 = Clock::now();
        const int it32 = std::max(1, iters / 2);
        for (int i = 0; i < it32; ++i) {
            (void)hp.next_byte_log_probs(32);
        }
        return elapsed_us(t32, it32);
    }();
    out["next_byte_log_probs32_iters"] = std::max(1, iters / 2);
    out["note"] =
        "observe_next_byte = compress/adapt; next_byte_log_probs* = delta-undo bit-tree (default)";

    std::cout << out.dump(2) << '\n';
    return 0;
}
